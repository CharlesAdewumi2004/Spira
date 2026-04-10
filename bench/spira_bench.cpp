// ============================================================================
// Spira Stage 4 (Multi-Threaded) — bench/spira_bench.cpp
//
// Benchmarks:
//   Insert  : parallel_fill() → lock(); 256 entries pre-partitioned by
//             thread.  Each thread merges only its own partition.
//             Cold LLC before each timed section.
//   SpMV    : parallel::algorithms::spmv on a locked matrix; each thread
//             covers its own CSR partition independently.  Cold LLC.
//   Scaling : SpMV at 1 % density, random pattern, varying thread count
//             (1 → 2 → 4 → 8 → hardware_concurrency).
//
// Matrix   : 10 000 × 10 000, SoA layout, double, hash_map_buffer.
// Densities: range(0) = nnz_per_row — 10 (0.1 %), 100 (1 %), 1000 (10 %).
// Patterns : range(1) = 0 random | 1 strided.
// ============================================================================
#include <benchmark/benchmark.h>
#include <spira/spira.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/parallel/parallel_matrix.hpp>
#include <spira/parallel/algorithms/spmv.hpp>

#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
static constexpr size_t   N     = 10'000;
static constexpr size_t   BATCH = 256;
static constexpr unsigned SEED  = 42;
// ---------------------------------------------------------------------------

static void flush_cache()
{
    static const size_t flush_size = [] {
#if defined(_WIN32)
        size_t llc = 32UL * 1024 * 1024;
        DWORD len = 0;
        GetLogicalProcessorInformation(nullptr, &len);
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
            len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buf.data(), &len))
            for (auto &e : buf)
                if (e.Relationship == RelationCache && e.Cache.Level == 3)
                    llc = e.Cache.Size;
        return llc;
#else
        long s = sysconf(_SC_LEVEL3_CACHE_SIZE);
        return static_cast<size_t>(s > 0 ? s : 32L * 1024 * 1024);
#endif
    }();
    static std::vector<char> arena(flush_size, 1);
    volatile char sink = 0;
    for (size_t i = 0; i < flush_size; i += 64)
        sink ^= arena[i];
    (void)sink;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using L  = spira::layout::tags::soa_tag;
using BT = spira::buffer::tags::hash_map_buffer;
using PM = spira::parallel::parallel_matrix<L, uint32_t, double, BT>;

struct Triple { uint32_t row, col; double val; };

// ---------------------------------------------------------------------------
// Triple generation
// ---------------------------------------------------------------------------
static std::vector<Triple> make_full_triples(size_t nnz_per_row, bool rnd)
{
    std::vector<Triple> v;
    v.reserve(N * nnz_per_row);
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> vd(0.0, 1.0);
    if (rnd) {
        std::uniform_int_distribution<uint32_t> cd(0, static_cast<uint32_t>(N - 1));
        for (size_t r = 0; r < N; ++r)
            for (size_t k = 0; k < nnz_per_row; ++k)
                v.push_back({static_cast<uint32_t>(r), cd(rng), vd(rng)});
    } else {
        const size_t stride = std::max<size_t>(N / nnz_per_row, 1);
        for (size_t r = 0; r < N; ++r)
            for (size_t k = 0; k < nnz_per_row; ++k)
                v.push_back({static_cast<uint32_t>(r),
                             static_cast<uint32_t>((k * stride) % N), vd(rng)});
    }
    return v;
}

static std::vector<Triple> make_batch(size_t nnz_per_row, bool rnd)
{
    std::vector<Triple> v;
    v.reserve(BATCH);
    std::mt19937 rng(SEED ^ 0xDEADBEEFu);
    std::uniform_real_distribution<double> vd(0.0, 1.0);
    if (rnd) {
        std::uniform_int_distribution<uint32_t> rd(0, static_cast<uint32_t>(N - 1));
        std::uniform_int_distribution<uint32_t> cd(0, static_cast<uint32_t>(N - 1));
        for (size_t k = 0; k < BATCH; ++k)
            v.push_back({rd(rng), cd(rng), vd(rng)});
    } else {
        const size_t rs = std::max<size_t>(N / BATCH, 1);
        const size_t cs = std::max<size_t>(N / nnz_per_row, 1);
        for (size_t k = 0; k < BATCH; ++k)
            v.push_back({static_cast<uint32_t>((k * rs) % N),
                         static_cast<uint32_t>((k * cs) % N), vd(rng)});
    }
    return v;
}

// ---------------------------------------------------------------------------
// Partition-aware distribution helpers
// ---------------------------------------------------------------------------

// Build a by-thread partitioning of triples given a constructed (open) matrix.
// Uses the actual partition boundaries from partition_at() — safe to call
// before or after lock() since boundaries don't change unless rebalance() is
// called.
static std::vector<std::vector<Triple>>
distribute_by_thread(PM &mat, const std::vector<Triple> &triples, size_t n_threads)
{
    // Collect partition start rows (sorted ascending)
    std::vector<size_t> starts(n_threads);
    for (size_t t = 0; t < n_threads; ++t)
        starts[t] = mat.partition_at(t).row_start;

    std::vector<std::vector<Triple>> by_thread(n_threads);
    for (const auto &tri : triples) {
        // upper_bound gives the first start strictly greater than tri.row;
        // back off one to get the owning partition.
        auto it  = std::upper_bound(starts.begin(), starts.end(),
                                    static_cast<size_t>(tri.row));
        size_t t = static_cast<size_t>(std::distance(starts.begin(), it)) - 1;
        by_thread[t].push_back(tri);
    }
    return by_thread;
}

// Fill matrix with parallel_fill then lock
static void fill_and_lock(PM &mat,
                          const std::vector<std::vector<Triple>> &by_thread)
{
    mat.parallel_fill([&](auto &rows, size_t r_start, size_t, size_t tid) {
        for (const auto &t : by_thread[tid])
            rows[static_cast<size_t>(t.row) - r_start].insert(t.col, t.val);
    });
    mat.lock();
}

// ============================================================================
// Insertion — open() + parallel_fill(256 entries) + lock()
// ============================================================================
class InsertFixture : public benchmark::Fixture
{
public:
    std::unique_ptr<PM>              mat;
    std::vector<std::vector<Triple>> batch_by_thread;
    size_t                           n_threads{1};

    void SetUp(const benchmark::State &state) override
    {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t nnz = static_cast<size_t>(state.range(0));
        const bool   rnd = (state.range(1) == 0);

        mat = std::make_unique<PM>(N, N, n_threads);

        auto full = make_full_triples(nnz, rnd);
        auto full_by_thread = distribute_by_thread(*mat, full, n_threads);
        fill_and_lock(*mat, full_by_thread);   // locked

        auto batch      = make_batch(nnz, rnd);
        batch_by_thread = distribute_by_thread(*mat, batch, n_threads);
    }

    void TearDown(const benchmark::State &) override { mat.reset(); }
};

BENCHMARK_DEFINE_F(InsertFixture, Insert)(benchmark::State &state)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        mat->open();
        mat->parallel_fill([&](auto &rows, size_t r_start, size_t, size_t tid) {
            for (const auto &t : batch_by_thread[tid])
                rows[static_cast<size_t>(t.row) - r_start].insert(t.col, t.val);
        });
        mat->lock();
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BATCH));
}

BENCHMARK_REGISTER_F(InsertFixture, Insert)
    ->Args({10,   0})->Args({100,  0})->Args({1000, 0})
    ->Args({10,   1})->Args({100,  1})->Args({1000, 1})
    ->ArgNames({"nnz_per_row", "pattern"})
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// SpMV — parallel::algorithms::spmv; cold LLC each iteration
// ============================================================================
class SpMVFixture : public benchmark::Fixture
{
public:
    std::unique_ptr<PM> mat;
    std::vector<double> x, y;
    size_t              n_threads{1};

    void SetUp(const benchmark::State &state) override
    {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t nnz = static_cast<size_t>(state.range(0));
        const bool   rnd = (state.range(1) == 0);

        mat = std::make_unique<PM>(N, N, n_threads);

        auto full = make_full_triples(nnz, rnd);
        auto by_t = distribute_by_thread(*mat, full, n_threads);
        fill_and_lock(*mat, by_t);

        std::mt19937 rng(SEED ^ 0xC0FFEEu);
        std::uniform_real_distribution<double> vd(0.0, 1.0);
        x.resize(N);
        y.assign(N, 0.0);
        for (auto &v : x) v = vd(rng);
    }

    void TearDown(const benchmark::State &) override { mat.reset(); }
};

BENCHMARK_DEFINE_F(SpMVFixture, SpMV)(benchmark::State &state)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        spira::parallel::algorithms::spmv(*mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    const size_t nnz = static_cast<size_t>(state.range(0));
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(N) * static_cast<int64_t>(nnz) * 2);
}

BENCHMARK_REGISTER_F(SpMVFixture, SpMV)
    ->Args({10,   0})->Args({100,  0})->Args({1000, 0})
    ->Args({10,   1})->Args({100,  1})->Args({1000, 1})
    ->ArgNames({"nnz_per_row", "pattern"})
    ->Unit(benchmark::kNanosecond);

// ============================================================================
// Thread scaling — SpMV at 1 % density, random pattern
// range(0) = n_threads
// ============================================================================
class ThreadScalingFixture : public benchmark::Fixture
{
public:
    std::unique_ptr<PM> mat;
    std::vector<double> x, y;

    void SetUp(const benchmark::State &state) override
    {
        const size_t n_threads = static_cast<size_t>(state.range(0));

        mat = std::make_unique<PM>(N, N, n_threads);

        auto full = make_full_triples(100, /*rnd=*/true);  // 1 % density
        auto by_t = distribute_by_thread(*mat, full, n_threads);
        fill_and_lock(*mat, by_t);

        std::mt19937 rng(SEED ^ 0xC0FFEEu);
        std::uniform_real_distribution<double> vd(0.0, 1.0);
        x.resize(N);
        y.assign(N, 0.0);
        for (auto &v : x) v = vd(rng);
    }

    void TearDown(const benchmark::State &) override { mat.reset(); }
};

BENCHMARK_DEFINE_F(ThreadScalingFixture, SpMV_Scaling)(benchmark::State &state)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        spira::parallel::algorithms::spmv(*mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    // 1 % density = 100 nnz/row; 2 flops per non-zero
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(N) * 100 * 2);
}

BENCHMARK_REGISTER_F(ThreadScalingFixture, SpMV_Scaling)
    ->Apply([](benchmark::internal::Benchmark *b) {
        const int hw = static_cast<int>(
            std::max(1u, std::thread::hardware_concurrency()));
        for (int t : {1, 2, 4, 8})
            if (t <= hw) b->Arg(t);
        if (hw > 8) b->Arg(hw);
    })
    ->ArgName("n_threads")
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
