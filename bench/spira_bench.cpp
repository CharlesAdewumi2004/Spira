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
// Both AoS and SoA layouts are benchmarked across all three benchmark types.
// SoA picks the SIMD kernel; AoS runs scalar — showing the SIMD gain on top
// of the threading gain.
//
// Matrix   : 10 000 × 10 000, double, hash_map_buffer.
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
// Layout aliases.
// SoA + uint32_t + double → SIMD kernel path.
// AoS + uint32_t + double → scalar fallback path.
// ---------------------------------------------------------------------------
using AoS = spira::layout::tags::aos_tag;
using SoA = spira::layout::tags::soa_tag;
using BT  = spira::buffer::tags::hash_map_buffer;

template <typename LayoutTag>
using PM = spira::parallel::parallel_matrix<LayoutTag, uint32_t, double, BT>;

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
template <typename LayoutTag>
static std::vector<std::vector<Triple>>
distribute_by_thread(PM<LayoutTag> &mat,
                     const std::vector<Triple> &triples,
                     size_t n_threads)
{
    std::vector<size_t> starts(n_threads);
    for (size_t t = 0; t < n_threads; ++t)
        starts[t] = mat.partition_at(t).row_start;

    std::vector<std::vector<Triple>> by_thread(n_threads);
    for (const auto &tri : triples) {
        auto it  = std::upper_bound(starts.begin(), starts.end(),
                                    static_cast<size_t>(tri.row));
        size_t t = static_cast<size_t>(std::distance(starts.begin(), it)) - 1;
        by_thread[t].push_back(tri);
    }
    return by_thread;
}

template <typename LayoutTag>
static void fill_and_lock(PM<LayoutTag> &mat,
                          const std::vector<std::vector<Triple>> &by_thread)
{
    mat.parallel_fill([&](auto &rows, size_t r_start, size_t, size_t tid) {
        for (const auto &t : by_thread[tid])
            rows[static_cast<size_t>(t.row) - r_start].insert(t.col, t.val);
    });
    mat.lock();
}

// ============================================================================
// Insertion — shared logic, templated on layout
// ============================================================================
template <typename LayoutTag>
class InsertFixtureBase : public benchmark::Fixture
{
public:
    std::unique_ptr<PM<LayoutTag>>   mat;
    std::vector<std::vector<Triple>> batch_by_thread;
    size_t                           n_threads{1};

    void SetUp(const benchmark::State &state) override
    {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t nnz = static_cast<size_t>(state.range(0));
        const bool   rnd = (state.range(1) == 0);

        mat = std::make_unique<PM<LayoutTag>>(N, N, n_threads);

        auto full         = make_full_triples(nnz, rnd);
        auto full_by_thread = distribute_by_thread(*mat, full, n_threads);
        fill_and_lock(*mat, full_by_thread);

        auto batch      = make_batch(nnz, rnd);
        batch_by_thread = distribute_by_thread(*mat, batch, n_threads);
    }

    void TearDown(const benchmark::State &) override { mat.reset(); }
};

class InsertFixture_SoA : public InsertFixtureBase<SoA> {};
class InsertFixture_AoS : public InsertFixtureBase<AoS> {};

template <typename LayoutTag>
static void run_insert(benchmark::State &state,
                       PM<LayoutTag> &mat,
                       const std::vector<std::vector<Triple>> &batch_by_thread)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        mat.open();
        mat.parallel_fill([&](auto &rows, size_t r_start, size_t, size_t tid) {
            for (const auto &t : batch_by_thread[tid])
                rows[static_cast<size_t>(t.row) - r_start].insert(t.col, t.val);
        });
        mat.lock();
    }
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(BATCH));
    // Each insert writes a (col, val) pair to the buffer (4 + 8 = 12 B)
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(BATCH) *
        static_cast<int64_t>(sizeof(uint32_t) + sizeof(double)));
}

BENCHMARK_DEFINE_F(InsertFixture_SoA, Insert)(benchmark::State &state)
{
    run_insert(state, *mat, batch_by_thread);
}
BENCHMARK_DEFINE_F(InsertFixture_AoS, Insert)(benchmark::State &state)
{
    run_insert(state, *mat, batch_by_thread);
}

#define REGISTER_INSERT(FIXTURE)               \
    BENCHMARK_REGISTER_F(FIXTURE, Insert)      \
        ->Args({10,   0})                      \
        ->Args({100,  0})                      \
        ->Args({1000, 0})                      \
        ->Args({10,   1})                      \
        ->Args({100,  1})                      \
        ->Args({1000, 1})                      \
        ->ArgNames({"nnz_per_row", "pattern"}) \
        ->Unit(benchmark::kNanosecond)

REGISTER_INSERT(InsertFixture_SoA);
REGISTER_INSERT(InsertFixture_AoS);

// ============================================================================
// SpMV — shared logic, templated on layout
// SoA: parallel SIMD kernel.  AoS: parallel scalar fallback.
// ============================================================================
template <typename LayoutTag>
class SpMVFixtureBase : public benchmark::Fixture
{
public:
    std::unique_ptr<PM<LayoutTag>> mat;
    std::vector<double>            x, y;
    size_t                         n_threads{1};

    void SetUp(const benchmark::State &state) override
    {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t nnz = static_cast<size_t>(state.range(0));
        const bool   rnd = (state.range(1) == 0);

        mat = std::make_unique<PM<LayoutTag>>(N, N, n_threads);

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

class SpMVFixture_SoA : public SpMVFixtureBase<SoA> {};
class SpMVFixture_AoS : public SpMVFixtureBase<AoS> {};

template <typename LayoutTag>
static void run_spmv(benchmark::State &state,
                     PM<LayoutTag> &mat,
                     std::vector<double> &x,
                     std::vector<double> &y,
                     size_t nnz_per_row)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        spira::parallel::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    const size_t nnz_tot = N * nnz_per_row;
    // FLOPs: 2 per non-zero (multiply + accumulate)
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(nnz_tot) * 2);
    // Memory bandwidth: CSR values (8B) + col indices (4B) + row offsets (8B)
    //                   + input vector x (8B) + output vector y (8B)
    const int64_t bytes_per_iter =
        static_cast<int64_t>(nnz_tot) * (sizeof(double) + sizeof(uint32_t)) +
        static_cast<int64_t>(N + 1)   *  sizeof(size_t) +
        static_cast<int64_t>(N)       * (sizeof(double) + sizeof(double));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * bytes_per_iter);
}

BENCHMARK_DEFINE_F(SpMVFixture_SoA, SpMV)(benchmark::State &state)
{
    run_spmv(state, *mat, x, y, static_cast<size_t>(state.range(0)));
}
BENCHMARK_DEFINE_F(SpMVFixture_AoS, SpMV)(benchmark::State &state)
{
    run_spmv(state, *mat, x, y, static_cast<size_t>(state.range(0)));
}

#define REGISTER_SPMV(FIXTURE)                 \
    BENCHMARK_REGISTER_F(FIXTURE, SpMV)        \
        ->Args({10,   0})                      \
        ->Args({100,  0})                      \
        ->Args({1000, 0})                      \
        ->Args({10,   1})                      \
        ->Args({100,  1})                      \
        ->Args({1000, 1})                      \
        ->ArgNames({"nnz_per_row", "pattern"}) \
        ->Unit(benchmark::kNanosecond)

REGISTER_SPMV(SpMVFixture_SoA);
REGISTER_SPMV(SpMVFixture_AoS);

// ============================================================================
// Thread scaling — SpMV at 1 % density, random pattern, both layouts
// range(0) = n_threads
// ============================================================================
template <typename LayoutTag>
class ThreadScalingFixtureBase : public benchmark::Fixture
{
public:
    std::unique_ptr<PM<LayoutTag>> mat;
    std::vector<double>            x, y;

    void SetUp(const benchmark::State &state) override
    {
        const size_t n_threads = static_cast<size_t>(state.range(0));

        mat = std::make_unique<PM<LayoutTag>>(N, N, n_threads);

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

class ThreadScalingFixture_SoA : public ThreadScalingFixtureBase<SoA> {};
class ThreadScalingFixture_AoS : public ThreadScalingFixtureBase<AoS> {};

template <typename LayoutTag>
static void run_scaling(benchmark::State &state,
                        PM<LayoutTag> &mat,
                        std::vector<double> &x,
                        std::vector<double> &y)
{
    for (auto _ : state) {
        state.PauseTiming();
        flush_cache();
        state.ResumeTiming();

        spira::parallel::algorithms::spmv(mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    // 1 % density = 100 nnz/row; 2 flops per non-zero
    static constexpr size_t nnz_tot = N * 100;
    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(nnz_tot) * 2);
    const int64_t bytes =
        static_cast<int64_t>(nnz_tot) * (sizeof(double) + sizeof(uint32_t)) +
        static_cast<int64_t>(N + 1)   *  sizeof(size_t) +
        static_cast<int64_t>(N)       * (sizeof(double) + sizeof(double));
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) * bytes);
}

BENCHMARK_DEFINE_F(ThreadScalingFixture_SoA, SpMV_Scaling)(benchmark::State &state)
{
    run_scaling(state, *mat, x, y);
}
BENCHMARK_DEFINE_F(ThreadScalingFixture_AoS, SpMV_Scaling)(benchmark::State &state)
{
    run_scaling(state, *mat, x, y);
}

#define REGISTER_SCALING(FIXTURE)                                        \
    BENCHMARK_REGISTER_F(FIXTURE, SpMV_Scaling)                          \
        ->Apply([](benchmark::internal::Benchmark *b) {                  \
            const int hw = static_cast<int>(                             \
                std::max(1u, std::thread::hardware_concurrency()));       \
            for (int t : {1, 2, 4, 8})                                   \
                if (t <= hw) b->Arg(t);                                  \
            if (hw > 8) b->Arg(hw);                                      \
        })                                                               \
        ->ArgName("n_threads")                                           \
        ->Unit(benchmark::kNanosecond)

REGISTER_SCALING(ThreadScalingFixture_SoA);
REGISTER_SCALING(ThreadScalingFixture_AoS);

BENCHMARK_MAIN();
