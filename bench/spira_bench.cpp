// ============================================================================
// Spira Stage 2 (SIMD Kernels & Hardware Detection) — bench/spira_bench.cpp
//
// SpMV only: insertion design is unchanged from Stage 1.
// The runtime kernel dispatch (AVX-512 > AVX2+FMA > SSE4.2 > NEON > scalar)
// is installed by dispatch() at program start.  SoA + uint32_t + double
// selects the SIMD double overload of spira::algorithms::spmv.
//
// Matrix   : 10 000 × 10 000, SoA layout, double precision.
// Densities: range(0) = nnz_per_row — 10 (0.1 %), 100 (1 %), 1000 (10 %).
// Patterns : range(1) = 0 random | 1 strided.
// ============================================================================
#include <benchmark/benchmark.h>
#include <spira/spira.hpp>

#include <algorithm>
#include <random>
#include <vector>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
static constexpr size_t   N     = 10'000;
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

// SoA + double → picks the sparse_dot_double SIMD overload at runtime
using L   = spira::layout::tags::soa_tag;
using Mat = spira::matrix<L, uint32_t, double>;

struct Triple { uint32_t row, col; double val; };

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

static void fill_and_flush(Mat &mat, const std::vector<Triple> &triples)
{
    mat.set_mode(spira::mode::matrix_mode::insert_heavy);
    for (const auto &t : triples)
        mat.insert(t.row, t.col, t.val);
    mat.flush();
}

// ============================================================================
// SpMV — SIMD kernel active for soa_tag + uint32_t + double
// ============================================================================
class SpMVFixture : public benchmark::Fixture
{
public:
    std::unique_ptr<Mat> mat;
    std::vector<double>  x, y;

    void SetUp(const benchmark::State &state) override
    {
        const size_t nnz = static_cast<size_t>(state.range(0));
        const bool   rnd = (state.range(1) == 0);
        auto full = make_full_triples(nnz, rnd);
        mat       = std::make_unique<Mat>(N, N);
        fill_and_flush(*mat, full);
        mat->set_mode(spira::mode::matrix_mode::spmv);

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

        spira::algorithms::spmv(*mat, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    const size_t nnz     = static_cast<size_t>(state.range(0));
    const size_t nnz_tot = N * nnz;
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

BENCHMARK_REGISTER_F(SpMVFixture, SpMV)
    ->Args({10,   0})->Args({100,  0})->Args({1000, 0})
    ->Args({10,   1})->Args({100,  1})->Args({1000, 1})
    ->ArgNames({"nnz_per_row", "pattern"})
    ->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
