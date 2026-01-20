// bench/google-benchmark/BM_spira_matrix_all.cpp
#include <benchmark/benchmark.h>

#include <spira/spira.hpp>
#include <spira/algorithms/spmv.hpp> // adjust include if your spmv lives elsewhere

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

// ------------------------------
// Layout tags (match your aliases)
// ------------------------------
using AoS = spira::layout::tags::aos_tag;
using SoA = spira::layout::tags::soa_tag;

using I = std::size_t;

// ------------------------------
// Helpers
// ------------------------------
template <class V>
static V non_zero_value(std::uint64_t x)
{
    if constexpr (std::is_same_v<V, std::complex<double>>)
        return V{double((x % 13) + 1), double((x % 7))}; // non-zero
    else
        return V((x % 13) + 1); // 1..13
}

template <class V, class Layout>
static void build_matrix_insert_heavy(
    spira::matrix<Layout, I, V> &m,
    const std::vector<I> &rs,
    const std::vector<I> &cs)
{
    m.set_mode(spira::mode::matrix_mode::insert_heavy);
    for (std::size_t k = 0; k < rs.size(); ++k)
        m.add(rs[k], cs[k], non_zero_value<V>(k));
    // leave unflushed for insert-heavy add benches, unless caller flushes
}

static void gen_coords_uniform(std::vector<I> &rs, std::vector<I> &cs, I rows, I cols, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<I> rdist(0, rows - 1);
    std::uniform_int_distribution<I> cdist(0, cols - 1);
    for (std::size_t k = 0; k < rs.size(); ++k)
    {
        rs[k] = rdist(rng);
        cs[k] = cdist(rng);
    }
}

static void gen_coords_hot_row(std::vector<I> &rs, std::vector<I> &cs, I hot_row, I cols, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<I> cdist(0, cols - 1);
    for (std::size_t k = 0; k < rs.size(); ++k)
    {
        rs[k] = hot_row;
        cs[k] = cdist(rng);
    }
}

static void gen_coords_row_scan(std::vector<I> &rs, std::vector<I> &cs, I rows, I cols)
{
    // sequential rows, pseudo-random cols (deterministic)
    for (std::size_t k = 0; k < rs.size(); ++k)
    {
        rs[k] = I(k % rows);
        cs[k] = I((k * 1315423911ULL) % cols);
    }
}

static void add_common_counters(benchmark::State &state)
{
    state.counters["rows"] = double(state.range(1));
    state.counters["cols"] = double(state.range(2));
    state.counters["ops"]  = benchmark::Counter(double(state.iterations()) * double(state.range(0)),
                                                benchmark::Counter::kIsRate);
}

// ------------------------------
// BEST-MODE ADD BENCH
// Inserts in insert_heavy mode (best for heavy random writes)
// ------------------------------
template <class V, class Layout>
static void BM_add_insert_heavy(benchmark::State &state)
{
    const std::size_t n = (std::size_t)state.range(0);
    const I rows = (I)state.range(1);
    const I cols = (I)state.range(2);

    std::vector<I> rs(n), cs(n);
    gen_coords_uniform(rs, cs, rows, cols, 12345);

    for (auto _ : state)
    {
        state.PauseTiming();
        spira::matrix<Layout, I, V> m(rows, cols);
        m.set_mode(spira::mode::matrix_mode::insert_heavy);
        state.ResumeTiming();

        for (std::size_t k = 0; k < n; ++k)
            m.add(rs[k], cs[k], non_zero_value<V>(k));

        benchmark::DoNotOptimize(&m);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)n);
    add_common_counters(state);
}

// ------------------------------
// BEST-MODE GET BENCH (query-ready)
// Build in insert_heavy, then switch to balanced and flush,
// then time get() (best way to benchmark slab-only/steady reads)
// ------------------------------
enum class GetPattern { Uniform, HotRow, RowScan };

template <class V, class Layout, GetPattern P>
static void BM_get_query_ready(benchmark::State &state)
{
    const std::size_t n = (std::size_t)state.range(0);
    const I rows = (I)state.range(1);
    const I cols = (I)state.range(2);

    std::vector<I> rs(n), cs(n);
    if constexpr (P == GetPattern::Uniform)  gen_coords_uniform(rs, cs, rows, cols, 12345);
    if constexpr (P == GetPattern::HotRow)   gen_coords_hot_row(rs, cs, /*hot_row=*/0, cols, 12345);
    if constexpr (P == GetPattern::RowScan)  gen_coords_row_scan(rs, cs, rows, cols);

    // Build once
    spira::matrix<Layout, I, V> m(rows, cols);
    build_matrix_insert_heavy(m, rs, cs);

    // Make it query-ready (slab canonical)
    m.set_mode(spira::mode::matrix_mode::balanced);
    m.flush();

    for (auto _ : state)
    {
        for (std::size_t k = 0; k < n; ++k)
            benchmark::DoNotOptimize(m.get(rs[k], cs[k]));

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)n);
    add_common_counters(state);
}

// ------------------------------
// SPmV BENCH (best mode = spmv)
// Build in insert_heavy, flush, set spmv mode, then time spmv()
// We report nnz-rate as well (after finalization)
// ------------------------------
enum class SpmvPattern { Diagonal, RandomKPerRow };

template <class V, class Layout, SpmvPattern P>
static void BM_spmv_best_mode(benchmark::State &state)
{
    const I rows = (I)state.range(0);
    const I cols = (I)state.range(1);
    const I k_per_row = (I)state.range(2); // used for RandomKPerRow

    spira::matrix<Layout, I, V> m(rows, cols);
    m.set_mode(spira::mode::matrix_mode::insert_heavy);

    std::mt19937_64 rng(12345);

    if constexpr (P == SpmvPattern::Diagonal)
    {
        const I n = (rows < cols) ? rows : cols;
        for (I i = 0; i < n; ++i)
            m.add(i, i, non_zero_value<V>(i));
    }
    else if constexpr (P == SpmvPattern::RandomKPerRow)
    {
        std::uniform_int_distribution<I> cdist(0, cols - 1);
        for (I r = 0; r < rows; ++r)
        {
            for (I t = 0; t < k_per_row; ++t)
            {
                const I c = cdist(rng);
                m.add(r, c, non_zero_value<V>(std::uint64_t(r) * 1315423911ULL + t));
            }
        }
    }

    // Finalize for spmv
    m.set_mode(spira::mode::matrix_mode::spmv);
    m.flush();

    // Vectors
    std::vector<V> x(cols), y(rows);
    for (I i = 0; i < cols; ++i) x[i] = non_zero_value<V>(i + 999);
    for (I i = 0; i < rows; ++i) y[i] = V{};

    // One-time nnz measurement (can be expensive; do it once)
    const std::size_t nnz = m.nnz();

    for (auto _ : state)
    {
        spira::algorithms::spmv(m, x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    // items processed = nnz * iterations
    state.SetItemsProcessed((int64_t)state.iterations() * (int64_t)nnz);
    state.counters["rows"] = double(rows);
    state.counters["cols"] = double(cols);
    state.counters["nnz"]  = double(nnz);
    state.counters["nnz_per_s"] =
        benchmark::Counter(double(state.iterations()) * double(nnz), benchmark::Counter::kIsRate);
}

// ------------------------------
// Registration helpers
// ------------------------------
static void ApplyGetScales(benchmark::internal::Benchmark *b)
{
    // n_coords, rows, cols
    // (keep n_coords big; rows/cols big; feel free to tune)
    b->Args({1'000'000, 50'000, 50'000});
    b->Args({2'000'000, 100'000, 100'000});
    b->Args({8'000'000, 500'000, 500'000});
}

static void ApplyAddScales(benchmark::internal::Benchmark *b)
{
    // n_inserts, rows, cols
    b->Args({1'000'000, 50'000, 50'000});
    b->Args({2'000'000, 100'000, 100'000});
    b->Args({8'000'000, 500'000, 500'000});
}

static void ApplySpmvScales(benchmark::internal::Benchmark *b)
{
    // rows, cols, k_per_row
    b->Args({50'000, 50'000, 16});
    b->Args({100'000, 100'000, 16});
    b->Args({500'000, 500'000, 16}); // ~8M nnz if no duplicates per row
}

// ------------------------------
// ADD benches (best = insert_heavy)
// ------------------------------
BENCHMARK_TEMPLATE(BM_add_insert_heavy, float, SoA)->Apply(ApplyAddScales)->Name("add_insert_heavy/soa/float");
BENCHMARK_TEMPLATE(BM_add_insert_heavy, float, AoS)->Apply(ApplyAddScales)->Name("add_insert_heavy/aos/float");
BENCHMARK_TEMPLATE(BM_add_insert_heavy, double, SoA)->Apply(ApplyAddScales)->Name("add_insert_heavy/soa/double");
BENCHMARK_TEMPLATE(BM_add_insert_heavy, double, AoS)->Apply(ApplyAddScales)->Name("add_insert_heavy/aos/double");
BENCHMARK_TEMPLATE(BM_add_insert_heavy, std::complex<double>, SoA)->Apply(ApplyAddScales)->Name("add_insert_heavy/soa/cdouble");
BENCHMARK_TEMPLATE(BM_add_insert_heavy, std::complex<double>, AoS)->Apply(ApplyAddScales)->Name("add_insert_heavy/aos/cdouble");

// ------------------------------
// GET benches (best = query-ready slab; balanced+flush)
// Patterns: uniform random, hot-row, row-scan
// ------------------------------
BENCHMARK_TEMPLATE(BM_get_query_ready, float, SoA, GetPattern::Uniform)->Apply(ApplyGetScales)->Name("get_query_ready/uniform/soa/float");
BENCHMARK_TEMPLATE(BM_get_query_ready, float, AoS, GetPattern::Uniform)->Apply(ApplyGetScales)->Name("get_query_ready/uniform/aos/float");
BENCHMARK_TEMPLATE(BM_get_query_ready, float, SoA, GetPattern::HotRow)->Apply(ApplyGetScales)->Name("get_query_ready/hotrow/soa/float");
BENCHMARK_TEMPLATE(BM_get_query_ready, float, AoS, GetPattern::HotRow)->Apply(ApplyGetScales)->Name("get_query_ready/hotrow/aos/float");
BENCHMARK_TEMPLATE(BM_get_query_ready, float, SoA, GetPattern::RowScan)->Apply(ApplyGetScales)->Name("get_query_ready/rowscan/soa/float");
BENCHMARK_TEMPLATE(BM_get_query_ready, float, AoS, GetPattern::RowScan)->Apply(ApplyGetScales)->Name("get_query_ready/rowscan/aos/float");

BENCHMARK_TEMPLATE(BM_get_query_ready, double, SoA, GetPattern::Uniform)->Apply(ApplyGetScales)->Name("get_query_ready/uniform/soa/double");
BENCHMARK_TEMPLATE(BM_get_query_ready, double, AoS, GetPattern::Uniform)->Apply(ApplyGetScales)->Name("get_query_ready/uniform/aos/double");

// ------------------------------
// SPMV benches (best = spmv mode)
// Patterns: diagonal, random k-per-row
// ------------------------------
BENCHMARK_TEMPLATE(BM_spmv_best_mode, double, SoA, SpmvPattern::Diagonal)->Args({500'000, 500'000, 0})->Name("spmv/diag/soa/double");
BENCHMARK_TEMPLATE(BM_spmv_best_mode, double, AoS, SpmvPattern::Diagonal)->Args({500'000, 500'000, 0})->Name("spmv/diag/aos/double");

BENCHMARK_TEMPLATE(BM_spmv_best_mode, double, SoA, SpmvPattern::RandomKPerRow)->Apply(ApplySpmvScales)->Name("spmv/kperrow/soa/double");
BENCHMARK_TEMPLATE(BM_spmv_best_mode, double, AoS, SpmvPattern::RandomKPerRow)->Apply(ApplySpmvScales)->Name("spmv/kperrow/aos/double");

BENCHMARK_MAIN();
