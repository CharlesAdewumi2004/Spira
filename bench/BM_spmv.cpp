#include "config.hpp"

template <class V, class Layout>
static void BM_spmv(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0)); // nnz
    const std::size_t rows = static_cast<std::size_t>(state.range(1));
    const std::size_t cols = static_cast<std::size_t>(state.range(2));

    spira::matrix<Layout, I, V> m(rows, cols);
    std::vector<V> x;
    std::vector<V> y(rows, V{});

    x.reserve(cols);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> num_dist(minMaxtrixValue, maxMatrixValue);

    for (std::size_t i = 0; i < cols; ++i)
    {
        x.push_back(static_cast<V>(num_dist(rng)));
    }

    for (std::size_t k = 0; k < n_coordinates; ++k)
    {
        m.add(row_dist(rng), col_dist(rng), static_cast<V>(num_dist(rng)));
    }

    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());

    for (auto _ : state)
    {
        spira::algorithms::spmv(m, x, y);
        benchmark::ClobberMemory();
    }

    // Each SPMV touches ~2 * nnz scalar ops; we at least record nnz
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n_coordinates));
}

// --- registrations: SPMV, weak + strong, AoS + SoA, float/double/complex -----

// AOS weak
BENCHMARK_TEMPLATE(BM_spmv, float, AoS)
    ->Apply(weakScaling)
    ->Name("aos-spmv-float-weak");
BENCHMARK_TEMPLATE(BM_spmv, double, AoS)
    ->Apply(weakScaling)
    ->Name("aos-spmv-double-weak");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, AoS)
    ->Apply(weakScaling)
    ->Name("aos-spmv-complex-weak");

// SOA weak
BENCHMARK_TEMPLATE(BM_spmv, float, SoA)
    ->Apply(weakScaling)
    ->Name("soa-spmv-float-weak");
BENCHMARK_TEMPLATE(BM_spmv, double, SoA)
    ->Apply(weakScaling)
    ->Name("soa-spmv-double-weak");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, SoA)
    ->Apply(weakScaling)
    ->Name("soa-spmv-complex-weak");

// AOS strong
BENCHMARK_TEMPLATE(BM_spmv, float, AoS)
    ->Apply(strongScaling)
    ->Name("aos-spmv-float-strong");
BENCHMARK_TEMPLATE(BM_spmv, double, AoS)
    ->Apply(strongScaling)
    ->Name("aos-spmv-double-strong");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, AoS)
    ->Apply(strongScaling)
    ->Name("aos-spmv-complex-strong");

// SOA strong
BENCHMARK_TEMPLATE(BM_spmv, float, SoA)
    ->Apply(strongScaling)
    ->Name("soa-spmv-float-strong");
BENCHMARK_TEMPLATE(BM_spmv, double, SoA)
    ->Apply(strongScaling)
    ->Name("soa-spmv-double-strong");
BENCHMARK_TEMPLATE(BM_spmv, std::complex<double>, SoA)
    ->Apply(strongScaling)
    ->Name("soa-spmv-complex-strong");