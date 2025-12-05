#include "config.hpp"

template <class V, class Layout>
static void BM_streamGetOperation(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0));
    const std::size_t rows = static_cast<std::size_t>(state.range(1));
    const std::size_t cols = static_cast<std::size_t>(state.range(2));

    spira::matrix<Layout, I, V> m(rows, cols);

    // Fill the matrix with something (dense zero for now; adapt if you want sparsity)
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            m.add(i, j, V{}); // V{} == 0 for float/double/complex
        }
    }

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    std::vector<I> xs;
    std::vector<I> ys;
    xs.reserve(n_coordinates);
    ys.reserve(n_coordinates);

    for (std::size_t k = 0; k < n_coordinates; ++k)
    {
        xs.push_back(row_dist(rng));
        ys.push_back(col_dist(rng));
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    for (auto _ : state)
    {
        for (std::size_t k = 0; k < n_coordinates; ++k)
        {
            benchmark::DoNotOptimize(m.get(xs[k], ys[k]));
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n_coordinates));
}

// --- registrations: GET, weak + strong, AoS + SoA, float/double/complex -----

// SOA weak
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, SoA)
    ->Apply(weakScaling)
    ->Name("soa-get-float-weak");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, SoA)
    ->Apply(weakScaling)
    ->Name("soa-get-double-weak");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, SoA)
    ->Apply(weakScaling)
    ->Name("soa-get-complex-weak");

// AOS weak
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, AoS)
    ->Apply(weakScaling)
    ->Name("aos-get-float-weak");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, AoS)
    ->Apply(weakScaling)
    ->Name("aos-get-double-weak");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, AoS)
    ->Apply(weakScaling)
    ->Name("aos-get-complex-weak");

// SOA strong
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, SoA)
    ->Apply(strongScaling)
    ->Name("soa-get-float-strong");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, SoA)
    ->Apply(strongScaling)
    ->Name("soa-get-double-strong");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, SoA)
    ->Apply(strongScaling)
    ->Name("soa-get-complex-strong");

// AOS strong
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, AoS)
    ->Apply(strongScaling)
    ->Name("aos-get-float-strong");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, AoS)
    ->Apply(strongScaling)
    ->Name("aos-get-double-strong");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, AoS)
    ->Apply(strongScaling)
    ->Name("aos-get-complex-strong");