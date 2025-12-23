#include <benchmark/benchmark.h>
#include "config.hpp"
#include "spira/spira.hpp"

#include <random>
#include <vector>
#include <complex>

template <class V, class Layout>
static void BM_streamGetOperation(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0));
    const std::size_t rows          = static_cast<std::size_t>(state.range(1));
    const std::size_t cols          = static_cast<std::size_t>(state.range(2));

    spira::matrix<Layout, I, V> m(rows, cols);

    // RNG for coordinates
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    // Pre-generate coordinates
    std::vector<I> xs(n_coordinates);
    std::vector<I> ys(n_coordinates);

    for (std::size_t k = 0; k < n_coordinates; ++k)
    {
        xs[k] = row_dist(rng);
        ys[k] = col_dist(rng);
    }

    // Insert only those points
    for (std::size_t k = 0; k < n_coordinates; ++k)
        m.add(xs[k], ys[k], V{});

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    for (auto _ : state)
    {
        for (std::size_t k = 0; k < n_coordinates; ++k)
            benchmark::DoNotOptimize(m.get(xs[k], ys[k]));

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_coordinates)
    );
}

// ------------------------------------------------------------
// REGISTRATION: problemSizeScaling + sparsityScaling
// ------------------------------------------------------------

// SOA problemSizeScaling
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-get-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-get-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-get-complex-problemSizeScaling");

// AOS problemSizeScaling
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-get-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-get-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-get-complex-problemSizeScaling");

// SOA sparsityScaling
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-get-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-get-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-get-complex-sparsityScaling");

// AOS sparsityScaling
BENCHMARK_TEMPLATE(BM_streamGetOperation, float, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-get-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, double, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-get-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamGetOperation, std::complex<double>, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-get-complex-sparsityScaling");
