#include <benchmark/benchmark.h>
#include "config.hpp"
#include "spira/spira.hpp"

#include <random>
#include <vector>
#include <complex>

template <class V, class Layout>
static void BM_streamInsertElements(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0));
    const std::size_t rows          = static_cast<std::size_t>(state.range(1));
    const std::size_t cols          = static_cast<std::size_t>(state.range(2));

    // RNG
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    // Pre-generate all coordinates
    std::vector<I> xs(n_coordinates);
    std::vector<I> ys(n_coordinates);

    for (std::size_t k = 0; k < n_coordinates; ++k)
    {
        xs[k] = row_dist(rng);
        ys[k] = col_dist(rng);
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    for (auto _ : state)
    {
        state.PauseTiming();
        spira::matrix<Layout, I, V> m(rows, cols);  // fresh matrix each iteration
        state.ResumeTiming();

        for (std::size_t k = 0; k < n_coordinates; ++k)
            m.add(xs[k], ys[k], V{});

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_coordinates)
    );
}

// SOA problemSizeScaling
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-insert-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-insert-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, SoA)
    ->Apply(problemSizeScaling)
    ->Name("soa-insert-complex-problemSizeScaling");

// AOS problemSizeScaling
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-insert-float-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-insert-double-problemSizeScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, AoS)
    ->Apply(problemSizeScaling)
    ->Name("aos-insert-complex-problemSizeScaling");

// SOA sparsityScaling
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-insert-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-insert-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, SoA)
    ->Apply(sparsityScaling)
    ->Name("soa-insert-complex-sparsityScaling");

// AOS sparsityScaling
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-insert-float-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-insert-double-sparsityScaling");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, AoS)
    ->Apply(sparsityScaling)
    ->Name("aos-insert-complex-sparsityScaling");
