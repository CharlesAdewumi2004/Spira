#include "config.hpp"

template <class V, class Layout>
static void BM_streamInsertElements(benchmark::State &state)
{
    using I = std::size_t;

    const std::size_t n_coordinates = static_cast<std::size_t>(state.range(0));
    const std::size_t rows = static_cast<std::size_t>(state.range(1));
    const std::size_t cols = static_cast<std::size_t>(state.range(2));

    spira::matrix<Layout, I, V> m(rows, cols);

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

    std::size_t idx = 0;

    for (auto _ : state)
    {
        I x = xs[idx];
        I y = ys[idx];

        idx++;
        if (idx == n_coordinates)
        {
            idx = 0;
        }

        m.add(x, y, V{});
        benchmark::ClobberMemory();
    }

    // One insert per outer iteration
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// --- registrations: INSERT AoS/SoA, weak + strong, float/double/complex ------

// SOA weak
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, SoA)
    ->Apply(weakScaling)
    ->Name("soa-insert-float-weak");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, SoA)
    ->Apply(weakScaling)
    ->Name("soa-insert-double-weak");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, SoA)
    ->Apply(weakScaling)
    ->Name("soa-insert-complex-weak");

// AOS weak
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, AoS)
    ->Apply(weakScaling)
    ->Name("aos-insert-float-weak");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, AoS)
    ->Apply(weakScaling)
    ->Name("aos-insert-double-weak");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, AoS)
    ->Apply(weakScaling)
    ->Name("aos-insert-complex-weak");

// SOA strong
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, SoA)
    ->Apply(strongScaling)
    ->Name("soa-insert-float-strong");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, SoA)
    ->Apply(strongScaling)
    ->Name("soa-insert-double-strong");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, SoA)
    ->Apply(strongScaling)
    ->Name("soa-insert-complex-strong");

// AOS strong
BENCHMARK_TEMPLATE(BM_streamInsertElements, float, AoS)
    ->Apply(strongScaling)
    ->Name("aos-insert-float-strong");
BENCHMARK_TEMPLATE(BM_streamInsertElements, double, AoS)
    ->Apply(strongScaling)
    ->Name("aos-insert-double-strong");
BENCHMARK_TEMPLATE(BM_streamInsertElements, std::complex<double>, AoS)
    ->Apply(strongScaling)
    ->Name("aos-insert-complex-strong");