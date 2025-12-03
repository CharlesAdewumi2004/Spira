#include <benchmark/benchmark.h>
#include "spira/spira.hpp"
#include <random>
#include <vector>

#define rowsSize 10000
#define columnsSize 10000

template<class V>
static void BM_soa_streamInsertElements(benchmark::State& state) {
    using LayoutTag = spira::layout::tags::soa_tag;
    using I = std::size_t;

    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2);  

    spira::matrix<LayoutTag, I, V> m(rows, cols);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    std::vector<I> xs;
    std::vector<I> ys;
    xs.reserve(n_coordinates);
    ys.reserve(n_coordinates);

    for (std::size_t k = 0; k < n_coordinates; ++k) {
        xs.push_back(row_dist(rng));
        ys.push_back(col_dist(rng));
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    std::size_t idx = 0;

    for (auto _ : state) {
        I x = xs[idx];
        I y = ys[idx];

        idx++;
        if (idx == n_coordinates) idx = 0;

        m.add(x, y, 0.0);

        benchmark::ClobberMemory();
    }
}

template<class V>
static void BM_aos_streamInsertElements(benchmark::State& state) {
    using LayoutTag = spira::layout::tags::aos_tag;
    using I = std::size_t;

    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2);  

    spira::matrix<LayoutTag, I, V> m(rows, cols);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    std::vector<I> xs;
    std::vector<I> ys;
    xs.reserve(n_coordinates);
    ys.reserve(n_coordinates);

    for (std::size_t k = 0; k < n_coordinates; ++k) {
        xs.push_back(row_dist(rng));
        ys.push_back(col_dist(rng));
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    std::size_t idx = 0;

    for (auto _ : state) {
        I x = xs[idx];
        I y = ys[idx];

        idx++;
        if (idx == n_coordinates) idx = 0;

        m.add(x, y, 0.0);

        benchmark::ClobberMemory();
    }
}

BENCHMARK_TEMPLATE(BM_soa_streamInsertElements, float)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_streamInsertElements, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_streamInsertElements, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});     

BENCHMARK_TEMPLATE(BM_aos_streamInsertElements, float)
    ->Args({1000, 10000, 10000});
BENCHMARK_TEMPLATE(BM_aos_streamInsertElements, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_aos_streamInsertElements, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});   
      
