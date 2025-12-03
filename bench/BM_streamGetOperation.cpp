#include <benchmark/benchmark.h>
#include "spira/spira.hpp"
#include <random>
#include <vector>

#define rowsSize 10000
#define columnsSize 10000

template<class V>
static void BM_soa_streamGetOperation(benchmark::State& state){
    using I = size_t;
    using Layout = spira::layout::tags::soa_tag;
    
    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2); 
    
    spira::matrix<Layout, I, V> m(rowsSize, columnsSize);
    for(size_t i; i < rowsSize; i++){
        for(size_t j; j < columnsSize; j++){
            m.add(i,j,0.0);
        }
    }

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    std::vector<I> xs;
    std::vector<I> ys;
    xs.reserve(n_coordinates);
    ys.reserve(n_coordinates);

    for(size_t i = 0; i < n_coordinates; i++){
        xs.push_back(row_dist(rng));
        ys.push_back(col_dist(rng));
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    for (auto _ : state) {
       for(size_t i = 0; i < n_coordinates; i++){
            m.get(xs[i], ys[i]);
       }

        benchmark::ClobberMemory();
    }
}

template<class V>
static void BM_aos_streamGetOperation(benchmark::State& state){
    using I = size_t;
    using Layout = spira::layout::tags::aos_tag;
    
    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2); 
    
    spira::matrix<Layout, I, V> m(rowsSize, columnsSize);
    for(size_t i; i < rowsSize; i++){
        for(size_t j; j < columnsSize; j++){
            m.add(i,j,0.0);
        }
    }

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);

    std::vector<I> xs;
    std::vector<I> ys;
    xs.reserve(n_coordinates);
    ys.reserve(n_coordinates);

    for(size_t i = 0; i < n_coordinates; i++){
        xs.push_back(row_dist(rng));
        ys.push_back(col_dist(rng));
    }

    benchmark::DoNotOptimize(xs.data());
    benchmark::DoNotOptimize(ys.data());

    for (auto _ : state) {
       for(size_t i = 0; i < n_coordinates; i++){
            m.get(xs[i], ys[i]);
       }

        benchmark::ClobberMemory();
    }
}

BENCHMARK_TEMPLATE(BM_soa_streamGetOperation, float)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_streamGetOperation, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_streamGetOperation, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});

BENCHMARK_TEMPLATE(BM_aos_streamGetOperation, float)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_aos_streamGetOperation, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_aos_streamGetOperation, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});

