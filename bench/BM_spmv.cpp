#include <benchmark/benchmark.h>
#include "spira/spira.hpp"
#include <random>
#include <vector>

#define rowsSize 100000
#define columnsSize 100000
#define maxMatrixValue 10000
#define minMaxtrixValue -10000


template<class V>
static void BM_soa_spmv(benchmark::State& state){
    using Layout = spira::layout::tags::soa_tag;
    using I = size_t;
    
    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2); 
    
    spira::matrix<Layout, I, V> m(rows, cols);
    std::vector<V> x,y(rows, 0);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> num_dist(minMaxtrixValue, maxMatrixValue);

    for(size_t i = 0; i < cols; i++){
        x.push_back(num_dist(rng));
    }

    for(size_t i = 0; i < n_coordinates; i++){
        m.add(row_dist(rng), col_dist(rng), num_dist(rng));
    }

    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());

    for (auto _ : state) {
        spira::algorithms::spmv(m,x,y);
        benchmark::ClobberMemory();
    }
}

template<class V>
static void BM_aos_spmv(benchmark::State& state){
    using Layout = spira::layout::tags::aos_tag;
    using I = size_t;
    
    std::size_t rows          = state.range(0);
    std::size_t cols          = state.range(1);
    std::size_t n_coordinates = state.range(2); 
    
    spira::matrix<Layout, I, V> m(rows, cols);
    std::vector<V> x,y(rows, 0);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<I> row_dist(0, rows - 1);
    std::uniform_int_distribution<I> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> num_dist(minMaxtrixValue, maxMatrixValue);

    for(size_t i = 0; i < cols; i++){
        x.push_back(num_dist(rng));
    }

    for(size_t i = 0; i < n_coordinates; i++){
        m.add(row_dist(rng), col_dist(rng), num_dist(rng));
    }

    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());

    for (auto _ : state) {
        spira::algorithms::spmv(m,x,y);
        benchmark::ClobberMemory();
    }
}

BENCHMARK_TEMPLATE(BM_aos_spmv, float)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_aos_spmv, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_aos_spmv, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});

BENCHMARK_TEMPLATE(BM_soa_spmv, float)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_spmv, double)
    ->Args({1000, rowsSize, columnsSize});
BENCHMARK_TEMPLATE(BM_soa_spmv, std::complex<double>)
    ->Args({1000, rowsSize, columnsSize});