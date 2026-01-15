#include <benchmark/benchmark.h>
#include <spira/spira.hpp>
#include <random>
#include <vector>
#define maxMatrixValue 10000
#define minMaxtrixValue -10000

using SoA = spira::layout::tags::soa_tag;
using AoS = spira::layout::tags::aos_tag;

static void problemSizeScaling(benchmark::internal::Benchmark* b){
    const int nnz_per_row = 16;

    for(long rows : {100, 500, 1000, 5000, 10000, 50000, 100000, 500000}){
        long cols = rows;
        long nnz = rows * nnz_per_row;
        b->Args({nnz, rows, cols}); 
    }
}

static void sparsityScaling(benchmark::internal::Benchmark* b){
    long rows = 10000, cols = 10000;

    for(long nnz = 1; nnz < 2048; nnz*=2){
        b->Args({nnz*rows ,rows, cols});
    }
}


