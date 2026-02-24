#include "../src/kernels/simd_aliases.h"
#include <cstddef>


namespace spira::kernel::dot {
    using namespace spira::kernel::simd;

    double sparse_dot_double_avx2(const double* vals, const uint32_t* cols, const double* x, size_t n){}

    float  sparse_dot_float_avx2(const float* vals, const uint32_t* cols, const float* x, size_t n){
        size_t i = 0;
    
    }
}