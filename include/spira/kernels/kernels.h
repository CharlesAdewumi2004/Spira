#include <cstdint>
#include <stddef.h>

namespace spira::kernel {
    extern double (*sparse_dot_double)(const double* vals, const uint32_t* cols, const double* x, size_t n, size_t x_size);
    extern float  (*sparse_dot_float)(const float* vals, const uint32_t* cols, const float* x, size_t n, size_t x_size);
}