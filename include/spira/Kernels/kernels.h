#include <stddef.h>

namespace spira::kernel {
    extern double (*sparse_dot_f64)(const double* vals, const int* cols, const double* x, size_t n);
    extern float  (*sparse_dot_f32)(const float* vals, const int* cols, const float* x, size_t n);
}