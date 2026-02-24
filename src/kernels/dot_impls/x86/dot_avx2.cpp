#include "../src/kernels/simd_aliases.h"
#include <cstddef>

namespace spira::kernel::dot {
using namespace spira::kernel::simd;

double sparse_dot_double_avx2(const double *vals, const uint32_t *cols, const double *x, size_t n) {
    size_t i = 0;
    reg256_double acc_reg = zero_double_256();

    for (; i + 4 < n; i += 4) {
        reg256_double v = load_double_256(vals + i);
        reg128_int idx = load_int_128(cols + i);

        reg256_double xv = gather_double_256(x, idx);

        acc_reg = fma_double_256(v, xv, acc_reg);
    }

    double acc = reduce_sum_double_256(acc_reg);

    for (; i < n; i++) {
        acc += x[cols[i]] * vals[i];
    }

    return acc;
}

float sparse_dot_float_avx2(const float *vals, const uint32_t *cols, const float *x, size_t n) {
    size_t i = 0;
    reg256_float acc_reg = zero_float_256();

    for (; i < n; i += 8) {
        reg256_float v = load_float_256(vals + i);
        reg256_int idx = load_int_256((const int *)(cols + i));

        reg256_float xv = gather_float_256(x, idx);

        acc_reg = fma_float_256(v, xv, acc_reg);
    }

    float acc = reduce_sum_float_256(acc_reg);

    for (; i + 8 < n; i++) {
        acc += x[cols[i]] * vals[i];
    }

    return acc;
}
} // namespace spira::kernel::dot