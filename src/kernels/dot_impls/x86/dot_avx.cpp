
#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases_x86/simd_avx_aliases.h"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols, const double *x, size_t n) {
    size_t i = 0;
    spira::kernel::simd::reg256_double acc_reg = spira::kernel::simd::zero_double_256();
    
    for (; i + 4 <= n; i += 4) {
        spira::kernel::simd::reg256_double v = spira::kernel::simd::load_double_256(vals + i);

        float vx0 = x[cols[i]], vx1 = x[cols[i+1]], vx2 = x[cols[i+2]], vx3 = x[cols[i+3]];

        spira::kernel::simd::reg256_double xv = _mm256_setr_pd(vx0, vx1, vx2, vx3);

        acc_reg = spira::kernel::simd::fma_double_256(v, xv, acc_reg);
    }

    double acc = spira::kernel::simd::reduce_sum_double_256(acc_reg);

    for (; i < n; i++) {
        acc += vals[i] * x[cols[i]];
    }

    return acc;
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols, const float *x, size_t n) {
    size_t i = 0;
    spira::kernel::simd::reg256_float acc_reg = spira::kernel::simd::zero_float_256();

    for (; i + 8 <= n; i += 8) {
        spira::kernel::simd::reg256_float v = spira::kernel::simd::load_float_256(vals + i);
        float vx0 = x[cols[i]], vx1 = x[cols[i+1]], vx2 = x[cols[i+2]], vx3 = x[cols[i+3]], 
        vx4 = x[cols[i+4]], vx5 = x[cols[i+5]], vx6 = x[cols[i+6]], vx7 = x[cols[i+7]];

        spira::kernel::simd::reg256_float xv = _mm256_setr_ps(vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7);

        acc_reg = spira::kernel::simd::fma_float_256(v, xv, acc_reg);
    }

    float acc = spira::kernel::simd::reduce_sum_float_256(acc_reg);

    for (; i < n; i++) { // i < n, not i + 8 < n
        acc += vals[i] * x[cols[i]];
    }

    return acc;
}

#endif