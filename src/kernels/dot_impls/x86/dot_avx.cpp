#include "../src/kernels/cpu_detect.h"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases_x86/simd_avx_aliases.h"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols, const double *x, size_t n) {
    size_t i = 0;
    spira::kernel::simd::reg256_double acc_reg = spira::kernel::simd::zero_double_256();
    
    for (; i + 4 <= n; i += 4) { // <= not
        spira::kernel::simd::reg256_double v = spira::kernel::simd::load_double_256(vals + i);
        spira::kernel::simd::reg128_int idx = spira::kernel::simd::load_int_128(cols + i);
        spira::kernel::simd::reg256_double xv = spira::kernel::simd::gather_double_256(x, idx);
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

    for (; i + 8 <= n; i += 8) { // i + 8 <= n, not i < n
        spira::kernel::simd::reg256_float v = spira::kernel::simd::load_float_256(vals + i);
        spira::kernel::simd::reg256_int idx = spira::kernel::simd::load_int_256((const int *)(cols + i));
        spira::kernel::simd::reg256_float xv = spira::kernel::simd::gather_float_256(x, idx);
        acc_reg = spira::kernel::simd::fma_float_256(v, xv, acc_reg);
    }

    float acc = spira::kernel::simd::reduce_sum_float_256(acc_reg);

    for (; i < n; i++) { // i < n, not i + 8 < n
        acc += vals[i] * x[cols[i]];
    }

    return acc;
}

#endif