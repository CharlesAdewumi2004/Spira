
#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_avx_aliases.hpp"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols,
                             const double *x, size_t n, size_t /*x_size*/)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_double acc_reg = zero_double_256();

    for (; i + 4 <= n; i += 4)
    {
        reg256_double v = load_double_256(vals + i);
        reg128_int idx = _mm_loadu_si128((const __m128i *)(cols + i));
        reg256_double xv = gather_double_256(x, idx);
        acc_reg = fma_double_256(v, xv, acc_reg);
    }

    double acc = reduce_sum_double_256(acc_reg);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols,
                           const float *x, size_t n, size_t /*x_size*/)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_float acc_reg = zero_float_256();

    for (; i + 8 <= n; i += 8)
    {
        reg256_float v = load_float_256(vals + i);
        reg256_int idx = _mm256_loadu_si256((const __m256i *)(cols + i));
        reg256_float xv = gather_float_256(x, idx);
        acc_reg = fma_float_256(v, xv, acc_reg);
    }

    float acc = reduce_sum_float_256(acc_reg);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}

#endif
