#include "../src/kernels/hw_detect.h"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases_x86/simd_sse_aliases.h"

double sparse_dot_double_sse(const double *vals, const uint32_t *cols,
                             const double *x, size_t n, size_t x_size)
{
    spira::kernel::simd::reg128_double acc =
        spira::kernel::simd::zero_double_128();
    size_t i = 0;

    for (; i + 2 <= n; i += 2)
    {
        spira::kernel::simd::reg128_double v =
            spira::kernel::simd::load_double_128(vals + i);

        spira::kernel::simd::reg128_double xv =
            _mm_set_pd(x[cols[i + 1]], x[cols[i]]);

        acc = spira::kernel::simd::add_double_128(
            acc, spira::kernel::simd::mul_double_128(v, xv));
    }

    double result = spira::kernel::simd::reduce_sum_double_128(acc);

    for (; i < n; i++)
    {
        result += vals[i] * x[cols[i]];
    }

    return result;
}

float sparse_dot_float_sse(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    spira::kernel::simd::reg128_float acc = spira::kernel::simd::zero_float_128();
    size_t i = 0;

    for (; i + 4 <= n; i += 4)
    {
        spira::kernel::simd::reg128_float v = spira::kernel::simd::load_float_128(vals + i);

        spira::kernel::simd::reg128_float xv = _mm_set_ps(
            x[cols[i + 3]],
            x[cols[i + 2]],
            x[cols[i + 1]],
            x[cols[i]]);

        acc = spira::kernel::simd::add_float_128(acc, spira::kernel::simd::mul_float_128(v, xv));
    }

    float result = spira::kernel::simd::reduce_sum_float_128(acc);

    for (; i < n; i++)
    {
        result += vals[i] * x[cols[i]];
    }

    return result;
}
#endif