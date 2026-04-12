
#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_avx_aliases.hpp"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols,
                             const double *x, size_t n, size_t /*x_size*/)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_double acc0 = zero_double_256();
    reg256_double acc1 = zero_double_256();
    reg256_double acc2 = zero_double_256();
    reg256_double acc3 = zero_double_256();

    for (; i + 16 <= n; i += 16)
    {
        reg256_double v0 = load_double_256(vals + i);
        reg128_int idx0 = _mm_loadu_si128((const __m128i *)(cols + i));
        reg256_double xv0 = gather_double_256(x, idx0);
        acc0 = fma_double_256(v0, xv0, acc0);

        reg256_double v1 = load_double_256(vals + i + 4);
        reg128_int idx1 = _mm_loadu_si128((const __m128i *)(cols + i + 4));
        reg256_double xv1 = gather_double_256(x, idx1);
        acc1 = fma_double_256(v1, xv1, acc1);

        reg256_double v2 = load_double_256(vals + i + 8);
        reg128_int idx2 = _mm_loadu_si128((const __m128i *)(cols + i + 8));
        reg256_double xv2 = gather_double_256(x, idx2);
        acc2 = fma_double_256(v2, xv2, acc2);

        reg256_double v3 = load_double_256(vals + i + 12);
        reg128_int idx3 = _mm_loadu_si128((const __m128i *)(cols + i + 12));
        reg256_double xv3 = gather_double_256(x, idx3);
        acc3 = fma_double_256(v3, xv3, acc3);
    }

    for (; i + 4 <= n; i += 4)
    {
        reg256_double v = load_double_256(vals + i);
        reg128_int idx = _mm_loadu_si128((const __m128i *)(cols + i));
        reg256_double xv = gather_double_256(x, idx);
        acc0 = fma_double_256(v, xv, acc0);
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    double acc = reduce_sum_double_256(acc0);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols,
                           const float *x, size_t n, size_t /*x_size*/)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_float acc0 = zero_float_256();
    reg256_float acc1 = zero_float_256();
    reg256_float acc2 = zero_float_256();
    reg256_float acc3 = zero_float_256();

    for (; i + 32 <= n; i += 32)
    {
        reg256_float v0 = load_float_256(vals + i);
        reg256_int idx0 = _mm256_loadu_si256((const __m256i *)(cols + i));
        reg256_float xv0 = gather_float_256(x, idx0);
        acc0 = fma_float_256(v0, xv0, acc0);

        reg256_float v1 = load_float_256(vals + i + 8);
        reg256_int idx1 = _mm256_loadu_si256((const __m256i *)(cols + i + 8));
        reg256_float xv1 = gather_float_256(x, idx1);
        acc1 = fma_float_256(v1, xv1, acc1);

        reg256_float v2 = load_float_256(vals + i + 16);
        reg256_int idx2 = _mm256_loadu_si256((const __m256i *)(cols + i + 16));
        reg256_float xv2 = gather_float_256(x, idx2);
        acc2 = fma_float_256(v2, xv2, acc2);

        reg256_float v3 = load_float_256(vals + i + 24);
        reg256_int idx3 = _mm256_loadu_si256((const __m256i *)(cols + i + 24));
        reg256_float xv3 = gather_float_256(x, idx3);
        acc3 = fma_float_256(v3, xv3, acc3);
    }

    for (; i + 8 <= n; i += 8)
    {
        reg256_float v = load_float_256(vals + i);
        reg256_int idx = _mm256_loadu_si256((const __m256i *)(cols + i));
        reg256_float xv = gather_float_256(x, idx);
        acc0 = fma_float_256(v, xv, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float acc = reduce_sum_float_256(acc0);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}

#endif
