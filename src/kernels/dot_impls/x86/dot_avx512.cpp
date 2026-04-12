#include <immintrin.h>
#include <cstddef>
#include <cstdint>

#if defined(SPIRA_ARCH_X86)

double sparse_dot_double_avx512(const double *vals, const uint32_t *cols,
                                const double *x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();

    for (; i + 32 <= n; i += 32)
    {
        __m512d v0 = _mm512_loadu_pd(vals + i);
        __m256i idx0 = _mm256_loadu_si256((const __m256i *)(cols + i));
        __m512d xv0 = _mm512_i32gather_pd(idx0, x, 8);
        acc0 = _mm512_fmadd_pd(v0, xv0, acc0);

        __m512d v1 = _mm512_loadu_pd(vals + i + 8);
        __m256i idx1 = _mm256_loadu_si256((const __m256i *)(cols + i + 8));
        __m512d xv1 = _mm512_i32gather_pd(idx1, x, 8);
        acc1 = _mm512_fmadd_pd(v1, xv1, acc1);

        __m512d v2 = _mm512_loadu_pd(vals + i + 16);
        __m256i idx2 = _mm256_loadu_si256((const __m256i *)(cols + i + 16));
        __m512d xv2 = _mm512_i32gather_pd(idx2, x, 8);
        acc2 = _mm512_fmadd_pd(v2, xv2, acc2);

        __m512d v3 = _mm512_loadu_pd(vals + i + 24);
        __m256i idx3 = _mm256_loadu_si256((const __m256i *)(cols + i + 24));
        __m512d xv3 = _mm512_i32gather_pd(idx3, x, 8);
        acc3 = _mm512_fmadd_pd(v3, xv3, acc3);
    }

    for (; i + 8 <= n; i += 8)
    {
        __m512d v = _mm512_loadu_pd(vals + i);
        __m256i idx = _mm256_loadu_si256((const __m256i *)(cols + i));
        __m512d xv = _mm512_i32gather_pd(idx, x, 8);
        acc0 = _mm512_fmadd_pd(v, xv, acc0);
    }

    acc0 = _mm512_add_pd(acc0, acc1);
    acc2 = _mm512_add_pd(acc2, acc3);
    acc0 = _mm512_add_pd(acc0, acc2);
    double acc = _mm512_reduce_add_pd(acc0);

    for (; i < n; i++)
        acc += x[cols[i]] * vals[i];

    return acc;
}

float sparse_dot_float_avx512(const float *vals, const uint32_t *cols,
                              const float *x, size_t n, size_t /*x_size*/)
{
    size_t i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    for (; i + 64 <= n; i += 64)
    {
        __m512 v0 = _mm512_loadu_ps(vals + i);
        __m512i idx0 = _mm512_loadu_si512((const __m512i *)(cols + i));
        __m512 xv0 = _mm512_i32gather_ps(idx0, x, 4);
        acc0 = _mm512_fmadd_ps(v0, xv0, acc0);

        __m512 v1 = _mm512_loadu_ps(vals + i + 16);
        __m512i idx1 = _mm512_loadu_si512((const __m512i *)(cols + i + 16));
        __m512 xv1 = _mm512_i32gather_ps(idx1, x, 4);
        acc1 = _mm512_fmadd_ps(v1, xv1, acc1);

        __m512 v2 = _mm512_loadu_ps(vals + i + 32);
        __m512i idx2 = _mm512_loadu_si512((const __m512i *)(cols + i + 32));
        __m512 xv2 = _mm512_i32gather_ps(idx2, x, 4);
        acc2 = _mm512_fmadd_ps(v2, xv2, acc2);

        __m512 v3 = _mm512_loadu_ps(vals + i + 48);
        __m512i idx3 = _mm512_loadu_si512((const __m512i *)(cols + i + 48));
        __m512 xv3 = _mm512_i32gather_ps(idx3, x, 4);
        acc3 = _mm512_fmadd_ps(v3, xv3, acc3);
    }

    for (; i + 16 <= n; i += 16)
    {
        __m512 v = _mm512_loadu_ps(vals + i);
        __m512i idx = _mm512_loadu_si512((const __m512i *)(cols + i));
        __m512 xv = _mm512_i32gather_ps(idx, x, 4);
        acc0 = _mm512_fmadd_ps(v, xv, acc0);
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float acc = _mm512_reduce_add_ps(acc0);

    for (; i < n; i++)
        acc += x[cols[i]] * vals[i];

    return acc;
}

#endif
