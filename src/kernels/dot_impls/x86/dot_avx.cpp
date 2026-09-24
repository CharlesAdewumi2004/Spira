
#include <immintrin.h>

#include "kernels/dot_impls.hpp"

static double hsum(__m256d v)
{
    const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

static float hsum(__m256 v)
{
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pairs = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55)));
}

double sparse_dot_double_avx(const double *vals, const uint32_t *cols,
                             const double *x, size_t n)
{
    size_t i = 0;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    for (; i + 16 <= n; i += 16)
    {
        __m256d v0 = _mm256_loadu_pd(vals + i);
        __m128i idx0 = _mm_loadu_si128((const __m128i *)(cols + i));
        __m256d xv0 = _mm256_i32gather_pd(x, idx0, 8);
        acc0 = _mm256_fmadd_pd(v0, xv0, acc0);

        __m256d v1 = _mm256_loadu_pd(vals + i + 4);
        __m128i idx1 = _mm_loadu_si128((const __m128i *)(cols + i + 4));
        __m256d xv1 = _mm256_i32gather_pd(x, idx1, 8);
        acc1 = _mm256_fmadd_pd(v1, xv1, acc1);

        __m256d v2 = _mm256_loadu_pd(vals + i + 8);
        __m128i idx2 = _mm_loadu_si128((const __m128i *)(cols + i + 8));
        __m256d xv2 = _mm256_i32gather_pd(x, idx2, 8);
        acc2 = _mm256_fmadd_pd(v2, xv2, acc2);

        __m256d v3 = _mm256_loadu_pd(vals + i + 12);
        __m128i idx3 = _mm_loadu_si128((const __m128i *)(cols + i + 12));
        __m256d xv3 = _mm256_i32gather_pd(x, idx3, 8);
        acc3 = _mm256_fmadd_pd(v3, xv3, acc3);
    }

    for (; i + 4 <= n; i += 4)
    {
        __m256d v = _mm256_loadu_pd(vals + i);
        __m128i idx = _mm_loadu_si128((const __m128i *)(cols + i));
        __m256d xv = _mm256_i32gather_pd(x, idx, 8);
        acc0 = _mm256_fmadd_pd(v, xv, acc0);
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    double acc = hsum(acc0);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols,
                           const float *x, size_t n)
{
    size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 32 <= n; i += 32)
    {
        __m256 v0 = _mm256_loadu_ps(vals + i);
        __m256i idx0 = _mm256_loadu_si256((const __m256i *)(cols + i));
        __m256 xv0 = _mm256_i32gather_ps(x, idx0, 4);
        acc0 = _mm256_fmadd_ps(v0, xv0, acc0);

        __m256 v1 = _mm256_loadu_ps(vals + i + 8);
        __m256i idx1 = _mm256_loadu_si256((const __m256i *)(cols + i + 8));
        __m256 xv1 = _mm256_i32gather_ps(x, idx1, 4);
        acc1 = _mm256_fmadd_ps(v1, xv1, acc1);

        __m256 v2 = _mm256_loadu_ps(vals + i + 16);
        __m256i idx2 = _mm256_loadu_si256((const __m256i *)(cols + i + 16));
        __m256 xv2 = _mm256_i32gather_ps(x, idx2, 4);
        acc2 = _mm256_fmadd_ps(v2, xv2, acc2);

        __m256 v3 = _mm256_loadu_ps(vals + i + 24);
        __m256i idx3 = _mm256_loadu_si256((const __m256i *)(cols + i + 24));
        __m256 xv3 = _mm256_i32gather_ps(x, idx3, 4);
        acc3 = _mm256_fmadd_ps(v3, xv3, acc3);
    }

    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(vals + i);
        __m256i idx = _mm256_loadu_si256((const __m256i *)(cols + i));
        __m256 xv = _mm256_i32gather_ps(x, idx, 4);
        acc0 = _mm256_fmadd_ps(v, xv, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float acc = hsum(acc0);

    for (; i < n; i++)
        acc += vals[i] * x[cols[i]];

    return acc;
}
