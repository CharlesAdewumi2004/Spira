#include <immintrin.h>

#include "kernels/dot_impls.hpp"

// Horizontal sums using SSE2 shuffles only, so the file needs nothing beyond
// its -msse4.2 (or MSVC default) flags.
static double hsum(__m128d v)
{
    return _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)));
}

static float hsum(__m128 v)
{
    const __m128 pairs = _mm_add_ps(v, _mm_movehl_ps(v, v)); // [0+2, 1+3, ..]
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55)));
}

double sparse_dot_double_sse(const double *vals, const uint32_t *cols,
                             const double *x, size_t n)
{
    size_t i = 0;
    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    __m128d acc2 = _mm_setzero_pd();
    __m128d acc3 = _mm_setzero_pd();

    for (; i + 8 <= n; i += 8)
    {
        __m128d v0 = _mm_loadu_pd(vals + i);
        __m128d xv0 = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(v0, xv0));

        __m128d v1 = _mm_loadu_pd(vals + i + 2);
        __m128d xv1 = _mm_set_pd(x[cols[i + 3]], x[cols[i + 2]]);
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(v1, xv1));

        __m128d v2 = _mm_loadu_pd(vals + i + 4);
        __m128d xv2 = _mm_set_pd(x[cols[i + 5]], x[cols[i + 4]]);
        acc2 = _mm_add_pd(acc2, _mm_mul_pd(v2, xv2));

        __m128d v3 = _mm_loadu_pd(vals + i + 6);
        __m128d xv3 = _mm_set_pd(x[cols[i + 7]], x[cols[i + 6]]);
        acc3 = _mm_add_pd(acc3, _mm_mul_pd(v3, xv3));
    }

    for (; i + 2 <= n; i += 2)
    {
        __m128d v = _mm_loadu_pd(vals + i);
        __m128d xv = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(v, xv));
    }

    acc0 = _mm_add_pd(acc0, acc1);
    acc2 = _mm_add_pd(acc2, acc3);
    acc0 = _mm_add_pd(acc0, acc2);
    double result = hsum(acc0);

    for (; i < n; i++)
        result += vals[i] * x[cols[i]];

    return result;
}

float sparse_dot_float_sse(const float *vals, const uint32_t *cols,
                           const float *x, size_t n)
{
    size_t i = 0;
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();

    for (; i + 16 <= n; i += 16)
    {
        __m128 v0 = _mm_loadu_ps(vals + i);
        __m128 xv0 = _mm_set_ps(
            x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(v0, xv0));

        __m128 v1 = _mm_loadu_ps(vals + i + 4);
        __m128 xv1 = _mm_set_ps(
            x[cols[i + 7]], x[cols[i + 6]], x[cols[i + 5]], x[cols[i + 4]]);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(v1, xv1));

        __m128 v2 = _mm_loadu_ps(vals + i + 8);
        __m128 xv2 = _mm_set_ps(
            x[cols[i + 11]], x[cols[i + 10]], x[cols[i + 9]], x[cols[i + 8]]);
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(v2, xv2));

        __m128 v3 = _mm_loadu_ps(vals + i + 12);
        __m128 xv3 = _mm_set_ps(
            x[cols[i + 15]], x[cols[i + 14]], x[cols[i + 13]], x[cols[i + 12]]);
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(v3, xv3));
    }

    for (; i + 4 <= n; i += 4)
    {
        __m128 v = _mm_loadu_ps(vals + i);
        __m128 xv = _mm_set_ps(
            x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(v, xv));
    }

    acc0 = _mm_add_ps(acc0, acc1);
    acc2 = _mm_add_ps(acc2, acc3);
    acc0 = _mm_add_ps(acc0, acc2);
    float result = hsum(acc0);

    for (; i < n; i++)
        result += vals[i] * x[cols[i]];

    return result;
}
