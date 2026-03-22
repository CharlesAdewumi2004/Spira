#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_X86)

double sparse_dot_double_avx512(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size)
{
    size_t i = 0;
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n, 8, 16))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(8, 16);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 32 <= prefetch_end; i += 32)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 2]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 3]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 4]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 5]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 6]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 7]], 0, 3);

            __m512d v0 = _mm512_loadu_pd(vals + i);
            __m512d xv0 = _mm512_setr_pd(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]);
            acc0 = _mm512_fmadd_pd(v0, xv0, acc0);

            __m512d v1 = _mm512_loadu_pd(vals + i + 8);
            __m512d xv1 = _mm512_setr_pd(
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc1 = _mm512_fmadd_pd(v1, xv1, acc1);

            __m512d v2 = _mm512_loadu_pd(vals + i + 16);
            __m512d xv2 = _mm512_setr_pd(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]]);
            acc2 = _mm512_fmadd_pd(v2, xv2, acc2);

            __m512d v3 = _mm512_loadu_pd(vals + i + 24);
            __m512d xv3 = _mm512_setr_pd(
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc3 = _mm512_fmadd_pd(v3, xv3, acc3);
        }

        for (; i + 32 <= n; i += 32)
        {
            __m512d v0 = _mm512_loadu_pd(vals + i);
            __m512d xv0 = _mm512_setr_pd(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]);
            acc0 = _mm512_fmadd_pd(v0, xv0, acc0);

            __m512d v1 = _mm512_loadu_pd(vals + i + 8);
            __m512d xv1 = _mm512_setr_pd(
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc1 = _mm512_fmadd_pd(v1, xv1, acc1);

            __m512d v2 = _mm512_loadu_pd(vals + i + 16);
            __m512d xv2 = _mm512_setr_pd(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]]);
            acc2 = _mm512_fmadd_pd(v2, xv2, acc2);

            __m512d v3 = _mm512_loadu_pd(vals + i + 24);
            __m512d xv3 = _mm512_setr_pd(
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc3 = _mm512_fmadd_pd(v3, xv3, acc3);
        }

        // single-register cleanup
        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(vals + i);
            __m512d xv = _mm512_setr_pd(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]);
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
    else
    {
        for (; i + 32 <= n; i += 32)
        {
            __m512d v0 = _mm512_loadu_pd(vals + i);
            __m512d xv0 = _mm512_setr_pd(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]);
            acc0 = _mm512_fmadd_pd(v0, xv0, acc0);

            __m512d v1 = _mm512_loadu_pd(vals + i + 8);
            __m512d xv1 = _mm512_setr_pd(
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc1 = _mm512_fmadd_pd(v1, xv1, acc1);

            __m512d v2 = _mm512_loadu_pd(vals + i + 16);
            __m512d xv2 = _mm512_setr_pd(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]]);
            acc2 = _mm512_fmadd_pd(v2, xv2, acc2);

            __m512d v3 = _mm512_loadu_pd(vals + i + 24);
            __m512d xv3 = _mm512_setr_pd(
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc3 = _mm512_fmadd_pd(v3, xv3, acc3);
        }

        for (; i + 8 <= n; i += 8)
        {
            __m512d v = _mm512_loadu_pd(vals + i);
            __m512d xv = _mm512_setr_pd(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]]);
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
}

float sparse_dot_float_avx512(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    size_t i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n, 16, 32))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(16, 32);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 64 <= prefetch_end; i += 64)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 2]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 3]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 4]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 5]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 6]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 7]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 8]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 9]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 10]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 11]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 12]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 13]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 14]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 15]], 0, 3);

            __m512 v0 = _mm512_loadu_ps(vals + i);
            __m512 xv0 = _mm512_setr_ps(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]],
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc0 = _mm512_fmadd_ps(v0, xv0, acc0);

            __m512 v1 = _mm512_loadu_ps(vals + i + 16);
            __m512 xv1 = _mm512_setr_ps(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]],
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc1 = _mm512_fmadd_ps(v1, xv1, acc1);

            __m512 v2 = _mm512_loadu_ps(vals + i + 32);
            __m512 xv2 = _mm512_setr_ps(
                x[cols[i + 32]], x[cols[i + 33]], x[cols[i + 34]], x[cols[i + 35]],
                x[cols[i + 36]], x[cols[i + 37]], x[cols[i + 38]], x[cols[i + 39]],
                x[cols[i + 40]], x[cols[i + 41]], x[cols[i + 42]], x[cols[i + 43]],
                x[cols[i + 44]], x[cols[i + 45]], x[cols[i + 46]], x[cols[i + 47]]);
            acc2 = _mm512_fmadd_ps(v2, xv2, acc2);

            __m512 v3 = _mm512_loadu_ps(vals + i + 48);
            __m512 xv3 = _mm512_setr_ps(
                x[cols[i + 48]], x[cols[i + 49]], x[cols[i + 50]], x[cols[i + 51]],
                x[cols[i + 52]], x[cols[i + 53]], x[cols[i + 54]], x[cols[i + 55]],
                x[cols[i + 56]], x[cols[i + 57]], x[cols[i + 58]], x[cols[i + 59]],
                x[cols[i + 60]], x[cols[i + 61]], x[cols[i + 62]], x[cols[i + 63]]);
            acc3 = _mm512_fmadd_ps(v3, xv3, acc3);
        }

        for (; i + 64 <= n; i += 64)
        {
            __m512 v0 = _mm512_loadu_ps(vals + i);
            __m512 xv0 = _mm512_setr_ps(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]],
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc0 = _mm512_fmadd_ps(v0, xv0, acc0);

            __m512 v1 = _mm512_loadu_ps(vals + i + 16);
            __m512 xv1 = _mm512_setr_ps(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]],
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc1 = _mm512_fmadd_ps(v1, xv1, acc1);

            __m512 v2 = _mm512_loadu_ps(vals + i + 32);
            __m512 xv2 = _mm512_setr_ps(
                x[cols[i + 32]], x[cols[i + 33]], x[cols[i + 34]], x[cols[i + 35]],
                x[cols[i + 36]], x[cols[i + 37]], x[cols[i + 38]], x[cols[i + 39]],
                x[cols[i + 40]], x[cols[i + 41]], x[cols[i + 42]], x[cols[i + 43]],
                x[cols[i + 44]], x[cols[i + 45]], x[cols[i + 46]], x[cols[i + 47]]);
            acc2 = _mm512_fmadd_ps(v2, xv2, acc2);

            __m512 v3 = _mm512_loadu_ps(vals + i + 48);
            __m512 xv3 = _mm512_setr_ps(
                x[cols[i + 48]], x[cols[i + 49]], x[cols[i + 50]], x[cols[i + 51]],
                x[cols[i + 52]], x[cols[i + 53]], x[cols[i + 54]], x[cols[i + 55]],
                x[cols[i + 56]], x[cols[i + 57]], x[cols[i + 58]], x[cols[i + 59]],
                x[cols[i + 60]], x[cols[i + 61]], x[cols[i + 62]], x[cols[i + 63]]);
            acc3 = _mm512_fmadd_ps(v3, xv3, acc3);
        }

        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(vals + i);
            __m512 xv = _mm512_setr_ps(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]],
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
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
    else
    {
        for (; i + 64 <= n; i += 64)
        {
            __m512 v0 = _mm512_loadu_ps(vals + i);
            __m512 xv0 = _mm512_setr_ps(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]],
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
            acc0 = _mm512_fmadd_ps(v0, xv0, acc0);

            __m512 v1 = _mm512_loadu_ps(vals + i + 16);
            __m512 xv1 = _mm512_setr_ps(
                x[cols[i + 16]], x[cols[i + 17]], x[cols[i + 18]], x[cols[i + 19]],
                x[cols[i + 20]], x[cols[i + 21]], x[cols[i + 22]], x[cols[i + 23]],
                x[cols[i + 24]], x[cols[i + 25]], x[cols[i + 26]], x[cols[i + 27]],
                x[cols[i + 28]], x[cols[i + 29]], x[cols[i + 30]], x[cols[i + 31]]);
            acc1 = _mm512_fmadd_ps(v1, xv1, acc1);

            __m512 v2 = _mm512_loadu_ps(vals + i + 32);
            __m512 xv2 = _mm512_setr_ps(
                x[cols[i + 32]], x[cols[i + 33]], x[cols[i + 34]], x[cols[i + 35]],
                x[cols[i + 36]], x[cols[i + 37]], x[cols[i + 38]], x[cols[i + 39]],
                x[cols[i + 40]], x[cols[i + 41]], x[cols[i + 42]], x[cols[i + 43]],
                x[cols[i + 44]], x[cols[i + 45]], x[cols[i + 46]], x[cols[i + 47]]);
            acc2 = _mm512_fmadd_ps(v2, xv2, acc2);

            __m512 v3 = _mm512_loadu_ps(vals + i + 48);
            __m512 xv3 = _mm512_setr_ps(
                x[cols[i + 48]], x[cols[i + 49]], x[cols[i + 50]], x[cols[i + 51]],
                x[cols[i + 52]], x[cols[i + 53]], x[cols[i + 54]], x[cols[i + 55]],
                x[cols[i + 56]], x[cols[i + 57]], x[cols[i + 58]], x[cols[i + 59]],
                x[cols[i + 60]], x[cols[i + 61]], x[cols[i + 62]], x[cols[i + 63]]);
            acc3 = _mm512_fmadd_ps(v3, xv3, acc3);
        }

        for (; i + 16 <= n; i += 16)
        {
            __m512 v = _mm512_loadu_ps(vals + i);
            __m512 xv = _mm512_setr_ps(
                x[cols[i]], x[cols[i + 1]], x[cols[i + 2]], x[cols[i + 3]],
                x[cols[i + 4]], x[cols[i + 5]], x[cols[i + 6]], x[cols[i + 7]],
                x[cols[i + 8]], x[cols[i + 9]], x[cols[i + 10]], x[cols[i + 11]],
                x[cols[i + 12]], x[cols[i + 13]], x[cols[i + 14]], x[cols[i + 15]]);
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
}

#endif
