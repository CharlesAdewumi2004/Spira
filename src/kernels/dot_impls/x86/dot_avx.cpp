
#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_avx_aliases.hpp"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols,
                             const double *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_double acc0 = zero_double_256();
    reg256_double acc1 = zero_double_256();
    reg256_double acc2 = zero_double_256();
    reg256_double acc3 = zero_double_256();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n, 4, 8))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(4, 8);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 16 <= prefetch_end; i += 16)
        {
            __builtin_prefetch(&x[cols[i + d]],     0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 2]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 3]], 0, 3);

            reg256_double v0 = load_double_256(vals + i);
            reg256_double xv0 = _mm256_setr_pd(
                x[cols[i]], x[cols[i+1]], x[cols[i+2]], x[cols[i+3]]);
            acc0 = fma_double_256(v0, xv0, acc0);

            reg256_double v1 = load_double_256(vals + i + 4);
            reg256_double xv1 = _mm256_setr_pd(
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc1 = fma_double_256(v1, xv1, acc1);

            reg256_double v2 = load_double_256(vals + i + 8);
            reg256_double xv2 = _mm256_setr_pd(
                x[cols[i+8]], x[cols[i+9]], x[cols[i+10]], x[cols[i+11]]);
            acc2 = fma_double_256(v2, xv2, acc2);

            reg256_double v3 = load_double_256(vals + i + 12);
            reg256_double xv3 = _mm256_setr_pd(
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc3 = fma_double_256(v3, xv3, acc3);
        }

        for (; i + 16 <= n; i += 16)
        {
            reg256_double v0 = load_double_256(vals + i);
            reg256_double xv0 = _mm256_setr_pd(
                x[cols[i]], x[cols[i+1]], x[cols[i+2]], x[cols[i+3]]);
            acc0 = fma_double_256(v0, xv0, acc0);

            reg256_double v1 = load_double_256(vals + i + 4);
            reg256_double xv1 = _mm256_setr_pd(
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc1 = fma_double_256(v1, xv1, acc1);

            reg256_double v2 = load_double_256(vals + i + 8);
            reg256_double xv2 = _mm256_setr_pd(
                x[cols[i+8]], x[cols[i+9]], x[cols[i+10]], x[cols[i+11]]);
            acc2 = fma_double_256(v2, xv2, acc2);

            reg256_double v3 = load_double_256(vals + i + 12);
            reg256_double xv3 = _mm256_setr_pd(
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc3 = fma_double_256(v3, xv3, acc3);
        }

        for (; i + 4 <= n; i += 4)
        {
            reg256_double v = load_double_256(vals + i);
            reg256_double xv = _mm256_setr_pd(
                x[cols[i]], x[cols[i+1]], x[cols[i+2]], x[cols[i+3]]);
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
    else
    {
        for (; i + 16 <= n; i += 16)
        {
            reg256_double v0 = load_double_256(vals + i);
            reg256_double xv0 = _mm256_setr_pd(
                x[cols[i]], x[cols[i+1]], x[cols[i+2]], x[cols[i+3]]);
            acc0 = fma_double_256(v0, xv0, acc0);

            reg256_double v1 = load_double_256(vals + i + 4);
            reg256_double xv1 = _mm256_setr_pd(
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc1 = fma_double_256(v1, xv1, acc1);

            reg256_double v2 = load_double_256(vals + i + 8);
            reg256_double xv2 = _mm256_setr_pd(
                x[cols[i+8]], x[cols[i+9]], x[cols[i+10]], x[cols[i+11]]);
            acc2 = fma_double_256(v2, xv2, acc2);

            reg256_double v3 = load_double_256(vals + i + 12);
            reg256_double xv3 = _mm256_setr_pd(
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc3 = fma_double_256(v3, xv3, acc3);
        }

        for (; i + 4 <= n; i += 4)
        {
            reg256_double v = load_double_256(vals + i);
            reg256_double xv = _mm256_setr_pd(
                x[cols[i]], x[cols[i+1]], x[cols[i+2]], x[cols[i+3]]);
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
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols,
                           const float *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_float acc0 = zero_float_256();
    reg256_float acc1 = zero_float_256();
    reg256_float acc2 = zero_float_256();
    reg256_float acc3 = zero_float_256();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n, 8, 16))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(8, 16);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 32 <= prefetch_end; i += 32)
        {
            __builtin_prefetch(&x[cols[i + d]],     0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 2]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 3]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 4]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 5]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 6]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 7]], 0, 3);

            reg256_float v0 = load_float_256(vals + i);
            reg256_float xv0 = _mm256_setr_ps(
                x[cols[i]],   x[cols[i+1]], x[cols[i+2]], x[cols[i+3]],
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc0 = fma_float_256(v0, xv0, acc0);

            reg256_float v1 = load_float_256(vals + i + 8);
            reg256_float xv1 = _mm256_setr_ps(
                x[cols[i+8]],  x[cols[i+9]],  x[cols[i+10]], x[cols[i+11]],
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc1 = fma_float_256(v1, xv1, acc1);

            reg256_float v2 = load_float_256(vals + i + 16);
            reg256_float xv2 = _mm256_setr_ps(
                x[cols[i+16]], x[cols[i+17]], x[cols[i+18]], x[cols[i+19]],
                x[cols[i+20]], x[cols[i+21]], x[cols[i+22]], x[cols[i+23]]);
            acc2 = fma_float_256(v2, xv2, acc2);

            reg256_float v3 = load_float_256(vals + i + 24);
            reg256_float xv3 = _mm256_setr_ps(
                x[cols[i+24]], x[cols[i+25]], x[cols[i+26]], x[cols[i+27]],
                x[cols[i+28]], x[cols[i+29]], x[cols[i+30]], x[cols[i+31]]);
            acc3 = fma_float_256(v3, xv3, acc3);
        }

        for (; i + 32 <= n; i += 32)
        {
            reg256_float v0 = load_float_256(vals + i);
            reg256_float xv0 = _mm256_setr_ps(
                x[cols[i]],   x[cols[i+1]], x[cols[i+2]], x[cols[i+3]],
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc0 = fma_float_256(v0, xv0, acc0);

            reg256_float v1 = load_float_256(vals + i + 8);
            reg256_float xv1 = _mm256_setr_ps(
                x[cols[i+8]],  x[cols[i+9]],  x[cols[i+10]], x[cols[i+11]],
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc1 = fma_float_256(v1, xv1, acc1);

            reg256_float v2 = load_float_256(vals + i + 16);
            reg256_float xv2 = _mm256_setr_ps(
                x[cols[i+16]], x[cols[i+17]], x[cols[i+18]], x[cols[i+19]],
                x[cols[i+20]], x[cols[i+21]], x[cols[i+22]], x[cols[i+23]]);
            acc2 = fma_float_256(v2, xv2, acc2);

            reg256_float v3 = load_float_256(vals + i + 24);
            reg256_float xv3 = _mm256_setr_ps(
                x[cols[i+24]], x[cols[i+25]], x[cols[i+26]], x[cols[i+27]],
                x[cols[i+28]], x[cols[i+29]], x[cols[i+30]], x[cols[i+31]]);
            acc3 = fma_float_256(v3, xv3, acc3);
        }

        for (; i + 8 <= n; i += 8)
        {
            reg256_float v = load_float_256(vals + i);
            reg256_float xv = _mm256_setr_ps(
                x[cols[i]],   x[cols[i+1]], x[cols[i+2]], x[cols[i+3]],
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
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
    else
    {
        for (; i + 32 <= n; i += 32)
        {
            reg256_float v0 = load_float_256(vals + i);
            reg256_float xv0 = _mm256_setr_ps(
                x[cols[i]],   x[cols[i+1]], x[cols[i+2]], x[cols[i+3]],
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
            acc0 = fma_float_256(v0, xv0, acc0);

            reg256_float v1 = load_float_256(vals + i + 8);
            reg256_float xv1 = _mm256_setr_ps(
                x[cols[i+8]],  x[cols[i+9]],  x[cols[i+10]], x[cols[i+11]],
                x[cols[i+12]], x[cols[i+13]], x[cols[i+14]], x[cols[i+15]]);
            acc1 = fma_float_256(v1, xv1, acc1);

            reg256_float v2 = load_float_256(vals + i + 16);
            reg256_float xv2 = _mm256_setr_ps(
                x[cols[i+16]], x[cols[i+17]], x[cols[i+18]], x[cols[i+19]],
                x[cols[i+20]], x[cols[i+21]], x[cols[i+22]], x[cols[i+23]]);
            acc2 = fma_float_256(v2, xv2, acc2);

            reg256_float v3 = load_float_256(vals + i + 24);
            reg256_float xv3 = _mm256_setr_ps(
                x[cols[i+24]], x[cols[i+25]], x[cols[i+26]], x[cols[i+27]],
                x[cols[i+28]], x[cols[i+29]], x[cols[i+30]], x[cols[i+31]]);
            acc3 = fma_float_256(v3, xv3, acc3);
        }

        for (; i + 8 <= n; i += 8)
        {
            reg256_float v = load_float_256(vals + i);
            reg256_float xv = _mm256_setr_ps(
                x[cols[i]],   x[cols[i+1]], x[cols[i+2]], x[cols[i+3]],
                x[cols[i+4]], x[cols[i+5]], x[cols[i+6]], x[cols[i+7]]);
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
}

#endif
