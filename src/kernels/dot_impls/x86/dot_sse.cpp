#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_sse_aliases.hpp"

double sparse_dot_double_sse(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg128_double acc0 = zero_double_128();
    reg128_double acc1 = zero_double_128();
    reg128_double acc2 = zero_double_128();
    reg128_double acc3 = zero_double_128();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n, 2, 4))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(2, 4);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 8 <= prefetch_end; i += 8)
        {
            __builtin_prefetch(&x[cols[i + d]],     0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);

            reg128_double v0 = load_double_128(vals + i);
            reg128_double xv0 = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
            acc0 = add_double_128(acc0, mul_double_128(v0, xv0));

            reg128_double v1 = load_double_128(vals + i + 2);
            reg128_double xv1 = _mm_set_pd(x[cols[i + 3]], x[cols[i + 2]]);
            acc1 = add_double_128(acc1, mul_double_128(v1, xv1));

            reg128_double v2 = load_double_128(vals + i + 4);
            reg128_double xv2 = _mm_set_pd(x[cols[i + 5]], x[cols[i + 4]]);
            acc2 = add_double_128(acc2, mul_double_128(v2, xv2));

            reg128_double v3 = load_double_128(vals + i + 6);
            reg128_double xv3 = _mm_set_pd(x[cols[i + 7]], x[cols[i + 6]]);
            acc3 = add_double_128(acc3, mul_double_128(v3, xv3));
        }

        for (; i + 8 <= n; i += 8)
        {
            reg128_double v0 = load_double_128(vals + i);
            reg128_double xv0 = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
            acc0 = add_double_128(acc0, mul_double_128(v0, xv0));

            reg128_double v1 = load_double_128(vals + i + 2);
            reg128_double xv1 = _mm_set_pd(x[cols[i + 3]], x[cols[i + 2]]);
            acc1 = add_double_128(acc1, mul_double_128(v1, xv1));

            reg128_double v2 = load_double_128(vals + i + 4);
            reg128_double xv2 = _mm_set_pd(x[cols[i + 5]], x[cols[i + 4]]);
            acc2 = add_double_128(acc2, mul_double_128(v2, xv2));

            reg128_double v3 = load_double_128(vals + i + 6);
            reg128_double xv3 = _mm_set_pd(x[cols[i + 7]], x[cols[i + 6]]);
            acc3 = add_double_128(acc3, mul_double_128(v3, xv3));
        }

        for (; i + 2 <= n; i += 2)
        {
            reg128_double v = load_double_128(vals + i);
            reg128_double xv = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
            acc0 = add_double_128(acc0, mul_double_128(v, xv));
        }

        acc0 = add_double_128(acc0, acc1);
        acc2 = add_double_128(acc2, acc3);
        acc0 = add_double_128(acc0, acc2);
        double result = reduce_sum_double_128(acc0);

        for (; i < n; i++)
            result += vals[i] * x[cols[i]];

        return result;
    }
    else
    {
        for (; i + 8 <= n; i += 8)
        {
            reg128_double v0 = load_double_128(vals + i);
            reg128_double xv0 = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
            acc0 = add_double_128(acc0, mul_double_128(v0, xv0));

            reg128_double v1 = load_double_128(vals + i + 2);
            reg128_double xv1 = _mm_set_pd(x[cols[i + 3]], x[cols[i + 2]]);
            acc1 = add_double_128(acc1, mul_double_128(v1, xv1));

            reg128_double v2 = load_double_128(vals + i + 4);
            reg128_double xv2 = _mm_set_pd(x[cols[i + 5]], x[cols[i + 4]]);
            acc2 = add_double_128(acc2, mul_double_128(v2, xv2));

            reg128_double v3 = load_double_128(vals + i + 6);
            reg128_double xv3 = _mm_set_pd(x[cols[i + 7]], x[cols[i + 6]]);
            acc3 = add_double_128(acc3, mul_double_128(v3, xv3));
        }

        for (; i + 2 <= n; i += 2)
        {
            reg128_double v = load_double_128(vals + i);
            reg128_double xv = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);
            acc0 = add_double_128(acc0, mul_double_128(v, xv));
        }

        acc0 = add_double_128(acc0, acc1);
        acc2 = add_double_128(acc2, acc3);
        acc0 = add_double_128(acc0, acc2);
        double result = reduce_sum_double_128(acc0);

        for (; i < n; i++)
            result += vals[i] * x[cols[i]];

        return result;
    }
}

float sparse_dot_float_sse(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg128_float acc0 = zero_float_128();
    reg128_float acc1 = zero_float_128();
    reg128_float acc2 = zero_float_128();
    reg128_float acc3 = zero_float_128();

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n, 4, 8))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(4, 8);
        const size_t prefetch_end = (n > d) ? n - d : 0;

        for (; i + 16 <= prefetch_end; i += 16)
        {
            __builtin_prefetch(&x[cols[i + d]],     0, 3);
            __builtin_prefetch(&x[cols[i + d + 1]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 2]], 0, 3);
            __builtin_prefetch(&x[cols[i + d + 3]], 0, 3);

            reg128_float v0 = load_float_128(vals + i);
            reg128_float xv0 = _mm_set_ps(
                x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
            acc0 = add_float_128(acc0, mul_float_128(v0, xv0));

            reg128_float v1 = load_float_128(vals + i + 4);
            reg128_float xv1 = _mm_set_ps(
                x[cols[i + 7]], x[cols[i + 6]], x[cols[i + 5]], x[cols[i + 4]]);
            acc1 = add_float_128(acc1, mul_float_128(v1, xv1));

            reg128_float v2 = load_float_128(vals + i + 8);
            reg128_float xv2 = _mm_set_ps(
                x[cols[i + 11]], x[cols[i + 10]], x[cols[i + 9]], x[cols[i + 8]]);
            acc2 = add_float_128(acc2, mul_float_128(v2, xv2));

            reg128_float v3 = load_float_128(vals + i + 12);
            reg128_float xv3 = _mm_set_ps(
                x[cols[i + 15]], x[cols[i + 14]], x[cols[i + 13]], x[cols[i + 12]]);
            acc3 = add_float_128(acc3, mul_float_128(v3, xv3));
        }

        for (; i + 16 <= n; i += 16)
        {
            reg128_float v0 = load_float_128(vals + i);
            reg128_float xv0 = _mm_set_ps(
                x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
            acc0 = add_float_128(acc0, mul_float_128(v0, xv0));

            reg128_float v1 = load_float_128(vals + i + 4);
            reg128_float xv1 = _mm_set_ps(
                x[cols[i + 7]], x[cols[i + 6]], x[cols[i + 5]], x[cols[i + 4]]);
            acc1 = add_float_128(acc1, mul_float_128(v1, xv1));

            reg128_float v2 = load_float_128(vals + i + 8);
            reg128_float xv2 = _mm_set_ps(
                x[cols[i + 11]], x[cols[i + 10]], x[cols[i + 9]], x[cols[i + 8]]);
            acc2 = add_float_128(acc2, mul_float_128(v2, xv2));

            reg128_float v3 = load_float_128(vals + i + 12);
            reg128_float xv3 = _mm_set_ps(
                x[cols[i + 15]], x[cols[i + 14]], x[cols[i + 13]], x[cols[i + 12]]);
            acc3 = add_float_128(acc3, mul_float_128(v3, xv3));
        }

        for (; i + 4 <= n; i += 4)
        {
            reg128_float v = load_float_128(vals + i);
            reg128_float xv = _mm_set_ps(
                x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
            acc0 = add_float_128(acc0, mul_float_128(v, xv));
        }

        acc0 = add_float_128(acc0, acc1);
        acc2 = add_float_128(acc2, acc3);
        acc0 = add_float_128(acc0, acc2);
        float result = reduce_sum_float_128(acc0);

        for (; i < n; i++)
            result += vals[i] * x[cols[i]];

        return result;
    }
    else
    {
        for (; i + 16 <= n; i += 16)
        {
            reg128_float v0 = load_float_128(vals + i);
            reg128_float xv0 = _mm_set_ps(
                x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
            acc0 = add_float_128(acc0, mul_float_128(v0, xv0));

            reg128_float v1 = load_float_128(vals + i + 4);
            reg128_float xv1 = _mm_set_ps(
                x[cols[i + 7]], x[cols[i + 6]], x[cols[i + 5]], x[cols[i + 4]]);
            acc1 = add_float_128(acc1, mul_float_128(v1, xv1));

            reg128_float v2 = load_float_128(vals + i + 8);
            reg128_float xv2 = _mm_set_ps(
                x[cols[i + 11]], x[cols[i + 10]], x[cols[i + 9]], x[cols[i + 8]]);
            acc2 = add_float_128(acc2, mul_float_128(v2, xv2));

            reg128_float v3 = load_float_128(vals + i + 12);
            reg128_float xv3 = _mm_set_ps(
                x[cols[i + 15]], x[cols[i + 14]], x[cols[i + 13]], x[cols[i + 12]]);
            acc3 = add_float_128(acc3, mul_float_128(v3, xv3));
        }

        for (; i + 4 <= n; i += 4)
        {
            reg128_float v = load_float_128(vals + i);
            reg128_float xv = _mm_set_ps(
                x[cols[i + 3]], x[cols[i + 2]], x[cols[i + 1]], x[cols[i]]);
            acc0 = add_float_128(acc0, mul_float_128(v, xv));
        }

        acc0 = add_float_128(acc0, acc1);
        acc2 = add_float_128(acc2, acc3);
        acc0 = add_float_128(acc0, acc2);
        float result = reduce_sum_float_128(acc0);

        for (; i < n; i++)
            result += vals[i] * x[cols[i]];

        return result;
    }
}
#endif
