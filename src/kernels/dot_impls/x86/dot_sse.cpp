#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_sse_aliases.hpp"

double sparse_dot_double_sse(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size)
{
    spira::kernel::simd::reg128_double acc = spira::kernel::simd::zero_double_128();
    size_t i = 0;
    double result;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = n - d;

        for (; i + 2 <= prefetch_end; i += 2)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 1 + d]], 0, 3);

            spira::kernel::simd::reg128_double v = spira::kernel::simd::load_double_128(vals + i);

            spira::kernel::simd::reg128_double xv = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);

            acc = spira::kernel::simd::add_double_128(acc, spira::kernel::simd::mul_double_128(v, xv));
        }

        for (; i + 2 <= n; i += 2)
        {
            spira::kernel::simd::reg128_double v = spira::kernel::simd::load_double_128(vals + i);

            spira::kernel::simd::reg128_double xv = _mm_set_pd(x[cols[i + 1]], x[cols[i]]);

            acc = spira::kernel::simd::add_double_128(acc, spira::kernel::simd::mul_double_128(v, xv));
        }

        result = spira::kernel::simd::reduce_sum_double_128(acc);

        for (; i < n; i++)
        {
            result += vals[i] * x[cols[i]];
        }
    }
    else
    {
        for (; i + 2 <= n; i += 2)
        {
            spira::kernel::simd::reg128_double v =
                spira::kernel::simd::load_double_128(vals + i);

            spira::kernel::simd::reg128_double xv =
                _mm_set_pd(x[cols[i + 1]], x[cols[i]]);

            acc = spira::kernel::simd::add_double_128(
                acc, spira::kernel::simd::mul_double_128(v, xv));
        }

        result = spira::kernel::simd::reduce_sum_double_128(acc);

        for (; i < n; i++)
        {
            result += vals[i] * x[cols[i]];
        }
    }

    return result;
}

float sparse_dot_float_sse(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    spira::kernel::simd::reg128_float acc = spira::kernel::simd::zero_float_128();
    size_t i = 0;

    float result;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = n - d;

        for (; i + 4 <= prefetch_end; i += 4)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 1 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 2 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 3 + d]], 0, 3);

            spira::kernel::simd::reg128_float v = spira::kernel::simd::load_float_128(vals + i);

            spira::kernel::simd::reg128_float xv = _mm_set_ps(
                x[cols[i + 3]],
                x[cols[i + 2]],
                x[cols[i + 1]],
                x[cols[i]]);

            acc = spira::kernel::simd::add_float_128(acc, spira::kernel::simd::mul_float_128(v, xv));
        }

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

        result = spira::kernel::simd::reduce_sum_float_128(acc);

        for (; i < n; i++)
        {
            result += vals[i] * x[cols[i]];
        }
    }
    else
    {
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

        result = spira::kernel::simd::reduce_sum_float_128(acc);

        for (; i < n; i++)
        {
            result += vals[i] * x[cols[i]];
        }
    }

    return result;
}
#endif