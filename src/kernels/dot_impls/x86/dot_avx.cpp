
#include "../src/kernels/runtime_config.hpp"

#if defined(SPIRA_ARCH_X86)

#include "../src/kernels/simd_aliases/x86/simd_avx_aliases.hpp"

double sparse_dot_double_avx(const double *vals, const uint32_t *cols,
                             const double *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_double acc_reg = zero_double_256();
    double acc;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n, 4, 8))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(4, 8);
        const size_t prefetch_end = n - d;

        for (; i + 4 <= prefetch_end; i += 4)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 1 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 2 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 3 + d]], 0, 3);

            reg256_double v = load_double_256(vals + i);
            reg128_int idx = _mm_loadu_si128((const __m128i *)(cols + i));
            reg256_double xv = gather_double_256(x, idx);
            acc_reg = fma_double_256(v, xv, acc_reg);
        }

        for (; i + 4 <= n; i += 4)
        {
            reg256_double v = load_double_256(vals + i);
            reg128_int idx = _mm_loadu_si128((const __m128i *)(cols + i));
            reg256_double xv = gather_double_256(x, idx);
            acc_reg = fma_double_256(v, xv, acc_reg);
        }

        acc = reduce_sum_double_256(acc_reg);

        for (; i < n; i++)
        {
            acc += vals[i] * x[cols[i]];
        }
    }
    else
    {
        for (; i + 4 <= n; i += 4)
        {
            reg256_double v = load_double_256(vals + i);
            reg128_int idx = _mm_loadu_si128((const __m128i *)(cols + i));
            reg256_double xv = gather_double_256(x, idx);
            acc_reg = fma_double_256(v, xv, acc_reg);
        }

        acc = reduce_sum_double_256(acc_reg);

        for (; i < n; i++)
        {
            acc += vals[i] * x[cols[i]];
        }
    }

    return acc;
}

float sparse_dot_float_avx(const float *vals, const uint32_t *cols,
                           const float *x, size_t n, size_t x_size)
{
    using namespace spira::kernel::simd;
    size_t i = 0;
    reg256_float acc_reg = zero_float_256();
    float acc;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n, 8, 16))
    {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.prefetch_distance_for(8, 16);
        const size_t prefetch_end = n - d;

        for (; i + 8 <= prefetch_end; i += 8)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 1 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 2 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 3 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 4 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 5 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 6 + d]], 0, 3);
            __builtin_prefetch(&x[cols[i + 7 + d]], 0, 3);

            reg256_float v = load_float_256(vals + i);
            reg256_int idx = _mm256_loadu_si256((const __m256i *)(cols + i));
            reg256_float xv = gather_float_256(x, idx);
            acc_reg = fma_float_256(v, xv, acc_reg);
        }

        for (; i + 8 <= n; i += 8)
        {
            reg256_float v = load_float_256(vals + i);
            reg256_int idx = _mm256_loadu_si256((const __m256i *)(cols + i));
            reg256_float xv = gather_float_256(x, idx);
            acc_reg = fma_float_256(v, xv, acc_reg);
        }

        acc = reduce_sum_float_256(acc_reg);

        for (; i < n; i++)
        {
            acc += vals[i] * x[cols[i]];
        }
    }
    else
    {
        for (; i + 8 <= n; i += 8)
        {
            reg256_float v = load_float_256(vals + i);
            reg256_int idx = _mm256_loadu_si256((const __m256i *)(cols + i));
            reg256_float xv = gather_float_256(x, idx);
            acc_reg = fma_float_256(v, xv, acc_reg);
        }

        acc = reduce_sum_float_256(acc_reg);

        for (; i < n; i++)
        {
            acc += vals[i] * x[cols[i]];
        }
    }

    return acc;
}

#endif
