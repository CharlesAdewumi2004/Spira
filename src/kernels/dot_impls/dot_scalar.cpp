#include <cstddef>
#include <cstdint>
#include <stddef.h>
#include "../runtime_config.hpp"

double sparse_dot_double_scalar(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size)
{
    double acc = 0.0;
    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n)){
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = n - d;

        for (size_t i = 0; i < prefetch_end; i++)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            acc += x[cols[i]] * vals[i];
        }
        for (size_t i = prefetch_end; i < n; i++)
        {
            acc += x[cols[i]] * vals[i];
        }
    } else {
        for (size_t i = 0; i < n; i++)
        {
            acc += x[cols[i]] * vals[i];
        }
    }
    return acc;
}

float sparse_dot_float_scalar(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    float acc = 0.0f;
    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n)){
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = n - d;

        for (size_t i = 0; i < prefetch_end; i++)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            acc += x[cols[i]] * vals[i];
        }
        for (size_t i = prefetch_end; i < n; i++)
        {
            acc += x[cols[i]] * vals[i];
        }
    } else {
        for (size_t i = 0; i < n; i++)
        {
            acc += x[cols[i]] * vals[i];
        }
    }
    return acc;
}