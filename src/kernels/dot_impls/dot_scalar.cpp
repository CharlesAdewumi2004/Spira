#include <cstddef>
#include <cstdint>
#include <stddef.h>
#include "../runtime_config.hpp"

double sparse_dot_double_scalar(const double *vals, const uint32_t *cols, const double *x, size_t n, size_t x_size)
{
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(double), n)) {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = (n > d) ? n - d : 0;

        size_t i = 0;
        for (; i + 4 <= prefetch_end; i += 4)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        for (; i + 4 <= n; i += 4)
        {
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        double acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++)
            acc += x[cols[i]] * vals[i];
        return acc;
    } else {
        size_t i = 0;
        for (; i + 4 <= n; i += 4)
        {
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        double acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++)
            acc += x[cols[i]] * vals[i];
        return acc;
    }
}

float sparse_dot_float_scalar(const float *vals, const uint32_t *cols, const float *x, size_t n, size_t x_size)
{
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    if (spira::kernel::RuntimeConfig::dotRunPrefetch(x_size * sizeof(float), n)) {
        const size_t d = (size_t)spira::kernel::RuntimeConfig::get().memory.estimated_prefetch_distance;
        const size_t prefetch_end = (n > d) ? n - d : 0;

        size_t i = 0;
        for (; i + 4 <= prefetch_end; i += 4)
        {
            __builtin_prefetch(&x[cols[i + d]], 0, 3);
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        for (; i + 4 <= n; i += 4)
        {
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++)
            acc += x[cols[i]] * vals[i];
        return acc;
    } else {
        size_t i = 0;
        for (; i + 4 <= n; i += 4)
        {
            acc0 += x[cols[i]]     * vals[i];
            acc1 += x[cols[i + 1]] * vals[i + 1];
            acc2 += x[cols[i + 2]] * vals[i + 2];
            acc3 += x[cols[i + 3]] * vals[i + 3];
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < n; i++)
            acc += x[cols[i]] * vals[i];
        return acc;
    }
}
