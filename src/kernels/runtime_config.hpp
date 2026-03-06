#pragma once
#include "hw_detect.hpp"

namespace spira::kernel
{

    struct RuntimeConfig
    {
        CpuFeatures cpu;
        MemoryLatencyInfo memory;

        static const RuntimeConfig &get()
        {
            static RuntimeConfig instance;
            return instance;
        }

        static bool dotRunPrefetch(size_t sizeInBytesOfVector, size_t rowSize,
                                   int stride = 1, int cycles_per_iter = 4)
        {
            int d = get().memory.prefetch_distance_for(stride, cycles_per_iter);
            return sizeInBytesOfVector > (size_t)get().cpu.cache.l2_size &&
                   rowSize >= (size_t)(d * 2);
        }

        RuntimeConfig(const RuntimeConfig &) = delete;
        RuntimeConfig &operator=(const RuntimeConfig &) = delete;

    private:
        RuntimeConfig() : cpu{}, memory{measure_memory_latency()} {}
    };

} // namespace spira::kernel
