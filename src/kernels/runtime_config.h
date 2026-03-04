#pragma once
#include "hw_detect.h"

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

        static bool dotRunPrefetch(size_t sizeInBytesOfVector, size_t rowSize){
            return sizeInBytesOfVector > (size_t)get().cpu.cache.l2_size &&
                   rowSize >= (size_t)(get().memory.estimated_prefetch_distance * 2);
        }

        RuntimeConfig(const RuntimeConfig &) = delete;
        RuntimeConfig &operator=(const RuntimeConfig &) = delete;

    private:
        RuntimeConfig() : cpu{}, memory{measure_memory_latency()} {}
    };

} // namespace spira::kernel
