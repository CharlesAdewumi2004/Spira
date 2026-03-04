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

        RuntimeConfig(const RuntimeConfig &) = delete;
        RuntimeConfig &operator=(const RuntimeConfig &) = delete;

    private:
        RuntimeConfig() : cpu{}, memory{measure_memory_latency()} {}
    };

} // namespace spira::kernel
