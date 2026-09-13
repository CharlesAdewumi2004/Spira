#pragma once
#include "hw_detect.hpp"

namespace spira::kernel
{

    struct RuntimeConfig
    {
        CpuFeatures cpu;

        static const RuntimeConfig &get()
        {
            static RuntimeConfig instance;
            return instance;
        }

        RuntimeConfig(const RuntimeConfig &) = delete;
        RuntimeConfig &operator=(const RuntimeConfig &) = delete;

    private:
        RuntimeConfig() : cpu{} {}
    };

} // namespace spira::kernel
