#pragma once
#include <cstddef>

namespace spira::config
{

    struct mode_policy
    {
        size_t buffersize;
        std::size_t max_runs;
        std::size_t slab_merge_threshold;
        double compact_run_ratio;
    };

    inline constexpr mode_policy spmv{.buffersize = 32, .max_runs = 0, .slab_merge_threshold = 0, .compact_run_ratio = 0.0};
    inline constexpr mode_policy balanced{.buffersize = 128, .max_runs = 1, .slab_merge_threshold = 2048, .compact_run_ratio = 0.25};
    inline constexpr mode_policy insert_heavy{.buffersize = 2048, .max_runs = 6, .slab_merge_threshold = 0, .compact_run_ratio = 0.75};

}