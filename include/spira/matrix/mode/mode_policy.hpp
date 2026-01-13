#pragma once

#include <spira/matrix/mode/matrix_mode.hpp>
#include <cstddef>

namespace spira::mode{
     struct mode_policy
    {
        std::size_t buffer_cap;
        std::size_t max_runs;
        std::size_t slab_merge_threshold;
        double compact_run_ratio;
    };

    static constexpr mode_policy policy_for(matrix_mode m)
    {
        switch (m)
        {
        case matrix_mode::spmv:
            return {.buffer_cap = 32, .max_runs = 0, .slab_merge_threshold = 0, .compact_run_ratio = 0.0};
        case matrix_mode::balanced:
            return {.buffer_cap = 128, .max_runs = 1, .slab_merge_threshold = 2048, .compact_run_ratio = 0.25};
        case matrix_mode::insert_heavy:
            return {.buffer_cap = 2048, .max_runs = 6, .slab_merge_threshold = 0, .compact_run_ratio = 0.75};
        }
        return {.buffer_cap = 32, .max_runs = 0, .slab_merge_threshold = 0, .compact_run_ratio = 0.0};
    }
}