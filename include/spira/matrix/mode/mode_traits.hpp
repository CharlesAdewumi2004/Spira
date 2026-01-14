#include <spira/matrix/mode/matrix_mode.hpp>
#include <cstddef>
namespace spira::mode
{
    struct mode_policy
    {
        std::size_t max_runs;             // max runs allowed before compaction
        std::size_t slab_merge_threshold; // if slab is small, merge directly instead of making a run
        double compact_run_ratio;         // compact if run_entries > ratio * slab_entries
    };

    static constexpr mode_policy policy_for(matrix_mode m)
    {
        switch (m)
        {
        case matrix_mode::spmv:
            return {.max_runs = 0, .slab_merge_threshold = 0, .compact_run_ratio = 0.0};
        case matrix_mode::balanced:
            return {.max_runs = 1, .slab_merge_threshold = 2048, .compact_run_ratio = 0.25};
        case matrix_mode::insert_heavy:
            return {.max_runs = 6, .slab_merge_threshold = 0, .compact_run_ratio = 0.75};
        }
        return {.max_runs = 0, .slab_merge_threshold = 0, .compact_run_ratio = 0.0};
    }

}