#pragma once

#include <cstddef>
#include <functional>
#include <string_view>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // job
    //
    // Describes a single unit of parallel work broadcast to all worker threads.
    // The callable fn is invoked as fn(thread_id) on every thread in the pool.
    //
    // Only one job is active at a time — thread_pool::execute() blocks until all
    // threads complete, then the job is discarded.
    //
    // Metadata fields (id, name) are for debugging and profiling only; they have
    // no effect on execution.
    // ─────────────────────────────────────────────────────────────────────────────

    struct job
    {
        // Incremented by the pool on each execute() call.
        std::size_t id{0};

        // Optional human-readable label — e.g. "lock", "spmv".
        // Must outlive the job (use a string literal or a stable std::string).
        std::string_view name{};

        // The work to run on each thread.
        // Signature: fn(thread_id) where thread_id ∈ [0, n_threads).
        std::move_only_function<void(std::size_t)> fn{};

        [[nodiscard]] bool empty() const noexcept { return !fn; }
    };

} // namespace spira::parallel
