#pragma once

#include <cstddef>
#include <cstdint>


namespace spira::config {

// ─────────────────────────────────────────────
// Matrix modes
// ─────────────────────────────────────────────

enum class matrix_mode : uint8_t {
    open,   // mutable: inserts staged in per-row buffer, slab preserved from prior cycles
    locked  // frozen: buffer merged into slab, one sorted array per row, zero-overhead reads
};

// ─────────────────────────────────────────────
// Insert policy (parallel_matrix only)
// ─────────────────────────────────────────────

enum class insert_policy : uint8_t {
    direct, // write straight to partition row buffers — zero overhead, cache-hostile
            // under random row arrival order
    staged  // accumulate inserts in a per-partition staging array on the main thread;
            // burst-flush to row buffers when staging capacity is reached or at lock().
            // keeps the active staging array hot in L1/L2 by writing to one partition
            // at a time
};

inline constexpr std::size_t default_row_reserve_hint = 0;

} // namespace spira::config
