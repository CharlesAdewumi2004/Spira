#pragma once

#include <cstddef>
#include <cstdint>


namespace spira::config {

// ─────────────────────────────────────────────
// Matrix modes
// ─────────────────────────────────────────────

enum class matrix_mode : uint8_t {
    open,   // mutable: edits staged in per-row buffers; the committed CSR is kept
    locked  // frozen: edits merged into the CSR; reads go straight to it
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

} // namespace spira::config
