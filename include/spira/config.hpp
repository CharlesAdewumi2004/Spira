#pragma once

#include <cstddef>
#include <cstdint>

#include <boundcraft/searcher.hpp>

namespace spira::config {

// ─────────────────────────────────────────────
// Matrix modes
// ─────────────────────────────────────────────

enum class matrix_mode : uint8_t {
    open,   // mutable: inserts staged in per-row buffer, slab preserved from prior cycles
    locked  // frozen: buffer merged into slab, one sorted array per row, zero-overhead reads
};

// ─────────────────────────────────────────────
// Lock policy
// ─────────────────────────────────────────────

enum class lock_policy : uint8_t {
    no_compact,       // locked mode uses per-row buffers directly; no CSR built
    compact_preserve, // build CSR at lock(), keep per-row buffers (2x memory, O(1) open())
    compact_move      // build CSR at lock(), free per-row buffers (1x memory, O(n) open())
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

// ─────────────────────────────────────────────
// Search policies
// ─────────────────────────────────────────────

using aos_search_policy = boundcraft::policy::hybrid<32>;
using soa_search_policy = boundcraft::policy::hybrid<32>;

inline constexpr std::size_t default_row_reserve_hint = 0;

} // namespace spira::config
