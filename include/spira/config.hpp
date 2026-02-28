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
// Search policies
// ─────────────────────────────────────────────

using aos_search_policy = boundcraft::policy::hybrid<32>;
using soa_search_policy = boundcraft::policy::hybrid<32>;

inline constexpr std::size_t default_row_reserve_hint = 0;

} // namespace spira::config
