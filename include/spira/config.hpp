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

// ─────────────────────────────────────────────
// Row slack
// ─────────────────────────────────────────────

// How much spare room the locked CSR keeps, so that small edits land in place
// instead of rebuilding the matrix.
//
//   Rows — which rows get slack:
//       slack_rows::edited (default): rows are laid out packed, and a row gets
//           slack only once it has grown past its slot (it then moves to the
//           free tail with slack, and keeps slack through later repacks).
//           Rows that are never edited, or only have values changed or
//           entries deleted, stay packed, so SpMV reads no wasted slots.
//       slack_rows::all: every non-empty row gets slack at layout time.
//   Percent, Base — slots a row gets when the CSR is laid out (first lock,
//       repack, transpose):
//           capacity(len) = len + max(Base, len * Percent / 100)   for len > 0
//           capacity(0)   = 0
//       Percent is headroom as a percentage of the row length; Base is the
//       minimum spare slots for any non-empty row. Empty rows get none, so a
//       hypersparse matrix does not pay for rows it never fills; they are
//       placed in the free tail on their first insert. row_slack<0, 0> packs
//       rows with no slack, so every growing row relocates.
//
//   TailPercent — size of the free tail reserved after the row slots, as a
//       percentage of them. A row that outgrows its slot moves there.
//
//   RepackPercent — once the slots left behind by moved rows exceed this
//       percentage of all assigned slots, the next re-lock rebuilds the whole
//       CSR instead (a repack also happens whenever the tail runs out).
enum class slack_rows : uint8_t
{
    edited, // slack only for rows that have grown
    all     // slack for every non-empty row
};

template <std::size_t Percent, std::size_t Base,
          std::size_t TailPercent = 12, std::size_t RepackPercent = 25,
          slack_rows Rows = slack_rows::edited>
struct row_slack
{
    static_assert(RepackPercent <= 100, "RepackPercent is a percentage of the assigned slots");

    static constexpr slack_rows rows = Rows;
    static constexpr std::size_t percent = Percent;
    static constexpr std::size_t base = Base;
    static constexpr std::size_t tail_percent = TailPercent;
    static constexpr std::size_t repack_percent = RepackPercent;

    /// Slots for a row of len entries.
    static constexpr std::size_t capacity(std::size_t len) noexcept
    {
        if (len == 0)
            return 0;
        const std::size_t extra = len * Percent / 100;
        return len + (extra > Base ? extra : Base);
    }

    /// Slots for a row of len entries at layout time; grown says whether the
    /// row has ever outgrown its slot.
    static constexpr std::size_t layout_capacity(std::size_t len, bool grown) noexcept
    {
        return (Rows == slack_rows::all || grown) ? capacity(len) : len;
    }

    /// Free tail to reserve after assigned row slots.
    static constexpr std::size_t tail(std::size_t assigned) noexcept
    {
        return assigned * TailPercent / 100;
    }

    /// True if holes abandoned slots out of assigned are too many to keep.
    static constexpr bool should_repack(std::size_t holes, std::size_t assigned) noexcept
    {
        return holes * 100 > assigned * RepackPercent;
    }
};

template <class T>
inline constexpr bool is_row_slack_v = false;
template <std::size_t P, std::size_t B, std::size_t T, std::size_t R, slack_rows S>
inline constexpr bool is_row_slack_v<row_slack<P, B, T, R, S>> = true;

/// Slack only for rows that grow: 25 % headroom, at least one spare slot;
/// 12 % free tail; repack at 25 % holes.
using default_row_slack = row_slack<25, 1>;

} // namespace spira::config
