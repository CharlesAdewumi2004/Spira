#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <spira/concepts.hpp>
#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/row.hpp>
#include <spira/matrix/storage/csr_storage.hpp>

namespace spira::parallel
{

    // ─────────────────────────────────────────────────────────────────────────────
    // partition<LayoutTag, I, V, BufferTag, BufferN, LP>
    //
    // Owns everything one thread needs to work independently:
    //   - [row_start, row_end): the global row range this thread owns
    //   - rows: the row objects for those rows (buffer + CSR slice state)
    //   - csr:  the flat CSR storage for this partition's rows
    //
    // Row indices are 0-based within the partition (local). Use local_row() to
    // convert a global row index to a local one.
    //
    // No synchronisation primitives — this is pure per-thread state.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag,
              concepts::Indexable I = uint32_t,
              concepts::Valueable V = double,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64,
              config::lock_policy LP = config::lock_policy::compact_preserve>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag>
    struct partition
    {
        using row_type = row<LayoutTag, I, V, BufferTag, BufferN>;

        std::size_t row_start{0};
        std::size_t row_end{0};
        std::vector<row_type> rows{};
        csr_storage<LayoutTag, I, V> csr{};

        [[nodiscard]] std::size_t size() const noexcept
        {
            return row_end - row_start;
        }

        [[nodiscard]] std::size_t local_row(std::size_t global_row) const noexcept
        {
            return global_row - row_start;
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // compute_partition_boundaries
    //
    // Given per-row nnz counts and a thread count, returns a boundary vector of
    // size n_threads + 1 where:
    //   boundaries[t]     = first global row owned by thread t
    //   boundaries[t + 1] = one-past-last global row owned by thread t
    //
    // Partitioning is nnz-balanced: each thread receives approximately
    // total_nnz / n_threads non-zeros. Boundary placement uses a prefix-sum
    // binary search — O(n_rows + n_threads * log(n_rows)).
    //
    // Edge cases:
    //   total_nnz == 0    → uniform row-count split.
    //   n_threads >= n_rows → excess threads receive empty partitions [x, x).
    //   n_threads == 1    → boundaries = {0, n_rows}.
    // ─────────────────────────────────────────────────────────────────────────────

    inline std::vector<std::size_t> compute_partition_boundaries(
        const std::vector<std::size_t> &row_nnz,
        std::size_t n_threads)
    {
        const std::size_t n_rows = row_nnz.size();

        // Default: all threads point to n_rows (safe empty range), then fix up.
        std::vector<std::size_t> boundaries(n_threads + 1, n_rows);
        boundaries[0] = 0;

        if (n_threads <= 1 || n_rows == 0)
            return boundaries;

        // Prefix sum: prefix[i+1] = cumulative nnz for rows 0..i.
        std::vector<std::size_t> prefix(n_rows + 1, 0);
        for (std::size_t i = 0; i < n_rows; ++i)
            prefix[i + 1] = prefix[i] + row_nnz[i];

        const std::size_t total_nnz = prefix[n_rows];

        if (total_nnz == 0)
        {
            // Uniform row-count split: spread rows evenly across threads.
            for (std::size_t t = 1; t < n_threads; ++t)
                boundaries[t] = t * n_rows / n_threads;
            return boundaries;
        }

        // Place each interior boundary at the first row r where the cumulative
        // nnz prefix[r] reaches the per-thread target.  lower_bound gives the
        // smallest r satisfying prefix[r] >= target, naturally placing the
        // boundary so thread t-1 accumulates exactly (or just over) its share.
        //
        // target is clamped to at least 1 to avoid stalling on threads whose
        // integer-divided share rounds to zero (happens when total_nnz < n_threads).
        for (std::size_t t = 1; t < n_threads; ++t)
        {
            const std::size_t target = std::max(std::size_t{1}, t * total_nnz / n_threads);
            const auto it = std::lower_bound(
                prefix.cbegin() + static_cast<std::ptrdiff_t>(boundaries[t - 1]),
                prefix.cend(),
                target);
            boundaries[t] = std::min(
                static_cast<std::size_t>(it - prefix.cbegin()),
                n_rows);
        }

        return boundaries;
    }

} // namespace spira::parallel
