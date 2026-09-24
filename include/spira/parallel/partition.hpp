#pragma once

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
    // partition<LayoutTag, I, V, BufferTag, BufferN>
    //
    // Owns everything one thread needs to work independently:
    //   - [row_start, row_end): the global row range this thread owns
    //   - rows: the row objects for those rows (buffer + CSR slice state)
    //   - csr:  the flat CSR storage for this partition's rows
    //   - dirty bookkeeping: which rows have pending edits for the next lock
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
              std::size_t BufferN = 64>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> &&
                 layout::ValidLayoutTag<LayoutTag>
    struct partition
    {
        using row_type = row<LayoutTag, I, V, BufferTag, BufferN>;

        std::size_t row_start{0};
        std::size_t row_end{0};
        std::vector<row_type> rows{};
        csr_storage<LayoutTag, I, V> csr{};

        std::vector<bool> dirty{};             // local row i is in dirty_rows
        std::vector<std::size_t> dirty_rows{}; // local rows edited since the last lock
        bool scan_all{false};                  // rows were handed out wholesale (parallel_fill)

        /// Size the dirty bookkeeping for local_n rows.
        void reset_dirty(std::size_t local_n)
        {
            dirty.assign(local_n, false);
            dirty_rows.clear();
            scan_all = false;
        }

        /// Record that local row i has pending edits: reopen it and queue it
        /// for the next lock. Each row is queued at most once per cycle.
        void mark_dirty(std::size_t i)
        {
            if (dirty[i])
                return;
            dirty[i] = true;
            dirty_rows.push_back(i);
            rows[i].open();
        }

        /// Local row i, queued for the next lock and open for writing. Every
        /// algorithm that writes into a partition's rows goes through this.
        [[nodiscard]] row_type &writable_row(std::size_t i)
        {
            mark_dirty(i);
            return rows[i];
        }

        /// Open every row for direct writes; the next lock finds the edited
        /// rows by scanning their buffers.
        void open_all()
        {
            for (auto &r : rows)
                r.open();
            scan_all = true;
        }

        /// Drop every pending edit and return the touched rows to locked mode.
        void clear_pending()
        {
            auto reset = [](row_type &r)
            {
                r.clear();
                r.lock();
            };
            if (scan_all)
                for (auto &r : rows)
                    reset(r);
            else
                for (const std::size_t i : dirty_rows)
                    reset(rows[i]);
            end_cycle();
        }

        /// Forget the rows queued in this cycle (after a lock or clear).
        void end_cycle()
        {
            for (const std::size_t i : dirty_rows)
                dirty[i] = false;
            dirty_rows.clear();
            scan_all = false;
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return row_end - row_start;
        }

        [[nodiscard]] std::size_t local_row(std::size_t global_row) const noexcept
        {
            return global_row - row_start;
        }
    };

} // namespace spira::parallel
