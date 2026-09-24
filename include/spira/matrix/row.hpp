#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <spira/concepts.hpp>
#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/matrix/storage/csr_storage.hpp>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ─────────────────────────────────────────────────────────────────────────────
    // row<LayoutTag, I, V, BufferTag, BufferN>
    //
    // Two-mode buffer+CSR design:
    //
    //   Open mode  — inserts stage in buffer_ (unsorted, growable).
    //                Reads check buffer first (last-write wins), then the
    //                committed CSR slice from the previous lock cycle.
    //
    //   Locked mode — buffer_ is sorted, deduplicated, and zero-filtered in-place.
    //                 matrix::lock() then builds/merges a flat CSR and calls
    //                 set_csr_slice() to install the slice, then calls
    //                 clear_buffer_content() to free the staging area.
    //                 Reads use the CSR slice.
    //
    // lock()  — sort+dedup+filter buffer in-place; set locked.  O(k log k)
    // open()  — set flag to open; CSR slice and buffer left as-is.  O(1)
    //
    // The CSR slice type depends on LayoutTag (soa_tag -> separate cols/vals
    // pointers; aos_tag -> interleaved elementPair pointer).  Slice pointers are
    // owned by the parent matrix::csr_ object and remain valid until the next
    // lock() call on the matrix (which may reallocate csr_).
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V,
              class BufferTag = buffer::tags::array_buffer<layout::tags::aos_tag>,
              std::size_t BufferN = 64>
        requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V> && layout::ValidLayoutTag<LayoutTag>
    class row
    {
    public:
        using buffer_t = buffer::traits::traits_of_type<BufferTag, I, V, BufferN>;
        using index_type = I;
        using value_type = V;
        using size_type = std::size_t;

        // ─────────────────────────────────────────
        // Construction
        // ─────────────────────────────────────────

        row() = default;

        explicit row(size_type column_limit) : column_limit_{column_limit} {}

        // ─────────────────────────────────────────
        // Mode
        // ─────────────────────────────────────────

        [[nodiscard]] bool is_locked() const noexcept
        {
            return mode_ == config::matrix_mode::locked;
        }

        /// Sort + dedup buffer in-place, then freeze.  O(k log k)
        /// Zeros are kept: they survive to relock_rows, which uses them to delete
        /// matching old CSR entries via its collision handler.
        void lock()
        {
            if (mode_ == config::matrix_mode::locked)
                return;
            buffer_.sort_and_dedup();
            mode_ = config::matrix_mode::locked;
        }

        /// Reopen for mutations.  O(1) — CSR slice and buffer left as-is.
        void open() { mode_ = config::matrix_mode::open; }

        // ─────────────────────────────────────────
        // CSR slice management (called by matrix)
        // ─────────────────────────────────────────

        /// Install a layout-appropriate CSR slice from the parent matrix's flat CSR.
        /// The slice remains valid until the next matrix::lock() call.
        void set_csr_slice(csr_slice<LayoutTag, I, V> s) noexcept
        {
            csr_slice_ = s;
        }

        /// Clear staging buffer content (but keep allocation).
        /// Called by matrix::lock() after the CSR has been built.
        ///
        /// The empty() guard matters: a buffer's column-index map keeps the
        /// bucket array it grew during the initial bulk fill, and clear() memsets
        /// that array whatever its size. Without the guard every lock() memsets
        /// every row's buckets even when no row changed.
        void clear_buffer_content() noexcept
        {
            if (!buffer_.empty())
                buffer_.clear();
        }

        // ─────────────────────────────────────────
        // Size / capacity
        // ─────────────────────────────────────────

        /// Locked (matrix-owned): csr_slice_.nnz (exact; buffer cleared by lock()).
        /// Locked (standalone): buffer_.size() (sorted+deduped, exact).
        /// Open: csr_slice_.nnz + buffer_.size() (upper bound; buffer may have dups).
        [[nodiscard]] size_type size() const noexcept
        {
            return csr_slice_.nnz + buffer_.size();
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return csr_slice_.nnz == 0 && buffer_.empty();
        }

        /// True if the buffer holds entries not yet committed to the CSR.
        [[nodiscard]] bool has_buffered() const noexcept { return !buffer_.empty(); }

        void clear() noexcept
        {
            assert(mode_ == config::matrix_mode::open &&
                   "row::clear() requires open mode");
            buffer_.clear();
            // CSR slice not touched — committed history persists.
        }

        // ─────────────────────────────────────────
        // Mutation (open mode only)
        // ─────────────────────────────────────────

        void insert(index_type col, const value_type &val)
        {
            assert(mode_ == config::matrix_mode::open &&
                   "row::insert() requires open mode");
            if (to_size(col) >= column_limit_)
                throw std::out_of_range("Column index out of range");
            buffer_.push_back(col, val);
        }

        /// Add val onto the entry's current value (staged or committed); a
        /// missing entry counts as zero. The sum is staged like an insert, so
        /// a sum of exactly zero deletes the entry at the next lock().
        void add(index_type col, const value_type &val)
        {
            assert(mode_ == config::matrix_mode::open &&
                   "row::add() requires open mode");
            if (to_size(col) >= column_limit_)
                throw std::out_of_range("Column index out of range");
            if (value_type *p = buffer_.get_ptr(col); p != nullptr)
            {
                *p += val;
                return;
            }
            const value_type *current = get(col);
            value_type sum = current ? *current : traits::ValueTraits<value_type>::zero();
            sum += val;
            buffer_.push_back(col, sum);
        }

        // ─────────────────────────────────────────
        // Queries (both modes)
        //
        // Open:   buffer first (reverse linear, last-write wins), then CSR.
        // Locked: CSR slice if installed, else the sorted buffer.
        // ─────────────────────────────────────────

        [[nodiscard]] bool contains(index_type col) const
        {
            return buffer_.contains(col) || csr_slice_.binary_search(col) != nullptr;
        }

        [[nodiscard]] const value_type *get(index_type col) const
        {
            if (to_size(col) >= column_limit_)
                return nullptr;
            if (const value_type *p = buffer_.get_ptr(col); p != nullptr)
                return p;
            return csr_slice_.binary_search(col);
        }

        [[nodiscard]] value_type accumulate() const noexcept
        {
            if (buffer_.empty())
                return csr_slice_.accumulate();
            value_type acc = buffer_.accumulate();
            csr_slice_.for_each([&](I col, V val)
                                { if (!buffer_.contains(col)) acc += val; });
            return acc;
        }

        // ─────────────────────────────────────────
        // Iteration
        //
        // begin()/end() return buffer iterators.
        //   — In locked mode (before set_csr_slice), buffer is sorted+deduped
        //     and ready to be consumed by build_csr / relock_rows.
        //   — In open mode, buffer is unsorted insertion-order.
        //
        // for_each_element() visits the committed entries: the CSR slice once
        // the row belongs to a locked matrix, else the (sorted) buffer of a
        // standalone row that has never had a slice.
        // ─────────────────────────────────────────

        auto begin() noexcept { return buffer_.begin(); }
        auto end() noexcept { return buffer_.end(); }
        auto begin() const noexcept { return buffer_.begin(); }
        auto end() const noexcept { return buffer_.end(); }

        template <class Fn>
        void for_each_element(Fn &&f) const
        {
            if (csr_slice_.is_set())
            {
                csr_slice_.for_each(std::forward<Fn>(f));
            }
            else
            {
                // Standalone row — the buffer is the committed store.
                for (const auto &entry : buffer_)
                    std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
            }
        }

    private:
        static constexpr size_type to_size(index_type i) noexcept
        {
            return static_cast<size_type>(i);
        }

    private:
        buffer_t buffer_{};
        csr_slice<LayoutTag, I, V> csr_slice_{};
        config::matrix_mode mode_{config::matrix_mode::open};
        size_type column_limit_{0};
    };

} // namespace spira
