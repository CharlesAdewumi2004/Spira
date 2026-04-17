#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <spira/concepts.hpp>
#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
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
    //                 For no_compact the CSR slice is never set; reads fall back
    //                 to the sorted buffer.
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

        // reserve_hint is accepted for API compatibility; buffer capacity is
        // controlled by BufferN (the initial vector reserve hint).
        row(size_type /*reserve_hint*/, size_type column_limit)
            : column_limit_{column_limit}
        {
        }

        // ─────────────────────────────────────────
        // Mode
        // ─────────────────────────────────────────

        [[nodiscard]] config::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] bool is_locked() const noexcept
        {
            return mode_ == config::matrix_mode::locked;
        }

        /// Sort + dedup + filter buffer in-place, then freeze.  O(k log k)
        void lock()
        {
            if (mode_ == config::matrix_mode::locked)
                return;
            buffer_.sort_and_dedup();
            mode_ = config::matrix_mode::locked;
        }

        /// Sort + dedup buffer in-place (keeping zeros), then freeze.  O(k log k)
        /// For compact_* policies only: zeros survive to merge_csr, which uses them
        /// to delete matching old CSR entries via its collision handler.
        void lock_for_compact()
        {
            if (mode_ == config::matrix_mode::locked)
                return;
            buffer_.sort_and_dedup_keep_zeros();
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

        /// Detach CSR slice (called at start of matrix::lock() before rebuild).
        void reset_csr_slice() noexcept { csr_slice_.reset(); }

        /// Clear staging buffer content (but keep allocation).
        /// Called by matrix::lock() after the CSR has been built.
        void clear_buffer_content() noexcept { buffer_.clear(); }

        /// Release staging buffer storage (compact_move policy).
        void release_buffer() noexcept { buffer_t{}.swap(buffer_); }

        // ─────────────────────────────────────────
        // Size / capacity
        // ─────────────────────────────────────────

        /// Locked compact_*: csr_slice_.nnz (exact).
        /// Locked no_compact: buffer_.size() (sorted+deduped, exact).
        /// Open: csr_slice_.nnz + buffer_.size() (upper bound; buffer may have dups).
        [[nodiscard]] size_type size() const noexcept
        {
            return csr_slice_.nnz + buffer_.size();
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return csr_slice_.nnz == 0 && buffer_.empty();
        }

        void clear() noexcept
        {
            assert(mode_ == config::matrix_mode::open &&
                   "row::clear() requires open mode");
            buffer_.clear();
            // CSR slice not touched — committed history persists.
        }

        void reserve(size_type /*n*/) noexcept {} // no-op: buffer is growable

        [[nodiscard]] size_type capacity() const noexcept { return 0; }

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

        // ─────────────────────────────────────────
        // Queries (both modes)
        //
        // Open:   buffer first (reverse linear, last-write wins), then CSR.
        // Locked: CSR slice if set; else sorted buffer (no_compact fallback).
        // ─────────────────────────────────────────

        [[nodiscard]] bool contains(index_type col) const
        {
            if (mode_ == config::matrix_mode::open)
            {
                if (buffer_.contains(col))
                    return true;
                return csr_slice_.binary_search(col) != nullptr;
            }
            // Locked
            if (csr_slice_.is_set())
                return csr_slice_.binary_search(col) != nullptr;
            return buffer_.contains(col);
        }

        [[nodiscard]] const value_type *get(index_type col) const
        {
            if (to_size(col) >= column_limit_)
                return nullptr;
            if (mode_ == config::matrix_mode::open)
            {
                if (const value_type *p = buffer_.get_ptr(col); p != nullptr)
                    return p;
                return csr_slice_.binary_search(col);
            }
            // Locked
            if (csr_slice_.is_set())
                return csr_slice_.binary_search(col);
            return buffer_.get_ptr(col);
        }

        [[nodiscard]] value_type accumulate() const noexcept
        {
            if (mode_ == config::matrix_mode::locked)
            {
                if (csr_slice_.is_set())
                    return csr_slice_.accumulate();
                // no_compact — sorted buffer
                value_type acc = traits::ValueTraits<value_type>::zero();
                for (const auto &entry : buffer_)
                    acc += entry.second_ref();
                return acc;
            }
            // Open mode — buffer (deduped, last-write wins) + CSR entries not
            // shadowed by the buffer.
            value_type acc = buffer_.accumulate();
            if (csr_slice_.is_set())
            {
                csr_slice_.for_each([&](I col, V val)
                                    {
                    if (!buffer_.contains(col))
                        acc += val; });
            }
            return acc;
        }

        // ─────────────────────────────────────────
        // Iteration
        //
        // begin()/end() return buffer iterators.
        //   — In locked mode (before set_csr_slice), buffer is sorted+deduped
        //     and ready to be consumed by build_csr / merge_csr.
        //   — In open mode, buffer is unsorted insertion-order.
        //
        // for_each_element() (const, locked) iterates the CSR slice if set,
        // otherwise the sorted buffer (no_compact fallback).
        // ─────────────────────────────────────────

        auto begin() noexcept { return buffer_.begin(); }
        auto end() noexcept { return buffer_.end(); }
        auto begin() const noexcept { return buffer_.begin(); }
        auto end() const noexcept { return buffer_.end(); }
        auto cbegin() const noexcept { return buffer_.cbegin(); }
        auto cend() const noexcept { return buffer_.cend(); }

        template <class Fn>
        void for_each_element(Fn &&f) const
        {
            assert(mode_ == config::matrix_mode::locked &&
                   "row::for_each_element() requires locked mode");
            if (csr_slice_.is_set())
            {
                csr_slice_.for_each(std::forward<Fn>(f));
            }
            else
            {
                for (const auto &entry : buffer_)
                    std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
            }
        }

        template <class Fn>
        void for_each_element(Fn &&f)
        {
            assert(mode_ == config::matrix_mode::open &&
                   "mutable row::for_each_element() requires open mode");
            for (auto &entry : buffer_)
                std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
        }

        /// Iterate committed (CSR slice or sorted buffer) entries, works in any mode.
        /// Used by scalar multiplication and similar algorithms that need to read
        /// committed data while in open mode.
        template <class Fn>
        void for_each_committed_element(Fn &&f) const
        {
            if (csr_slice_.is_set())
            {
                csr_slice_.for_each(std::forward<Fn>(f));
            }
            else
            {
                // no_compact: committed data lives in the sorted buffer.
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
