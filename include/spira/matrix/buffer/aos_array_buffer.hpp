#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/layout/element_pair.hpp>
#include <spira/traits.hpp>

namespace spira::buffer::impls
{

    template <class I, class V, std::size_t N>
    class aos_array_buffer : public spira::buffer::base_buffer<aos_array_buffer<I, V, N>, I, V>
    {
    public:
        using entry_type = spira::layout::elementPair<I, V>;
        using size_type = std::size_t;

        aos_array_buffer() { buf_.reserve(N); }

        [[nodiscard]] bool empty_impl() const noexcept { return buf_.empty(); }
        [[nodiscard]] size_type size_impl() const noexcept { return buf_.size(); }
        [[nodiscard]] size_type remaining_capacity_impl() const noexcept
        {
            return std::numeric_limits<size_type>::max();
        }

        void clear_impl() noexcept { buf_.clear(); index_.clear(); }

        [[nodiscard]] const I &key_at(size_type idx) const noexcept { return buf_[idx].column; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return buf_[idx].value; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return buf_[idx].value; }

        [[nodiscard]] entry_type *begin_impl() noexcept { return buf_.data(); }
        [[nodiscard]] entry_type *end_impl() noexcept { return buf_.data() + buf_.size(); }
        [[nodiscard]] const entry_type *begin_impl() const noexcept { return buf_.data(); }
        [[nodiscard]] const entry_type *end_impl() const noexcept { return buf_.data() + buf_.size(); }

        void push_back_impl(const I &col, const V &v)
        {
            buf_.push_back(entry_type{col, v});
            index_[col] = buf_.size() - 1;
        }

        bool contains_impl(I col) const noexcept
        {
            return index_.count(col) != 0;
        }

        const V *get_ptr_impl(I col) const noexcept
        {
            auto it = index_.find(col);
            if (it == index_.end())
                return nullptr;
            return &buf_[it->second].value;
        }

        // O(unique columns) — index_ always points to the last-written entry per column.
        V accumulate_impl() const noexcept
        {
            V acc = traits::ValueTraits<V>::zero();
            for (const auto &[col, idx] : index_)
                acc += buf_[idx].value;
            return acc;
        }

        /// Sort by column, deduplicate (last-write wins), keeping zero values.
        /// Identical to sort_and_dedup() but zeros are not removed, so that
        /// merge_csr can see them as deletion signals during compact_* lock cycles.
        void sort_and_dedup_keep_zeros() const
        {
            if (buf_.empty())
                return;

            std::reverse(buf_.begin(), buf_.end());
            std::stable_sort(buf_.begin(), buf_.end(), [](const auto &a, const auto &b)
                             { return a.first_ref() < b.first_ref(); });

            auto write = buf_.begin();
            I last_col{};
            bool first = true;
            for (const auto &e : buf_)
            {
                if (!first && e.first_ref() == last_col)
                    continue;
                last_col = e.first_ref();
                first = false;
                *write++ = e;
            }
            buf_.erase(write, buf_.end());

            index_.clear();
            for (std::size_t i = 0; i < buf_.size(); ++i)
                index_[buf_[i].column] = i;
        }

        /// Sort by column, deduplicate (last-write wins), and filter zero values.
        /// No temporary allocations: after reverse+stable_sort the dedup and
        /// zero-filter step uses an in-place write pointer (write <= read always).
        void sort_and_dedup() const
        {
            if (buf_.empty())
                return;

            // Reverse so last-inserted element sorts first on equal columns,
            // giving last-write-wins semantics after stable_sort.
            std::reverse(buf_.begin(), buf_.end());
            std::stable_sort(buf_.begin(), buf_.end(), [](const auto &a, const auto &b)
                             { return a.first_ref() < b.first_ref(); });

            // Compact in-place: write pointer is always <= read pointer, so no
            // element is overwritten before it is read.
            auto write = buf_.begin();
            I last_col{};
            bool first = true;
            for (const auto &e : buf_)
            {
                if (!first && e.first_ref() == last_col)
                    continue; // duplicate — last-written already kept
                last_col = e.first_ref();
                first = false;
                if (traits::ValueTraits<V>::is_zero(e.second_ref()))
                    continue; // last-write was zero = deletion
                *write++ = e;
            }
            buf_.erase(write, buf_.end());

            // Rebuild index map to match the compacted buffer.
            index_.clear();
            for (std::size_t i = 0; i < buf_.size(); ++i)
                index_[buf_[i].column] = i;
        }

    private:
        mutable std::vector<entry_type> buf_;
        mutable ankerl::unordered_dense::map<I, std::size_t> index_;
    };

} // namespace spira::buffer::impls
