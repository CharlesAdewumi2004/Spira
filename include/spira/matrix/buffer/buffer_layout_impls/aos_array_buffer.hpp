#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/layouts/element_pair.hpp>
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

        void clear_impl() noexcept { buf_.clear(); }

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
        }

        bool contains_impl(I col) const noexcept
        {
            for (auto i = buf_.size(); i-- > 0;)
            {
                if (buf_[i].column == col)
                    return true;
            }
            return false;
        }

        const V *get_ptr_impl(I col) const noexcept
        {
            for (auto i = buf_.size(); i-- > 0;)
            {
                if (buf_[i].column == col)
                    return &buf_[i].value;
            }
            return nullptr;
        }

        V accumulate_impl() const noexcept
        {
            sort_and_dedup();
            V acc = traits::ValueTraits<V>::zero();
            for (const auto &e : buf_)
                acc += e.value;
            return acc;
        }

        template <class layout_policy>
        layout_policy normalize_buffer_impl()
        {
            sort_and_dedup();
            layout_policy chunk;
            chunk.reserve(buf_.size());
            for (const auto &e : buf_)
                chunk.push_back(e.column, e.value);
            clear_impl();
            return chunk;
        }

        /// Sort by column, deduplicate (last-write wins), and filter zero values.
        void sort_and_dedup() const
        {
            if (buf_.empty())
                return;

            // Reverse so last-inserted element sorts first on equal columns.
            std::reverse(buf_.begin(), buf_.end());

            std::stable_sort(buf_.begin(), buf_.end(), [](const auto &a, const auto &b)
                             { return a.first_ref() < b.first_ref(); });

            // Track the last column processed (regardless of zero-filtering) so
            // that a zero last-write truly erases all prior writes for that column.
            std::vector<entry_type> out;
            out.reserve(buf_.size());
            I last_col{};
            bool first = true;
            for (const auto &e : buf_)
            {
                if (!first && e.first_ref() == last_col)
                    continue; // duplicate — last-written already handled
                last_col = e.first_ref();
                first = false;
                if (traits::ValueTraits<V>::is_zero(e.second_ref()))
                    continue; // last-write was zero = deletion
                out.push_back(e);
            }
            buf_ = std::move(out);
        }

    private:
        mutable std::vector<entry_type> buf_;
    };

} // namespace spira::buffer::impls
