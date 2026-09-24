#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
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

        void clear_impl() noexcept { buf_.clear(); index_.clear(); }

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

        V *get_ptr_impl(I col) noexcept
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
        /// Zeros survive to relock_rows, which reads them as deletion signals and
        /// filters them when writing the CSR.
        void sort_and_dedup()
        {
            if (buf_.empty())
                return;
            // index_ already holds the last write for each column. Sorting in
            // thread-local scratch and assigning back keeps buf_'s capacity.
            thread_local std::vector<entry_type> out;
            out.clear();
            for (const auto &[col, idx] : index_)
                out.push_back(buf_[idx]);
            std::sort(out.begin(), out.end(), [](const auto &a, const auto &b)
                      { return a.column < b.column; });
            buf_.assign(out.begin(), out.end());
            for (std::size_t i = 0; i < buf_.size(); ++i)
                index_[buf_[i].column] = i;
        }

    private:
        std::vector<entry_type> buf_;
        ankerl::unordered_dense::map<I, std::size_t> index_;
    };

} // namespace spira::buffer::impls
