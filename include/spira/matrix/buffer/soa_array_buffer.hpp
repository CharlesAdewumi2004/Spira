#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/traits.hpp>

namespace spira::buffer::impls
{

    template <class I, class V, std::size_t N>
    class soa_array_buffer : public spira::buffer::base_buffer<soa_array_buffer<I, V, N>, I, V>
    {
    public:
        using size_type = std::size_t;

        template <class VRef>
        struct entry
        {
            const I &column;
            VRef value;
            const I &first_ref() const noexcept { return column; }
            VRef second_ref() const noexcept { return value; }
        };

        soa_array_buffer()
        {
            col_.reserve(N);
            val_.reserve(N);
        }

        [[nodiscard]] bool empty_impl() const noexcept { return col_.empty(); }
        [[nodiscard]] size_type size_impl() const noexcept { return col_.size(); }

        void clear_impl() noexcept
        {
            col_.clear();
            val_.clear();
            index_.clear();
        }

        void push_back_impl(const I &col, const V &val)
        {
            col_.push_back(col);
            val_.push_back(val);
            index_[col] = col_.size() - 1;
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
            return &val_[it->second];
        }

        V *get_ptr_impl(I col) noexcept
        {
            auto it = index_.find(col);
            if (it == index_.end())
                return nullptr;
            return &val_[it->second];
        }

        // O(unique columns) — index_ always points to the last-written entry per column.
        V accumulate_impl() const noexcept
        {
            V acc = traits::ValueTraits<V>::zero();
            for (const auto &[col, idx] : index_)
                acc += val_[idx];
            return acc;
        }

        /// Sort by column, deduplicate (last-write wins), keeping zero values.
        /// Zeros survive to relock_rows, which reads them as deletion signals and
        /// filters them when writing the CSR.
        void sort_and_dedup()
        {
            if (col_.empty())
                return;
            // index_ already holds the last write for each column. Sorting in
            // thread-local scratch and assigning back keeps the capacity of
            // col_ and val_.
            thread_local std::vector<std::pair<I, size_type>> order;
            thread_local std::vector<V> vals;
            order.assign(index_.begin(), index_.end());
            std::sort(order.begin(), order.end());
            vals.clear();
            for (const auto &[col, idx] : order)
                vals.push_back(val_[idx]);
            col_.resize(order.size());
            for (std::size_t i = 0; i < order.size(); ++i)
                col_[i] = order[i].first;
            val_.assign(vals.begin(), vals.end());
            for (std::size_t i = 0; i < col_.size(); ++i)
                index_[col_[i]] = i;
        }

        // Forward iterator over (column, value) pairs held in two arrays.
        template <class VT>
        struct basic_iterator
        {
            const I *c{nullptr};
            VT *v{nullptr};
            entry<VT &> operator*() const noexcept { return {*c, *v}; }
            basic_iterator &operator++() noexcept { ++c; ++v; return *this; }
            bool operator==(const basic_iterator &o) const noexcept { return c == o.c; }
            std::ptrdiff_t operator-(const basic_iterator &o) const noexcept { return c - o.c; }
        };
        using iterator = basic_iterator<V>;
        using const_iterator = basic_iterator<const V>;

        [[nodiscard]] iterator begin_impl() noexcept { return {col_.data(), val_.data()}; }
        [[nodiscard]] iterator end_impl() noexcept { return {col_.data() + col_.size(), val_.data() + val_.size()}; }
        [[nodiscard]] const_iterator begin_impl() const noexcept { return {col_.data(), val_.data()}; }
        [[nodiscard]] const_iterator end_impl() const noexcept { return {col_.data() + col_.size(), val_.data() + val_.size()}; }

    private:
        std::vector<I> col_;
        std::vector<V> val_;
        ankerl::unordered_dense::map<I, std::size_t> index_;
    };

} // namespace spira::buffer::impls
