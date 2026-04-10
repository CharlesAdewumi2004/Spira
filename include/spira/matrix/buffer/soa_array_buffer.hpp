#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
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

        struct entry_ref
        {
            const I &column;
            V &value;
            const I &first_ref() const noexcept { return column; }
            V &second_ref() noexcept { return value; }
            const V &second_ref() const noexcept { return value; }
        };
        struct entry_cref
        {
            const I &column;
            const V &value;
            const I &first_ref() const noexcept { return column; }
            const V &second_ref() const noexcept { return value; }
        };

        soa_array_buffer()
        {
            col_.reserve(N);
            val_.reserve(N);
        }

        [[nodiscard]] bool empty_impl() const noexcept { return col_.empty(); }
        [[nodiscard]] size_type size_impl() const noexcept { return col_.size(); }
        [[nodiscard]] size_type remaining_capacity_impl() const noexcept
        {
            return std::numeric_limits<size_type>::max();
        }

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

        // O(unique columns) — index_ always points to the last-written entry per column.
        V accumulate_impl() const noexcept
        {
            V acc = traits::ValueTraits<V>::zero();
            for (const auto &[col, idx] : index_)
                acc += val_[idx];
            return acc;
        }

        /// Sort by column, deduplicate (last-write wins), and filter zero values.
        /// Only one allocation (idx). The sort permutation is applied to col_/val_
        /// in-place via cycle decomposition, then dedup+filter uses a write pointer.
        void sort_and_dedup() const
        {
            const std::size_t sz = col_.size();
            if (sz == 0)
                return;

            // Build reversed index so stable_sort gives last-write-wins on equal columns.
            thread_local std::vector<size_type> idx;
            idx.resize(sz);
            for (size_type i = 0; i < sz; ++i)
                idx[i] = sz - 1 - i;

            std::stable_sort(idx.begin(), idx.end(),
                             [&](size_type a, size_type b)
                             { return col_[a] < col_[b]; });

            // Apply permutation in-place via cycle decomposition.
            // Each cycle is traced from its lowest unprocessed index; idx[j] is
            // overwritten with j once element j has been placed, so visited
            // elements are naturally skipped.
            for (std::size_t i = 0; i < sz; ++i)
            {
                if (idx[i] == i)
                    continue;
                I tmp_col = col_[i];
                V tmp_val = val_[i];
                std::size_t j = i;
                while (idx[j] != i)
                {
                    col_[j] = col_[idx[j]];
                    val_[j] = val_[idx[j]];
                    const std::size_t k = idx[j];
                    idx[j] = j; // mark placed
                    j = k;
                }
                col_[j] = tmp_col;
                val_[j] = tmp_val;
                idx[j] = j; // close cycle
            }

            // Compact in-place: dedup + zero-filter with a write pointer.
            std::size_t write = 0;
            I last_col{};
            bool first = true;
            for (std::size_t i = 0; i < sz; ++i)
            {
                if (!first && col_[i] == last_col)
                    continue;
                last_col = col_[i];
                first = false;
                if (traits::ValueTraits<V>::is_zero(val_[i]))
                    continue;
                if (write != i)
                {
                    col_[write] = col_[i];
                    val_[write] = val_[i];
                }
                ++write;
            }
            col_.resize(write);
            val_.resize(write);

            // Rebuild index map to match the compacted buffer.
            index_.clear();
            for (std::size_t i = 0; i < col_.size(); ++i)
                index_[col_[i]] = i;
        }

        class iterator
        {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = entry_ref;
            using reference = entry_ref;

            iterator() = default;
            iterator(I *c, V *v) : col_ptr(c), val_ptr(v) {}

            reference operator*() const noexcept { return {*col_ptr, *val_ptr}; }

            iterator &operator++() noexcept
            {
                ++col_ptr;
                ++val_ptr;
                return *this;
            }
            iterator operator++(int) noexcept
            {
                auto t = *this;
                ++(*this);
                return t;
            }
            iterator &operator--() noexcept
            {
                --col_ptr;
                --val_ptr;
                return *this;
            }
            iterator operator--(int) noexcept
            {
                auto t = *this;
                --(*this);
                return t;
            }

            iterator &operator+=(difference_type n) noexcept
            {
                col_ptr += n;
                val_ptr += n;
                return *this;
            }
            iterator &operator-=(difference_type n) noexcept
            {
                col_ptr -= n;
                val_ptr -= n;
                return *this;
            }

            friend iterator operator+(iterator it, difference_type n) noexcept
            {
                it += n;
                return it;
            }
            friend iterator operator+(difference_type n, iterator it) noexcept
            {
                it += n;
                return it;
            }
            friend iterator operator-(iterator it, difference_type n) noexcept
            {
                it -= n;
                return it;
            }
            friend difference_type operator-(const iterator &a, const iterator &b) noexcept
            {
                return a.col_ptr - b.col_ptr;
            }

            friend bool operator==(const iterator &a, const iterator &b) noexcept { return a.col_ptr == b.col_ptr; }
            friend bool operator!=(const iterator &a, const iterator &b) noexcept { return !(a == b); }
            friend bool operator<(const iterator &a, const iterator &b) noexcept { return a.col_ptr < b.col_ptr; }
            friend bool operator>(const iterator &a, const iterator &b) noexcept { return b < a; }
            friend bool operator<=(const iterator &a, const iterator &b) noexcept { return !(b < a); }
            friend bool operator>=(const iterator &a, const iterator &b) noexcept { return !(a < b); }

            reference operator[](difference_type n) const noexcept { return *(*this + n); }

            // Public raw pointer access (used by some tests and SIMD paths).
            I *col_ptr{nullptr};
            V *val_ptr{nullptr};

        private:
        };

        class const_iterator
        {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = entry_cref;
            using reference = entry_cref;

            const_iterator() = default;
            const_iterator(const I *c, const V *v) : c_(c), v_(v) {}

            reference operator*() const noexcept { return {*c_, *v_}; }

            const_iterator &operator++() noexcept
            {
                ++c_;
                ++v_;
                return *this;
            }
            const_iterator operator++(int) noexcept
            {
                auto t = *this;
                ++(*this);
                return t;
            }
            const_iterator &operator--() noexcept
            {
                --c_;
                --v_;
                return *this;
            }
            const_iterator operator--(int) noexcept
            {
                auto t = *this;
                --(*this);
                return t;
            }

            const_iterator &operator+=(difference_type n) noexcept
            {
                c_ += n;
                v_ += n;
                return *this;
            }
            const_iterator &operator-=(difference_type n) noexcept
            {
                c_ -= n;
                v_ -= n;
                return *this;
            }

            friend const_iterator operator+(const_iterator it, difference_type n) noexcept
            {
                it += n;
                return it;
            }
            friend const_iterator operator+(difference_type n, const_iterator it) noexcept
            {
                it += n;
                return it;
            }
            friend const_iterator operator-(const_iterator it, difference_type n) noexcept
            {
                it -= n;
                return it;
            }
            friend difference_type operator-(const const_iterator &a, const const_iterator &b) noexcept
            {
                return a.c_ - b.c_;
            }

            friend bool operator==(const const_iterator &a, const const_iterator &b) noexcept { return a.c_ == b.c_; }
            friend bool operator!=(const const_iterator &a, const const_iterator &b) noexcept { return !(a == b); }
            friend bool operator<(const const_iterator &a, const const_iterator &b) noexcept { return a.c_ < b.c_; }
            friend bool operator>(const const_iterator &a, const const_iterator &b) noexcept { return b < a; }
            friend bool operator<=(const const_iterator &a, const const_iterator &b) noexcept { return !(b < a); }
            friend bool operator>=(const const_iterator &a, const const_iterator &b) noexcept { return !(a < b); }

            reference operator[](difference_type n) const noexcept { return *(*this + n); }

        private:
            const I *c_{nullptr};
            const V *v_{nullptr};
        };

        [[nodiscard]] iterator begin_impl() noexcept { return iterator(col_.data(), val_.data()); }
        [[nodiscard]] iterator end_impl() noexcept { return iterator(col_.data() + col_.size(), val_.data() + val_.size()); }

        [[nodiscard]] const_iterator begin_impl() const noexcept { return const_iterator(col_.data(), val_.data()); }
        [[nodiscard]] const_iterator end_impl() const noexcept
        {
            return const_iterator(col_.data() + col_.size(), val_.data() + val_.size());
        }

        // Raw pointer access for SIMD paths (valid after sort_and_dedup).
        [[nodiscard]] const I *col_data() const noexcept { return col_.data(); }
        [[nodiscard]] const V *val_data() const noexcept { return val_.data(); }

    private:
        mutable std::vector<I> col_;
        mutable std::vector<V> val_;
        mutable ankerl::unordered_dense::map<I, std::size_t> index_;
    };

} // namespace spira::buffer::impls
