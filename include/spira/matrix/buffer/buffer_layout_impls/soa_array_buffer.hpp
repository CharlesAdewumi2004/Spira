#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

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
        }

        void push_back_impl(const I &col, const V &val)
        {
            col_.push_back(col);
            val_.push_back(val);
        }

        bool contains_impl(I col) const noexcept
        {
            for (auto i = col_.size(); i-- > 0;)
            {
                if (col_[i] == col)
                    return true;
            }
            return false;
        }

        const V *get_ptr_impl(I col) const noexcept
        {
            for (auto i = col_.size(); i-- > 0;)
            {
                if (col_[i] == col)
                    return &val_[i];
            }
            return nullptr;
        }

        V accumulate_impl() const noexcept
        {
            sort_and_dedup();
            V acc = traits::ValueTraits<V>::zero();
            for (std::size_t i = 0; i < col_.size(); ++i)
                acc += val_[i];
            return acc;
        }

        template <class layout_policy>
        layout_policy normalize_buffer_impl()
        {
            sort_and_dedup();
            layout_policy chunk;
            chunk.reserve(col_.size());
            for (std::size_t i = 0; i < col_.size(); ++i)
                chunk.push_back(col_[i], val_[i]);
            clear_impl();
            return chunk;
        }

        /// Sort by column, deduplicate (last-write wins), and filter zero values.
        /// After this call the buffer is sorted, unique, and zero-free.
        void sort_and_dedup() const
        {
            const std::size_t sz = col_.size();
            if (sz == 0)
                return;

            // Build index array reversed so last-inserted element appears first
            // after sort, giving last-write-wins semantics on equal columns.
            std::vector<size_type> idx(sz);
            for (size_type i = 0; i < sz; ++i)
                idx[i] = sz - 1 - i;

            std::stable_sort(idx.begin(), idx.end(),
                             [&](size_type a, size_type b)
                             { return col_[a] < col_[b]; });

            std::vector<I> new_col;
            std::vector<V> new_val;
            new_col.reserve(sz);
            new_val.reserve(sz);

            // Track the last column processed (regardless of zero-filtering) so
            // that a zero last-write truly erases all prior writes for that column.
            I last_col{};
            bool first = true;
            for (size_type i = 0; i < sz; ++i)
            {
                const size_type j = idx[i];
                if (!first && col_[j] == last_col)
                    continue; // duplicate — last-written (first occurrence) already handled
                last_col = col_[j];
                first = false;
                if (traits::ValueTraits<V>::is_zero(val_[j]))
                    continue; // last-write was zero = deletion, don't emit
                new_col.push_back(col_[j]);
                new_val.push_back(val_[j]);
            }

            col_ = std::move(new_col);
            val_ = std::move(new_val);
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

            friend iterator operator+(iterator it, difference_type n) noexcept { it += n; return it; }
            friend iterator operator+(difference_type n, iterator it) noexcept { it += n; return it; }
            friend iterator operator-(iterator it, difference_type n) noexcept { it -= n; return it; }
            friend difference_type operator-(const iterator &a, const iterator &b) noexcept {
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
    };

} // namespace spira::buffer::impls
