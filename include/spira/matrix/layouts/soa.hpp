#pragma once
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <boundcraft/searcher.hpp>

#include "../include/spira/concepts.hpp"

namespace spira::layout
{

    template <concepts::Indexable I, concepts::Valueable V>
    class soa
    {
    public:
        struct iterator
        {
            using iterator_concept = std::random_access_iterator_tag; 
            using iterator_category = std::random_access_iterator_tag; 
            using difference_type = std::ptrdiff_t;
            using value_type = std::pair<I, V>;

            struct ref
            {
                I &first;
                V &second;
                ref(I &i, V &v) : first(i), second(v) {}
                explicit operator value_type() const { return {first, second}; }
            };

            using pointer = void;
            using reference = ref;

            I *col_ptr{};
            V *val_ptr{};

            struct arrow_proxy
            {
                value_type val;
                arrow_proxy(I &i, V &v) : val{i, v} {}
                const value_type *operator->() const noexcept { return &val; }
            };

            iterator() = default;
            explicit iterator(I *c, V *v) : col_ptr(c), val_ptr(v) {}

            reference operator*() const noexcept { return reference{*col_ptr, *val_ptr}; }
            arrow_proxy operator->() const noexcept { return arrow_proxy{*col_ptr, *val_ptr}; }
            reference operator[](difference_type n) const noexcept { return reference{col_ptr[n], val_ptr[n]}; }

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
            friend bool operator<(const iterator &a, const iterator &b) noexcept { return a.col_ptr < b.col_ptr; }
            friend bool operator>(const iterator &a, const iterator &b) noexcept { return b < a; }
            friend bool operator<=(const iterator &a, const iterator &b) noexcept { return !(b < a); }
            friend bool operator>=(const iterator &a, const iterator &b) noexcept { return !(a < b); }
        };

        struct const_iterator
        {
            using iterator_concept = std::random_access_iterator_tag;
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = std::pair<I, V>;

            struct cref
            {
                const I &first;
                const V &second;
                cref(const I &i, const V &v) : first(i), second(v) {}
                explicit operator value_type() const { return {first, second}; }
            };

            using pointer = void;
            using reference = cref;

            const I *col_ptr{};
            const V *val_ptr{};

            struct arrow_proxy
            {
                value_type val;
                arrow_proxy(const I &i, const V &v) : val{i, v} {}
                const value_type *operator->() const noexcept { return &val; }
            };

            const_iterator() = default;
            explicit const_iterator(const I *c, const V *v) : col_ptr(c), val_ptr(v) {}

            reference operator*() const noexcept { return reference{*col_ptr, *val_ptr}; }
            arrow_proxy operator->() const noexcept { return arrow_proxy{*col_ptr, *val_ptr}; }
            reference operator[](difference_type n) const noexcept { return reference{col_ptr[n], val_ptr[n]}; }

            const_iterator &operator++() noexcept
            {
                ++col_ptr;
                ++val_ptr;
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
                --col_ptr;
                --val_ptr;
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
                col_ptr += n;
                val_ptr += n;
                return *this;
            }
            
            const_iterator &operator-=(difference_type n) noexcept
            {
                col_ptr -= n;
                val_ptr -= n;
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
                return a.col_ptr - b.col_ptr;
            }

            friend bool operator==(const const_iterator &a, const const_iterator &b) noexcept { return a.col_ptr == b.col_ptr; }
            friend bool operator<(const const_iterator &a, const const_iterator &b) noexcept { return a.col_ptr < b.col_ptr; }
            friend bool operator>(const const_iterator &a, const const_iterator &b) noexcept { return b < a; }
            friend bool operator<=(const const_iterator &a, const const_iterator &b) noexcept { return !(b < a); }
            friend bool operator>=(const const_iterator &a, const const_iterator &b) noexcept { return !(a < b); }
        };

        [[nodiscard]] bool empty() const noexcept { return columns.empty(); }
        [[nodiscard]] std::size_t size() const noexcept { return columns.size(); }

        void clear() noexcept
        {
            columns.clear();
            values.clear();
        }
        void reserve(std::size_t n)
        {
            columns.reserve(n);
            values.reserve(n);
        }
        [[nodiscard]] std::size_t capacity() const noexcept { return columns.capacity(); }

        [[nodiscard]] I key_at(std::size_t idx) const noexcept { return columns[idx]; }
        [[nodiscard]] V &value_at(std::size_t idx) noexcept { return values[idx]; }
        [[nodiscard]] const V &value_at(std::size_t idx) const noexcept { return values[idx]; }


        void insert_at(std::size_t idx, I col, V val)
        {
            columns.insert(columns.begin() + idx, col);
            values.insert(values.begin() + idx, val);
        }

        [[nodiscard]] std::size_t lower_bound(I col) const noexcept
        {
            auto s = boundcraft::searcher<boundcraft::policy::hybrid<32>>();
            auto it = s.lower_bound(columns.begin(), columns.end(), col);
            return static_cast<std::size_t>(it - columns.begin());
        }

        iterator begin() noexcept { return iterator(columns.data(), values.data()); }
        iterator end() noexcept { return iterator(columns.data() + columns.size(), values.data() + values.size()); }

        const_iterator begin() const noexcept { return cbegin(); }
        const_iterator end() const noexcept { return cend(); }
        const_iterator cbegin() const noexcept { return const_iterator(columns.data(), values.data()); }
        const_iterator cend() const noexcept { return const_iterator(columns.data() + columns.size(), values.data() + values.size()); }

    private:
        std::vector<I> columns;
        std::vector<V> values;
    };

}
