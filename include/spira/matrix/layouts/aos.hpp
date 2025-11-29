#pragma once
#include <vector>
#include <algorithm>
#include <cstddef>  // std::ptrdiff_t
#include <iterator> // iterator tags

#include "../include/spira/concepts.hpp"

namespace spira::layout
{

    template <concepts::Indexable I, concepts::Valueable V>
    struct elementPair
    {
        I column;
        V value;
    };

    template <concepts::Indexable I, concepts::Valueable V>
    class aos
    {
    public:
        struct iterator
        {
            using value_type = elementPair<I, V>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type *;
            using reference = value_type &;
            using iterator_concept = std::contiguous_iterator_tag;
            using iterator_category = std::contiguous_iterator_tag;

            pointer ptr{};

            constexpr iterator() noexcept = default;
            explicit constexpr iterator(pointer p) noexcept : ptr(p) {}

            constexpr reference operator*() const noexcept { return *ptr; }
            constexpr pointer operator->() const noexcept { return ptr; }
            constexpr reference operator[](difference_type n) const noexcept { return ptr[n]; }

            constexpr iterator &operator++() noexcept
            {
                ++ptr;
                return *this;
            }
            constexpr iterator operator++(int) noexcept
            {
                auto t = *this;
                ++(*this);
                return t;
            }
            constexpr iterator &operator--() noexcept
            {
                --ptr;
                return *this;
            }
            constexpr iterator operator--(int) noexcept
            {
                auto t = *this;
                --(*this);
                return t;
            }

            constexpr iterator &operator+=(difference_type n) noexcept
            {
                ptr += n;
                return *this;
            }
            constexpr iterator &operator-=(difference_type n) noexcept
            {
                ptr -= n;
                return *this;
            }
            friend constexpr iterator operator+(iterator it, difference_type n) noexcept
            {
                it += n;
                return it;
            }
            friend constexpr iterator operator+(difference_type n, iterator it) noexcept
            {
                it += n;
                return it;
            }
            friend constexpr iterator operator-(iterator it, difference_type n) noexcept
            {
                it -= n;
                return it;
            }

            friend constexpr difference_type operator-(iterator a, iterator b) noexcept { return a.ptr - b.ptr; }

            friend constexpr bool operator==(const iterator &, const iterator &) noexcept = default;
            friend constexpr auto operator<=>(const iterator &, const iterator &) noexcept = default;
        };

        struct const_iterator
        {
            using value_type = const elementPair<I, V>;
            using difference_type = std::ptrdiff_t;
            using pointer = const value_type *;
            using reference = const value_type &;
            using iterator_concept = std::contiguous_iterator_tag;
            using iterator_category = std::contiguous_iterator_tag;

            pointer ptr{};

            constexpr const_iterator() noexcept = default;
            explicit constexpr const_iterator(pointer p) noexcept : ptr(p) {}
            constexpr const_iterator(iterator it) noexcept : ptr(it.ptr) {}

            constexpr reference operator*() const noexcept { return *ptr; }
            constexpr pointer operator->() const noexcept { return ptr; }
            constexpr reference operator[](difference_type n) const noexcept { return ptr[n]; }

            constexpr const_iterator &operator++() noexcept
            {
                ++ptr;
                return *this;
            }
            constexpr const_iterator operator++(int) noexcept
            {
                auto t = *this;
                ++(*this);
                return t;
            }
            constexpr const_iterator &operator--() noexcept
            {
                --ptr;
                return *this;
            }
            constexpr const_iterator operator--(int) noexcept
            {
                auto t = *this;
                --(*this);
                return t;
            }

            constexpr const_iterator &operator+=(difference_type n) noexcept
            {
                ptr += n;
                return *this;
            }
            constexpr const_iterator &operator-=(difference_type n) noexcept
            {
                ptr -= n;
                return *this;
            }
            friend constexpr const_iterator operator+(const_iterator it, difference_type n) noexcept
            {
                it += n;
                return it;
            }
            friend constexpr const_iterator operator+(difference_type n, const_iterator it) noexcept
            {
                it += n;
                return it;
            }
            friend constexpr const_iterator operator-(const_iterator it, difference_type n) noexcept
            {
                it -= n;
                return it;
            }

            friend constexpr difference_type operator-(const_iterator a, const_iterator b) noexcept { return a.ptr - b.ptr; }

            friend constexpr bool operator==(const const_iterator &, const const_iterator &) noexcept = default;
            friend constexpr auto operator<=>(const const_iterator &, const const_iterator &) noexcept = default;
        };

        [[nodiscard]] bool empty() const noexcept { return elements.empty(); }
        [[nodiscard]] size_t size() const noexcept { return elements.size(); }
        [[nodiscard]] size_t capacity() const noexcept { return elements.capacity(); }
        void reserve(size_t n) { elements.reserve(n); }
        void clear() { elements.clear(); }

        [[nodiscard]] I key_at(size_t idx) const noexcept { return elements[idx].column; }
        [[nodiscard]] V &value_at(size_t idx) noexcept { return elements[idx].value; }
        [[nodiscard]] const V &value_at(size_t idx) const noexcept { return elements[idx].value; }

        void set_at(size_t index, I col, const V &val)
        {
            elements[index] = elementPair<I, V>{col, val};
        }

        void insert_at(size_t index, I col, const V &val)
        {
            elements.insert(elements.begin() + index, elementPair<I, V>{col, val});
        }

        void erase_at(size_t index)
        {
            elements.erase(elements.begin() + index);
        }

        [[nodiscard]] size_t lower_bound(I col) const noexcept
        {
            auto it = std::lower_bound(
                elements.begin(), elements.end(), col,
                [](auto const &e, I key)
                { return e.column < key; });
            return static_cast<size_t>(std::distance(elements.begin(), it));
        }

        iterator begin() noexcept { return iterator(elements.data()); }
        iterator end() noexcept { return iterator(elements.data() + elements.size()); }

        const_iterator begin() const noexcept { return cbegin(); }
        const_iterator end() const noexcept { return cend(); }
        const_iterator cbegin() const noexcept { return const_iterator(elements.data()); }
        const_iterator cend() const noexcept { return const_iterator(elements.data() + elements.size()); }

    private:
        std::vector<elementPair<I, V>> elements;
    };

}
