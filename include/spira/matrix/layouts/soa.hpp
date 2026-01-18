#pragma once
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <type_traits>

#include <boundcraft/boundcraft.hpp>
#include <spira/concepts.hpp>

namespace spira::layout
{

    template <class I, class V>
    struct soa_ref
    {
        I *first{};
        V *second{};

        soa_ref() = default;
        soa_ref(I &i, V &v) : first(&i), second(&v) {}

        explicit operator std::pair<I, V>() const { return {*first, *second}; }

        soa_ref &operator=(std::pair<I, V> const &x)
        {
            *first = x.first;
            *second = x.second;
            return *this;
        }

        I &first_ref() const { return *first; }
        V &second_ref() const { return *second; }

        template <std::size_t K>
        friend decltype(auto) get(soa_ref &r) noexcept
        {
            static_assert(K < 2, "soa_ref get<K>: K must be 0 or 1");
            if constexpr (K == 0)
                return (r.first_ref());
            else
                return (r.second_ref());
        }

        template <std::size_t K>
        friend decltype(auto) get(soa_ref const &r) noexcept
        {
            static_assert(K < 2, "soa_ref get<K>: K must be 0 or 1");
            if constexpr (K == 0)
                return (r.first_ref());
            else
                return (r.second_ref());
        }
    };

    template <class I, class V>
    struct soa_cref
    {
        I const *first{};
        V const *second{};

        soa_cref() = default;
        soa_cref(I const &i, V const &v) : first(&i), second(&v) {}

        explicit operator std::pair<I, V>() const { return {*first, *second}; }

        I const &first_ref() const { return *first; }
        V const &second_ref() const { return *second; }

        template <std::size_t K>
        friend decltype(auto) get(soa_cref const &r) noexcept
        {
            static_assert(K < 2, "soa_cref get<K>: K must be 0 or 1");
            if constexpr (K == 0)
                return (r.first_ref());
            else
                return (r.second_ref());
        }
    };

    template <concepts::Indexable I, concepts::Valueable V>
    class soa
    {
    public:
        using size_type = std::size_t;
        using value_type = std::pair<I, V>;

        template <class T>
        static decltype(auto) key_of(T const &x)
        {
            if constexpr (requires { x.first_ref(); })
                return x.first_ref();
            else
                return x.first;
        }

        struct iterator
        {
            using iterator_concept = std::random_access_iterator_tag;
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = std::pair<I, V>;
            using pointer = void;
            using reference = soa_ref<I, V>;

            I *col_ptr{};
            V *val_ptr{};

            iterator() = default;
            iterator(I *c, V *v) : col_ptr(c), val_ptr(v) {}

            reference operator*() const noexcept { return reference{*col_ptr, *val_ptr}; }
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

            friend difference_type operator-(iterator a, iterator b) noexcept { return a.col_ptr - b.col_ptr; }

            friend bool operator==(iterator a, iterator b) noexcept { return a.col_ptr == b.col_ptr; }
            friend bool operator!=(iterator a, iterator b) noexcept { return !(a == b); }
            friend bool operator<(iterator a, iterator b) noexcept { return a.col_ptr < b.col_ptr; }
            friend bool operator>(iterator a, iterator b) noexcept { return b < a; }
            friend bool operator<=(iterator a, iterator b) noexcept { return !(b < a); }
            friend bool operator>=(iterator a, iterator b) noexcept { return !(a < b); }

            // enable std::sort/std::stable_sort for proxy iterators
            friend void iter_swap(iterator a, iterator b) noexcept
            {
                using std::swap;
                swap(*a.col_ptr, *b.col_ptr);
                swap(*a.val_ptr, *b.val_ptr);
            }
        };

        struct const_iterator
        {
            using iterator_concept = std::random_access_iterator_tag;
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = std::pair<I, V>;
            using pointer = void;
            using reference = soa_cref<I, V>;

            I const *col_ptr{};
            V const *val_ptr{};

            const_iterator() = default;
            const_iterator(I const *c, V const *v) : col_ptr(c), val_ptr(v) {}
            const_iterator(iterator it) : col_ptr(it.col_ptr), val_ptr(it.val_ptr) {}

            reference operator*() const noexcept { return reference{*col_ptr, *val_ptr}; }
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

            friend difference_type operator-(const_iterator a, const_iterator b) noexcept { return a.col_ptr - b.col_ptr; }

            friend bool operator==(const_iterator a, const_iterator b) noexcept { return a.col_ptr == b.col_ptr; }
            friend bool operator!=(const_iterator a, const_iterator b) noexcept { return !(a == b); }
            friend bool operator<(const_iterator a, const_iterator b) noexcept { return a.col_ptr < b.col_ptr; }
            friend bool operator>(const_iterator a, const_iterator b) noexcept { return b < a; }
            friend bool operator<=(const_iterator a, const_iterator b) noexcept { return !(b < a); }
            friend bool operator>=(const_iterator a, const_iterator b) noexcept { return !(a < b); }
        };

        [[nodiscard]] bool empty() const noexcept { return columns_.empty(); }
        [[nodiscard]] size_type size() const noexcept { return columns_.size(); }
        [[nodiscard]] size_type capacity() const noexcept { return columns_.capacity(); }

        void clear()
        {
            columns_.clear();
            values_.clear();
        }
        void reserve(size_type n)
        {
            columns_.reserve(n);
            values_.reserve(n);
        }
        void resize(size_type n)
        {
            columns_.resize(n);
            values_.resize(n);
        }

        void swap(soa &other) noexcept
        {
            columns_.swap(other.columns_);
            values_.swap(other.values_);
        }

        [[nodiscard]] const I &key_at(size_type idx) const noexcept { return columns_[idx]; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return values_[idx]; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return values_[idx]; }

        void insert_at(size_type idx, I col, V val)
        {
            columns_.insert(columns_.begin() + static_cast<std::ptrdiff_t>(idx), col);
            values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(idx), val);
        }

        void push_back(I col, V const &val)
        {
            columns_.push_back(col);
            values_.push_back(val);
        }

        [[nodiscard]] size_type lower_bound(I col) const noexcept
        {
            auto s = boundcraft::searcher<boundcraft::policy::hybrid<32>>();
            auto it = s.lower_bound(columns_.begin(), columns_.end(), col);
            return static_cast<size_type>(it - columns_.begin());
        }

        iterator begin() noexcept { return iterator(columns_.data(), values_.data()); }
        iterator end() noexcept
        {
            return iterator(columns_.data() + columns_.size(),
                            values_.data() + values_.size());
        }

        const_iterator begin() const noexcept { return cbegin(); }
        const_iterator end() const noexcept { return cend(); }
        const_iterator cbegin() const noexcept { return const_iterator(columns_.data(), values_.data()); }
        const_iterator cend() const noexcept
        {
            return const_iterator(columns_.data() + columns_.size(),
                                  values_.data() + values_.size());
        }

        template <class It, class F>
        static decltype(auto) with_entry(It it, F &&f)
        {
            auto r = *it;
            auto &&[col, val] = r;
            return std::forward<F>(f)(col, val);
        }

    private:
        std::vector<I> columns_;
        std::vector<V> values_;
    };

}

namespace std
{
    template <class I, class V>
    struct tuple_size<spira::layout::soa_ref<I, V>>
        : integral_constant<std::size_t, 2>
    {
    };

    template <class I, class V>
    struct tuple_element<0, spira::layout::soa_ref<I, V>>
    {
        using type = I;
    };
    template <class I, class V>
    struct tuple_element<1, spira::layout::soa_ref<I, V>>
    {
        using type = V;
    };

    template <class I, class V>
    struct tuple_size<spira::layout::soa_cref<I, V>>
        : integral_constant<std::size_t, 2>
    {
    };

    template <class I, class V>
    struct tuple_element<0, spira::layout::soa_cref<I, V>>
    {
        using type = const I;
    };

    template <class I, class V>
    struct tuple_element<1, spira::layout::soa_cref<I, V>>
    {
        using type = const V;
    };
}
