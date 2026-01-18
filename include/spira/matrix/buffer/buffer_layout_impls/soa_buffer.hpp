#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <spira/matrix/layouts/aos.hpp>

namespace spira::buffer::impls
{
    template <class I, class V, std::size_t N>
    class soa_buffer
    {
    public:
        using size_type = std::size_t;
        using entry_type = spira::layout::elementPair<I, V>;

        struct entry_ref
        {
            const I &column;
            V &value;
        };
        struct entry_cref
        {
            const I &column;
            const V &value;
        };

        [[nodiscard]] bool empty() const noexcept { return sz_ == 0; }
        [[nodiscard]] size_type size() const noexcept
        {
            deduplicate();
            return sz_;
        }
        [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
        [[nodiscard]] size_type remaining_capacity() const noexcept { return N - sz_; }

        void clear() noexcept { sz_ = 0; }

        [[nodiscard]] const I &key_at(size_type idx) const noexcept { return col_[idx]; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return val_[idx]; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return val_[idx]; }

        [[nodiscard]] entry_ref at(size_type idx) noexcept { return {col_[idx], val_[idx]}; }
        [[nodiscard]] entry_cref at(size_type idx) const noexcept { return {col_[idx], val_[idx]}; }

        void push_back(const I &col, const V &v) noexcept(
            std::is_nothrow_copy_assignable_v<I> && std::is_nothrow_copy_assignable_v<V>)
        {
            assert(sz_ < N && "soa_buffer overflow: caller must ensure capacity");
            col_[sz_] = col;
            val_[sz_] = v;
            ++sz_;
        }

        bool contains(I col) const noexcept
        {
            for (size_type i = sz_; i-- > 0;)
            {
                if (col_[i] == col)
                    return true;
            }
            return false;
        }

        const V *get_ptr(I col) const noexcept
        {
            for (size_type i = sz_; i-- > 0;)
            {
                if (col_[i] == col)
                    return &val_[i];
            }
            return nullptr;
        }

        [[nodiscard]] std::vector<entry_type> flush_buffer()
        {
            std::vector<entry_type> chunk;
            chunk.reserve(sz_);
            for (size_type i = 0; i < sz_; ++i)
            {
                chunk.push_back(entry_type{col_[i], val_[i]});
            }
            clear();
            return chunk;
        }

        void deduplicate() const noexcept
        {
            std::array<I, N> tmpCol;
            std::array<V, N> tmpVal;
            size_t out = 0;

            for (size_t i = sz_; i-- > 0;)
            {
                bool seen = false;

                for (size_t j = 0; j < out; j++)
                {
                    if (tmpCol[j] == col_[i])
                    {
                        seen = true;
                        break;
                    }
                }

                if (seen == false)
                {
                    tmpVal[out] = val_[i];
                    tmpCol[out++] = col_[i];
                }
            }

            for (size_t i = 0; i < out; i++)
            {
                val_[i] = tmpVal[i];
                col_[i] = tmpCol[i];
            }

            sz_ = out;
        }

        V accumulate() const noexcept
        {
            deduplicate();

            V acc = traits::ValueTraits<V>::zero();

            for (size_t i = 0; i < sz_; i++)
            {
                acc += val_[i];
            }

            return acc;
        }

        template <class layout_policy>
        layout_policy normalize_buffer()
        {
            layout_policy chunk;
            chunk.reserve(sz_);

            for (size_t i = sz_; i-- > 0;)
            {
                bool seen = false;

                for (auto const &entry : chunk)
                {
                    if (entry.first_ref() == col_[i])
                    {
                        seen = true;
                        break;
                    }
                }

                if (!seen)
                {
                    chunk.push_back(col_[i], val_[i]);
                }
            }

            auto key_of = [](auto const &x) -> decltype(auto)
            {
                if constexpr (requires { x.first_ref(); })
                    return x.first_ref();
                else
                    return x.first;
            };

            std::stable_sort(chunk.begin(), chunk.end(), [&](auto const &a, auto const &b){ return key_of(a) < key_of(b); });

            clear();
            return chunk;
        }

        class iterator
        {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = entry_ref;
            using reference = entry_ref;

            iterator() = default;
            iterator(I *c, V *v) : c_(c), v_(v) {}

            reference operator*() const noexcept { return {*c_, *v_}; }

            iterator &operator++() noexcept
            {
                ++c_;
                ++v_;
                return *this;
            }
            iterator operator++(int) noexcept
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            iterator &operator--() noexcept
            {
                --c_;
                --v_;
                return *this;
            }
            iterator operator--(int) noexcept
            {
                auto tmp = *this;
                --(*this);
                return tmp;
            }

            iterator &operator+=(difference_type n) noexcept
            {
                c_ += n;
                v_ += n;
                return *this;
            }
            iterator &operator-=(difference_type n) noexcept
            {
                c_ -= n;
                v_ -= n;
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
                return a.c_ - b.c_;
            }

            friend bool operator==(const iterator &a, const iterator &b) noexcept { return a.c_ == b.c_; }
            friend bool operator!=(const iterator &a, const iterator &b) noexcept { return !(a == b); }
            friend bool operator<(const iterator &a, const iterator &b) noexcept { return a.c_ < b.c_; }
            friend bool operator>(const iterator &a, const iterator &b) noexcept { return b < a; }
            friend bool operator<=(const iterator &a, const iterator &b) noexcept { return !(b < a); }
            friend bool operator>=(const iterator &a, const iterator &b) noexcept { return !(a < b); }

            reference operator[](difference_type n) const noexcept { return *(*this + n); }

        private:
            I *c_{nullptr};
            V *v_{nullptr};
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
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            const_iterator &operator--() noexcept
            {
                --c_;
                --v_;
                return *this;
            }
            const_iterator operator--(int) noexcept
            {
                auto tmp = *this;
                --(*this);
                return tmp;
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

        [[nodiscard]] iterator begin() noexcept { return iterator(col_.data(), val_.data()); }
        [[nodiscard]] iterator end() noexcept { return iterator(col_.data() + sz_, val_.data() + sz_); }

        [[nodiscard]] const_iterator begin() const noexcept { return const_iterator(col_.data(), val_.data()); }
        [[nodiscard]] const_iterator end() const noexcept { return const_iterator(col_.data() + sz_, val_.data() + sz_); }

        [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
        [[nodiscard]] const_iterator cend() const noexcept { return end(); }

    private:
        mutable std::array<I, N> col_{};
        mutable std::array<V, N> val_{};
        mutable size_type sz_{0};
    };

}