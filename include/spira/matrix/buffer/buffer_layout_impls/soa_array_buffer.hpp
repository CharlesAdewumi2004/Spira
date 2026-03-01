#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/traits.hpp>

namespace spira::buffer::impls {

template <class I, class V, std::size_t N>
class soa_array_buffer : public spira::buffer::base_buffer<soa_array_buffer<I, V, N>, I, V> {
public:
    using size_type = std::size_t;

    struct entry_ref {
        const I &column;
        V &value;
    };
    struct entry_cref {
        const I &column;
        const V &value;
    };

    [[nodiscard]] bool empty_impl() const noexcept { return sz_ == 0; }
    [[nodiscard]] size_type size_impl() const noexcept {
        deduplicate();
        return sz_;
    }
    [[nodiscard]] size_type remaining_capacity_impl() const noexcept { return N - sz_; }

    void clear_impl() noexcept { sz_ = 0; }

    void push_back_impl(const I &col, const V &val) noexcept(std::is_nothrow_copy_assignable_v<I> &&
                                                             std::is_nothrow_copy_assignable_v<V>) {
        assert(sz_ < N && "soa_array_buffer overflow: caller must ensure capacity");
        col_[sz_] = col;
        val_[sz_] = val;
        ++sz_;
    }

    bool contains_impl(I col) const noexcept {
        for (size_type i = sz_; i-- > 0;) {
            if (col_[i] == col)
                return true;
        }
        return false;
    }

    const V *get_ptr_impl(I col) const noexcept {
        for (size_type i = sz_; i-- > 0;) {
            if (col_[i] == col)
                return &val_[i];
        }
        return nullptr;
    }

    V accumulate_impl() const noexcept {
        deduplicate();

        V acc = traits::ValueTraits<V>::zero();

        for (size_t i = 0; i < sz_; i++) {
            acc += val_[i];
        }

        return acc;
    }

    template <class layout_policy> layout_policy normalize_buffer_impl() {
        layout_policy chunk;
        chunk.reserve(sz_);
        deduplicate();

        for (size_t i = 0; i < sz_; i++) {
            chunk.push_back(col_[i], val_[i]);
        }

        clear_impl();
        return chunk;
    }

    class iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = entry_ref;
        using reference = entry_ref;

        iterator() = default;
        iterator(I *c, V *v) : c_(c), v_(v) {}

        reference operator*() const noexcept { return {*c_, *v_}; }

        iterator &operator++() noexcept {
            ++c_;
            ++v_;
            return *this;
        }
        iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator &operator--() noexcept {
            --c_;
            --v_;
            return *this;
        }
        iterator operator--(int) noexcept {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        iterator &operator+=(difference_type n) noexcept {
            c_ += n;
            v_ += n;
            return *this;
        }
        iterator &operator-=(difference_type n) noexcept {
            c_ -= n;
            v_ -= n;
            return *this;
        }

        friend iterator operator+(iterator it, difference_type n) noexcept {
            it += n;
            return it;
        }
        friend iterator operator+(difference_type n, iterator it) noexcept {
            it += n;
            return it;
        }
        friend iterator operator-(iterator it, difference_type n) noexcept {
            it -= n;
            return it;
        }
        friend difference_type operator-(const iterator &a, const iterator &b) noexcept { return a.c_ - b.c_; }

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

    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = entry_cref;
        using reference = entry_cref;

        const_iterator() = default;
        const_iterator(const I *c, const V *v) : c_(c), v_(v) {}

        reference operator*() const noexcept { return {*c_, *v_}; }

        const_iterator &operator++() noexcept {
            ++c_;
            ++v_;
            return *this;
        }
        const_iterator operator++(int) noexcept {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        const_iterator &operator--() noexcept {
            --c_;
            --v_;
            return *this;
        }
        const_iterator operator--(int) noexcept {
            auto tmp = *this;
            --(*this);
            return tmp;
        }

        const_iterator &operator+=(difference_type n) noexcept {
            c_ += n;
            v_ += n;
            return *this;
        }
        const_iterator &operator-=(difference_type n) noexcept {
            c_ -= n;
            v_ -= n;
            return *this;
        }

        friend const_iterator operator+(const_iterator it, difference_type n) noexcept {
            it += n;
            return it;
        }
        friend const_iterator operator+(difference_type n, const_iterator it) noexcept {
            it += n;
            return it;
        }
        friend const_iterator operator-(const_iterator it, difference_type n) noexcept {
            it -= n;
            return it;
        }
        friend difference_type operator-(const const_iterator &a, const const_iterator &b) noexcept {
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
    [[nodiscard]] iterator end_impl() noexcept { return iterator(col_.data() + sz_, val_.data() + sz_); }

    [[nodiscard]] const_iterator begin_impl() const noexcept { return const_iterator(col_.data(), val_.data()); }
    [[nodiscard]] const_iterator end_impl() const noexcept { return const_iterator(col_.data() + sz_, val_.data() + sz_); }

private:
    mutable std::array<I, N> col_{};
    mutable std::array<V, N> val_{};
    mutable size_type sz_{0};

    void deduplicate() const noexcept {
        if (sz_ == 0) return;

        std::array<size_type, N> idx;
        for (size_type i = 0; i < sz_; ++i)
            idx[i] = sz_ - 1 - i;

        std::stable_sort(idx.begin(), idx.begin() + sz_,
            [&](size_type a, size_type b) { return col_[a] < col_[b]; });

        std::array<I, N> tmpCol;
        std::array<V, N> tmpVal;
        size_type out = 0;

        for (size_type i = 0; i < sz_; ++i) {
            if (out == 0 || tmpCol[out - 1] != col_[idx[i]]) {
                tmpCol[out] = col_[idx[i]];
                tmpVal[out] = val_[idx[i]];
                ++out;
            }
        }

        for (size_type i = 0; i < out; ++i) {
            col_[i] = tmpCol[i];
            val_[i] = tmpVal[i];
        }
        sz_ = out;
    }
};

} // namespace spira::buffer::impls
