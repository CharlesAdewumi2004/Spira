#pragma once

#include <cstddef>
#include <type_traits>

namespace spira::layout {

template <class Derived, class I, class V>
class layout_base {
public:
    using size_type = std::size_t;

    [[nodiscard]] bool empty() const noexcept { return self().empty_impl(); }
    [[nodiscard]] size_type size() const noexcept { return self().size_impl(); }
    [[nodiscard]] size_type capacity() const noexcept { return self().capacity_impl(); }

    void reserve(size_type n) { self().reserve_impl(n); }
    void clear() noexcept { self().clear_impl(); }
    void resize(size_type n) { self().resize_impl(n); }
    void swap(Derived &other) noexcept { self().swap_impl(other); }

    [[nodiscard]] decltype(auto) key_at(size_type idx) const noexcept { return self().key_at_impl(idx); }
    [[nodiscard]] decltype(auto) value_at(size_type idx) noexcept { return self().value_at_impl(idx); }
    [[nodiscard]] decltype(auto) value_at(size_type idx) const noexcept { return self().value_at_impl(idx); }

    void insert_at(size_type idx, I col, const V &val) { self().insert_at_impl(idx, col, val); }
    void push_back(I col, const V &val) { self().push_back_impl(col, val); }

    [[nodiscard]] size_type lower_bound(I col) const noexcept { return self().lower_bound_impl(col); }

    [[nodiscard]] decltype(auto) data() noexcept { return self().data_impl(); }
    [[nodiscard]] decltype(auto) data() const noexcept { return self().data_impl(); }

    [[nodiscard]] auto begin() noexcept { return self().begin_impl(); }
    [[nodiscard]] auto end() noexcept { return self().end_impl(); }
    [[nodiscard]] auto begin() const noexcept { return self().begin_impl(); }
    [[nodiscard]] auto end() const noexcept { return self().end_impl(); }
    [[nodiscard]] auto cbegin() const noexcept { return self().begin_impl(); }
    [[nodiscard]] auto cend() const noexcept { return self().end_impl(); }

private:
    Derived &self() { return static_cast<Derived &>(*this); }
    const Derived &self() const { return static_cast<const Derived &>(*this); }
};

template <class T, class I, class V>
concept Layout = std::is_base_of_v<layout_base<T, I, V>, T>;

} // namespace spira::layout
