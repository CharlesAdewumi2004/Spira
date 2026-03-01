#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/layouts/element_pair.hpp>

namespace spira::buffer::impls {

template <class I, class V, std::size_t N>
class aos_array_buffer : public spira::buffer::base_buffer<aos_array_buffer<I, V, N>, I, V> {
public:
    using entry_type = spira::layout::elementPair<I, V>;
    using size_type = std::size_t;

    [[nodiscard]] bool empty_impl() const noexcept { return sz_ == 0; }
    [[nodiscard]] size_type size_impl() const noexcept {
        return sz_;
    }
    [[nodiscard]] size_type remaining_capacity_impl() const noexcept { return N - sz_; }

    void clear_impl() noexcept { sz_ = 0; }

    [[nodiscard]] const I &key_at(size_type idx) const noexcept { return buf_[idx].column; }
    [[nodiscard]] V &value_at(size_type idx) noexcept { return buf_[idx].value; }
    [[nodiscard]] const V &value_at(size_type idx) const noexcept { return buf_[idx].value; }

    [[nodiscard]] entry_type *begin_impl() noexcept { return buf_.data(); }
    [[nodiscard]] entry_type *end_impl() noexcept { return buf_.data() + sz_; }
    [[nodiscard]] const entry_type *begin_impl() const noexcept { return buf_.data(); }
    [[nodiscard]] const entry_type *end_impl() const noexcept { return buf_.data() + sz_; }

    void push_back_impl(const I &col, const V &v) noexcept(
        std::is_nothrow_copy_assignable_v<I> && std::is_nothrow_copy_assignable_v<V>) {
        assert(sz_ < N && "aos_array_buffer overflow: caller must ensure capacity");
        buf_[sz_++] = entry_type{col, v};
    }

    bool contains_impl(I col) const noexcept {
        for (size_type i = sz_; i-- > 0;) {
            if (buf_[i].column == col)
                return true;
        }
        return false;
    }

    const V *get_ptr_impl(I col) const noexcept {
        for (size_type i = sz_; i-- > 0;) {
            if (buf_[i].column == col)
                return &buf_[i].value;
        }
        return nullptr;
    }

    V accumulate_impl() const noexcept {
        deduplicate();

        V acc = traits::ValueTraits<V>::zero();

        for (size_t i = 0; i < sz_; i++) {
            acc += buf_[i].value;
        }
        return acc;
    }

    template <class layout_policy>
    layout_policy normalize_buffer_impl() {
        layout_policy chunk;
        chunk.reserve(sz_);
        deduplicate();

        for (size_t i = 0; i < sz_; i++) {
            chunk.push_back(buf_[i].column, buf_[i].value);
        }

        clear_impl();
        return chunk;
    }

private:
    mutable std::array<entry_type, N> buf_{};
    mutable size_type sz_{0};

    void deduplicate() const noexcept {
        if (sz_ == 0) return;

        auto first = buf_.begin();
        auto last = first + sz_;

        std::reverse(first, last);

        std::stable_sort(first, last, [](const auto &a, const auto &b) {
            return a.first_ref() < b.first_ref();
        });

        auto it = std::unique(first, last, [](const auto &a, const auto &b) {
            return a.first_ref() == b.first_ref();
        });

        sz_ = static_cast<size_t>(std::distance(first, it));
    }
};

} // namespace spira::buffer::impls
