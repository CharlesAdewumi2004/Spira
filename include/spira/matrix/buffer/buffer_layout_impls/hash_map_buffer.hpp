#pragma once
#include <limits>
#include <numeric>

#include <ankerl/unordered_dense.h>

#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/traits.hpp>
#include <spira/config.hpp>
#include <spira/matrix/layouts/element_pair.hpp>

namespace spira::buffer::impls {

template <class I, class V>
class hash_map_buffer : public spira::buffer::base_buffer<hash_map_buffer<I, V>, I, V> {
public:
    using size_type = std::size_t;
    using entry_type = spira::layout::elementPair<I, V>;

    bool empty_impl() const noexcept { return buf_.empty(); }
    size_type size_impl() const noexcept { return buf_.size(); }
    [[nodiscard]] size_type remaining_capacity_impl() const noexcept { return std::numeric_limits<size_type>::max(); }

    void clear_impl() noexcept { buf_.clear(); }
    void push_back_impl(const I &col, const V &val) noexcept { buf_[col] = val; }

    bool contains_impl(I col) const noexcept { return buf_.contains(col); }

    const V *get_ptr_impl(I col) const noexcept {
        auto it = buf_.find(col);
        if (it == buf_.end())
            return nullptr;
        return &it->second;
    }

    V accumulate_impl() const noexcept {
        return std::accumulate(buf_.begin(), buf_.end(), traits::ValueTraits<V>::zero(),
                               [](V acc, auto const &kv) { return acc + kv.second; });
    }

    template <class layout_policy>
    layout_policy normalize_buffer_impl() {
        layout_policy chunk;
        chunk.reserve(buf_.size());

        for (auto &[col, val] : buf_) {
            chunk.push_back(col, val);
        }

        auto key_of = [](auto const &x) -> decltype(auto) {
            if constexpr (requires { x.first_ref(); })
                return x.first_ref();
            else
                return x.first;
        };

        std::stable_sort(chunk.begin(), chunk.end(),
                         [&](auto const &a, auto const &b) { return key_of(a) < key_of(b); });

        buf_.clear();

        return chunk;
    }

    auto begin_impl() noexcept { return buf_.begin(); }
    auto end_impl() noexcept { return buf_.end(); }
    auto begin_impl() const noexcept { return buf_.cbegin(); }
    auto end_impl() const noexcept { return buf_.cend(); }

private:
    ankerl::unordered_dense::map<I, V> buf_{};
};

} // namespace spira::buffer::impls
