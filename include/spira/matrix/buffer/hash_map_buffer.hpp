#pragma once
#include <algorithm>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/traits.hpp>
#include <spira/config.hpp>
#include <spira/matrix/layout/element_pair.hpp>

namespace spira::buffer::impls {

template <class I, class V>
class hash_map_buffer : public spira::buffer::base_buffer<hash_map_buffer<I, V>, I, V> {
public:
    using size_type = std::size_t;
    using entry_type = spira::layout::elementPair<I, V>;

    bool empty_impl() const noexcept { return buf_.empty() && sorted_.empty(); }
    size_type size_impl() const noexcept { return buf_.size() + sorted_.size(); }

    void clear_impl() noexcept { buf_.clear(); sorted_.clear(); }
    void push_back_impl(const I &col, const V &val) noexcept { buf_[col] = val; }

    bool contains_impl(I col) const noexcept {
        return buf_.contains(col) || find_sorted(col) != nullptr;
    }

    const V *get_ptr_impl(I col) const noexcept {
        auto it = buf_.find(col);
        if (it != buf_.end())
            return &it->second;
        return find_sorted(col);
    }

    /// Mutable lookup covers only the staging map: sorted_ is rebuilt from buf_
    /// on the next sort_and_dedup(), so a write into it would be lost.
    V *get_ptr_impl(I col) noexcept {
        auto it = buf_.find(col);
        return it != buf_.end() ? &it->second : nullptr;
    }

    V accumulate_impl() const noexcept {
        V acc = traits::ValueTraits<V>::zero();
        for (auto const &kv : buf_)
            acc += kv.second;
        for (const auto &e : sorted_)
            acc += e.value;
        return acc;
    }

    /// Sort by column and materialize into sorted_, keeping zero values.
    /// The hash map already deduplicates (last-write wins) on insert, so no
    /// dedup pass is needed. Zeros survive to merge_csr as deletion signals.
    void sort_and_dedup() {
        sorted_.clear();
        // buf_.clear() below memsets the map's bucket array regardless of how
        // many entries it holds, so skip the whole pass for an untouched row.
        if (buf_.empty())
            return;
        sorted_.reserve(buf_.size());
        for (auto &[col, val] : buf_)
            sorted_.push_back(entry_type{col, val});
        std::sort(sorted_.begin(), sorted_.end(),
                  [](const entry_type &a, const entry_type &b) {
                      return a.first_ref() < b.first_ref();
                  });
        buf_.clear();
    }

    // Iterators return raw pointers into sorted_ (random-access, expose first_ref/second_ref).
    // Valid after sort_and_dedup(); in open mode sorted_ is empty so this is an empty range.
    entry_type *begin_impl() noexcept { return sorted_.data(); }
    entry_type *end_impl() noexcept { return sorted_.data() + sorted_.size(); }
    const entry_type *begin_impl() const noexcept { return sorted_.data(); }
    const entry_type *end_impl() const noexcept { return sorted_.data() + sorted_.size(); }

private:
    /// Binary search the materialized entries. push_back only touches buf_ and
    /// sort_and_dedup() drains it, so buf_ and sorted_ are never both
    /// non-empty — a miss in one is authoritative for the other.
    const V *find_sorted(I col) const noexcept {
        const auto it = std::lower_bound(
            sorted_.begin(), sorted_.end(), col,
            [](const entry_type &e, I c) { return e.first_ref() < c; });
        if (it != sorted_.end() && it->first_ref() == col)
            return &it->second_ref();
        return nullptr;
    }

    ankerl::unordered_dense::map<I, V> buf_{};
    std::vector<entry_type> sorted_{};
};

} // namespace spira::buffer::impls
