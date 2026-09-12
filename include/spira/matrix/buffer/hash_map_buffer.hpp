#pragma once
#include <algorithm>
#include <limits>
#include <numeric>
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
    [[nodiscard]] size_type remaining_capacity_impl() const noexcept { return std::numeric_limits<size_type>::max(); }

    void clear_impl() noexcept { buf_.clear(); sorted_.clear(); }
    void push_back_impl(const I &col, const V &val) noexcept { buf_[col] = val; }

    bool contains_impl(I col) const noexcept {
        return buf_.contains(col) || find_sorted(col) != nullptr;
    }

    const V *get_ptr_impl(I col) const noexcept {
        auto it = buf_.find(col);
        if (it != buf_.end())
            return &it->second;
        // no_compact keeps committed data in sorted_ after lock(); buf_ is empty.
        return find_sorted(col);
    }

    V accumulate_impl() const noexcept {
        V acc = traits::ValueTraits<V>::zero();
        for (auto const &kv : buf_)
            acc += kv.second;
        for (const auto &e : sorted_)
            acc += e.value;
        return acc;
    }

    /// Sort by column, filter zero values, and materialize into sorted_.
    /// The hash map already deduplicates (last-write wins) on insert.
    /// After this call: buf_ is empty, sorted_ has sorted, unique, non-zero entries.
    void sort_and_dedup() { materialize_sorted(/*keep_zeros=*/false); }

    /// Sort by column and materialize into sorted_, keeping zero values.
    /// Identical to sort_and_dedup() but zeros are not removed, so that
    /// merge_csr can see them as deletion signals during compact_* lock cycles.
    void sort_and_dedup_keep_zeros() { materialize_sorted(/*keep_zeros=*/true); }

    // Iterators return raw pointers into sorted_ (random-access, expose first_ref/second_ref).
    // Valid after sort_and_dedup(); in open mode sorted_ is empty so this is an empty range.
    entry_type *begin_impl() noexcept { return sorted_.data(); }
    entry_type *end_impl() noexcept { return sorted_.data() + sorted_.size(); }
    const entry_type *begin_impl() const noexcept { return sorted_.data(); }
    const entry_type *end_impl() const noexcept { return sorted_.data() + sorted_.size(); }

private:
    /// Binary search the materialized entries. push_back only touches buf_ and
    /// materialize_sorted() drains it, so buf_ and sorted_ are never both
    /// non-empty — a miss in one is authoritative for the other.
    const V *find_sorted(I col) const noexcept {
        const auto it = std::lower_bound(
            sorted_.begin(), sorted_.end(), col,
            [](const entry_type &e, I c) { return e.first_ref() < c; });
        if (it != sorted_.end() && it->first_ref() == col)
            return &it->second_ref();
        return nullptr;
    }

    /// Drain buf_ into sorted_, ordered by column. The hash map has already
    /// deduplicated (last-write wins) on insert, so no dedup pass is needed.
    void materialize_sorted(bool keep_zeros) {
        sorted_.clear();
        sorted_.reserve(buf_.size());
        for (auto &[col, val] : buf_) {
            if (keep_zeros || !traits::ValueTraits<V>::is_zero(val))
                sorted_.push_back(entry_type{col, val});
        }
        std::sort(sorted_.begin(), sorted_.end(),
                  [](const entry_type &a, const entry_type &b) {
                      return a.first_ref() < b.first_ref();
                  });
        buf_.clear();
    }

    ankerl::unordered_dense::map<I, V> buf_{};
    std::vector<entry_type> sorted_{};
};

} // namespace spira::buffer::impls
