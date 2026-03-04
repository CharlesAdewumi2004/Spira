#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <spira/concepts.hpp>
#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer.hpp>
#include <spira/matrix/buffer/buffer_base.hpp>
#include <spira/matrix/buffer/buffer_tag_traits.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/matrix/layouts/layout_base.hpp>
#include <spira/matrix/layouts/layout_of.hpp>
#include <spira/traits.hpp>

namespace spira {

// ─────────────────────────────────────────────────────────────────────────────
// row<LayoutTag, I, V, BufferTag, BufferN>
//
// Two-mode slab+buffer design:
//
//   Open mode  — inserts stage in buffer_ (unsorted); slab_ from prior lock
//                cycles is preserved and readable via get()/contains().
//   Locked mode — buffer_ merged into slab_; one sorted array per row;
//                 zero-overhead binary-search reads; no mutations allowed.
//
// lock()  — sort buffer, merge into slab (buffer wins on key collision),
//           clear buffer, set locked.  O(k log k + n)
// open()  — set flag to open; slab untouched, buffer ready.  O(1)
//
// BufferTag selects the staging buffer implementation:
//   buffer::tags::array_buffer<LayoutTag>  — fixed-size, cache-friendly
//   buffer::tags::hash_map_buffer          — unbounded, O(1) dedup on insert
//
// When an array buffer fills up, insert() automatically flushes it into the
// slab so the user never needs to manage buffer capacity manually.
// ─────────────────────────────────────────────────────────────────────────────

template <class LayoutTag, concepts::Indexable I, concepts::Valueable V,
          class BufferTag = buffer::tags::array_buffer<LayoutTag>,
          std::size_t BufferN = 64>
  requires buffer::Buffer<buffer::traits::traits_of_type<BufferTag, I, V, BufferN>, I, V>
        && layout::Layout<layout::detail::storage_of_t<LayoutTag, I, V>, I, V>
class row {
public:
  using layout_policy = layout::detail::storage_of_t<LayoutTag, I, V>;
  using buffer_t = buffer::traits::traits_of_type<BufferTag, I, V, BufferN>;
  using index_type = I;
  using value_type = V;
  using size_type = std::size_t;

  // ─────────────────────────────────────────
  // Construction
  // ─────────────────────────────────────────

  row() = default;

  explicit row(size_type column_limit) : column_limit_{column_limit} {}

  row(size_type reserve_hint, size_type column_limit)
      : column_limit_{column_limit} {
    slab_.reserve(reserve_hint);
  }

  // ─────────────────────────────────────────
  // Mode
  // ─────────────────────────────────────────

  [[nodiscard]] config::matrix_mode mode() const noexcept { return mode_; }
  [[nodiscard]] bool is_locked() const noexcept {
    return mode_ == config::matrix_mode::locked;
  }

  /// Merge buffer into slab and freeze.  O(k log k + n)
  void lock() {
    if (mode_ == config::matrix_mode::locked)
      return;
    if (!buffer_.empty()) {
      flush_buffer_to_slab();
    }
    mode_ = config::matrix_mode::locked;
  }

  /// Reopen for mutations.  O(1) — slab untouched, buffer already empty.
  void open() { mode_ = config::matrix_mode::open; }

  // ─────────────────────────────────────────
  // Size / capacity
  // ─────────────────────────────────────────

  /// Locked: exact deduplicated count.
  /// Open:   slab_size + buffer_size (upper bound; may include duplicates).
  [[nodiscard]] size_type size() const noexcept {
    return slab_.size() + buffer_.size();
  }

  [[nodiscard]] bool empty() const noexcept {
    return slab_.size() == 0 && buffer_.empty();
  }

  [[nodiscard]] size_type capacity() const noexcept { return slab_.capacity(); }

  void reserve(size_type n) { slab_.reserve(n); }

  void clear() noexcept {
    assert(mode_ == config::matrix_mode::open &&
           "row::clear() requires open mode");
    slab_.clear();
    buffer_.clear();
  }

  // ─────────────────────────────────────────
  // Mutation (open mode only)
  // ─────────────────────────────────────────

  void insert(index_type col, const value_type &val) {
    assert(mode_ == config::matrix_mode::open &&
           "row::insert() requires open mode");
    if (to_size(col) >= column_limit_) {
      throw std::out_of_range("Column index out of range");
    }
    // If array buffer is full, merge into slab to make room.
    if (buffer_.remaining_capacity() == 0) {
      flush_buffer_to_slab();
    }
    buffer_.push_back(col, val);
  }

  // ─────────────────────────────────────────
  // Queries (both modes)
  //
  // Open:   buffer first (linear, last-write wins), then slab (binary).
  // Locked: slab only (binary search).
  // ─────────────────────────────────────────

  [[nodiscard]] bool contains(index_type col) const {
    if (mode_ == config::matrix_mode::open) {
      if (buffer_.contains(col))
        return true;
    }
    const auto pos = slab_.lower_bound(col);
    return pos < slab_.size() && slab_.key_at(pos) == col;
  }

  [[nodiscard]] const value_type *get(index_type col) const {
    if (to_size(col) >= column_limit_)
      return nullptr;
    if (mode_ == config::matrix_mode::open) {
      if (const value_type *p = buffer_.get_ptr(col); p != nullptr)
        return p;
    }
    const auto pos = slab_.lower_bound(col);
    if (pos < slab_.size() && slab_.key_at(pos) == col)
      return &slab_.value_at(pos);
    return nullptr;
  }

  [[nodiscard]] value_type accumulate() const noexcept {
    if (mode_ == config::matrix_mode::locked) {
      value_type acc = traits::ValueTraits<value_type>::zero();
      for (const auto &entry : slab_) {
        acc += val_of(entry);
      }
      return acc;
    } else {
      // Open mode: buffer sum (deduped) + slab entries not shadowed by buffer.
      value_type acc = buffer_.accumulate();
      for (const auto &entry : slab_) {
        if (!buffer_.contains(key_of(entry)))
          acc += val_of(entry);
      }
      return acc;
    }
  }

  // ─────────────────────────────────────────
  // Iteration (locked mode — sorted slab)
  // ─────────────────────────────────────────

  auto begin() noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::begin() requires locked mode");
    return slab_.begin();
  }
  auto end() noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::end() requires locked mode");
    return slab_.end();
  }
  auto begin() const noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::begin() requires locked mode");
    return slab_.begin();
  }
  auto end() const noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::end() requires locked mode");
    return slab_.end();
  }
  auto cbegin() const noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::cbegin() requires locked mode");
    return slab_.cbegin();
  }
  auto cend() const noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::cend() requires locked mode");
    return slab_.cend();
  }

  auto data() noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::data() requires locked mode");
    return slab_.data();
  }
  auto data() const noexcept {
    assert(mode_ == config::matrix_mode::locked &&
           "row::data() requires locked mode");
    return slab_.data();
  }

  template <class Fn> void for_each_element(Fn &&f) const {
    assert(mode_ == config::matrix_mode::locked &&
           "row::for_each_element() requires locked mode");
    for (auto it = slab_.cbegin(); it != slab_.cend(); ++it) {
      const auto &entry = *it;
      std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
    }
  }

  template <class Fn> void for_each_element(Fn &&f) {
    assert(mode_ == config::matrix_mode::locked &&
           "row::for_each_element() requires locked mode");
    for (auto it = slab_.begin(); it != slab_.end(); ++it) {
      auto &entry = *it;
      std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
    }
  }

private:
  static constexpr size_type to_size(index_type i) noexcept {
    return static_cast<size_type>(i);
  }

  // Extract column key from an iterator element (aos elementPair and soa
  // proxy types both expose first_ref()).
  static decltype(auto) key_of(const auto &entry) {
    if constexpr (requires { entry.first_ref(); })
      return entry.first_ref();
    else
      return entry.first;
  }

  static decltype(auto) val_of(const auto &entry) {
    if constexpr (requires { entry.second_ref(); })
      return entry.second_ref();
    else
      return entry.second;
  }

  // Sort + dedup buffer via normalize_buffer, merge into slab, clear buffer.
  void flush_buffer_to_slab() {
    layout_policy chunk = buffer_.template normalize_buffer<layout_policy>();
    buffer_.clear();
    merge_slab_with_chunk(chunk);
  }

  // Two-pointer merge: sorted slab_ + sorted chunk → new slab_.
  // chunk wins on key collision (buffer is more recent than slab).
  // Zero values are filtered out during merge.
  void merge_slab_with_chunk(const layout_policy &chunk) {
    if (chunk.size() == 0)
      return;
    if (slab_.size() == 0) {
      slab_.reserve(chunk.size());
      for (auto bit = chunk.cbegin(), bend = chunk.cend(); bit != bend; ++bit) {
        auto be = *bit;
        if (!traits::ValueTraits<V>::is_zero(val_of(be)))
          slab_.push_back(key_of(be), val_of(be));
      }
      return;
    }

    layout_policy new_slab;
    new_slab.reserve(slab_.size() + chunk.size());

    auto sit = slab_.cbegin(), send = slab_.cend();
    auto bit = chunk.cbegin(), bend = chunk.cend();

    while (sit != send && bit != bend) {
      auto se = *sit;
      auto be = *bit;

      const auto s_col = key_of(se);
      const auto b_col = key_of(be);

      if (s_col < b_col) {
        if (!traits::ValueTraits<V>::is_zero(val_of(se)))
          new_slab.push_back(s_col, val_of(se));
        ++sit;
      } else if (b_col < s_col) {
        if (!traits::ValueTraits<V>::is_zero(val_of(be)))
          new_slab.push_back(b_col, val_of(be));
        ++bit;
      } else {
        // Same column: chunk (buffer) wins — it carries the newer value.
        if (!traits::ValueTraits<V>::is_zero(val_of(be)))
          new_slab.push_back(b_col, val_of(be));
        ++sit;
        ++bit;
      }
    }
    while (sit != send) {
      auto se = *sit++;
      if (!traits::ValueTraits<V>::is_zero(val_of(se)))
        new_slab.push_back(key_of(se), val_of(se));
    }
    while (bit != bend) {
      auto be = *bit++;
      if (!traits::ValueTraits<V>::is_zero(val_of(be)))
        new_slab.push_back(key_of(be), val_of(be));
    }

    slab_ = std::move(new_slab);
  }

private:
  layout_policy slab_{}; // sorted; canonical data from prior lock cycles
  buffer_t buffer_{};    // unsorted; new inserts since last open()
  config::matrix_mode mode_{config::matrix_mode::open};
  size_type column_limit_{0};
};

} // namespace spira
