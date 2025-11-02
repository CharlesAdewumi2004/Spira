#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

#include "spira/concepts.hpp"
#include "spira/traits.hpp"
#include "spira/layouts/layout_of.hpp"

namespace spira{

template<class LayoutTag, concepts::Indexable I, concepts::Valueable V>
class row {
    using policy_t = layout::of::storage_of_t<LayoutTag, I, V>;

public:
    using policy_type = policy_t;

    row() : column_limit_(0) {}
    explicit row(std::size_t reserve_hint, size_t const column_limit) : column_limit_(column_limit)
    {
        storage_.reserve(reserve_hint);
    }

    [[nodiscard]] bool        empty()    const noexcept {
        return storage_.empty();
    }
    [[nodiscard]] std::size_t size()     const noexcept {
        return storage_.size();
    }
    [[nodiscard]] std::size_t capacity() const noexcept {
        return storage_.capacity();
    }
    void reserve(std::size_t n) {
        storage_.reserve(n);
    }
    void clear() noexcept {
        storage_.clear();
    }

    void add(I col, const V& val) {
        if (col >= column_limit_) return;

        const bool is_zero = traits::ValueTraits<V>::is_zero(val);
        std::size_t pos = storage_.lower_bound(col);

        if (pos < storage_.size() && storage_.key_at(pos) == col) {
            if (is_zero) {
                storage_.erase_at(pos);
            } else {
                storage_.value_at(pos) = val;
            }
        } else {
            if (!is_zero) {
                storage_.insert_at(pos, col, val);
            }
        }
    }

    void remove(I col) {
        if (col >= column_limit_) return;
        std::size_t pos = storage_.lower_bound(col);
        if (pos < storage_.size() && storage_.key_at(pos) == col) {
            storage_.erase_at(pos);
        }
    }

    template<class PairRange>
    void set_row(const PairRange& pairs) {
        std::vector<std::pair<I,V>> tmp;
        tmp.reserve(std::distance(std::begin(pairs), std::end(pairs)));

        for (auto&& p : pairs) {
            const I c = static_cast<I>(p.first);
            const V& v = p.second;
            if (c < column_limit_ && !traits::ValueTraits<V>::is_zero(v)) {
                tmp.emplace_back(c, v);
            }
        }

        std::sort(tmp.begin(), tmp.end(),
                  [](auto const& a, auto const& b){ return a.first < b.first; });

        std::vector<std::pair<I,V>> cleaned;
        cleaned.reserve(tmp.size());
        for (std::size_t i = 0; i < tmp.size(); ) {
            I c = tmp[i].first;
            V v = tmp[i].second;
            std::size_t j = i + 1;
            while (j < tmp.size() && tmp[j].first == c) {
                v = tmp[j].second; // overwrite-last policy
                ++j;
            }
            if (!traits::ValueTraits<V>::is_zero(v)) {
                cleaned.emplace_back(c, v);
            }
            i = j;
        }

        storage_.clear();
        storage_.reserve(cleaned.size());
        for (auto& [c, v] : cleaned) {
            storage_.insert_at(storage_.size(), c, v);
        }
    }

    [[nodiscard]] bool contains(I col) const {
        if (col >= column_limit_) return false;
        std::size_t pos = storage_.lower_bound(col);
        return (pos < storage_.size() && storage_.key_at(pos) == col);
    }

    [[nodiscard]] const V* get(I col) const {
        if (col >= column_limit_) return nullptr;
        std::size_t pos = storage_.lower_bound(col);
        if (pos < storage_.size() && storage_.key_at(pos) == col) {
            return &storage_.value_at(pos);
        }
        return nullptr;
    }

private:
    policy_t storage_;
    size_t const column_limit_;
};

}
