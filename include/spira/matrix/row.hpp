#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

#include "spira/concepts.hpp"
#include "spira/traits.hpp"
#include "layouts/layout_of.hpp"

namespace spira::row
{

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    class row
    {
    public:
        using layout_policy = layout::of::storage_of_t<LayoutTag, I, V>;

        row();
        explicit row(std::size_t reserve_hint, size_t const column_limit);

        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] std::size_t capacity() const noexcept;

        void reserve(std::size_t n);
        void clear() noexcept;

        void add(I col, const V &val);
        void remove(I col);

        template <class PairRange>
        void set_row(const PairRange &pairs);

        [[nodiscard]] bool contains(I col) const;
        [[nodiscard]] const V *get(I col) const;

        template <class Func>
        void row<LayoutTag, I, V>::for_each_nnz(const Func &c) const
        {

        private:
            layout_policy storage_;
            size_t const column_limit_;
        };

        // -----------------------------------------------------------------------------
        // Implementation
        // -----------------------------------------------------------------------------

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        row<LayoutTag, I, V>::row()
            : column_limit_(0)
        {
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        row<LayoutTag, I, V>::row(std::size_t reserve_hint, size_t const column_limit)
            : column_limit_(column_limit)
        {
            storage_.reserve(reserve_hint);
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        bool row<LayoutTag, I, V>::empty() const noexcept
        {
            return storage_.empty();
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        std::size_t row<LayoutTag, I, V>::size() const noexcept
        {
            return storage_.size();
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        std::size_t row<LayoutTag, I, V>::capacity() const noexcept
        {
            return storage_.capacity();
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        void row<LayoutTag, I, V>::reserve(std::size_t n)
        {
            storage_.reserve(n);
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        void row<LayoutTag, I, V>::clear() noexcept
        {
            storage_.clear();
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        void row<LayoutTag, I, V>::add(I col, const V &val)
        {
            if (col >= column_limit_)
                return;

            const bool is_zero = traits::ValueTraits<V>::is_zero(val);
            std::size_t pos = storage_.lower_bound(col);

            if (pos < storage_.size() && storage_.key_at(pos) == col)
            {
                if (is_zero)
                {
                    storage_.erase_at(pos);
                }
                else
                {
                    storage_.value_at(pos) = val;
                }
            }
            else
            {
                if (!is_zero)
                {
                    storage_.insert_at(pos, col, val);
                }
            }
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        void row<LayoutTag, I, V>::remove(I col)
        {
            if (col >= column_limit_)
                return;

            std::size_t pos = storage_.lower_bound(col);
            if (pos < storage_.size() && storage_.key_at(pos) == col)
            {
                storage_.erase_at(pos);
            }
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        template <class PairRange>
        void row<LayoutTag, I, V>::set_row(const PairRange &pairs)
        {
            std::vector<std::pair<I, V>> tmp;
            tmp.reserve(std::distance(std::begin(pairs), std::end(pairs)));

            for (auto &&p : pairs)
            {
                const I c = static_cast<I>(p.first);
                const V &v = p.second;

                if (c < column_limit_ && !traits::ValueTraits<V>::is_zero(v))
                {
                    tmp.emplace_back(c, v);
                }
            }

            std::sort(
                tmp.begin(), tmp.end(),
                [](auto const &a, auto const &b)
                {
                    return a.first < b.first;
                });

            std::vector<std::pair<I, V>> cleaned;
            cleaned.reserve(tmp.size());

            for (std::size_t i = 0; i < tmp.size();)
            {
                I c = tmp[i].first;
                V v = tmp[i].second;

                std::size_t j = i + 1;
                while (j < tmp.size() && tmp[j].first == c)
                {
                    v = tmp[j].second;
                    ++j;
                }

                if (!traits::ValueTraits<V>::is_zero(v))
                {
                    cleaned.emplace_back(c, v);
                }

                i = j;
            }

            storage_.clear();
            storage_.reserve(cleaned.size());

            for (auto &[c, v] : cleaned)
            {
                storage_.insert_at(storage_.size(), c, v);
            }
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        bool row<LayoutTag, I, V>::contains(I col) const
        {
            if (col >= column_limit_)
                return false;

            std::size_t pos = storage_.lower_bound(col);
            return (pos < storage_.size() && storage_.key_at(pos) == col);
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        const V *row<LayoutTag, I, V>::get(I col) const
        {
            if (col >= column_limit_)
                return nullptr;

            std::size_t pos = storage_.lower_bound(col);

            if (pos < storage_.size() && storage_.key_at(pos) == col)
            {
                return &storage_.value_at(pos);
            }

            return nullptr;
        }

        template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
        template <class Func>
        void row<LayoutTag, I, V>::for_each_nnz(Func &&f) const
        {
            for (auto &elem : storage_)
            {
                std::forward<Func>(f)(elem);
            }
        }
    };