#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <spira/algorithms/merge.hpp>
#include <spira/concepts.hpp>
#include <spira/config.hpp>
#include <spira/matrix/buffer/buffer.hpp>
#include <spira/matrix/layouts/layout_of.hpp>
#include <spira/matrix/mode/matrix_mode.hpp>
#include <spira/matrix/mode/mode_traits.hpp>
#include <spira/traits.hpp>

namespace spira
{
    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    class row
    {
    public:
        using layout_policy = layout::of::storage_of_t<LayoutTag, I, V>;
        using entry_type = spira::layout::elementPair<I, V>;

        using small_buffer =
            buffer::traits::traits_of_type<LayoutTag, I, V, config::spmv.buffersize>;

        using balanced_buffer =
            buffer::traits::traits_of_type<LayoutTag, I, V,
                                           config::balanced.buffersize>;

        using insert_heavy_buffer =
            buffer::traits::traits_of_type<LayoutTag, I, V,
                                           config::insert_heavy.buffersize>;

        using buffer_variant =
            std::variant<small_buffer, balanced_buffer, insert_heavy_buffer>;

        row();
        explicit row(std::size_t reserve_hint, std::size_t column_limit);

        [[nodiscard]] mode::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] config::mode_policy const &traits() const noexcept
        {
            return traits_;
        }

        void set_mode(mode::matrix_mode m);

        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        [[nodiscard]] size_t capacity() const noexcept;

        [[nodiscard]] size_t buffer_size() const noexcept
        {
            return std::visit([](auto const &buf) noexcept
                              { return buf.size(); },
                              buffer_);
        }

        [[nodiscard]] size_t number_of_runs() const noexcept { return runs_.size(); }
        [[nodiscard]] size_t slab_size() const noexcept { return slab_.size(); }

        void reserve(std::size_t n);
        void clear() noexcept;

        void add(I col, const V &val);

        [[nodiscard]] bool contains(I col) const;
        [[nodiscard]] const V *get(I col) const;

        [[nodiscard]] V accumulate() const noexcept;

        void flush() const;
        bool is_dirty() const noexcept { return dirty_; }

        template <class Fn>
        void for_each_element(Fn &&f) const
            noexcept(noexcept(std::declval<Fn &>()(std::declval<I>(),
                                                   std::declval<const V &>())));

        auto begin() noexcept { return slab_.begin(); }
        auto end() noexcept { return slab_.end(); }
        auto begin() const noexcept { return slab_.begin(); }
        auto end() const noexcept { return slab_.end(); }
        auto cbegin() const noexcept { return slab_.cbegin(); }
        auto cend() const noexcept { return slab_.cend(); }

    private:
        static std::vector<entry_type> &tls_chunk()
        {
            thread_local std::vector<entry_type> chunk;
            return chunk;
        }

        static layout_policy &tls_layout_tmp()
        {
            thread_local layout_policy tmp;
            return tmp;
        }

        static I const &key_of(entry_type const &e) noexcept
        {
            if constexpr (requires { e.column; })
                return e.column;
            else
                return e.first;
        }

        static V const &val_of(entry_type const &e) noexcept
        {
            if constexpr (requires { e.value; })
                return e.value;
            else
                return e.second;
        }

        template <class Fn>
        void iterate_over_elements(Fn &&f) const
            noexcept(noexcept(std::declval<Fn &>()(std::declval<I>(),
                                                   std::declval<V const &>())))
        {
            for (auto it = slab_.cbegin(); it != slab_.cend(); ++it)
            {
                auto entry = *it;

                auto const &col = entry.first_ref();
                auto const &val = entry.second_ref();

                std::forward<Fn>(f)(col, val);
            }
        }

        bool buffer_has_live() const noexcept
        {
            return std::visit([](auto const &buf)
                              { return !buf.empty(); }, buffer_);
        }

        void recompute_dirty() const noexcept
        {
            dirty_ = buffer_has_live() || !runs_.empty();
        }

        void full_flush() const noexcept;

    private:
        mutable layout_policy slab_{};
        mutable buffer_variant buffer_{};
        mutable std::vector<layout_policy> runs_{};

        mutable bool dirty_{false};

        mode::matrix_mode mode_{mode::matrix_mode::balanced};
        config::mode_policy traits_{mode::policy_for(mode::matrix_mode::balanced)};
        std::size_t column_limit_{0};
    };

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row()
        : slab_{}, buffer_{balanced_buffer{}}, runs_{}, dirty_{false},
          mode_{mode::matrix_mode::balanced},
          traits_{mode::policy_for(mode::matrix_mode::balanced)}, column_limit_{0}
    {
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row(std::size_t reserve_hint, std::size_t column_limit)
        : slab_{}, buffer_{balanced_buffer{}}, runs_{}, dirty_{false},
          mode_{mode::matrix_mode::balanced},
          traits_{mode::policy_for(mode::matrix_mode::balanced)},
          column_limit_{column_limit}
    {
        slab_.reserve(reserve_hint);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::set_mode(mode::matrix_mode m)
    {
        if (mode_ == m)
            return;

        mode_ = m;
        traits_ = mode::policy_for(m);

        flush();

        switch (m)
        {
        case mode::matrix_mode::spmv:
            if (dirty_)
            {
                full_flush();
            }
            buffer_.template emplace<small_buffer>();
            break;
        case mode::matrix_mode::balanced:
            buffer_.template emplace<balanced_buffer>();
            break;
        case mode::matrix_mode::insert_heavy:
            buffer_.template emplace<insert_heavy_buffer>();
            break;
        }
        recompute_dirty();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::empty() const noexcept
    {
        if (!slab_.empty())
            return false;
        for (auto const &run : runs_)
            if (!run.empty())
                return false;

        return !buffer_has_live();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t row<LayoutTag, I, V>::size() const noexcept
    {
        recompute_dirty();
        if(dirty_){
            full_flush();
        }
        
        return slab_.size();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t row<LayoutTag, I, V>::capacity() const noexcept
    {
        return slab_.capacity();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::reserve(std::size_t n)
    {
        slab_.reserve(n);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::clear() noexcept
    {
        slab_.clear();
        runs_.clear();
        std::visit([](auto &buf)
                   { buf.clear(); }, buffer_);
        dirty_ = false;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::add(I col, const V &val)
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
            return;

        dirty_ = true;

        std::visit(
            [&](auto &buf)
            {
                if (buf.remaining_capacity() == 0)
                {
                    flush();
                }
                buf.push_back(col, val);
            },
            buffer_);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::contains(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
        {
            return false;
        }

        if (std::visit(
                [&](auto const &buf)
                {
                    return buf.contains(col);
                },
                buffer_))
        {
            return true;
        }

        for (auto it = runs_.rbegin(); it != runs_.rend(); ++it)
        {
            auto const &run = *it;
            auto pos = run.lower_bound(col);
            if (pos < run.size() && run.key_at(pos) == col)
            {
                return !traits::ValueTraits<V>::is_zero(run.value_at(pos));
            }
        }

        auto pos = slab_.lower_bound(col);
        if (pos < slab_.size() && slab_.key_at(pos) == col)
            return true;
        return false;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    const V *row<LayoutTag, I, V>::get(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
            return nullptr;

        if (auto p = std::visit([&](auto const &buf)
                                { return buf.get_ptr(col); },
                                buffer_))
        {
            if (traits::ValueTraits<V>::is_zero(*p))
                return nullptr;
            return p;
        }

        for (auto it = runs_.rbegin(); it != runs_.rend(); ++it)
        {
            auto const &run = *it;
            auto pos = run.lower_bound(col);
            if (pos < run.size() && run.key_at(pos) == col)
            {
                if (traits::ValueTraits<V>::is_zero(run.value_at(pos)))
                    return nullptr;
                return &run.value_at(pos);
            }
        }

        auto pos = slab_.lower_bound(col);
        if (pos < slab_.size() && slab_.key_at(pos) == col)
            return &slab_.value_at(pos);

        return nullptr;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V row<LayoutTag, I, V>::accumulate() const noexcept
    {
        if (dirty_)
        {
            full_flush();
        }

        V acc = traits::ValueTraits<V>::zero();

        for (auto const &entry : slab_)
        {
            acc += entry.second_ref();
        }

        return acc;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::flush() const
    {
        layout_policy chunk = std::visit(
            [](auto &buf) -> layout_policy
            {
                return buf.template normalize_buffer<layout_policy>();
            },
            buffer_);

        if (chunk.empty())
        {
            recompute_dirty();
            return;
        }

        bool runs_allowed = (traits_.max_runs != 0);

        if (!runs_allowed)
        {
            spira::algorithms::merge<layout_policy>(slab_, chunk);

            recompute_dirty();

            return;
        }
        else if (runs_.size() >= traits_.max_runs)
        {
            runs_.push_back(chunk);

            for (auto &run : runs_)
            {
                spira::algorithms::merge<layout_policy>(slab_, run);
            }

            runs_.clear();

            recompute_dirty();
        }
        else
        {
            runs_.push_back(chunk);
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::full_flush() const noexcept
    {
        layout_policy chunk = std::visit(
            [](auto &buf) -> layout_policy
            {
                return buf.template normalize_buffer<layout_policy>();
            },
            buffer_);

        runs_.push_back(chunk);

        for (auto &run : runs_)
        {
            spira::algorithms::merge(slab_, run);
        }

        runs_.clear();

        recompute_dirty();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Fn>
    void row<LayoutTag, I, V>::for_each_element(Fn &&f) const
        noexcept(noexcept(std::declval<Fn &>()(std::declval<I>(),
                                               std::declval<const V &>())))
    {
        if (mode_ == mode::matrix_mode::spmv)
        {
            if (dirty_)
                flush();
            iterate_over_elements(f);
        }
        else
        {
            if (dirty_)
                full_flush();
            iterate_over_elements(f);
        }
    }

}
