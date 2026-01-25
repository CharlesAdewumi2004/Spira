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

        using small_buffer =
            buffer::traits::traits_of_type<
                buffer::tags::array_buffer<LayoutTag>, I, V, config::spmv.buffersize>;

        using balanced_buffer =
            buffer::traits::traits_of_type<
                buffer::tags::array_buffer<LayoutTag>, I, V, config::balanced.buffersize>;

        using insert_heavy_buffer =
            buffer::traits::traits_of_type<
                buffer::tags::hash_map_buffer, I, V, config::insert_heavy.buffersize>;

        using buffer_variant = std::variant<small_buffer, balanced_buffer, insert_heavy_buffer>;

        row();
        explicit row(config::mode_policy mode_policy);

        row(std::size_t reserve_hint, std::size_t column_limit);
        row(std::size_t reserve_hint, std::size_t column_limit, config::mode_policy mode_policy);

        [[nodiscard]] mode::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] config::mode_policy const& traits() const noexcept { return traits_; }

        void set_mode(mode::matrix_mode m);

        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] std::size_t capacity() const noexcept;

        [[nodiscard]] std::size_t buffer_size() const noexcept;
        [[nodiscard]] std::size_t slab_size() const noexcept { return slab_.size(); }

        void reserve(std::size_t n);
        void clear() noexcept;

        void add(I col, V const& val);

        [[nodiscard]] bool contains(I col) const;
        [[nodiscard]] V const* get(I col) const;

        [[nodiscard]] V accumulate() const noexcept;

        void flush() const;
        [[nodiscard]] bool is_dirty() const noexcept { return dirty_; }

        template <class Fn>
        void for_each_element(Fn&& f) const noexcept(noexcept(std::declval<Fn&>()(std::declval<I>(), std::declval<V const&>())));

        auto begin() noexcept { return slab_.begin(); }
        auto end() noexcept { return slab_.end(); }
        auto begin() const noexcept { return slab_.begin(); }
        auto end() const noexcept { return slab_.end(); }
        auto cbegin() const noexcept { return slab_.cbegin(); }
        auto cend() const noexcept { return slab_.cend(); }

    private:
        template <class Fn>
        void for_each_slab_element(Fn&& f) const noexcept(
            noexcept(std::declval<Fn&>()(std::declval<I>(), std::declval<V const&>())));

        template <class Fn>
        decltype(auto) with_buffer_mut(Fn&& fn)
        {
            switch (buffer_.index())
            {
            case 0: return std::forward<Fn>(fn)(std::get<0>(buffer_));
            case 1: return std::forward<Fn>(fn)(std::get<1>(buffer_));
            case 2: return std::forward<Fn>(fn)(std::get<2>(buffer_));
            default: std::terminate();
            }
        }

        template <class Fn>
        decltype(auto) with_buffer_const(Fn&& fn) const
        {
            switch (buffer_.index())
            {
            case 0: return std::forward<Fn>(fn)(std::get<0>(buffer_));
            case 1: return std::forward<Fn>(fn)(std::get<1>(buffer_));
            case 2: return std::forward<Fn>(fn)(std::get<2>(buffer_));
            default: std::terminate();
            }
        }

        [[nodiscard]] bool buffer_has_live() const noexcept;
        void recompute_dirty() const noexcept { dirty_ = buffer_has_live(); }

        void reset_buffer_for_mode(mode::matrix_mode m);

    private:
        mutable layout_policy slab_{};
        mutable buffer_variant buffer_{balanced_buffer{}};

        mutable bool dirty_{false};

        mode::matrix_mode  mode_{mode::matrix_mode::balanced};
        config::mode_policy traits_{mode::policy_for(mode::matrix_mode::balanced)};
        std::size_t        column_limit_{0};
    };

    // -----------------------------
    // Constructors
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row() = default;

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row(config::mode_policy mode_policy)
        : slab_{}
        , buffer_{balanced_buffer{}}
        , dirty_{false}
        , mode_{mode::matrix_mode::balanced}
        , traits_{mode_policy}
        , column_limit_{0}
    {}

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row(std::size_t reserve_hint, std::size_t column_limit)
        : row{}
    {
        column_limit_ = column_limit;
        slab_.reserve(reserve_hint);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row(std::size_t reserve_hint,
                              std::size_t column_limit,
                              config::mode_policy mode_policy)
        : row{mode_policy}
    {
        column_limit_ = column_limit;
        slab_.reserve(reserve_hint);
    }

    // -----------------------------
    // Mode / configuration
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::reset_buffer_for_mode(mode::matrix_mode m)
    {
        switch (m)
        {
        case mode::matrix_mode::spmv:
            buffer_.template emplace<small_buffer>();
            break;
        case mode::matrix_mode::balanced:
            buffer_.template emplace<balanced_buffer>();
            break;
        case mode::matrix_mode::insert_heavy:
            buffer_.template emplace<insert_heavy_buffer>();
            break;
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::set_mode(mode::matrix_mode m)
    {
        if (mode_ == m)
        {
            return;
        }

        mode_   = m;
        traits_ = mode::policy_for(m);

        flush();
        reset_buffer_for_mode(m);
        recompute_dirty();
    }

    // -----------------------------
    // Size / capacity
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::empty() const noexcept
    {
        if (!slab_.empty())
        {
            return false;
        }
        return !buffer_has_live();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t row<LayoutTag, I, V>::size() const noexcept
    {
        recompute_dirty();
        if (dirty_)
        {
            flush();
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

    // -----------------------------
    // Mutations
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::clear() noexcept
    {
        slab_.clear();
        with_buffer_mut([](auto& buf) { buf.clear(); });
        dirty_ = false;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::add(I col, V const& val)
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
        {
            return;
        }

        dirty_ = true;

        with_buffer_mut(
            [&](auto& buf)
            {
                if (buf.remaining_capacity() == 0)
                {
                    flush();
                }
                buf.push_back(col, val);
            });
    }

    // -----------------------------
    // Queries
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::contains(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
        {
            return false;
        }

        const bool in_buffer = with_buffer_const(
            [&](auto const& buf)
            {
                return buf.contains(col);
            });

        if (in_buffer)
        {
            return true;
        }

        const auto pos = slab_.lower_bound(col);
        return (pos < slab_.size() && slab_.key_at(pos) == col);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V const* row<LayoutTag, I, V>::get(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
        {
            return nullptr;
        }

        if (auto p = with_buffer_const(
                [&](auto const& buf)
                {
                    return buf.get_ptr(col);
                }))
        {
            if (traits::ValueTraits<V>::is_zero(*p))
            {
                return nullptr;
            }
            return p;
        }

        const auto pos = slab_.lower_bound(col);
        if (pos < slab_.size() && slab_.key_at(pos) == col)
        {
            return &slab_.value_at(pos);
        }

        return nullptr;
    }

    // -----------------------------
    // Reductions
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V row<LayoutTag, I, V>::accumulate() const noexcept
    {
        if (dirty_)
        {
            flush();
        }

        V acc = traits::ValueTraits<V>::zero();
        for (auto const& entry : slab_)
        {
            acc += entry.second_ref();
        }
        return acc;
    }

    // -----------------------------
    // Flush / merge
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::flush() const
    {
        layout_policy chunk = std::visit(
            [](auto& buf) -> layout_policy
            {
                return buf.template normalize_buffer<layout_policy>();
            },
            buffer_);

        if (chunk.empty())
        {
            recompute_dirty();
            return;
        }

        spira::algorithms::merge<layout_policy>(slab_, chunk);
        recompute_dirty();
    }

    // -----------------------------
    // Iteration utilities
    // -----------------------------

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Fn>
    void row<LayoutTag, I, V>::for_each_slab_element(Fn&& f) const noexcept(
        noexcept(std::declval<Fn&>()(std::declval<I>(), std::declval<V const&>())))
    {
        for (auto it = slab_.cbegin(); it != slab_.cend(); ++it)
        {
            auto const entry = *it;
            std::forward<Fn>(f)(entry.first_ref(), entry.second_ref());
        }
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Fn>
    void row<LayoutTag, I, V>::for_each_element(Fn&& f) const noexcept(
        noexcept(std::declval<Fn&>()(std::declval<I>(), std::declval<V const&>())))
    {
        if (dirty_)
        {
            flush();
        }
        for_each_slab_element(std::forward<Fn>(f));
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t row<LayoutTag, I, V>::buffer_size() const noexcept
    {
        return with_buffer_const(
            [](auto const& buf) noexcept
            {
                return buf.size();
            });
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::buffer_has_live() const noexcept
    {
        return with_buffer_const(
            [](auto const& buf)
            {
                return !buf.empty();
            });
    }

} 
