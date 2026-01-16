#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <iostream>

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

        using small_buffer = buffer::traits::traits_of_type<LayoutTag, I, V, config::spmv.buffersize>;
        using balanced_buffer = buffer::traits::traits_of_type<LayoutTag, I, V, config::balanced.buffersize>;
        using insert_heavy_buffer = buffer::traits::traits_of_type<LayoutTag, I, V, config::insert_heavy.buffersize>;

        using buffer_variant = std::variant<small_buffer, balanced_buffer, insert_heavy_buffer>;

        row();
        explicit row(std::size_t reserve_hint, std::size_t column_limit);

        [[nodiscard]] mode::matrix_mode mode() const noexcept { return mode_; }
        [[nodiscard]] config::mode_policy const &traits() const noexcept { return traits_; }

        void set_mode(mode::matrix_mode m);

        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        [[nodiscard]] size_t capacity() const noexcept;

        [[nodiscard]] size_t buffer_size() const noexcept
        {
            return std::visit([](auto const &buf) noexcept
                              { return buf.size(); }, buffer_);
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

        template <class Fn>
        void for_each_element(Fn &&f) const
            noexcept(noexcept(std::declval<Fn &>()(std::declval<I>(), std::declval<const V &>())));

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
        void iterate_over_elements(Fn &&f) const noexcept
        {
            for (auto it = slab_.cbegin(); it != slab_.cend(); ++it)
            {
                auto &&[col, val] = *it;
                f(col, val);
            }
        }

        void normalize_chunk(std::vector<entry_type> &chunk) const;
        void merge_sorted_chunk_into_slab(std::vector<entry_type> const &chunk) const;
        void push_chunk_as_run(std::vector<entry_type> const &chunk) const;
        bool should_compact() const;
        void compact_runs_into_slab() const;

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
        : slab_{}, buffer_{balanced_buffer{}}, runs_{}, dirty_{false}, mode_{mode::matrix_mode::balanced}, traits_{mode::policy_for(mode::matrix_mode::balanced)}, column_limit_{0}
    {
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    row<LayoutTag, I, V>::row(std::size_t reserve_hint, std::size_t column_limit)
        : slab_{}, buffer_{balanced_buffer{}}, runs_{}, dirty_{false}, mode_{mode::matrix_mode::balanced}, traits_{mode::policy_for(mode::matrix_mode::balanced)}, column_limit_{column_limit}
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
        dirty_ = false;

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
    bool row<LayoutTag, I, V>::empty() const noexcept
    {
        if (!slab_.empty())
            return false;

        for (auto const &run : runs_)
            if (!run.empty())
                return false;

        const bool buffer_has_live =
            std::visit([&](auto const &buf) -> bool
                       {
                for (std::size_t i = 0; i < buf.size(); ++i)
                    if (!traits::ValueTraits<V>::is_zero(buf.value_at(i)))
                        return true;
                return false; }, buffer_);

        return !buffer_has_live;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    std::size_t row<LayoutTag, I, V>::size() const noexcept
    {
        std::size_t runs_size = 0;
        for (auto const &run : runs_)
            runs_size += run.size();

        const std::size_t buffer_size =
            std::visit([](auto const &buf)
                       { return buf.size(); }, buffer_);

        return slab_.size() + runs_size + buffer_size;
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

        std::visit([&](auto &buf)
                   {
            if (buf.remaining_capacity() == 0)
                flush();
            buf.push_back(col, val); }, buffer_);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::contains(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
            return false;

        if (std::visit([&](auto const &buf)
                       { return buf.contains(col); }, buffer_))
            return true;

        for (auto it = runs_.rbegin(); it != runs_.rend(); ++it)
        {
            auto const &run = *it;
            auto pos = run.lower_bound(col);
            if (pos < run.size() && run.key_at(pos) == col)
                return true;
        }

        auto pos = slab_.lower_bound(col);
        return (pos < slab_.size() && slab_.key_at(pos) == col);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    const V *row<LayoutTag, I, V>::get(I col) const
    {
        if (static_cast<std::size_t>(col) >= column_limit_)
            return nullptr;

        if (auto p = std::visit([&](auto const &buf)
                                { return buf.get_ptr(col); }, buffer_))
            return p;

        for (auto it = runs_.rbegin(); it != runs_.rend(); ++it)
        {
            auto const &run = *it;
            auto pos = run.lower_bound(col);
            if (pos < run.size() && run.key_at(pos) == col)
                return &run.value_at(pos);
        }

        auto pos = slab_.lower_bound(col);
        if (pos < slab_.size() && slab_.key_at(pos) == col)
            return &slab_.value_at(pos);

        return nullptr;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    V row<LayoutTag, I, V>::accumulate() const noexcept
    {
        V acc = traits::ValueTraits<V>::zero();

        acc += std::visit([](auto const &buf)
                          { return buf.accumulate(); }, buffer_);

        for (auto const &run : runs_)
            for (auto const &[c, v] : run)
                acc += v;

        for (auto const &[c, v] : slab_)
            acc += v;

        return acc;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::flush() const
    {

        auto &chunk = tls_chunk();
        chunk = std::visit(
            [](auto &buf) -> std::vector<entry_type>
            { return buf.flush_buffer(); },
            buffer_);

        if (chunk.empty() && runs_.empty())
        {
            dirty_ = false;
            return;
        }

        normalize_chunk(chunk);
        if (chunk.empty() && runs_.empty())
        {
            dirty_ = false;
            return;
        }

        const bool runs_allowed = (traits_.max_runs != 0);

        if (!runs_allowed)
        {
            merge_sorted_chunk_into_slab(chunk);
            chunk.clear();
        }

        if (slab_.size() < traits_.slab_merge_threshold)
        {
            merge_sorted_chunk_into_slab(chunk);
        }
        else
        {
            push_chunk_as_run(chunk);
        }
        if (should_compact())
        {
            compact_runs_into_slab();
        }
        chunk.clear();
        dirty_ = false;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::normalize_chunk(std::vector<entry_type> &chunk) const
    {
        std::sort(chunk.begin(), chunk.end(),
                  [](entry_type const &a, entry_type const &b)
                  { return key_of(a) < key_of(b); });

        std::size_t w = 0;

        for (std::size_t i = 0; i < chunk.size();)
        {
            const I c = key_of(chunk[i]);
            V v = val_of(chunk[i]);

            std::size_t j = i + 1;
            while (j < chunk.size() && key_of(chunk[j]) == c)
            {
                v = val_of(chunk[j]);
                ++j;
            }

            if (!traits::ValueTraits<V>::is_zero(v))
            {
                if constexpr (requires { chunk[w].column; chunk[w].value; })
                {
                    chunk[w].column = c;
                    chunk[w].value = v;
                }
                else
                {
                    chunk[w].first = c;
                    chunk[w].second = v;
                }
                ++w;
            }

            i = j;
        }

        chunk.resize(w);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::merge_sorted_chunk_into_slab(std::vector<entry_type> const &chunk) const
    {
        if (chunk.empty())
            return;

        if (slab_.empty())
        {
            slab_.clear();
            slab_.reserve(chunk.size());
            for (auto const &e : chunk)
                slab_.insert_at(slab_.size(), key_of(e), val_of(e));
            return;
        }

        auto &out = tls_layout_tmp();
        out.clear();
        out.reserve(slab_.size() + chunk.size());

        std::size_t i = 0; // slab index

        for (std::size_t j = 0; j < chunk.size(); ++j)
        {
            const I c = key_of(chunk[j]);
            const V v = val_of(chunk[j]);

            while (i < slab_.size() && slab_.key_at(i) < c)
            {
                out.insert_at(out.size(), slab_.key_at(i), slab_.value_at(i));
                ++i;
            }

            if (i < slab_.size() && slab_.key_at(i) == c)
            {
                out.insert_at(out.size(), c, v); // chunk overrides slab
                ++i;
            }
            else
            {
                out.insert_at(out.size(), c, v);
            }
        }

        while (i < slab_.size())
        {
            out.insert_at(out.size(), slab_.key_at(i), slab_.value_at(i));
            ++i;
        }

        using std::swap;
        swap(slab_, out);
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::push_chunk_as_run(std::vector<entry_type> const &chunk) const
    {
        layout_policy run;
        run.reserve(chunk.size());
        for (auto const &e : chunk)
            run.insert_at(run.size(), key_of(e), val_of(e));

        runs_.push_back(std::move(run));
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    bool row<LayoutTag, I, V>::should_compact() const
    {
        if (runs_.empty())
            return false;

        if (runs_.size() > traits_.max_runs)
            return true;

        if (traits_.compact_run_ratio > 0.0 && slab_.size() > 0)
        {
            std::size_t run_total = 0;
            for (auto const &r : runs_)
                run_total += r.size();

            const double ratio =
                static_cast<double>(run_total) / static_cast<double>(slab_.size());

            if (ratio > traits_.compact_run_ratio)
                return true;
        }

        return false;
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    void row<LayoutTag, I, V>::compact_runs_into_slab() const
    {
        if (runs_.empty())
            return;

        auto &out = tls_layout_tmp();

        for (auto &run : runs_)
        {
            if (run.empty())
                continue;

            out.clear();
            out.reserve(slab_.size() + run.size());

            std::size_t i = 0; // slab idx
            std::size_t j = 0; // run idx

            while (i < slab_.size() && j < run.size())
            {
                const I sc = slab_.key_at(i);
                const I rc = run.key_at(j);

                if (sc < rc)
                {
                    out.insert_at(out.size(), sc, slab_.value_at(i));
                    ++i;
                }
                else if (rc < sc)
                {
                    out.insert_at(out.size(), rc, run.value_at(j));
                    ++j;
                }
                else
                {
                    out.insert_at(out.size(), rc, run.value_at(j)); // run overrides slab
                    ++i;
                    ++j;
                }
            }

            while (i < slab_.size())
            {
                out.insert_at(out.size(), slab_.key_at(i), slab_.value_at(i));
                ++i;
            }
            while (j < run.size())
            {
                out.insert_at(out.size(), run.key_at(j), run.value_at(j));
                ++j;
            }

            using std::swap;
            swap(slab_, out);
        }

        runs_.clear();
    }

    template <class LayoutTag, concepts::Indexable I, concepts::Valueable V>
    template <class Fn>
    void row<LayoutTag, I, V>::for_each_element(Fn &&f) const noexcept(noexcept(std::declval<Fn &>()(std::declval<I>(), std::declval<const V &>())))
    {
        if (mode_ == mode::matrix_mode::spmv)
        {
            if (dirty_)
            {
                flush();
            }

            iterate_over_elements(f);
        }
        else if (mode_ == mode::matrix_mode::balanced || mode_ == mode::matrix_mode::insert_heavy)
        {
            if (dirty_)
            {
                flush();
            }

            if (!runs_.empty())
            {
                compact_runs_into_slab();
            }

            iterate_over_elements(f);
        }
    }

}