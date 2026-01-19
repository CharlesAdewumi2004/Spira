#pragma once

#include <unordered_map>
#include <numeric>

#include <spira/traits.hpp>
#include <spira/matrix/layouts/element_pair.hpp>

namespace spira::buffer::impls
{
    template <class I, class V>
    class hash_map_buffer
    {
    public:
        using size_type = std::size_t;
        using entry_type = spira::layout::elementPair<I, V>;

        bool empty() const noexcept { return buf_.empty(); }
        size_type size() const noexcept { return buf_.size(); }

        void clear() noexcept { buf_.clear(); }
        void push_back(const I &col, const V &val) noexcept { buf_[col] = val; }

        bool contains(I col) const noexcept
        {
            return buf_.contains(col);
        }
        const V *get_ptr(I col) const noexcept
        {
            if (contains(col))
            {
                return &buf_[col];
            }
            return nullptr;
        }

        V accumulate() const noexcept
        {
            return std::accumulate(buf_.begin(), buf_.end(), traits::ValueTraits<V>::zero(), [](V acc, auto const &kv)
                                   { return acc + kv.second; });
        }

        template <class layout_policy>
        layout_policy normalize_buffer()
        {
            layout_policy chunk;
            chunk.reserve(buf_.size());

            for (auto &[col, val] : buf_)
            {
                chunk.push_back(col, val);
            }

            auto key_of = [](auto const &x) -> decltype(auto)
            {
                if constexpr (requires { x.first_ref(); })
                    return x.first_ref();
                else
                    return x.first;
            };

            std::stable_sort(chunk.begin(), chunk.end(), [&](auto const &a, auto const &b)
                             { return key_of(a) < key_of(b); });

            return chunk;
        }

        std::unordered_map<I, V>::iterator begin() noexcept{return buf_.begin();}
        std::unordered_map<I, V>::iterator end() noexcept{return buf_.end();}

        std::unordered_map<I, V>::const_iterator begin() const noexcept{cbegin();}
        std::unordered_map<I, V>::const_iterator end() const noexcept{cend();}
        std::unordered_map<I, V>::const_iterator cbegin() const noexcept{buf_.cbegin();}
        std::unordered_map<I, V>::const_iterator cend() const noexcept{buf_.cend();}

    private:
        std::unordered_map<I, V> buf_;
    };
}