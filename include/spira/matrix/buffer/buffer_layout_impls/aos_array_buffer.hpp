#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include <spira/matrix/layouts/element_pair.hpp>

namespace spira::buffer::impls
{
    template <class I, class V, std::size_t N>
    class aos_array_buffer
    {
    public:
        using entry_type = spira::layout::elementPair<I, V>;
        using size_type = std::size_t;

        [[nodiscard]] bool empty() const noexcept { return sz_ == 0; }
        [[nodiscard]] size_type size() const noexcept
        {
            deduplicate();
            return sz_;
        }

        [[nodiscard]] size_type remaining_capacity() const noexcept { return N - sz_; }

        void clear() noexcept { sz_ = 0; }

        [[nodiscard]] const I &key_at(size_type idx) const noexcept { return buf_[idx].column; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return buf_[idx].value; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return buf_[idx].value; }

        // NOTE: begin/end expose only the live region [0, sz_)
        [[nodiscard]] entry_type *begin() noexcept { return buf_.data(); }
        [[nodiscard]] entry_type *end() noexcept { return buf_.data() + sz_; }
        [[nodiscard]] const entry_type *begin() const noexcept { return buf_.data(); }
        [[nodiscard]] const entry_type *end() const noexcept { return buf_.data() + sz_; }
        [[nodiscard]] const entry_type *cbegin() const noexcept { return begin(); }
        [[nodiscard]] const entry_type *cend() const noexcept { return end(); }

        void push_back(const I &col, const V &v) noexcept(
            std::is_nothrow_copy_assignable_v<I> && std::is_nothrow_copy_assignable_v<V>)
        {
            assert(sz_ < N && "aos_array_buffer overflow: caller must ensure capacity");
            buf_[sz_++] = entry_type{col, v};
        }

        bool contains(I col) const noexcept
        {
            for (size_type i = sz_; i-- > 0;)
            {
                if (buf_[i].column == col)
                {
                    return true;
                }
            }
            return false;
        }

        const V *get_ptr(I col) const noexcept
        {
            for (size_type i = sz_; i-- > 0;)
            {
                if (buf_[i].column == col)
                    return &buf_[i].value;
            }
            return nullptr;
        }

        V accumulate() const noexcept
        {
            deduplicate();

            V acc = traits::ValueTraits<V>::zero();

            for (size_t i = 0; i < sz_; i++)
            {
                acc += buf_[i].value;
            }
            return acc;
        }

        template <class layout_policy>
        layout_policy normalize_buffer()
        {
            layout_policy chunk;
            chunk.reserve(sz_);

            auto key_of = [](auto const &x) -> decltype(auto)
            {
                if constexpr (requires { x.column; })
                    return x.column;
                else if constexpr (requires { x.first_ref(); })
                    return x.first_ref();
                else
                    return x.first;
            };

            for (size_t i = sz_; i-- > 0;)
            {
                I col = buf_[i].column;

                bool seen = false;
                for (auto it = chunk.begin(); it != chunk.end(); ++it)
                {
                    if (key_of(*it) == col)
                    {
                        seen = true;
                        break;
                    }
                }
                if (!seen)
                    chunk.push_back(buf_[i].column, buf_[i].value);
            }

            if (chunk.empty())
                return chunk;

            std::stable_sort(chunk.begin(), chunk.end(),
                             [&](auto const &a, auto const &b)
                             { return key_of(a) < key_of(b); });

            clear();
            return chunk;
        }

    private:
        mutable std::array<entry_type, N> buf_{};
        mutable size_type sz_{0};

        void deduplicate() const noexcept
        {
            std::array<entry_type, N> tmp;
            size_t out = 0;

            for (size_t i = sz_; i-- > 0;)
            {
                bool seen = false;

                for (size_t j = 0; j < out; j++)
                {
                    if (tmp[j].column == buf_[i].column)
                    {
                        seen = true;
                        break;
                    }
                }

                if (seen == false)
                {
                    tmp[out++] = buf_[i];
                }
            }

            for (size_t i = 0; i < out; i++)
            {
                buf_[i] = tmp[i];
            }

            sz_ = out;
        }
    };
}