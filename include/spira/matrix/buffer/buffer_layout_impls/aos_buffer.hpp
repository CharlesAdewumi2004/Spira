#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <cassert>

namespace spira::buffer::impls
{

    template <class I, class V>
    struct bufferElementPair
    {
        I column;
        V value;
    };

    template <class I, class V, std::size_t N>
    class aos_buffer
    {
    public:
        using entry_type = bufferElementPair<I, V>;
        using size_type = std::size_t;

        [[nodiscard]] bool empty() const noexcept { return sz_ == 0; }
        [[nodiscard]] size_type size() const noexcept { return sz_; }
        [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
        [[nodiscard]] size_type remaining_capacity() const noexcept { return N - sz_; }

        void clear() noexcept { sz_ = 0; }

        [[nodiscard]] const I key_at(size_type idx) const noexcept { return buf_[idx].column; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return buf_[idx].value; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return buf_[idx].value; }

        [[nodiscard]] entry_type *data() noexcept { return buf_.data(); }
        [[nodiscard]] const entry_type *data() const noexcept { return buf_.data(); }

        [[nodiscard]] entry_type *begin() noexcept { return buf_.data(); }
        [[nodiscard]] entry_type *end() noexcept { return buf_.data() + sz_; }
        [[nodiscard]] const entry_type *begin() const noexcept { return buf_.data(); }
        [[nodiscard]] const entry_type *end() const noexcept { return buf_.data() + sz_; }

        void push_back(const I &col, const V &v)
        {
            buf_[sz_++] = entry_type{col, v};
        }

    private:
        std::array<entry_type, N> buf_{};
        size_type sz_ = 0;
    };

}
