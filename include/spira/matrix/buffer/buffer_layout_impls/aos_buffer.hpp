#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>

#include <spira/matrix/layouts/aos.hpp>

namespace spira::buffer::impls
{
    template <class I, class V, std::size_t N>
    class aos_buffer
    {
    public:
        using entry_type = spira::layout::elementPair<I, V>;
        using size_type = std::size_t;

        [[nodiscard]] bool empty() const noexcept { return sz_ == 0; }
        [[nodiscard]] size_type size() const noexcept { return sz_; }
        [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
        [[nodiscard]] size_type remaining_capacity() const noexcept { return N - sz_; }

        void clear() noexcept { sz_ = 0; }

        [[nodiscard]] const I &key_at(size_type idx) const noexcept { return buf_[idx].column; }
        [[nodiscard]] V &value_at(size_type idx) noexcept { return buf_[idx].value; }
        [[nodiscard]] const V &value_at(size_type idx) const noexcept { return buf_[idx].value; }

        [[nodiscard]] entry_type *data() noexcept { return buf_.data(); }
        [[nodiscard]] const entry_type *data() const noexcept { return buf_.data(); }

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
            assert(sz_ < N && "aos_buffer overflow: caller must ensure capacity");
            buf_[sz_++] = entry_type{col, v};
        }

        bool contains(I col) const noexcept
        {
            for (size_type i = sz_; i-- > 0;)
            {
                if (buf_[i].column == col)
                    return true;
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

        [[nodiscard]] std::vector<entry_type> flush_buffer()
        {
            std::vector<entry_type> chunk;
            chunk.reserve(sz_);
            for (size_type i = 0; i < sz_; ++i)
            {
                chunk.push_back(buf_[i]);
            }
            clear();
            return chunk;
        }

        void deduplicate(){
            std::array<entry_type, N> tmp;
            size_t out = 0;

            for(size_t i = sz_; i-- > 0;){
                bool seen = false;

                for(size_t j = 0; j < out; j++){
                    if(tmp[j].column == buf_[i].column){
                        seen = true;
                        break;
                    }
                }

                if(seen == false){
                    tmp[out++] = buf_[i];
                }
            }

            for(size_t i = 0; i < out; i++){
                buf_[i] = tmp[i];

            }

            sz_ = out;
        }

        V accumlate() const noexcept{
            deduplicate();
            
            V acc = traits::ValueTraits<V>::zero();

            for(size_t i = 0; i < sz_; i++){
                acc += buf_[i];
            }
            return acc;
        }

    private:
        std::array<entry_type, N> buf_{};
        size_type sz_{0};
    };
}