#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/layout/element_pair.hpp>
#include <spira/matrix/storage/csr_storage_detail.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ── AoS CSR storage ─────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_storage<layout::tags::aos_tag, I, V> : detail::csr_row_table
    {
        detail::csr_buf<layout::elementPair<I, V>> pairs; // [capacity], 64-byte aligned

        csr_storage() = default;

        // Row table for n_rows_ rows plus a data array of cap slots, all empty.
        csr_storage(std::size_t n_rows_, std::size_t cap)
            : detail::csr_row_table(n_rows_, cap),
              pairs{detail::alloc_csr_buf<layout::elementPair<I, V>>(cap)}
        {
        }

        // Copies the assigned slots [0, end); the free tail is not carried over.
        csr_storage(const csr_storage &other)
            : detail::csr_row_table(other),
              pairs{detail::alloc_csr_buf<layout::elementPair<I, V>>(other.end)}
        {
            if (end > 0)
                std::copy_n(other.pairs.get(), end, pairs.get());
        }

        csr_storage &operator=(const csr_storage &other)
        {
            if (this != &other)
            {
                csr_storage tmp(other);
                *this = std::move(tmp);
            }
            return *this;
        }

        csr_storage(csr_storage &&) = default;
        csr_storage &operator=(csr_storage &&) = default;

        [[nodiscard]] csr_slice<layout::tags::aos_tag, I, V> slice(std::size_t r) const noexcept
        {
            return {pairs.get() + row_start[r], row_len[r]};
        }

        // Slot access, the same for both layouts.
        [[nodiscard]] I col(std::size_t k) const noexcept { return pairs.get()[k].column; }
        [[nodiscard]] V val(std::size_t k) const noexcept { return pairs.get()[k].value; }
        void set(std::size_t k, I c, const V &v) noexcept { pairs.get()[k] = {c, v}; }
        // Copy n slots starting at src slot s into this storage at slot d.
        void copy_from(std::size_t d, const csr_storage &src, std::size_t s, std::size_t n) noexcept
        {
            std::copy_n(src.pairs.get() + s, n, pairs.get() + d);
        }
    };

    // ── AoS CSR slice ────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_slice<layout::tags::aos_tag, I, V>
    {
        const layout::elementPair<I, V> *pairs{nullptr};
        std::size_t nnz{0};

        [[nodiscard]] bool is_set() const noexcept { return pairs != nullptr; }

        [[nodiscard]] const V *binary_search(I col) const noexcept
        {
            if (!pairs || nnz == 0)
                return nullptr;
            std::size_t lo = 0, hi = nnz;
            while (lo < hi)
            {
                const std::size_t mid = lo + (hi - lo) / 2;
                if (pairs[mid].column < col)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < nnz && pairs[lo].column == col)
                return &pairs[lo].value;
            return nullptr;
        }

        template <class Fn>
        void for_each(Fn &&f) const
        {
            for (std::size_t k = 0; k < nnz; ++k)
                std::forward<Fn>(f)(pairs[k].column, pairs[k].value);
        }

        [[nodiscard]] V accumulate() const noexcept
        {
            V acc = traits::ValueTraits<V>::zero();
            for (std::size_t k = 0; k < nnz; ++k)
                acc += pairs[k].value;
            return acc;
        }
    };

} // namespace spira
