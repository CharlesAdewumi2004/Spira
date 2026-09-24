#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/storage/csr_storage_detail.hpp>
#include <spira/traits.hpp>

namespace spira
{

    // ── SoA CSR storage ─────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_storage<layout::tags::soa_tag, I, V> : detail::csr_row_table
    {
        detail::csr_buf<I> cols; // [capacity], 64-byte aligned
        detail::csr_buf<V> vals; // [capacity], 64-byte aligned

        csr_storage() = default;

        // Row table for n_rows_ rows plus data arrays of cap slots, all empty.
        csr_storage(std::size_t n_rows_, std::size_t cap)
            : detail::csr_row_table(n_rows_, cap),
              cols{detail::alloc_csr_buf<I>(cap)},
              vals{detail::alloc_csr_buf<V>(cap)}
        {
        }

        // Copies the assigned slots [0, end); the free tail is not carried over.
        csr_storage(const csr_storage &other)
            : detail::csr_row_table(other),
              cols{detail::alloc_csr_buf<I>(other.end)},
              vals{detail::alloc_csr_buf<V>(other.end)}
        {
            if (end > 0)
            {
                std::copy_n(other.cols.get(), end, cols.get());
                std::copy_n(other.vals.get(), end, vals.get());
            }
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

        [[nodiscard]] csr_slice<layout::tags::soa_tag, I, V> slice(std::size_t r) const noexcept
        {
            return {cols.get() + row_start[r], vals.get() + row_start[r], row_len[r]};
        }

        // Slot access, the same for both layouts.
        [[nodiscard]] I col(std::size_t k) const noexcept { return cols.get()[k]; }
        [[nodiscard]] V val(std::size_t k) const noexcept { return vals.get()[k]; }
        void set(std::size_t k, I c, const V &v) noexcept
        {
            cols.get()[k] = c;
            vals.get()[k] = v;
        }
        // Copy n slots starting at src slot s into this storage at slot d.
        void copy_from(std::size_t d, const csr_storage &src, std::size_t s, std::size_t n) noexcept
        {
            std::copy_n(src.cols.get() + s, n, cols.get() + d);
            std::copy_n(src.vals.get() + s, n, vals.get() + d);
        }
    };

    // ── SoA CSR slice ────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_slice<layout::tags::soa_tag, I, V>
    {
        const I *cols{nullptr};
        const V *vals{nullptr};
        std::size_t nnz{0};

        [[nodiscard]] bool is_set() const noexcept { return cols != nullptr; }

        [[nodiscard]] const V *binary_search(I col) const noexcept
        {
            if (!cols || nnz == 0)
                return nullptr;
            const I *lo = cols, *hi = cols + nnz;
            const I *it = std::lower_bound(lo, hi, col);
            if (it != hi && *it == col)
                return vals + (it - cols);
            return nullptr;
        }

        template <class Fn>
        void for_each(Fn &&f) const
        {
            for (std::size_t k = 0; k < nnz; ++k)
                std::forward<Fn>(f)(cols[k], vals[k]);
        }

        [[nodiscard]] V accumulate() const noexcept
        {
            V acc = traits::ValueTraits<V>::zero();
            for (std::size_t k = 0; k < nnz; ++k)
                acc += vals[k];
            return acc;
        }
    };

} // namespace spira
