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
    struct csr_storage<layout::tags::soa_tag, I, V>
    {
        std::size_t n_rows{0};
        std::size_t nnz{0};
        std::size_t capacity{0}; // allocated element count in cols/vals (>= nnz)

        std::unique_ptr<std::size_t[]> offsets;
        detail::csr_buf<I> cols;
        detail::csr_buf<V> vals;

        csr_storage() = default;

        // Allocate for exactly nnz_ elements.
        csr_storage(std::size_t n_rows_, std::size_t nnz_)
            : csr_storage(n_rows_, nnz_, nnz_)
        {
        }

        // Allocate for cap elements but record only nnz_ actual entries.
        // cap >= nnz_ allows merge_csr to reuse the allocation when it still fits.
        csr_storage(std::size_t n_rows_, std::size_t nnz_, std::size_t cap)
            : n_rows{n_rows_}, nnz{nnz_}, capacity{cap},
              offsets{std::make_unique<std::size_t[]>(n_rows_ + 1)},
              cols{detail::alloc_csr_buf<I>(cap)},
              vals{detail::alloc_csr_buf<V>(cap)}
        {
        }

        // Copy produces a tight copy (capacity == nnz); no excess is carried over.
        csr_storage(const csr_storage &other)
            : n_rows{other.n_rows}, nnz{other.nnz}, capacity{other.nnz},
              offsets{other.offsets ? std::make_unique<std::size_t[]>(other.n_rows + 1) : nullptr},
              cols{detail::alloc_csr_buf<I>(other.nnz)},
              vals{detail::alloc_csr_buf<V>(other.nnz)}
        {
            if (offsets)
                std::copy_n(other.offsets.get(), n_rows + 1, offsets.get());
            if (nnz > 0)
            {
                std::copy_n(other.cols.get(), nnz, cols.get());
                std::copy_n(other.vals.get(), nnz, vals.get());
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

        // Move preserves capacity — the reuse check in merge_csr depends on it.
        csr_storage(csr_storage &&) = default;
        csr_storage &operator=(csr_storage &&) = default;

        [[nodiscard]] bool is_built() const noexcept { return offsets != nullptr; }
    };

    // ── SoA CSR slice ────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_slice<layout::tags::soa_tag, I, V>
    {
        const I *cols{nullptr};
        const V *vals{nullptr};
        std::size_t nnz{0};

        [[nodiscard]] bool is_set() const noexcept { return cols != nullptr; }

        void reset() noexcept
        {
            cols = nullptr;
            vals = nullptr;
            nnz = 0;
        }

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
