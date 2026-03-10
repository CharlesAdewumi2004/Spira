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
    struct csr_storage<layout::tags::aos_tag, I, V>
    {
        std::size_t n_rows{0};
        std::size_t nnz{0};
        std::size_t capacity{0}; // allocated element count in pairs (>= nnz)

        std::unique_ptr<std::size_t[]> offsets;           // [n_rows + 1]
        detail::csr_buf<layout::elementPair<I, V>> pairs; // [capacity], 64-byte aligned

        csr_storage() = default;

        csr_storage(std::size_t n_rows_, std::size_t nnz_)
            : csr_storage(n_rows_, nnz_, nnz_)
        {
        }

        csr_storage(std::size_t n_rows_, std::size_t nnz_, std::size_t cap)
            : n_rows{n_rows_}, nnz{nnz_}, capacity{cap},
              offsets{std::make_unique<std::size_t[]>(n_rows_ + 1)},
              pairs{detail::alloc_csr_buf<layout::elementPair<I, V>>(cap)}
        {
        }

        csr_storage(const csr_storage &other)
            : n_rows{other.n_rows}, nnz{other.nnz}, capacity{other.nnz},
              offsets{other.offsets ? std::make_unique<std::size_t[]>(other.n_rows + 1) : nullptr},
              pairs{detail::alloc_csr_buf<layout::elementPair<I, V>>(other.nnz)}
        {
            if (offsets)
                std::copy_n(other.offsets.get(), n_rows + 1, offsets.get());
            if (nnz > 0)
                std::copy_n(other.pairs.get(), nnz, pairs.get());
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

        [[nodiscard]] bool is_built() const noexcept { return offsets != nullptr; }
    };

    // ── AoS CSR slice ────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_slice<layout::tags::aos_tag, I, V>
    {
        const layout::elementPair<I, V> *pairs{nullptr};
        std::size_t nnz{0};

        [[nodiscard]] bool is_set() const noexcept { return pairs != nullptr; }

        void reset() noexcept
        {
            pairs = nullptr;
            nnz = 0;
        }

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
