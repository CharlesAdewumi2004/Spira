#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>

#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/layout/element_pair.hpp>
#include <spira/traits.hpp>

namespace spira
{

    inline constexpr std::size_t csr_alignment = 64;

    namespace detail
    {

        struct csr_aligned_deleter
        {
            void operator()(void *p) const noexcept
            {
                ::operator delete(p, std::align_val_t{csr_alignment});
            }
        };

        template <class T>
        using csr_buf = std::unique_ptr<T, csr_aligned_deleter>;

        template <class T>
        csr_buf<T> alloc_csr_buf(std::size_t n)
        {
            if (n == 0)
                return {nullptr};
            return {
                static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{csr_alignment})), {}};
        }

    } // namespace detail

    // ─────────────────────────────────────────────────────────────────────────────
    // csr_storage<LayoutTag, I, V>
    //
    // The layout tag determines the flat memory arrangement of the locked CSR:
    //
    //   soa_tag — two separate flat arrays: cols[nnz] and vals[nnz].
    //             SIMD SpMV loads values and column indices independently
    //             with single aligned vector loads.
    //
    //   aos_tag — one interleaved array: elementPair<I,V> pairs[nnz].
    //             SIMD SpMV must gather/stride over interleaved data.
    //
    // Both variants share: offsets[n_rows+1] for row boundaries.
    // The layout policy only affects the locked (read) structure; open-mode
    // buffering is always a plain growable array of (col, val) pairs.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class I, class V>
    struct csr_storage; // primary template — specialised below

    // ── SoA CSR ─────────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_storage<layout::tags::soa_tag, I, V>
    {
        std::size_t n_rows{0};
        std::size_t nnz{0};

        std::unique_ptr<std::size_t[]> offsets;
        detail::csr_buf<I> cols;
        detail::csr_buf<V> vals;

        csr_storage() = default;

        csr_storage(std::size_t n_rows_, std::size_t nnz_)
            : n_rows{n_rows_}, nnz{nnz_},
              offsets{std::make_unique<std::size_t[]>(n_rows_ + 1)},
              cols{detail::alloc_csr_buf<I>(nnz_)},
              vals{detail::alloc_csr_buf<V>(nnz_)}
        {
        }

        csr_storage(const csr_storage &other)
            : n_rows{other.n_rows}, nnz{other.nnz},
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

        csr_storage(csr_storage &&) = default;
        csr_storage &operator=(csr_storage &&) = default;

        [[nodiscard]] bool is_built() const noexcept { return offsets != nullptr; }
    };

    // ── AoS CSR ─────────────────────────────────────────────────────────────────

    template <class I, class V>
    struct csr_storage<layout::tags::aos_tag, I, V>
    {
        std::size_t n_rows{0};
        std::size_t nnz{0};

        std::unique_ptr<std::size_t[]> offsets;           // [n_rows + 1]
        detail::csr_buf<layout::elementPair<I, V>> pairs; // [nnz], 64-byte aligned, interleaved

        csr_storage() = default;

        csr_storage(std::size_t n_rows_, std::size_t nnz_)
            : n_rows{n_rows_}, nnz{nnz_},
              offsets{std::make_unique<std::size_t[]>(n_rows_ + 1)},
              pairs{detail::alloc_csr_buf<layout::elementPair<I, V>>(nnz_)}
        {
        }

        csr_storage(const csr_storage &other)
            : n_rows{other.n_rows}, nnz{other.nnz},
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

    // ─────────────────────────────────────────────────────────────────────────────
    // csr_slice<LayoutTag, I, V>
    //
    // A non-owning view into one row's slice of a csr_storage.
    // Installed on each row by matrix::lock() via row::set_csr_slice().
    //
    // Provides: is_set(), reset(), binary_search(col), for_each(fn), accumulate().
    // The layout tag selects the underlying pointer type and access pattern.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class LayoutTag, class I, class V>
    struct csr_slice; // primary template — specialised below

    // ── SoA slice ───────────────────────────────────────────────────────────────

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

    // ── AoS slice ───────────────────────────────────────────────────────────────

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
