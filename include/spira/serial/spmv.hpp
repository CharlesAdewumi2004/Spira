#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#include <spira/kernels/kernels.h>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/matrix.hpp>
#include <spira/traits.hpp>

namespace spira::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // Generic SpMV — works for any layout / index / value combination.
    //
    // Dispatch order:
    //   1. CSR flat-buffer scalar loop  (compact_preserve or compact_move)
    //   2. Per-row for_each_element     (no_compact fallback)
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP>
    inline void spmv(const spira::matrix<L, I, V, BT, BN, LP> &mat,
                     const std::vector<V> &x, std::vector<V> &y)
    {
        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "The size of the input vector x does not match the number of columns of the matrix");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument(
                "The size of the output vector y does not match the number of rows of the matrix");

        assert(mat.is_locked() && "spmv: input matrix must be locked");

        if (const auto *csr = mat.csr(); csr != nullptr)
        {
            // CSR flat-buffer path: O(nnz) with sequential memory access.
            const std::size_t *offsets = csr->offsets.get();
            const V *xp = x.data();
            const std::size_t nr = mat.n_rows();

            if constexpr (std::is_same_v<L, layout::tags::soa_tag>)
            {
                const I *cols = csr->cols.get();
                const V *vals = csr->vals.get();
                for (std::size_t i = 0; i < nr; ++i)
                {
                    V acc = traits::ValueTraits<V>::zero();
                    for (std::size_t k = offsets[i]; k < offsets[i + 1]; ++k)
                        acc += xp[static_cast<std::size_t>(cols[k])] * vals[k];
                    y[i] = acc;
                }
            }
            else // aos_tag: interleaved pairs
            {
                const auto *pairs = csr->pairs.get();
                for (std::size_t i = 0; i < nr; ++i)
                {
                    V acc = traits::ValueTraits<V>::zero();
                    for (std::size_t k = offsets[i]; k < offsets[i + 1]; ++k)
                        acc += xp[static_cast<std::size_t>(pairs[k].column)] * pairs[k].value;
                    y[i] = acc;
                }
            }
        }
        else
        {
            // Per-row fallback — used when no_compact is in effect.
            mat.for_each_row([&y, &x](const auto &row, I rowIndex)
                             {
            V acc = traits::ValueTraits<V>::zero();
            row.for_each_element(
                [&acc, &x](I const &col, V const &val) { acc += x[col] * val; });
            y[static_cast<std::size_t>(rowIndex)] = acc; });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / float
    //
    // More specialised than the generic template: picked by overload resolution
    // when L=soa_tag, I=uint32_t, V=float.
    //
    // CSR path → flat cols/vals passed directly to the SIMD kernel.
    // Fallback  → scalar accumulation via for_each_element.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN, config::lock_policy LP>
    inline void spmv(
        const spira::matrix<layout::tags::soa_tag, uint32_t, float, BT, BN, LP> &mat,
        const std::vector<float> &x, std::vector<float> &y)
    {
        using L = layout::tags::soa_tag;
        using I = uint32_t;
        using V = float;

        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "The size of the input vector x does not match the number of columns of the matrix");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument(
                "The size of the output vector y does not match the number of rows of the matrix");

        assert(mat.is_locked() && "spmv: input matrix must be locked");

        if (const auto *csr = mat.csr(); csr != nullptr)
        {
            const uint32_t *cols = csr->cols.get();
            const float *vals = csr->vals.get();
            const std::size_t *offsets = csr->offsets.get();
            const std::size_t nr = mat.n_rows();

            for (std::size_t i = 0; i < nr; ++i)
            {
                const std::size_t row_nnz = offsets[i + 1] - offsets[i];
                y[i] = kernel::sparse_dot_float(
                    vals + offsets[i], cols + offsets[i],
                    x.data(), row_nnz, x.size());
            }
        }
        else
        {
            // no_compact fallback — scalar accumulation.
            mat.for_each_row([&y, &x](const row<L, I, V, BT, BN> &r, I rowIndex)
                             {
            V acc = traits::ValueTraits<V>::zero();
            r.for_each_element([&acc, &x](const I col, const V val) {
                acc += x[static_cast<std::size_t>(col)] * val;
            });
            y[static_cast<std::size_t>(rowIndex)] = acc; });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / double
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN, config::lock_policy LP>
    inline void spmv(
        const spira::matrix<layout::tags::soa_tag, uint32_t, double, BT, BN, LP> &mat,
        const std::vector<double> &x, std::vector<double> &y)
    {
        using L = layout::tags::soa_tag;
        using I = uint32_t;
        using V = double;

        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "The size of the input vector x does not match the number of columns of the matrix");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument(
                "The size of the output vector y does not match the number of rows of the matrix");

        assert(mat.is_locked() && "spmv: input matrix must be locked");

        if (const auto *csr = mat.csr(); csr != nullptr)
        {
            const uint32_t *cols = csr->cols.get();
            const double *vals = csr->vals.get();
            const std::size_t *offsets = csr->offsets.get();
            const std::size_t nr = mat.n_rows();

            for (std::size_t i = 0; i < nr; ++i)
            {
                const std::size_t row_nnz = offsets[i + 1] - offsets[i];
                y[i] = kernel::sparse_dot_double(
                    vals + offsets[i], cols + offsets[i],
                    x.data(), row_nnz, x.size());
            }
        }
        else
        {
            // no_compact fallback — scalar accumulation.
            mat.for_each_row([&y, &x](const row<L, I, V, BT, BN> &r, I rowIndex)
                             {
            V acc = traits::ValueTraits<V>::zero();
            r.for_each_element([&acc, &x](const I col, const V val) {
                acc += x[static_cast<std::size_t>(col)] * val;
            });
            y[static_cast<std::size_t>(rowIndex)] = acc; });
        }
    }

} // namespace spira::algorithms
