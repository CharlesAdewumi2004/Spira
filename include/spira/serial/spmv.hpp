#pragma once

#include <cstdint>
#include <vector>

#include <spira/kernels/kernels.h>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/matrix.hpp>
#include <spira/traits.hpp>

namespace spira::serial::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // Generic SpMV — works for any layout / index / value combination.
    //
    // The CSR is always built once the matrix is locked.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN>
    inline void spmv(const spira::matrix<L, I, V, BT, BN> &mat,
                     const std::vector<V> &x, std::vector<V> &y)
    {
        if (x.size() != mat.n_cols())
            throw std::invalid_argument(
                "The size of the input vector x does not match the number of columns of the matrix");
        if (y.size() != mat.n_rows())
            throw std::invalid_argument(
                "The size of the output vector y does not match the number of rows of the matrix");

        if (!mat.is_locked())
            throw std::logic_error("spmv: matrix must be locked");

        // The CSR is always built once the matrix is locked.
        const auto *csr = mat.csr();
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

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / float
    //
    // More specialised than the generic template: picked by overload resolution
    // when L=soa_tag, I=uint32_t, V=float.
    //
    // CSR path → flat cols/vals passed directly to the SIMD kernel.
    // Fallback  → scalar accumulation via for_each_element.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN>
    inline void spmv(
        const spira::matrix<layout::tags::soa_tag, uint32_t, float, BT, BN> &mat,
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

        if (!mat.is_locked())
            throw std::logic_error("spmv: matrix must be locked");

        // The CSR is always built once the matrix is locked.
        const auto *csr = mat.csr();
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

    // ─────────────────────────────────────────────────────────────────────────────
    // SIMD overload — soa_tag / uint32_t / double
    // ─────────────────────────────────────────────────────────────────────────────

    template <class BT, std::size_t BN>
    inline void spmv(
        const spira::matrix<layout::tags::soa_tag, uint32_t, double, BT, BN> &mat,
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

        if (!mat.is_locked())
            throw std::logic_error("spmv: matrix must be locked");

        // The CSR is always built once the matrix is locked.
        const auto *csr = mat.csr();
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

} // namespace spira::algorithms
