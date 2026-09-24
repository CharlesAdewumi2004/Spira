#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <spira/kernels/kernels.h>
#include <spira/matrix/layout/layout_tags.hpp>
#include <spira/matrix/matrix.hpp>
#include <spira/traits.hpp>

namespace spira
{
    namespace detail
    {
        // y[i] = row i of csr · x for the csr's n_rows rows. SoA storage with
        // uint32_t columns and float/double values goes through the dispatched
        // SIMD kernel; every other combination uses the scalar loop.
        template <class L, class I, class V>
        void csr_spmv(const csr_storage<L, I, V> &csr, const V *x, V *y)
        {
            constexpr bool soa = std::is_same_v<L, layout::tags::soa_tag>;
            constexpr bool simd = soa && std::is_same_v<I, uint32_t> &&
                                  (std::is_same_v<V, float> || std::is_same_v<V, double>);
            const std::size_t *row_start = csr.row_start.get();
            const std::size_t *row_len = csr.row_len.get();

            for (std::size_t i = 0; i < csr.n_rows; ++i)
            {
                const std::size_t beg = row_start[i];
                const std::size_t len = row_len[i];
                if constexpr (simd)
                {
                    if constexpr (std::is_same_v<V, float>)
                        y[i] = kernel::sparse_dot_float(csr.vals.get() + beg, csr.cols.get() + beg, x, len);
                    else
                        y[i] = kernel::sparse_dot_double(csr.vals.get() + beg, csr.cols.get() + beg, x, len);
                }
                else
                {
                    V acc = traits::ValueTraits<V>::zero();
                    for (std::size_t k = beg; k < beg + len; ++k)
                    {
                        if constexpr (soa)
                            acc += x[static_cast<std::size_t>(csr.cols.get()[k])] * csr.vals.get()[k];
                        else
                            acc += x[static_cast<std::size_t>(csr.pairs.get()[k].column)] * csr.pairs.get()[k].value;
                    }
                    y[i] = acc;
                }
            }
        }

        template <class M, class V>
        void check_spmv_args(const M &mat, const std::vector<V> &x, const std::vector<V> &y)
        {
            if (x.size() != mat.n_cols())
                throw std::invalid_argument("spmv: x size does not match matrix column count");
            if (y.size() != mat.n_rows())
                throw std::invalid_argument("spmv: y size does not match matrix row count");
            if (!mat.is_locked())
                throw std::logic_error("spmv: matrix must be locked");
        }
    } // namespace detail

    namespace serial::algorithms
    {
        /// y = mat · x. mat must be locked.
        template <class L, concepts::Indexable I, concepts::Valueable V, class BT, std::size_t BN, class S>
        inline void spmv(const spira::matrix<L, I, V, BT, BN, S> &mat,
                         const std::vector<V> &x, std::vector<V> &y)
        {
            spira::detail::check_spmv_args(mat, x, y);
            spira::detail::csr_spmv(*mat.csr(), x.data(), y.data());
        }
    } // namespace serial::algorithms

} // namespace spira
