#pragma once

#include <cstddef>
#include <vector>

#include <spira/parallel/parallel_matrix.hpp>
#include <spira/serial/spmv.hpp>

namespace spira::parallel::algorithms
{

    /// y = mat · x. Each worker computes y[row_start, row_end) for its own
    /// partition; the output ranges are disjoint, so no synchronisation is
    /// needed. mat must be locked.
    template <class L, concepts::Indexable I, concepts::Valueable V, class BT,
              std::size_t BN, config::insert_policy IP, std::size_t SN>
    inline void spmv(parallel_matrix<L, I, V, BT, BN, IP, SN> &mat,
                     const std::vector<V> &x, std::vector<V> &y)
    {
        spira::detail::check_spmv_args(mat, x, y);
        mat.execute([&x, &y](const auto &p, std::size_t)
        { spira::detail::csr_spmv(p.csr, x.data(), y.data() + p.row_start); });
    }

} // namespace spira::parallel::algorithms
