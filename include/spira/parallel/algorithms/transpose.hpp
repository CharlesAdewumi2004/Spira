#pragma once

#include <cstddef>
#include <stdexcept>

#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────
    // transpose — returns a new parallel_matrix with rows and columns swapped.
    //
    // The fill scan is serial (scattered writes to arbitrary output rows prevent
    // simple parallelisation without synchronisation). lock() of the output
    // runs in parallel as usual, sorting each partition independently.
    //
    // mat must be locked.
    // ─────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>
    transpose(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat)
    {
        if (!mat.is_locked())
            throw std::logic_error("transpose: matrix must be locked");

        parallel_matrix<L, I, V, BT, BN, LP, IP, SN> out(
            mat.n_cols(), mat.n_rows(), mat.n_threads());

        for (std::size_t t = 0; t < mat.n_threads(); ++t)
        {
            const auto &p = mat.partition_at(t);
            for (std::size_t i = 0; i < p.rows.size(); ++i)
            {
                const I global_row = static_cast<I>(p.row_start + i);
                p.rows[i].for_each_element([&out, global_row](I col, V val)
                { out.insert(static_cast<std::size_t>(col), global_row, val); });
            }
        }

        out.lock();
        return out;
    }

} // namespace spira::parallel::algorithms
