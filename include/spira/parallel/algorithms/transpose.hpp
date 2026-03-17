#pragma once

#include <cassert>
#include <cstddef>

#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // transpose — returns a new parallel_matrix with rows and columns swapped.
    //
    // NOTE: transpose cannot be parallelised without synchronisation on the output
    // because each input row (i, j, v) maps to output row j, which belongs to an
    // arbitrary output partition. The scan is therefore performed serially on the
    // main thread; lock() of the output runs in parallel as usual.
    //
    // mat must be locked.
    // Output uses the same thread count as the input.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>
    transpose(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat)
    {
        assert(mat.is_locked() && "transpose: matrix must be locked");

        parallel_matrix<L, I, V, BT, BN, LP, IP, SN> out(
            mat.n_cols(), mat.n_rows(), mat.n_threads());

        // Serial scan: each (row i, col j, val v) → output (row j, col i, val v).
        // insert() is single-threaded by design; no pool involvement.
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
