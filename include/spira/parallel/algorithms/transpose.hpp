#pragma once

#include <cstddef>
#include <stdexcept>

#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────
    // transpose — returns a new parallel_matrix with rows and columns swapped.
    //
    // Fully parallel fill: each output partition is populated independently by
    // its owning worker thread with no locks or shared state.
    //
    // Each worker scans all input partitions in ascending global row order.
    // For every nonzero (row=i, col=j, val) it checks whether j falls inside
    // its output row range [rs, re). If so, it inserts into output row j−rs at
    // column i with value val.
    //
    // Two key properties follow from this scan order:
    //
    //   1. No synchronisation — workers write to disjoint output row ranges.
    //
    //   2. Naturally sorted output — input rows are visited with strictly
    //      increasing i, so column indices written into each output row also
    //      increase monotonically. lock() sort is therefore a no-op pass.
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

        const std::size_t n_in = mat.n_threads();

        parallel_matrix<L, I, V, BT, BN, LP, IP, SN> out(
            mat.n_cols(), mat.n_rows(), mat.n_threads());

        out.execute([&mat, n_in](auto &p_out, std::size_t)
        {
            const std::size_t rs = p_out.row_start;
            const std::size_t re = p_out.row_end;

            for (std::size_t s = 0; s < n_in; ++s)
            {
                const auto &p_in = mat.partition_at(s);
                for (std::size_t i = 0; i < p_in.rows.size(); ++i)
                {
                    const I global_row = static_cast<I>(p_in.row_start + i);
                    p_in.rows[i].for_each_element([&](I col, V val)
                    {
                        const auto col_sz = static_cast<std::size_t>(col);
                        if (col_sz >= rs && col_sz < re)
                            p_out.rows[col_sz - rs].insert(global_row, val);
                    });
                }
            }
        });

        out.lock();
        return out;
    }

} // namespace spira::parallel::algorithms
