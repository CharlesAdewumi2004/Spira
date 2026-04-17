#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // accumulate — sum a single row (delegates to matrix method, no pool overhead)
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    V accumulate(const parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat, std::size_t row)
    {
        if (row >= mat.n_rows())
            throw std::out_of_range("accumulate: row index out of range");
        return mat.accumulate(row);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // accumulate — parallel sum of every row.
    // Each worker fills y[row_start..row_end) for its own partition.
    // Output ranges are disjoint — no synchronisation needed.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    std::vector<V> accumulate(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &mat)
    {
        std::vector<V> result(mat.n_rows());

        mat.execute([&result](const auto &p, std::size_t)
        {
            for (std::size_t i = 0; i < p.rows.size(); ++i)
                result[p.row_start + i] = p.rows[i].accumulate();
        });

        return result;
    }

} // namespace spira::parallel::algorithms
