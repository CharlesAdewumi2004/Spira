#pragma once

#include <cstddef>
#include <stdexcept>

#include <ankerl/unordered_dense.h>
#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // spgemm — C = A * B, parallel.
    //
    // A is m×k, B is k×n.  Both must be locked.
    // Each worker computes C[row_start..row_end) for its own partition of A,
    // reading from B concurrently (B is locked/read-only — no synchronisation
    // needed for reads).  Each worker writes only to its own partition of C
    // (disjoint row ranges).
    //
    // Output C uses A's thread count and has shape (A.n_rows(), B.n_cols()).
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>
    spgemm(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &A,
           parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &B)
    {
        if (A.n_cols() != B.n_rows())
            throw std::invalid_argument("spgemm: A.n_cols() must equal B.n_rows()");

        if (!A.is_locked())
            throw std::logic_error("spgemm: A must be locked");
        if (!B.is_locked())
            throw std::logic_error("spgemm: B must be locked");

        parallel_matrix<L, I, V, BT, BN, LP, IP, SN> C(
            A.n_rows(), B.n_cols(), A.n_threads());

        A.execute([&A, &B, &C](const auto &p_A, std::size_t t)
        {
            auto &p_C = C.partition_at(t);

            // Per-row accumulator — allocated once, cleared each row.
            ankerl::unordered_dense::map<I, V> acc;

            for (std::size_t i = 0; i < p_A.rows.size(); ++i)
            {
                acc.clear();

                p_A.rows[i].for_each_element([&B, &acc](I k, V a_ik)
                {
                    // B.row_at() is const and read-only — safe for concurrent access.
                    B.row_at(static_cast<std::size_t>(k)).for_each_element(
                        [&acc, a_ik](I j, V b_kj)
                        { acc[j] += a_ik * b_kj; });
                });

                for (auto &[j, v] : acc)
                    if (v != V{})
                        p_C.rows[i].insert(j, v);
            }
        });

        C.lock();
        return C;
    }

} // namespace spira::parallel::algorithms
