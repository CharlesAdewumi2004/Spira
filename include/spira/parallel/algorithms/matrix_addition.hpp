#pragma once

#include <cstddef>
#include <stdexcept>

#include <spira/parallel/parallel_matrix.hpp>
#include <spira/serial/matrix_addition.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // MatrixAddition — C = A + B, fully parallel.
    //
    // A and B must be locked and have identical shape and thread count so that
    // partition t of A and partition t of B contain exactly the same global rows.
    // Each worker merges its own partition's rows with serial::addRows
    // (disjoint row ranges, so no synchronisation).
    //
    // Returns a locked parallel_matrix with the same dimensions and thread count.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN,
              config::insert_policy IP, std::size_t SN, class S>
    parallel_matrix<L, I, V, BT, BN, IP, SN, S>
    MatrixAddition(parallel_matrix<L, I, V, BT, BN, IP, SN, S> &A,
                   parallel_matrix<L, I, V, BT, BN, IP, SN, S> &B)
    {
        if (A.shape() != B.shape())
            throw std::invalid_argument("MatrixAddition: matrices must have the same shape");
        if (A.n_threads() != B.n_threads())
            throw std::invalid_argument("MatrixAddition: matrices must have the same thread count");

        if (!A.is_locked())
            throw std::logic_error("MatrixAddition: A must be locked");
        if (!B.is_locked())
            throw std::logic_error("MatrixAddition: B must be locked");

        parallel_matrix<L, I, V, BT, BN, IP, SN, S> C(
            A.n_rows(), A.n_cols(), A.n_threads());

        A.execute([&B, &C](const auto &p_A, std::size_t t)
        {
            const auto &p_B = B.partition_at(t);
            auto       &p_C = C.partition_at(t);

            for (std::size_t i = 0; i < p_A.rows.size(); ++i)
                serial::algorithms::addRows(p_A.rows[i], p_B.rows[i], p_C.writable_row(i));
        });

        C.lock();
        return C;
    }

} // namespace spira::parallel::algorithms
