#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spira/traits.hpp>
#include <spira/parallel/parallel_matrix.hpp>

namespace spira::parallel::algorithms
{

    // ─────────────────────────────────────────────────────────────────────────────
    // MatrixAddition — C = A + B, fully parallel.
    //
    // A and B must be locked and have identical shape and thread count so that
    // partition t of A and partition t of B contain exactly the same global rows.
    // Each worker handles its own partition independently (disjoint row ranges).
    //
    // Returns a locked parallel_matrix with the same dimensions and thread count.
    // ─────────────────────────────────────────────────────────────────────────────

    template <class L, concepts::Indexable I, concepts::Valueable V,
              class BT, std::size_t BN, config::lock_policy LP,
              config::insert_policy IP, std::size_t SN>
    parallel_matrix<L, I, V, BT, BN, LP, IP, SN>
    MatrixAddition(parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &A,
                   parallel_matrix<L, I, V, BT, BN, LP, IP, SN> &B)
    {
        if (A.shape() != B.shape())
            throw std::invalid_argument("MatrixAddition: matrices must have the same shape");
        if (A.n_threads() != B.n_threads())
            throw std::invalid_argument("MatrixAddition: matrices must have the same thread count");

        if (!A.is_locked())
            throw std::logic_error("MatrixAddition: A must be locked");
        if (!B.is_locked())
            throw std::logic_error("MatrixAddition: B must be locked");

        parallel_matrix<L, I, V, BT, BN, LP, IP, SN> C(
            A.n_rows(), A.n_cols(), A.n_threads());

        A.execute([&B, &C](const auto &p_A, std::size_t t)
        {
            const auto &p_B = B.partition_at(t);
            auto       &p_C = C.partition_at(t);

            for (std::size_t i = 0; i < p_A.rows.size(); ++i)
            {
                // Two-pointer merge over sorted CSR entries of A row and B row.
                std::vector<std::pair<I, V>> a_entries, b_entries;
                a_entries.reserve(p_A.rows[i].size());
                b_entries.reserve(p_B.rows[i].size());

                p_A.rows[i].for_each_element([&a_entries](I col, V val)
                { a_entries.push_back({col, val}); });
                p_B.rows[i].for_each_element([&b_entries](I col, V val)
                { b_entries.push_back({col, val}); });

                std::size_t ai = 0, bi = 0;
                while (ai < a_entries.size() && bi < b_entries.size())
                {
                    const auto ac = a_entries[ai].first;
                    const auto bc = b_entries[bi].first;
                    if (ac == bc)
                    {
                        V sum = a_entries[ai].second + b_entries[bi].second;
                        if (!traits::ValueTraits<V>::is_zero(sum))
                            p_C.rows[i].insert(ac, sum);
                        ++ai; ++bi;
                    }
                    else if (ac < bc)
                    {
                        p_C.rows[i].insert(ac, a_entries[ai].second);
                        ++ai;
                    }
                    else
                    {
                        p_C.rows[i].insert(bc, b_entries[bi].second);
                        ++bi;
                    }
                }
                while (ai < a_entries.size())
                {
                    p_C.rows[i].insert(a_entries[ai].first, a_entries[ai].second);
                    ++ai;
                }
                while (bi < b_entries.size())
                {
                    p_C.rows[i].insert(b_entries[bi].first, b_entries[bi].second);
                    ++bi;
                }
            }
        });

        C.lock();
        return C;
    }

} // namespace spira::parallel::algorithms
