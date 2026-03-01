#pragma once

#include <cassert>
#include <stdexcept>

#include <ankerl/unordered_dense.h>
#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    spira::matrix<Layout, I, V> spgemm(const spira::matrix<Layout, I, V> &A, const spira::matrix<Layout, I, V> &B)
    {
        if (A.n_cols() != B.n_rows())
        {
            throw std::invalid_argument("A.cols must equal B.rows");
        }

        assert(A.is_locked() && "spgemm: A must be locked");
        assert(B.is_locked() && "spgemm: B must be locked");

        spira::matrix<Layout, I, V> C(A.n_rows(), B.n_cols());

        ankerl::unordered_dense::map<I, V> acc;

        for (I i = 0; i < static_cast<I>(A.n_rows()); ++i)
        {
            acc.clear();

            const auto &arow = A.row_at(i);

            arow.for_each_element([&](I k, const V &a_ik) {
                const auto &brow = B.row_at(k);

                brow.for_each_element([&](I j, const V &b_kj) {
                    acc[j] += a_ik * b_kj;
                });
            });

            for (auto &[j, v] : acc)
            {
                if (v != V{})
                {
                    C.insert(i, j, v);
                }
            }
        }

        C.lock();
        return C;
    }

} // namespace spira::algorithms
