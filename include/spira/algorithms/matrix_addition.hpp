#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{
    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    void addRows(const spira::row<Layout, I, V> &A, const spira::row<Layout, I, V> &B, spira::row<Layout, I, V> &out)
    {
        out.clear();
        A.flush();
        B.flush();

        auto itA = A.begin();
        auto itB = B.begin();
        const auto endA = A.end();
        const auto endB = B.end();

        while (itA != endA && itB != endB)
        {
            const auto &[a_col, a_val] = *itA;
            const auto &[b_col, b_val] = *itB;

            if (a_col == b_col)
            {
                V sum = a_val + b_val;
                if (!spira::traits::ValueTraits<V>::is_zero(sum))
                {
                    out.insert(a_col, sum);
                }
                ++itA;
                ++itB;
            }
            else if (a_col < b_col)
            {
                out.insert(a_col, a_val);
                ++itA;
            }
            else
            {
                out.insert(b_col, b_val);
                ++itB;
            }
        }

        while (itA != endA)
        {
            const auto &[a_col, a_val] = *itA;
            out.insert(a_col, a_val);
            ++itA;
        }

        while (itB != endB)
        {
            const auto &[b_col, b_val] = *itB;
            out.insert(b_col, b_val);
            ++itB;
        }
    }

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    spira::matrix<Layout, I, V> MatrixAddition(const spira::matrix<Layout, I, V> &A, const spira::matrix<Layout, I, V> &B)
    {
        if (A.shape() != B.shape())
        {
            throw std::invalid_argument("Matrices aren't the same size");
        }

        A.flush();
        B.flush();

        const auto [r, c] = A.shape();
        spira::matrix<Layout, I, V> out(r, c);

        out.set_mode(spira::mode::matrix_mode::insert_heavy);

        for (std::size_t i = 0; i < A.n_rows(); ++i)
        {
            const I ri = static_cast<I>(i);
            addRows(A.row_at(ri), B.row_at(ri), out.row_at_mut(ri));
        }

        return out;
    }

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    void MatrixAdditionInPlace(spira::matrix<Layout, I, V> &A, const spira::matrix<Layout, I, V> &B)
    {
        if (A.shape() != B.shape())
        {
            throw std::invalid_argument("Matrices aren't the same size");
        }

        A.flush();
        B.flush();

        A.set_mode(spira::mode::matrix_mode::insert_heavy);

        for (std::size_t i = 0; i < A.n_rows(); ++i)
        {
            const I ri = static_cast<I>(i);

            auto tmp = A.row_at(ri);
            tmp.clear();

            addRows(A.row_at(ri), B.row_at(ri), tmp);

            using std::swap;
            swap(A.row_at_mut(ri), tmp);
        }
    }

}
