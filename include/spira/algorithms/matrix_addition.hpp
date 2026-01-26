#pragma once

#include <spira/spira.hpp>

namespace spira::algorithms
{

    template <class Layout, spira::concepts::Indexable I, spira::concepts::Valueable V>
    void addRows(const spira::row<Layout, I, V> &A, const spira::row<Layout, I, V> &B, spira::row<Layout, I, V> &out)
    {
        if(A.is_dirty()){
            A.flush();
        }
        if(B.is_dirty()){
            B.flush();
        }
        
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
        if (A.get_shape() != B.get_shape())
        {
            throw std::invalid_argument("Mactices aren't the same size");
        }

        spira::matrix<Layout, I, V> out(A.get_shape().first, A.get_shape().second);

        out.set_mode(spira::mode::matrix_mode::insert_heavy);

        for (size_t i = 0; i < A.n_rows(); i++)
        {
            I ri = static_cast<I>(i);
            addRows(A.getRowAt(ri), B.getRowAt(ri), out.getMutableRowAt(ri));
        }

        out.set_mode(spira::mode::matrix_mode::balanced);

        return out;
    }
}