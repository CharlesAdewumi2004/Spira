#pragma once

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{
    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    spira::matrix<Layout, I, V> transpose(const spira::matrix<Layout, I, V>& mat)
    {
        auto [r, c] = mat.get_shape();
        spira::matrix<Layout, I, V> out(c, r);

        auto original_mode = mat.mode();
        out.set_mode(spira::mode::matrix_mode::insert_heavy);

        for (std::size_t i = 0; i < r; ++i)
        {
            I ri = static_cast<I>(i);
            const auto& row = mat.getRowAt(ri);

            row.for_each_element([&out, ri](I col, const V& val) {
                out.insert(col, ri, val);
            });
        }

        out.set_mode(original_mode);
        return out;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void transpose_itself(spira::matrix<Layout, I, V>& mat)
    {
        auto [r, c] = mat.get_shape();
        if (r != c)
            throw std::logic_error("in-place transpose requires square matrix");

        auto out = transpose(mat);
        mat.matrix_swap(out);
    }
}
