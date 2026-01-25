#pragma once

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{
    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    spira::matrix<Layout, I, V> transpose(const spira::matrix<Layout, I, V> &mat)
    {
        auto [r, c] = mat.get_shape();
        spira::matrix<Layout, I, V> out(c, r);

        for (I i = 0; i < static_cast<I>(r); ++i)
        {
            const auto &row = mat.getRowAt(i);
            row.for_each_element([&out, i](I col, const V &val){ out.insert(col, i, val); });
        }

        out.flush();
        return out;
    }

}