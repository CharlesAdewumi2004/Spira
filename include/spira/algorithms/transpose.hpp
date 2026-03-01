#pragma once

#include <cassert>
#include <stdexcept>

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{
    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    spira::matrix<Layout, I, V> transpose(const spira::matrix<Layout, I, V>& mat)
    {
        assert(mat.is_locked() && "transpose: input matrix must be locked");

        auto [r, c] = mat.shape();
        spira::matrix<Layout, I, V> out(c, r);

        for (std::size_t i = 0; i < r; ++i)
        {
            I ri = static_cast<I>(i);
            const auto& row = mat.row_at(ri);

            row.for_each_element([&out, ri](I col, const V& val) {
                out.insert(col, ri, val);
            });
        }

        out.lock();
        return out;
    }

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void transpose_itself(spira::matrix<Layout, I, V>& mat)
    {
        auto [r, c] = mat.shape();
        if (r != c){
            throw std::logic_error("in-place transpose requires square matrix");
        }

        auto out = transpose(mat);
        mat.swap(out);
    }
} // namespace spira::algorithms
