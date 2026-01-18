#pragma once

#include <vector>

#include <spira/matrix/matrix.hpp>

namespace spira::algorithms
{

    template <class layout_policy>
    static layout_policy &tls_layout_tmp()
    {
        thread_local layout_policy tmp;
        return tmp;
    }

    template <class layout_policy>
    void merge(layout_policy  &slab, layout_policy const &chunk)
    {
        auto &tmp = tls_layout_tmp<layout_policy>();

        tmp.clear();
        tmp.reserve(slab.size() + chunk.size());

        auto slab_it = slab.begin();
        auto slab_end = slab.end();
        auto chunk_it = chunk.begin();
        auto chunk_end = chunk.end();

        while (slab_it != slab_end && chunk_it != chunk_end)
        {
            auto [slab_col, slab_val] = *slab_it;
            auto [chunk_col, chunk_val] = *chunk_it;

            if (slab_col < chunk_col)
            {
                tmp.push_back(slab_col, slab_val);
                ++slab_it;
            }
            if(chunk_col < slab_col)
            {
                tmp.push_back(chunk_col, chunk_val);
                ++chunk_it;
            }else{
                tmp.push_back(chunk_col, chunk_val);
                chunk_it++;
                slab_it++;
            }
        }

        for (; slab_it != slab_end; ++slab_it)
        {
            auto [c, v] = *slab_it;
            tmp.push_back(c, v);
        }
        for (; chunk_it != chunk_end; ++chunk_it)
        {
            auto [c, v] = *chunk_it;
            tmp.push_back(c, v);
        }

        slab.swap(tmp);
        tmp.clear();
    }

}