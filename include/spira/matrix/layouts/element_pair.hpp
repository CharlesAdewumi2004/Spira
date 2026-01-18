#pragma once

#include <spira/concepts.hpp>

namespace spira::layout
{

    template <concepts::Indexable I, concepts::Valueable V>
    struct elementPair
    {
        I column;
        V value;
    };
}