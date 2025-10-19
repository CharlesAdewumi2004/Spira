#pragma once
#include "concepts.hpp"

namespace spira::element {
    template <concepts::Valueable V, concepts::Indexable I>
    struct element {
        V value;
        I col;
    };
}