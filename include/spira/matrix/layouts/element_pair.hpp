#pragma once

#include <spira/concepts.hpp>

namespace spira::layout
{

    template <concepts::Indexable I, concepts::Valueable V>
    struct elementPair
    {
        I column{};
        V value{};

        constexpr I &first_ref() noexcept { return column; }
        constexpr I const &first_ref() const noexcept { return column; }

        constexpr V &second_ref() noexcept { return value; }
        constexpr V const &second_ref() const noexcept { return value; }
    };

}