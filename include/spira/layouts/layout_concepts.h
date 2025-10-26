#pragma once

#include <concepts>
#include <type_traits>

#include "aos.hpp"
#include "soa.hpp"

namespace spira::layout::concepts {
    template<class T>
    inline constexpr bool is_layout_tag_v =
        std::is_same_v<std::remove_cv_t<T>, aos> ||
        std::is_same_v<std::remove_cv_t<T>, soa> ;

    template<class T>
    concept layout_tag = is_layout_tag_v<T>;
}
