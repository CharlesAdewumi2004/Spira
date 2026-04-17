#pragma once
#include <concepts>

namespace spira::layout::tags
{
    struct aos_tag
    {
    };
    struct soa_tag
    {
    };
}

namespace spira::layout
{
    template <class T>
    concept ValidLayoutTag = std::same_as<T, tags::soa_tag> || std::same_as<T, tags::aos_tag>;
}
