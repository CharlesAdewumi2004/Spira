#pragma once

#include "layout_tags.hpp"
#include "aos.hpp"
#include "soa.hpp"

namespace spira::layout::of
{

    template <class Tag, class I, class V>
    struct storage_of;

    template <class I, class V>
    struct storage_of<tags::aos_tag, I, V>
    {
        using type = aos<I, V>;
    };

    template <class I, class V>
    struct storage_of<tags::soa_tag, I, V>
    {
        using type = soa<I, V>;
    };

    template <class Tag, class I, class V>
    using storage_of_t = storage_of<Tag, I, V>::type;

}
