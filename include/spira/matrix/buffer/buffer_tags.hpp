#pragma once

namespace spira::buffer::tags
{
    template <class LayoutPolicy>
    struct array_buffer
    {
        using layout_policy = LayoutPolicy;
    };

    struct hash_map_buffer
    {
    };
}