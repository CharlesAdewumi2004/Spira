#pragma once

namespace spira::buffer::tags{
    template<class Layour_Policy>
    struct array_buffer
    {
        using layour_policy = Layour_Policy;
    };

    struct  hash_map_buffer
    {
    };   
}