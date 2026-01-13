#pragma once

#include <array>
#include <cstddef>

namespace spira::buffer::impls
{

    template<class I, class V, size_t N>
    class soa_buffer
    {
    public:
        
    private:
        std::array<I, N> colArray;
        std::array<V, N> valArray;
        size_t sz = 0; 
        size_t n = N;
    };

}