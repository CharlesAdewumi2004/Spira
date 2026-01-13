#pragma once

#include <array>
#include <cstddef>

namespace spira::buffer::impls
{

    template <class I, class V>
    struct bufferElementPair
    {
        I column;
        V value;
    };

    template<class I, class V, size_t N>
    class aos_buffer
    {
    public:
        
    private:
        std::array<bufferElementPair<I, V>, N> buf;
        size_t sz = 0; 
        size_t n = N;
    };

}