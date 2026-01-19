#pragma once

#include <cstddef>

#include <spira/matrix/layouts/layout_tags.hpp>
#include <spira/matrix/buffer/buffer_tags.hpp>
#include <spira/matrix/buffer/buffer_layout_impls/aos_array_buffer.hpp>
#include <spira/matrix/buffer/buffer_layout_impls/soa_array_buffer.hpp>
#include <spira/matrix/buffer/buffer_layout_impls/hash_map_buffer.hpp>

namespace spira::buffer::traits
{

    template <class Tag, class I, class V, std::size_t N>
    struct traits;

    template <class I, class V, std::size_t N>
    struct traits<buffer::tags::array_buffer<layout::tags::aos_tag>, I, V, N>
    {
        using type = spira::buffer::impls::aos_array_buffer<I, V, N>;
    };

    template <class I, class V, std::size_t N>
    struct traits<buffer::tags::array_buffer<layout::tags::soa_tag>, I, V, N>
    {
        using type = spira::buffer::impls::soa_array_buffer<I, V, N>;
    };

    template <class I, class V, std::size_t N>
    struct traits<buffer::tags::hash_map_buffer, I, V, N>
    {
        using type = spira::buffer::impls::hash_map_buffer<I, V>;
    };

    template <class Tag, class I, class V, std::size_t N>
    using traits_of_type = typename traits<Tag, I, V, N>::type;

}
