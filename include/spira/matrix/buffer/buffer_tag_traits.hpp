#pragma once

#include <spira/matrix/layouts/layout_tags.hpp>
#include <spira/matrix/buffer/buffer_layout_impls/aos_buffer.hpp>
#include <spira/matrix/buffer/buffer_layout_impls/soa_buffer.hpp>

namespace spira::buffer::traits{

    template<class Tag, class I, class V, size_t N>
    struct traits;

    template<class I, class V, size_t N>
    struct traits<layout::tags::aos_tag, I, V, N>{
        using type = spira::buffer::impls::aos_buffer<I, V, N>;
    };

    template<class I, class V, size_t N>
    struct traits<layout::tags::soa_tag, I, V, N>{
        using type = spira::buffer::impls::soa_buffer<I, V, N>;
    };

    template<class Tag ,class I, class V, size_t N>
    using traits_of_type = traits<Tag, I, V, N>::type;


}