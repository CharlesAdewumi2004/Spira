#pragma once

#include "layouts/layout_of.hpp"
#include "concepts.hpp"

namespace spira::row {
    template<class Layout,concepts::Indexable I, concepts::Valueable V>
    class row;

    template<concepts::Indexable I, concepts::Valueable V>
    class row<layout::tags::aos_tag, I, V>{
        using row_layout = layout::of::storage_of_t<layout::tags::aos_tag, I, V>;

    public:

    private:
        row_layout _row;


    };

    template<concepts::Indexable I, concepts::Valueable V>
    class row<layout::tags::soa_tag, I, V> {
        using row_layout = layout::of::storage_of_t<layout::tags::soa_tag, I, V>;
        private:
            row_layout _row;

    };
}
