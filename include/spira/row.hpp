#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include "element.hpp"

namespace spira::row {
    template<concepts::Valueable V, concepts::Indexable I>
    class row {
        private:
            std::vector<element::element<V, I>> _row;
        public:
            explicit row(std::vector<element::element<V, I>>&& row) {
                _row.reserve(row.size());
                _row = std::move(row);
                std::sort(_row.begin(), _row.end(), [](element::element<V, I> a, element::element<V, I> b) {
                    return a.col < b.col;
                });
            }
            void printRow() const {
                for (const auto& element : _row) {
                    std::cout <<"(" << element.value << ", " << element.col << ") ";
                }
            }
            ~row() = default;
};
}