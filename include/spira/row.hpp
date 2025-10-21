#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include "element.hpp"

namespace spira {
    template<concepts::Valueable V, concepts::Indexable I, I colSize>
    class row {
        private:
            std::shared_ptr<const std::vector<element::element<V,I>>> _row;
        public:
            row() noexcept = default;
            bool setRow(std::vector<element::element<V,I>> &&elems) {
                if (elems.size() >= colSize) {
                    throw std::out_of_range("Too many elements to fit in row");
                }
                if (_row != nullptr) {
                    return false;
                }
                std::sort(elems.begin(), elems.end(),
                          [](element::element<V,I> const &a, element::element<V,I>const &b) {
                              return a.col < b.col;
                          });
                _row = std::make_shared<const std::vector<element::element<V,I>>>(std::move(elems));
                return true;
            }

            [[nodiscard]] bool isSet() const noexcept {
                return _row != nullptr;
            }

            void printRow() const noexcept {
                if (_row == nullptr) {
                    std::cout << "row is empty" << std::endl;
                }else {
                    for (auto const &element : *_row) {
                        std::cout <<" (value: "<< element.value << ", col:" << element.col << ")," ;
                    }
                    std::cout << std::endl;
                }
            }
};
}