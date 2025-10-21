#pragma once
#include <vector>
#include <optional>

#include "element.hpp"

namespace spira {

    template <concepts::Valueable V, concepts::Indexable I, I rowSize, I colSize>
    class matrix {
    private:
        std::vector<std::vector<element<V, I>>> _matrix;

    public:
        matrix() noexcept;
        bool setRow(I rowIndex, std::vector<element<V, I>> elems);
        void printRow(I rowIndex) const noexcept;
        std::optional<V> at(I rowIndex, I colIndex) const;
        void printMatrix() const noexcept;
    };

}

#include "matrix.tpp"