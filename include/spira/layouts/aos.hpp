#pragma once
#include <vector>
#include <algorithm>

#include "spira/concepts.hpp"

namespace spira::layout {
    template<concepts::Indexable I, concepts::Valueable V>
    struct elementPair {
        I column;
        V value;
    };

    template<concepts::Indexable I, concepts::Valueable V>
    class aos {
        public:
            [[nodiscard]] bool empty() const noexcept {
                return elements.empty();
            }
            void reserve(size_t size) {
                elements.reserve(size);
            }
            [[nodiscard]] size_t capacity() const noexcept {
                return elements.capacity();
            }
            [[nodiscard]] size_t size() const noexcept {
                return elements.size();
            }
            void clear() {
                elements.clear();
            }
            I key_at(size_t index) const noexcept {
                return elements[index].column;
            }
            [[nodiscard]] V& value_at(size_t pos) noexcept {
                    return elements[pos].value;
                }

            [[nodiscard]] const V& value_at(size_t pos) const noexcept {
                    return elements[pos].value;
                }
            void set_at(size_t index, I col, const V& val) {
                elements.insert(elements.begin() + index, elementPair<I, V>{col, val});
            }
            void insert_at(size_t index, I col, const V& val) {
                elements.insert(elements.begin() + index, elementPair<I,V>{col, val});
            }
            void erase_at(size_t index) {
                elements.erase(elements.begin() + index);
            }
            [[nodiscard]] size_t lower_bound(I col) const noexcept {
                auto it = std::lower_bound(
                    elements.begin(), elements.end(), col,
                    [](auto const& e, I key){ return e.column < key; }
                );
                return static_cast<size_t>(distance(elements.begin(), it));
            }
    private:
        std::vector<elementPair<I, V>> elements;
    };
}
