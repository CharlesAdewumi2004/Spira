#include "spira/matrix.hpp"
#include "spira/spira.hpp"

int main() {
    spira::matrix<size_t, size_t, 10, 10> matrix{};
    if (bool happened = matrix.setRow(0, {{1,2}, {3,3}, {7,5}})) {
        std::cout << "happened" << std::endl;
    }else {
        std::cout << "not happened" << std::endl;
    }
    matrix.printRow(0);
    if (bool happened = matrix.setRow(0, {{1,2}, {3,3}, {7,5}})) {
        std::cout << "happened" << std::endl;
        }else {
            std::cout << "not happened" << std::endl;
        }
    if (const auto result = matrix.at(0,2)) {
        std::cout << *result << '\n';
    }
}