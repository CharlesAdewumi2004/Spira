#include "../include/spira/spira.hpp"
#include <iostream>

int main(){
    size_t rows = 10000;
    size_t cols = 10000;
    spira::matrix<spira::layout::tags::aos_tag, size_t, std::complex<double>> testMatrix(rows, cols);
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
             std::complex<double> z = 10;
            testMatrix.add(i,j,z);
        }
    }
    return 0; 
}