#include <cstddef>
#include <stddef.h>


namespace spira::spira::kernel::dot{
    double sparse_dot_double_scalar(const double* vals, const int* cols, const double* x, size_t n){
        double acc = 0.0;
        for(size_t i = 0; i < n; i++){
            acc += x[cols[i]] * vals[i]   ; 
        }

        return acc;
    }

    float sparse_dot_float_scalar(const float* vals, const int* cols, const float* x, size_t n){
        float acc = 0.0f;
        for(size_t i = 0; i < n; i++){
            acc += x[cols[i]] * vals[i]   ; 
        }

        return acc;
    }
}