#include <cstddef>
#include <cstdint>
#include <stddef.h>


    double sparse_dot_double_scalar(const double* vals, const uint32_t* cols, const double* x, size_t n, size_t x_size){
        double acc = 0.0;
        for(size_t i = 0; i < n; i++){
            acc += x[cols[i]] * vals[i]   ; 
        }

        return acc;
    }

    float sparse_dot_float_scalar(const float* vals, const uint32_t* cols, const float* x, size_t n, size_t x_size){
        float acc = 0.0f;
        for(size_t i = 0; i < n; i++){
            acc += x[cols[i]] * vals[i]   ; 
        }

        return acc;
    }