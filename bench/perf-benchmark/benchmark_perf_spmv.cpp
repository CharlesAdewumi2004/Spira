#include <../include/spira/spira.hpp>
#include <string.h>
#include <complex>
#include <random>
#include <iostream>

template <class layout, class I, class V>
void runSpmv(spira::matrix<layout, I, V> m, std::vector<V> x, std::vector<V> y)
{
    std::cout << "READY - attach perf now.\n";
    std::string _;
    std::getline(std::cin, _);
    spira::algorithms::spmv(m,x,y);
}

template <class layout, class V>
void constructBenchmark(size_t rows, size_t cols, size_t nnz)
{
    std::vector<V> x,y(rows, V{});
    spira::matrix<layout, size_t, V> m(rows, cols);

    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> row_dist(0, rows - 1);
    std::uniform_int_distribution<size_t> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> num_dist(-10000, 10000);

    for (std::size_t i = 0; i < cols; ++i)
    {
        x.push_back(static_cast<V>(num_dist(rng)));
    }
    for (std::size_t k = 0; k < nnz; ++k)
    {
        m.add(row_dist(rng), col_dist(rng), static_cast<V>(num_dist(rng)));
    }

    runSpmv(m, x, y);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        throw std::invalid_argument("Invalid number of arguements passed");
    }
    size_t nnz = std::stoull(argv[1]);
    size_t rows = std::stoull(argv[2]);
    size_t cols = std::stoull(argv[3]);
    std::string Inputlayout = argv[4];
    std::string dType = argv[5];

    if (Inputlayout == "aos")
    {
        if (dType == "double")
        {
            constructBenchmark<spira::layout::tags::aos_tag, double>(rows, cols, nnz);
        }
        else if (dType == "float")
        {
            constructBenchmark<spira::layout::tags::aos_tag, float>(rows, cols, nnz);
        }
        else if (dType == "complex")
        {
            constructBenchmark<spira::layout::tags::aos_tag, std::complex<double>>(rows, cols, nnz);
        }
        else
        {
            throw std::invalid_argument("Invalid type given");
        }
    }
    else if (Inputlayout == "soa")
    {
        if (dType == "double")
        {
            constructBenchmark<spira::layout::tags::soa_tag, double>(rows, cols, nnz);
        }
        else if (dType == "float")
        {
            constructBenchmark<spira::layout::tags::soa_tag, float>(rows, cols, nnz);
        }
        else if (dType == "complex")
        {
            constructBenchmark<spira::layout::tags::soa_tag, std::complex<double>>(rows, cols, nnz);
        }
        else
        {
            throw std::invalid_argument("Invalid type given");
        }
    }
    else
    {
        throw std::invalid_argument("Invalid layout given");
    }

    return 0;
}