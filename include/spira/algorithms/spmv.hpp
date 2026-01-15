#include "../matrix/matrix.hpp"

namespace spira::algorithms
{

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void spmv(const spira::matrix<Layout, I, V> &matrix, const std::vector<V> &x, std::vector<V> &y)
    {
        if (x.size() != matrix.n_cols())
        {
            throw std::invalid_argument("The size of the input vector x does not match the number of columns of the matrix");
        }
        if (y.size() != matrix.n_rows())
        {
            throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
        }

        auto SpMV = [&y, &x](const row<Layout, I, V> &row, I rowIndex)
        {
            V acc = traits::ValueTraits<V>::zero();

            auto op = [&acc, &x](I const col,V const &val) {
                acc += x[col] * val;
            };

            row.for_each_element(op);

            y[rowIndex] = acc;
        };

        matrix.for_each_row(SpMV);
    }

    template <class Layout, concepts::Indexable I>
    void spmv(const spira::matrix<Layout, I, double> &matrix, const std::vector<double> &x, std::vector<double> &y)
    {
        if (x.size() != matrix.n_cols())
        {
            throw std::invalid_argument("The size of the input vector x does not match the number of columns of the matrix");
        }
        if (y.size() != matrix.n_rows())
        {
            throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
        }

        auto SpMV = [&y, &x](const row<Layout, I, double> &row, I rowIndex)
        {
            double acc = traits::ValueTraits<double>::zero();

            auto op = [&acc, &x](I const col,double const &val) {
                acc += x[col] * val;
            };

            row.for_each_element(op);

            y[rowIndex] = acc;
        };

        matrix.for_each_row(SpMV);
    }

    template <class Layout, concepts::Indexable I>
    void spmv(const spira::matrix<Layout, I, float> &matrix, const std::vector<float> &x, std::vector<float> &y)
    {
        if (x.size() != matrix.n_cols())
        {
            throw std::invalid_argument("The size of the input vector x does not match the number of columns of the matrix");
        }
        if (y.size() != matrix.n_rows())
        {
            throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
        }

        auto SpMV = [&y, &x](const row<Layout, I, float> &row, I rowIndex)
        {
            float acc = traits::ValueTraits<float>::zero();

            auto op = [&acc, &x](I const col,float const &val) {
                acc += x[col] * val;
            };

            row.for_each_element(op);

            y[rowIndex] = acc;
        };

        matrix.for_each_row(SpMV);
    }

    template <class Layout, concepts::Indexable I>
    void spmv(const spira::matrix<Layout, I, std::complex<double>> &matrix, const std::vector<std::complex<double>> &x, std::vector<std::complex<double>> &y)
    {
        if (x.size() != matrix.n_cols())
        {
            throw std::invalid_argument("The size of the input vector x does not match the number of columns of the matrix");
        }
        if (y.size() != matrix.n_rows())
        {
            throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
        }

        auto SpMV = [&y, &x](const row<Layout, I, std::complex<double>> &row, I rowIndex)
        {
            std::complex<double> acc = traits::ValueTraits<std::complex<double>>::zero();

            auto op = [&acc, &x](I const col,std::complex<double> const &val) {
                acc += x[col] * val;
            };

            row.for_each_element(op);

            y[rowIndex] = acc;
        };

        matrix.for_each_row(SpMV);
    }

}
