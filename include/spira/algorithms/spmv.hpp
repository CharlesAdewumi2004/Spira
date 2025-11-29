#include "../matrix/matrix.hpp"

namespace spira::algorithms
{

    template <class Layout, concepts::Indexable I, concepts::Valueable V>
    void spmv(const spira::matrix<Layout, I, V> &matrix, const std::vector<V> &x, std::vector<V> &y)
    {
        if(x.size() != matrix.n_cols()){
            throw std::invalid_argument("The size of the input vector x does not match the number of columns of the matrix");
        }
        if(y.size() != matrix.n_rows()){
            throw std::invalid_argument("The size of the output vector y does not match the number of rows of the matrix");
        }

        auto applySpMV = [&y, &x](const row<Layout, I, V> &row, I rowIndex){
            V acc = traits::ValueTraits<V>::zero();
            for(auto it = row.cbegin(); it != row.cend(); it++){
                auto [col, val] = *it;
                acc += x[col] * val;
            }
            y[rowIndex] = acc;
        };        

        matrix.for_each_row(applySpMV);
    }

}
