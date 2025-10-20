#include "spira/row.hpp"
#include "spira/spira.hpp"
int main() {
    const spira::row::row<int, size_t> r({{3,4},{5,6}, {1,2}});
    r.printRow();
}