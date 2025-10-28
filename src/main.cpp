#include "spira/row.hpp"


int main() {
    spira::row::row<spira::layout::tags::aos_tag, unsigned, int> row;
    row.add(1, 2);
}