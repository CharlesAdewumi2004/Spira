#include <iostream>

#include "../include/spira/containers/matrix.hpp"

int main() {
    spira::matrix<spira::layout::tags::aos_tag, unsigned, int> m(10, 10);
    std::cout << m.get_shape().first<< "," << m.get_shape().second << std::endl;
    m.add(1,1,1000000000000000000);
    const int val = m.get(1,1);
    printf("%d", val);

}