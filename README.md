# Spira

Spira is a C++ library for experimenting with **dynamic sparse matrices**.  

## Build

```bash
# configure (Debug build with sanitizers)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# build library + tests
cmake --build build -j

# run tests
ctest --test-dir build --output-on-failure
