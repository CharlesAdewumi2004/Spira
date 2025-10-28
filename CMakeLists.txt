cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Your library include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# -------- Library (no test sources here) --------
add_library(spira
        src/main.cpp
        include/spira/traits.hpp
        include/spira/concepts.hpp
        include/spira/matrix.hpp
        include/spira/layouts/aos.hpp
        include/spira/layouts/soa.hpp
        include/spira/layouts/layout_of.hpp
        include/spira/row.hpp
        include/spira/layouts/layout_tags.hpp
)
target_include_directories(spira PUBLIC include)
target_compile_features(spira PUBLIC cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spira PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(spira PRIVATE $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
  target_link_options(spira PRIVATE $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
endif()

# -------- Dependencies (GoogleTest via FetchContent) --------
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For MSVC: use static runtime by default (optional)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# -------- Tests --------
enable_testing()

add_executable(unit_tests
        tests/test.cpp
)
target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)

# pthread for Linux
if (UNIX AND NOT APPLE)
  target_link_libraries(unit_tests PRIVATE pthread)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
  target_link_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)
