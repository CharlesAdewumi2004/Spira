cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sanitizers for Debug
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
  add_link_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

# Deps (GoogleTest via FetchContent)
include(${PROJECT_SOURCE_DIR}/cmake/FetchDeps.cmake)

# Library (placeholder for now)
add_library(spira src/dummy.cpp)
target_include_directories(spira PUBLIC include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spira PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
enable_testing()
add_executable(unit_tests tests/test_sanity.cpp)
target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(unit_tests)
