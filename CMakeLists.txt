cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Deps (GoogleTest via FetchContent)
include(${PROJECT_SOURCE_DIR}/cmake/FetchDeps.cmake)

# -------- Library --------
add_library(spira
        src/main.cpp
        # headers are optional to list; helps IDEs:
        include/spira/element.hpp
        include/spira/traits.hpp
        include/spira/concepts.hpp
        include/spira/matrix.hpp
        include/spira/matrix.tpp
)
target_include_directories(spira PUBLIC include)
target_compile_features(spira PUBLIC cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spira PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Sanitizers: apply to targets, Debug-only
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spira PRIVATE
          $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
  target_link_options(spira PRIVATE
          $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
endif()

# -------- Tests --------
enable_testing()

add_executable(unit_tests
        tests/element_test.cpp
        tests/test_matrix.cpp
)
target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)

# Sanitizers on tests too (Debug-only)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(unit_tests PRIVATE
          $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
  target_link_options(unit_tests PRIVATE
          $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>)
endif()

include(GoogleTest)
gtest_discover_tests(unit_tests)
