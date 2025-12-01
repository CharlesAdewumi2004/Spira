# Make sure no stale cache forces a local googletest path
unset(FETCHCONTENT_SOURCE_DIR_GOOGLETEST CACHE)
set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "" CACHE PATH "" FORCE)

cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ---------- Options ----------
option(SPIRA_BUILD_TESTS       "Build unit tests"           ON)
option(SPIRA_BUILD_BENCHMARKS  "Build performance benches"  ON)

# ---------- Library ----------
add_library(spira INTERFACE)
target_include_directories(spira INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(spira INTERFACE cxx_std_23)

# ---------- Tests ----------
if (SPIRA_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(unit_tests
      tests/test.cpp
      tests/soa_spmv_test.cpp
      tests/aos_spmv_test.cpp 
      tests/aos_test.cpp
      tests/soa_test.cpp
  )
  target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(unit_tests PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
    target_link_options(unit_tests PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
  endif()

  if (UNIX AND NOT APPLE)
    target_link_libraries(unit_tests PRIVATE pthread)
  endif()

  include(GoogleTest)
  gtest_discover_tests(unit_tests)
endif()

# ---------- Benchmarks ----------
if (SPIRA_BUILD_BENCHMARKS)
  add_executable(spira_bench_row
    benchmarks/benchmarks.cpp
  )

  target_link_libraries(spira_bench_row PRIVATE spira)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(spira_bench_row PRIVATE -O3 -march=native -DNDEBUG)
  elseif (MSVC)
    target_compile_options(spira_bench_row PRIVATE /O2)
  endif()
endif()
