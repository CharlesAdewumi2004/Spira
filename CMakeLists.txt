# Make sure no stale cache forces a local googletest path
unset(FETCHCONTENT_SOURCE_DIR_GOOGLETEST CACHE)
set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "" CACHE PATH "" FORCE)

cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
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

# We’ll use FetchContent for both GTest and Benchmark
include(FetchContent)

# ---------- Tests ----------
if (SPIRA_BUILD_TESTS)
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

  # Fetch Google Benchmark
  FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  # ------------------------------
  # 1. Main full benchmark suite
  # ------------------------------
  add_executable(spira_bench_row
      bench/BM_spmv.cpp
      bench/BM_streamGetOperation.cpp
      bench/BM_streamAddOperation.cpp
      bench/BM_main.cpp
      bench/BM_addOperation.cpp
      bench/BM_getOperation.cpp
  )

  target_link_libraries(spira_bench_row
      PRIVATE spira benchmark::benchmark pthread
  )

  target_compile_features(spira_bench_row PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(spira_bench_row PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(spira_bench_row PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
    target_link_options(spira_bench_row PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
  endif()

  # ------------------------------
  # 2. SpMV-only benchmark executable
  # ------------------------------
  add_executable(BM_spmv
      bench/BM_spmv.cpp
      bench/BM_main.cpp
  )

  target_link_libraries(BM_spmv
      PRIVATE spira benchmark::benchmark pthread
  )

  target_compile_features(BM_spmv PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(BM_spmv PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(BM_spmv PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
    target_link_options(BM_spmv PRIVATE
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer -fsanitize=address,undefined>
    )
  endif()

endif() # SPIRA_BUILD_BENCHMARKS

