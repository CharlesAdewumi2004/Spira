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

# We’ll use FetchContent for GTest, Benchmark, and Boundcraft
include(FetchContent)

# ---------- Perf-friendly flags for Debug / RelWithDebInfo ----------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # CMake already adds -g to Debug and RelWithDebInfo by default,
  # so we only need to force frame pointers.
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")

  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
endif()

# ---------- Boundcraft (your lower_bound/gallop lib) ----------
# NOTE: Boundcraft repo *already* defines a CMake target named `boundcraft`
# (typically as an INTERFACE library). So we must NOT re-add it here.
FetchContent_Declare(
  boundcraft_repo
  GIT_REPOSITORY https://github.com/CharlesAdewumi2004/hybridSearch.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(boundcraft_repo)

# Safety: if repo changes and stops providing `boundcraft`, create one.
if (NOT TARGET boundcraft)
  add_library(boundcraft INTERFACE)
  target_include_directories(boundcraft INTERFACE ${boundcraft_repo_SOURCE_DIR}/include)
  target_compile_features(boundcraft INTERFACE cxx_std_23)
endif()

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

  target_link_libraries(unit_tests PRIVATE spira boundcraft GTest::gtest_main)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(unit_tests PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(unit_tests PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
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

  add_executable(BM_spmv
      bench/google-benchmark/BM_spmv.cpp
      bench/google-benchmark/BM_main.cpp
      bench/google-benchmark/config.hpp
  )

  target_link_libraries(BM_spmv
      PRIVATE spira boundcraft benchmark::benchmark pthread
  )
  target_compile_features(BM_spmv PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(BM_spmv PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(BM_spmv PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(BM_spmv PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()

  add_executable(BM_insert_and_gets_stream
      bench/google-benchmark/BM_streamGetOperation.cpp
      bench/google-benchmark/BM_streamAddOperation.cpp
      bench/google-benchmark/BM_main.cpp
      bench/google-benchmark/config.hpp
  )

  target_link_libraries(BM_insert_and_gets_stream
      PRIVATE spira boundcraft benchmark::benchmark pthread
  )
  target_compile_features(BM_insert_and_gets_stream PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(BM_insert_and_gets_stream PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(BM_insert_and_gets_stream PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(BM_insert_and_gets_stream PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()

  add_executable(BM_perf_spmv
      bench/perf-benchmark/benchmark_perf_spmv.cpp
  )

  target_link_libraries(BM_perf_spmv
      PRIVATE spira boundcraft benchmark::benchmark pthread
  )
  target_compile_features(BM_perf_spmv PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(BM_perf_spmv PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(BM_perf_spmv PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(BM_perf_spmv PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()

endif() # SPIRA_BUILD_BENCHMARKS
