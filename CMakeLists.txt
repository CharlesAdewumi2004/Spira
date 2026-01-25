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

# We’ll use FetchContent for deps
include(FetchContent)

# ---------- Boundcraft (your lower_bound/gallop lib) ----------
# NOTE: Boundcraft repo *already* defines a CMake target named `boundcraft`
FetchContent_Declare(
  boundcraft_repo
  GIT_REPOSITORY https://github.com/CharlesAdewumi2004/Boundcraft.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(boundcraft_repo)

# Safety: if repo changes and stops providing `boundcraft`, create one.
if (NOT TARGET boundcraft)
  add_library(boundcraft INTERFACE)
  target_include_directories(boundcraft INTERFACE ${boundcraft_repo_SOURCE_DIR}/include)
  target_compile_features(boundcraft INTERFACE cxx_std_23)
endif()

# ---------- unordered_dense (ankerl::unordered_dense) ----------
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.5.0
)
FetchContent_MakeAvailable(unordered_dense)

# ---------- Library ----------
add_library(spira INTERFACE)
target_include_directories(spira INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(spira INTERFACE cxx_std_23)

# Make unordered_dense available to all spira consumers
target_link_libraries(spira INTERFACE unordered_dense::unordered_dense)

# ---------- Perf-friendly flags for Debug / RelWithDebInfo ----------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")

  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
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
      tests/layout_and_type_constraits_tests.cpp
      tests/matrix_tests.cpp
      tests/flush_behaviour_tests.cpp
      tests/mode_switch_tests.cpp
      tests/spmv_tests.cpp
      tests/large_matrix_stress_tests.cpp
      tests/spgemm_tests.cpp
      tests/transpose_tests.cpp
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

  FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(GB_spira
      bench/google-benchmark/spira_benchmark.cpp
      )

  target_link_libraries(GB_spira
      PRIVATE spira boundcraft benchmark::benchmark pthread
  )
  target_compile_features(GB_spira PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(GB_spira PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(GB_spira PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(GB_spira PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
  endif()

endif()
