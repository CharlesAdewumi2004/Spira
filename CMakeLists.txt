cmake_minimum_required(VERSION 3.22)
project(spira LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config generators default to Release (sanitizers are Debug-only).
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ---------- Options ----------
option(SPIRA_BUILD_TESTS       "Build unit tests"                  ON)
option(SPIRA_BUILD_BENCHMARKS  "Build performance benches"         ON)
option(SPIRA_BUILD_PARALLEL    "Build the parallel_matrix tests"   ON)

include(FetchContent)
find_package(Threads REQUIRED)

# ---------- unordered_dense (ankerl::unordered_dense) ----------
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.5.0
)
FetchContent_MakeAvailable(unordered_dense)

# ---------- Architecture Detection ----------
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  set(SPIRA_ARCH_X86 TRUE)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(SPIRA_ARCH_ARM64 TRUE)
endif()
# Any other architecture builds with the scalar kernel only.

# ---------- Kernel library ----------
# The sparse dot kernels (scalar fallback plus one file per ISA, each built
# with its own flags) and the startup dispatch that picks between them.
set(KERNEL_SOURCES
  src/kernels/dispatch.cpp
  src/kernels/dot_impls/dot_scalar.cpp
)
if (SPIRA_ARCH_X86)
  list(APPEND KERNEL_SOURCES
    src/kernels/dot_impls/x86/dot_sse.cpp
    src/kernels/dot_impls/x86/dot_avx.cpp
    src/kernels/dot_impls/x86/dot_avx512.cpp
  )
elseif (SPIRA_ARCH_ARM64)
  list(APPEND KERNEL_SOURCES src/kernels/dot_impls/arm/dot_neon.cpp)
endif()

add_library(spira_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(spira_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)

if (SPIRA_ARCH_X86)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(src/kernels/dot_impls/x86/dot_sse.cpp
      PROPERTIES COMPILE_OPTIONS "-msse4.2")
    set_source_files_properties(src/kernels/dot_impls/x86/dot_avx.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    # Dispatch only checks AVX-512F, so allow nothing beyond it.
    set_source_files_properties(src/kernels/dot_impls/x86/dot_avx512.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx512f")
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(src/kernels/dot_impls/x86/dot_avx.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/dot_impls/x86/dot_avx512.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(spira_kernels PRIVATE -fno-omit-frame-pointer)
endif()

# ---------- Library (header-only core + kernels) ----------
add_library(spira INTERFACE)
target_include_directories(spira INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(spira INTERFACE cxx_std_23)
target_link_libraries(spira INTERFACE
  unordered_dense::unordered_dense
  spira_kernels
  Threads::Threads
)

# ---------- Tests ----------
function(spira_test_target target)
  target_link_libraries(${target} PRIVATE spira GTest::gtest_main)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-omit-frame-pointer)
    if (NOT WIN32)
      target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
      target_link_options(${target} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    endif()
  endif()
  gtest_discover_tests(${target} DISCOVERY_TIMEOUT 30)
endfunction()

if (SPIRA_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  include(GoogleTest)

  add_executable(unit_tests
    tests/layout_and_type_constraints_tests.cpp
    tests/matrix_tests.cpp
    tests/add_tests.cpp
    tests/flush_behaviour_tests.cpp
    tests/mode_switch_tests.cpp
    tests/large_matrix_stress_tests.cpp
    tests/matrix_operator_overloads_tests.cpp
    tests/simd_dot_operator_tests.cpp
    tests/compaction_tests.cpp
    tests/hash_map_buffer_tests.cpp
    tests/serial/spmv_tests.cpp
    tests/serial/spgemm_tests.cpp
    tests/serial/transpose_tests.cpp
    tests/serial/matrix_addition_tests.cpp
    tests/serial/scalar_tests.cpp
    tests/serial/accumulate_algo_tests.cpp
  )
  spira_test_target(unit_tests)

  if (SPIRA_BUILD_PARALLEL)
    add_executable(parallel_tests
      tests/parallel/partition_tests.cpp
      tests/parallel/parallel_matrix_tests.cpp
      tests/parallel/parallel_algorithm_tests.cpp
      tests/parallel/hash_map_buffer_parallel_tests.cpp
    )
    spira_test_target(parallel_tests)
  endif()
endif()

# ---------- Benchmarks ----------
if (SPIRA_BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  foreach(bench spira_bench spira_cloth_bench)
    add_executable(${bench})
    target_link_libraries(${bench} PRIVATE spira benchmark::benchmark)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${bench} PRIVATE -march=native)
    endif()
  endforeach()
  target_sources(spira_bench PRIVATE bench/spira_bench.cpp)
  target_sources(spira_cloth_bench PRIVATE bench/cloth_bench.cpp)
endif()
