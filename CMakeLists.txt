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
option(SPIRA_ENABLE_SIMD       "Build SIMD-accelerated kernels" ON)

include(FetchContent)

# -------------------------------------------------------------------
# Boundcraft (dependency)
# IMPORTANT: force its tests/bench OFF when consumed via FetchContent
# -------------------------------------------------------------------
set(BOUNDCRAFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BOUNDCRAFT_BUILD_BENCH OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  boundcraft_repo
  GIT_REPOSITORY https://github.com/CharlesAdewumi2004/Boundcraft.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(boundcraft_repo)

if (TARGET boundcraft::boundcraft)
  set(_BOUNDCRAFT_TARGET boundcraft::boundcraft)
elseif (TARGET boundcraft)
  set(_BOUNDCRAFT_TARGET boundcraft)
else()
  message(FATAL_ERROR "Boundcraft did not define a target (expected boundcraft::boundcraft or boundcraft).")
endif()

# ---------- unordered_dense (ankerl::unordered_dense) ----------
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.5.0
)
FetchContent_MakeAvailable(unordered_dense)

# ---------- Library (header-only core) ----------
add_library(spira INTERFACE)
target_include_directories(spira INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(spira INTERFACE cxx_std_23)

# Make deps available to all spira consumers
target_link_libraries(spira INTERFACE unordered_dense::unordered_dense ${_BOUNDCRAFT_TARGET})

# ---------- Architecture Detection ----------
unset(SPIRA_ARCH_X86 CACHE)
unset(SPIRA_ARCH_ARM64 CACHE)
unset(SPIRA_ARCH_ARM32 CACHE)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  set(SPIRA_ARCH_X86 TRUE)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(SPIRA_ARCH_ARM64 TRUE)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "armv[78]")
  set(SPIRA_ARCH_ARM32 TRUE)
endif()

# ---------- SIMD Kernel Library ----------
if (SPIRA_ENABLE_SIMD)

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
  endif()

  if (SPIRA_ARCH_ARM64 OR SPIRA_ARCH_ARM32)
    list(APPEND KERNEL_SOURCES
      src/kernels/dot_impls/arm/dot_neon.cpp
    )
  endif()

  add_library(spira_kernels STATIC ${KERNEL_SOURCES})
  target_include_directories(spira_kernels PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )
  target_include_directories(spira_kernels PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_compile_features(spira_kernels PUBLIC cxx_std_23)

  if (SPIRA_ARCH_X86)
    target_compile_definitions(spira_kernels PRIVATE SPIRA_ARCH_X86)
  elseif (SPIRA_ARCH_ARM64)
    target_compile_definitions(spira_kernels PRIVATE SPIRA_ARCH_ARM64)
  elseif (SPIRA_ARCH_ARM32)
    target_compile_definitions(spira_kernels PRIVATE SPIRA_ARCH_ARM32)
  endif()

  # Per-file SIMD compile flags (x86)
  if (SPIRA_ARCH_X86)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      set_source_files_properties(src/kernels/dot_impls/x86/dot_sse.cpp
        PROPERTIES COMPILE_OPTIONS "-msse4.2"
      )
      set_source_files_properties(src/kernels/dot_impls/x86/dot_avx.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
      )
      set_source_files_properties(src/kernels/dot_impls/x86/dot_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512vl;-mavx512bw;-mavx512dq"
      )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set_source_files_properties(src/kernels/dot_impls/x86/dot_sse.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:SSE2"
      )
      set_source_files_properties(src/kernels/dot_impls/x86/dot_avx.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
      )
      set_source_files_properties(src/kernels/dot_impls/x86/dot_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
      )
    endif()
  endif()

  target_link_libraries(spira INTERFACE spira_kernels)
endif()

# ---------- Perf-friendly flags (apply only to your targets) ----------
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  if (TARGET spira_kernels)
    target_compile_options(spira_kernels PRIVATE -fno-omit-frame-pointer)
  endif()
endif()

# ---------- Tests ----------
if (SPIRA_BUILD_TESTS)
  unset(FETCHCONTENT_SOURCE_DIR_GOOGLETEST CACHE)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "" CACHE PATH "" FORCE)

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
    tests/matrix_addition_tests.cpp
    tests/scaler_tests.cpp
    tests/matrix_operator_overloads_tests.cpp
    tests/SIMD_dot_operator_tests.cpp
  )

  target_link_libraries(unit_tests PRIVATE spira GTest::gtest_main)

  if (SPIRA_ARCH_X86)
  target_compile_definitions(unit_tests PRIVATE SPIRA_ARCH_X86)
endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(unit_tests PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_compile_options(unit_tests PRIVATE -fno-omit-frame-pointer)
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

  target_link_libraries(GB_spira PRIVATE spira benchmark::benchmark)

  if (UNIX AND NOT APPLE)
    target_link_libraries(GB_spira PRIVATE pthread)
  endif()

  target_compile_features(GB_spira PRIVATE cxx_std_23)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(GB_spira PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(GB_spira PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(GB_spira PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_compile_options(GB_spira PRIVATE -fno-omit-frame-pointer)
  endif()
endif()