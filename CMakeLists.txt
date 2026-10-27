cmake_minimum_required(VERSION 3.16)
project(hnsw VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HNSW_BUILD_TESTS "Build unit tests" ON)
option(HNSW_BUILD_BENCH "Build the benchmark binary" ON)
option(HNSW_ENABLE_AVX2 "Compile AVX2 distance kernels" ON)
option(HNSW_ENABLE_ASAN "AddressSanitizer" OFF)
option(HNSW_ENABLE_UBSAN "UndefinedBehaviorSanitizer" OFF)
option(HNSW_ENABLE_TSAN "ThreadSanitizer" OFF)
option(HNSW_WERROR "Treat warnings as errors" OFF)

if(HNSW_ENABLE_TSAN AND (HNSW_ENABLE_ASAN OR HNSW_ENABLE_UBSAN))
  message(FATAL_ERROR "ThreadSanitizer cannot be combined with ASan or UBSan")
endif()

set(HNSW_SAN_FLAGS "")
if(HNSW_ENABLE_ASAN)
  list(APPEND HNSW_SAN_FLAGS -fsanitize=address)
endif()
if(HNSW_ENABLE_UBSAN)
  list(APPEND HNSW_SAN_FLAGS -fsanitize=undefined -fno-sanitize-recover=undefined)
endif()
if(HNSW_ENABLE_TSAN)
  list(APPEND HNSW_SAN_FLAGS -fsanitize=thread)
endif()
if(HNSW_SAN_FLAGS)
  add_compile_options(${HNSW_SAN_FLAGS} -fno-omit-frame-pointer)
  add_link_options(${HNSW_SAN_FLAGS})
endif()

set(HNSW_AVX2 OFF)
if(HNSW_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(HNSW_AVX2 ON)
endif()

add_library(hnsw
  src/distance_scalar.cpp
  src/distance_dispatch.cpp
  src/index.cpp
  src/bruteforce.cpp)
target_include_directories(hnsw PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hnsw PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
target_link_libraries(hnsw PUBLIC Threads::Threads)

if(HNSW_AVX2)
  target_sources(hnsw PRIVATE src/distance_avx2.cpp)
  if(MSVC)
    set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
  target_compile_definitions(hnsw PRIVATE HNSW_HAS_AVX2=1)
else()
  target_compile_definitions(hnsw PRIVATE HNSW_HAS_AVX2=0)
endif()

target_compile_options(hnsw PRIVATE -Wall -Wextra -Wpedantic)
if(HNSW_WERROR)
  target_compile_options(hnsw PRIVATE -Werror)
endif()
# GCC -fsanitize=thread warns that atomic_thread_fence is invisible to TSan.
# The seqlock fences stay. Clang has no -Wtsan flag.
if(HNSW_ENABLE_TSAN AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(hnsw PRIVATE -Wno-tsan)
endif()

add_executable(hnsw_example examples/example.cpp)
target_link_libraries(hnsw_example PRIVATE hnsw)
target_compile_options(hnsw_example PRIVATE -Wall -Wextra -Wpedantic)

if(HNSW_BUILD_BENCH)
  add_executable(hnsw_bench bench/bench.cpp)
  target_link_libraries(hnsw_bench PRIVATE hnsw)
  target_compile_options(hnsw_bench PRIVATE -Wall -Wextra -Wpedantic)
  add_executable(hnsw_delete_bench bench/delete_bench.cpp)
  target_link_libraries(hnsw_delete_bench PRIVATE hnsw)
  target_compile_options(hnsw_delete_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(HNSW_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(hnsw_tests
    tests/test_distance.cpp
    tests/test_index.cpp
    tests/test_recall.cpp
    tests/test_io.cpp
    tests/test_concurrent.cpp
    tests/test_delete.cpp)
  target_link_libraries(hnsw_tests PRIVATE hnsw GTest::gtest_main)
  target_compile_options(hnsw_tests PRIVATE -Wall -Wextra -Wpedantic)
  if(HNSW_WERROR)
    target_compile_options(hnsw_tests PRIVATE -Werror)
  endif()

  include(GoogleTest)
  gtest_discover_tests(hnsw_tests DISCOVERY_TIMEOUT 60)
  add_test(NAME example COMMAND hnsw_example)
endif()

include(GNUInstallDirs)
install(TARGETS hnsw ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
