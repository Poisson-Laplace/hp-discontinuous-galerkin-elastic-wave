cmake_minimum_required(VERSION 3.16)
project(dgwave VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

# ---------------------------------------------------------------------------
# Eigen
#   Preferred:  find_package(Eigen3 CONFIG)   (apt / vcpkg / conda install)
#   Fallback :  -DDGWAVE_EIGEN_INCLUDE_DIR=/path/to/eigen3
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 CONFIG QUIET)
if(TARGET Eigen3::Eigen)
  message(STATUS "Eigen: found via CONFIG (${EIGEN3_VERSION_STRING})")
  set(DGWAVE_EIGEN Eigen3::Eigen)
else()
  set(DGWAVE_EIGEN_INCLUDE_DIR "" CACHE PATH "Path to the eigen3 include directory")
  find_path(_eigen_inc
    NAMES signature_of_eigen3_matrix_library
    HINTS ${DGWAVE_EIGEN_INCLUDE_DIR}
    PATHS /usr/include/eigen3 /usr/local/include/eigen3 /opt/homebrew/include/eigen3)
  if(NOT _eigen_inc)
    message(FATAL_ERROR
      "Eigen not found. Install it (apt install libeigen3-dev) or pass "
      "-DDGWAVE_EIGEN_INCLUDE_DIR=/path/to/eigen3")
  endif()
  message(STATUS "Eigen: found via include dir (${_eigen_inc})")
  add_library(dgwave_eigen INTERFACE)
  target_include_directories(dgwave_eigen SYSTEM INTERFACE ${_eigen_inc})
  set(DGWAVE_EIGEN dgwave_eigen)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(dgwave
  src/Jacobi.cpp
  src/Dubiner.cpp
  src/RefTriangle.cpp
  src/Mesh.cpp
  src/Output.cpp
  src/Elastic.cpp
)
target_include_directories(dgwave PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(dgwave PUBLIC ${DGWAVE_EIGEN})
target_compile_options(dgwave PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
foreach(_t reference mesh elastic)
  add_executable(test_${_t} tests/test_${_t}.cpp)
  target_link_libraries(test_${_t} PRIVATE dgwave)
  add_test(NAME ${_t} COMMAND test_${_t})
endforeach()

# Visual output demo (not a pass/fail test -- writes .vtk files)
add_executable(viz_demo tests/viz_demo.cpp)
target_link_libraries(viz_demo PRIVATE dgwave)
