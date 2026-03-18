cmake_minimum_required(VERSION 3.24...4.2)

project(
  fullstaqd-compiler # TODO: rename according to the final name when it is decided
  DESCRIPTION "FullStaQD compiler project"
  LANGUAGES CXX C)

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 20)
  set(CMAKE_CXX_STANDARD
      20
      CACHE STRING "C++ standard to conform to")
endif()

if(DEFINED ENV{LLVM_EXTERNAL_LIT} AND NOT DEFINED LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT
      $ENV{LLVM_EXTERNAL_LIT}
      CACHE STRING "")
endif()

# ExternalDependencies MUST be included before SetupMLIR. `mqt-core` calls find_package(MLIR) internally. Setting up
# MLIR here before they are done with the build would create conflicts.
include(mlir/cmake/ExternalDependencies.cmake)
include(mlir/cmake/SetupMLIR.cmake)

add_subdirectory(mlir/include)
add_subdirectory(mlir/lib)
add_subdirectory(mlir/test)
add_subdirectory(mlir/tools)
