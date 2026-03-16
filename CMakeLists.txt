cmake_minimum_required(VERSION 3.20...4.2)

project(
    FullStaQD-mlir
    DESCRIPTION "FullStaQD project MLIR"
    LANGUAGES CXX C
)

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD
        17
        CACHE STRING "C++ standard to conform to"
    )
endif()

include(mlir/cmake/ExternalDependencies.cmake)
include(mlir/cmake/SetupMLIR.cmake)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)

add_subdirectory(mlir/include)
add_subdirectory(mlir/lib)
add_subdirectory(mlir/test)
add_subdirectory(mlir/tools/qcc)
add_subdirectory(mlir/tools/qcc-opt)