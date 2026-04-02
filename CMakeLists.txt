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

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile commands" FORCE)

include(GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})

# EarlyExternalDependencies MUST be included before SetupMLIR. `mqt-core` calls find_package(MLIR) internally. Setting
# up MLIR here before they are done with the build would create conflicts. Other dependencies require SetupMLIR to be
# finished. These are to be listed in LateExternalDependencies.
include(mlir/cmake/EarlyExternalDependencies.cmake)
include(mlir/cmake/SetupMLIR.cmake)
include(mlir/cmake/LateExternalDependencies.cmake)

add_subdirectory(mlir/include)
add_subdirectory(mlir/lib)
add_subdirectory(mlir/test)
add_subdirectory(mlir/tools)
