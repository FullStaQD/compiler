# Declare all external dependencies and make sure that they are available.

include(FetchContent)
include(CMakeDependentOption)
include(GNUInstallDirs)
set(FETCH_PACKAGES "")

FetchContent_Declare(
    mqt-core
    GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
    GIT_TAG 8747a89766dfb943d62ed100d383cd1823d2356c)
  list(APPEND FETCH_PACKAGES mqt-core)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})