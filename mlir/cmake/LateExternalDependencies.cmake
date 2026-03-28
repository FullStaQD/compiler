include(FetchContent)
set(FETCH_PACKAGES "")

FetchContent_Declare(
  stablehlo
  GIT_REPOSITORY https://github.com/openxla/stablehlo.git
  GIT_TAG v1.13.8)
set(STABLEHLO_BUILD_EMBEDDED
    ON
    CACHE BOOL "Build StableHLO as part of another project")
list(APPEND FETCH_PACKAGES stablehlo)

FetchContent_MakeAvailable(${FETCH_PACKAGES})
