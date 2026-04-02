include(FetchContent)
set(FETCH_PACKAGES "")

FetchContent_Declare(
  stablehlo
  GIT_REPOSITORY https://github.com/openxla/stablehlo.git
  GIT_TAG bdbe31e8a1a2f4884c29c1c685de36e74ba6a68d) # v1.13.8 - latest compatible with LLVM 22.1.0
set(STABLEHLO_BUILD_EMBEDDED
    ON
    CACHE BOOL "Build StableHLO as part of another project")
list(APPEND FETCH_PACKAGES stablehlo)

FetchContent_MakeAvailable(${FETCH_PACKAGES})
