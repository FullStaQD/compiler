# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCToQIR
  ${CONVERSION_SOURCES}
  DEPENDS
  QCCToQIRIncGen
  LINK_LIBS
  # Dialects
  MLIRQCDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  QCCAuxDialect
  # Conversion infra
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRFuncToLLVM
  # Core MLIR
  MLIRIR
  MLIRPass)
