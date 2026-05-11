# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCJaspToQC
  ${CONVERSION_SOURCES}
  DEPENDS
  JaspToQCIncGen
  LINK_LIBS
  MLIRQCDialect
  QCCJaspDialect
  MLIRArithDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms)

mqt_mlir_target_use_project_options(QCCJaspToQC)
