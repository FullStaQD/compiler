# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(
  QCCAuxDialect
  AuxDialect.cpp
  DEPENDS
  QCCAuxDialectIncGen
  QCCAuxOpsIncGen
  LINK_LIBS
  PRIVATE
  MLIRIR
  MLIRSupport)
