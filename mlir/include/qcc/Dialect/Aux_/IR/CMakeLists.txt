# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS AuxDialect.td)
mlir_tablegen(AuxDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AuxDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(QCCAuxDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AuxOps.td)
mlir_tablegen(AuxOps.h.inc -gen-op-decls)
mlir_tablegen(AuxOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QCCAuxOpsIncGen)
