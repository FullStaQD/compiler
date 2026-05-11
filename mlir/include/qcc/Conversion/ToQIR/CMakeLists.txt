# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS ToQIR.td)
mlir_tablegen(ToQIR.h.inc -gen-pass-decls -name ToQIR)
add_public_tablegen_target(QCCToQIRIncGen)

add_mlir_doc(ToQIR QCCToQIR Conversions/ -gen-pass-doc)
