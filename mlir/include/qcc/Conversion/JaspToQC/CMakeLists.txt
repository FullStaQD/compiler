# CMakeLists.txt
#
# Copyright (c) 2026 FullStaQD Project
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM_TARGET_DEFINITIONS JaspToQC.td)
mlir_tablegen(JaspToQC.h.inc -gen-pass-decls -name JaspToQC)
add_public_tablegen_target(JaspToQCIncGen)

add_mlir_doc(JaspToQC QCCJaspToQC Conversions/ -gen-pass-doc)
