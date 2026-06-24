// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertQCToIntrinsics : Pass<"convert-qc-to-intrinsics", "mlir::func::FuncOp"> {
  let summary = "Lowers QC conforming LLVM dialect to LLVM with intrinsics for HiSEP-Q.";

  let description = [{
      The pass anchors to a function and guarantees the following:

      - QC runtime and QIS function calls are translated to target specific intrinsics.
      - Only the llvm dialect (in the function body) is considered legal after this pass.
      }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}
