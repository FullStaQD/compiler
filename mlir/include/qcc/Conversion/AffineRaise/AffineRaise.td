// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def AffineRaiseFromSCF : Pass<"affine-raise-from-scf"> {
  let summary = "Raise SCF loops to Affine loops where possible.";

  let description = [{
      Attempts to convert `scf.for` loops into `affine.for` loops when the
      loop bounds and step are affine expressions.
   }];

  let dependentDialects = [
    "mlir::affine::AffineDialect"
  ];
}

def WhileToFor : Pass<"while-to-for"> {
  let summary = "Convert scf.while loops to scf.for loops using uplift patterns.";

  let description = [{
      Applies the SCF uplift patterns to rewrite eligible `scf.while` loops as
      `scf.for` loops.
   }];

  let dependentDialects = [
    "mlir::scf::SCFDialect"
  ];
}
