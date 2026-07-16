// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// A circuit using a T gate, which the HiSEP-Q device does not implement. Compiled by
// cli_options.mlir; not a test of its own.

func.func @main() attributes { qcc.entry_point } {
  %0 = qc.static 0 : !qc.qubit
  qc.h %0 : !qc.qubit
  qc.t %0 : !qc.qubit
  %m0 = qc.measure %0 : !qc.qubit -> i1
  aux.record_int %m0 : i1
  return
}
