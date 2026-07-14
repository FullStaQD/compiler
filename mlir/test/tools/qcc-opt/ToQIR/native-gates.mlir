// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// The device declares the QIS functions it implements. `prep-to-qir` declares only those, and
// `convert-qc-to-qir` rejects a gate that is not among them.

// A device with the gate set of HiSEP-Q declares four QIS functions:
// RUN: qcc-opt %s -prep-to-qir='native-gates=__quantum__qis__h__body,__quantum__qis__x__body,__quantum__qis__cx__body,__quantum__qis__mz__body' \
// RUN:   | FileCheck %s --check-prefix=CHECK-NATIVE

// With no gate given, the pass declares the whole QIR gate set:
// RUN: qcc-opt %s -prep-to-qir | FileCheck %s --check-prefix=CHECK-ALL

// A gate outside the set of the device is an error:
// RUN: not qcc-opt %s -prep-to-qir='native-gates=__quantum__qis__h__body,__quantum__qis__mz__body' \
// RUN:   -convert-qc-to-qir='native-gates=__quantum__qis__h__body,__quantum__qis__mz__body' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERR

func.func @main() attributes { qcc.entry_point } {
  %0 = qc.static 0 : !qc.qubit
  qc.h %0 : !qc.qubit
  qc.t %0 : !qc.qubit
  %m0 = qc.measure %0 : !qc.qubit -> i1
  return
}

// CHECK-NATIVE-DAG: llvm.func @__quantum__qis__h__body
// CHECK-NATIVE-DAG: llvm.func @__quantum__qis__x__body
// CHECK-NATIVE-DAG: llvm.func @__quantum__qis__cx__body
// CHECK-NATIVE-DAG: llvm.func @__quantum__qis__mz__body
// CHECK-NATIVE-NOT: llvm.func @__quantum__qis__t__body
// CHECK-NATIVE-NOT: llvm.func @__quantum__qis__rz__body
// CHECK-NATIVE-NOT: llvm.func @__quantum__qis__reset__body

// CHECK-ALL-DAG: llvm.func @__quantum__qis__h__body
// CHECK-ALL-DAG: llvm.func @__quantum__qis__t__body
// CHECK-ALL-DAG: llvm.func @__quantum__qis__rz__body
// CHECK-ALL-DAG: llvm.func @__quantum__qis__reset__body

// CHECK-ERR: error: the target device does not implement '__quantum__qis__t__body'
