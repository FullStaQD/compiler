// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// RUN: qcc-opt %s -convert-qc-to-intrinsics | FileCheck %s

// Functions without qcc.entry_point must be left unmodified.
func.func @not_an_entry_point() {
    %q0 = qc.static 0 : !qc.qubit
    qc.x %q0 : !qc.qubit
    return
}
// CHECK-LABEL: func.func @not_an_entry_point()
// CHECK:         qc.static 0
// CHECK:         qc.x

// Single-qubit gates (H, X) are lowered to the corresponding RISC-V intrinsics.
// Each use of a qubit value materializes a fresh inttoptr from the static index.
func.func @single_qubit_gates() attributes { qcc.entry_point } {
    %q0 = qc.static 0 : !qc.qubit
    %q1 = qc.static 1 : !qc.qubit

    qc.h %q0 : !qc.qubit
    qc.x %q1 : !qc.qubit
    return
}
// CHECK-LABEL: func.func @single_qubit_gates()
// CHECK-NOT:     qc.static
// CHECK-NOT:     qc.h
// CHECK-NOT:     qc.x
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[P0:.*]] = llvm.inttoptr %[[C0]] : i64 to !llvm.ptr
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[P0]])
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[P1:.*]] = llvm.inttoptr %[[C1]] : i64 to !llvm.ptr
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"(%[[P1]])
// CHECK:         return

// The CX gate is lowered to llvm.riscv.qv.cx with control before target.
func.func @cx_gate() attributes { qcc.entry_point } {
    %ctrl = qc.static 2 : !qc.qubit
    %tgt  = qc.static 3 : !qc.qubit

    qc.ctrl(%ctrl) { qc.x %tgt : !qc.qubit } : !qc.qubit
    return
}
// CHECK-LABEL: func.func @cx_gate()
// CHECK-DAG:   %[[CC:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:   %[[PC:.*]] = llvm.inttoptr %[[CC]] : i64 to !llvm.ptr
// CHECK-DAG:   %[[CT:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:   %[[PT:.*]] = llvm.inttoptr %[[CT]] : i64 to !llvm.ptr
// CHECK:       llvm.call_intrinsic "llvm.riscv.qv.cx"(%[[PC]], %[[PT]])
// CHECK:       return

// Measurements lower to qv.mz + qv.read_result; aux.record_int is erased.
func.func @measurement() attributes { qcc.entry_point } {
    %q0 = qc.static 0 : !qc.qubit

    %m = qc.measure %q0 : !qc.qubit -> i1
    aux.record_int %m : i1
    return
}
// CHECK-LABEL: func.func @measurement()
// CHECK-DAG:   %[[CQ:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:   %[[PQ:.*]] = llvm.inttoptr %[[CQ]] : i64 to !llvm.ptr
// CHECK-DAG:   %[[CR:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:   %[[PR:.*]] = llvm.inttoptr %[[CR]] : i64 to !llvm.ptr
// CHECK:       llvm.call_intrinsic "llvm.riscv.qv.mz"(%[[PQ]], %[[PR]])
// CHECK:       llvm.call_intrinsic "llvm.riscv.qv.read_result"(%[[PR]])
// CHECK-NOT:   aux.record_int
// CHECK:       return
