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

// Single-qubit gates (H, X) are lowered to QVSingleIntrinsic calls:
// (vs1: vector<1xi64>, tag: i64, block_imm: i64, vl: i64) -> void
// The qubit index is materialised as insertelement into a <1 x i64> vector.
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
// CHECK:         %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[UNDEF0:.*]] = llvm.mlir.undef : vector<1xi64>
// CHECK:         %[[LANE0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC0:.*]] = llvm.insertelement %[[IDX0]], %[[UNDEF0]][%[[LANE0]] : i32] : vector<1xi64>
// CHECK:         %[[TAG0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[WAIT0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[VL0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[VEC0]], %[[TAG0]], %[[WAIT0]], %[[VL0]])
// CHECK:         %[[IDX1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[UNDEF1:.*]] = llvm.mlir.undef : vector<1xi64>
// CHECK:         %[[LANE1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC1:.*]] = llvm.insertelement %[[IDX1]], %[[UNDEF1]][%[[LANE1]] : i32] : vector<1xi64>
// CHECK:         %[[TAG1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[WAIT1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[VL1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"(%[[VEC1]], %[[TAG1]], %[[WAIT1]], %[[VL1]])
// CHECK:         return

// The CX gate is a QVPairIntrinsic:
// (vs1: vector<1xi64>, vs2: vector<1xi64>, block_imm: i64, vl: i64) -> void
func.func @cx_gate() attributes { qcc.entry_point } {
    %ctrl = qc.static 2 : !qc.qubit
    %tgt  = qc.static 3 : !qc.qubit

    qc.ctrl(%ctrl) { qc.x %tgt : !qc.qubit } : !qc.qubit
    return
}
// CHECK-LABEL: func.func @cx_gate()
// CHECK-DAG:   %[[IDXC:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:   %[[UC:.*]] = llvm.mlir.undef : vector<1xi64>
// CHECK-DAG:   %[[LC:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   %[[VC:.*]] = llvm.insertelement %[[IDXC]], %[[UC]][%[[LC]] : i32] : vector<1xi64>
// CHECK-DAG:   %[[IDXT:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:   %[[UT:.*]] = llvm.mlir.undef : vector<1xi64>
// CHECK-DAG:   %[[LT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   %[[VT:.*]] = llvm.insertelement %[[IDXT]], %[[UT]][%[[LT]] : i32] : vector<1xi64>
// CHECK:       llvm.call_intrinsic "llvm.riscv.qv.cx"(%[[VC]], %[[VT]], {{.*}}, {{.*}})
// CHECK:       return

// qv.mz uses the same QVSingleIntrinsic signature; aux.record_int is erased.
// The measurement result is currently a placeholder undef pending qv.read_result
// being defined in IntrinsicsRISCVXQV.td.
func.func @measurement() attributes { qcc.entry_point } {
    %q0 = qc.static 0 : !qc.qubit

    %m = qc.measure %q0 : !qc.qubit -> i1
    aux.record_int %m : i1
    return
}
// CHECK-LABEL: func.func @measurement()
// CHECK:         %[[IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[UNDEF:.*]] = llvm.mlir.undef : vector<1xi64>
// CHECK:         %[[LANE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC:.*]] = llvm.insertelement %[[IDX]], %[[UNDEF]][%[[LANE]] : i32] : vector<1xi64>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"(%[[VEC]], {{.*}}, {{.*}}, {{.*}})
// CHECK:         llvm.mlir.undef : i1
// CHECK-NOT:     aux.record_int
// CHECK:         return
