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

// Single-qubit gates (H, X) lower to QVSingleIntrinsic calls:
// (vs1: vector<[1]xi8>, rs2: i32, block_imm: i32, vl: i32) -> void
// The qubit index is placed in lane 0 of a scalable <vscale x 1 x i8> register.
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
// CHECK:         %[[IDX0:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:         %[[UNDEF0:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[LANE0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC0:.*]] = llvm.insertelement %[[IDX0]], %[[UNDEF0]][%[[LANE0]] : i32] : vector<[1]xi8>
// CHECK:         %[[TAG0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[WAIT0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VL0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[VEC0]], %[[TAG0]], %[[WAIT0]], %[[VL0]])
// CHECK:         %[[IDX1:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:         %[[UNDEF1:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[LANE1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC1:.*]] = llvm.insertelement %[[IDX1]], %[[UNDEF1]][%[[LANE1]] : i32] : vector<[1]xi8>
// CHECK:         %[[TAG1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[WAIT1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VL1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"(%[[VEC1]], %[[TAG1]], %[[WAIT1]], %[[VL1]])
// CHECK:         return

// The CX gate is a QVPairIntrinsic:
// (vs1: vector<[1]xi8>, vs2: vector<[1]xi8>, block_imm: i32, vl: i32) -> void
func.func @cx_gate() attributes { qcc.entry_point } {
    %ctrl = qc.static 2 : !qc.qubit
    %tgt  = qc.static 3 : !qc.qubit

    qc.ctrl(%ctrl) { qc.x %tgt : !qc.qubit } : !qc.qubit
    return
}
// CHECK-LABEL: func.func @cx_gate()
// CHECK-DAG:   %[[IDXC:.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK-DAG:   %[[UC:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK-DAG:   %[[LC:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   %[[VC:.*]] = llvm.insertelement %[[IDXC]], %[[UC]][%[[LC]] : i32] : vector<[1]xi8>
// CHECK-DAG:   %[[IDXT:.*]] = llvm.mlir.constant(3 : i8) : i8
// CHECK-DAG:   %[[UT:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK-DAG:   %[[LT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:   %[[VT:.*]] = llvm.insertelement %[[IDXT]], %[[UT]][%[[LT]] : i32] : vector<[1]xi8>
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
// CHECK:         %[[IDX:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:         %[[UNDEF:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[LANE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC:.*]] = llvm.insertelement %[[IDX]], %[[UNDEF]][%[[LANE]] : i32] : vector<[1]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"(%[[VEC]], {{.*}}, {{.*}}, {{.*}})
// CHECK:         llvm.mlir.undef : i1
// CHECK-NOT:     aux.record_int
// CHECK:         return
