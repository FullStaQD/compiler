// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// RUN: qcc-opt %s -convert-qir-to-intrinsics | FileCheck %s

// Input: a module as produced by the ToQIR pipeline.
// Each qubit is an `!llvm.ptr` obtained via `llvm.inttoptr` of a constant index.
// QIS gate calls use those ptrs as operands.
//
// Expected output: every QIS call is replaced by an `llvm.call_intrinsic` call
// with the qubit encoded as a `vector<[1]xi8>` scalable vector in lane 0.

// ------------------------------------------------------------------
// Helper function declarations (produced by PrepToQIR).
// ------------------------------------------------------------------

llvm.func @__quantum__rt__initialize(!llvm.ptr) -> ()
llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1 attributes {arg_attrs = [{llvm.readonly}]}
llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr) -> ()
llvm.func @__quantum__qis__h__body(!llvm.ptr) -> ()
llvm.func @__quantum__qis__x__body(!llvm.ptr) -> ()
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr) -> ()
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) -> ()

llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}

// ------------------------------------------------------------------
// Single-qubit gates (H, X) lower to QVSingleIntrinsic:
// (vs1: vector<[1]xi8>, rs2: i32, block_imm: i32, vl: i32) -> void
// ------------------------------------------------------------------

llvm.func @single_qubit_gates() attributes { passthrough = ["entry_point"] } {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %q1 = llvm.inttoptr %c1 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__x__body(%q1) : (!llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @single_qubit_gates()
// CHECK-NOT:     llvm.call @__quantum__qis__h__body
// CHECK-NOT:     llvm.call @__quantum__qis__x__body
// CHECK:         %[[IDX0:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:         %[[UNDEF0:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[LANE0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC0:.*]] = llvm.insertelement %[[IDX0]], %[[UNDEF0]][%[[LANE0]] : i32] : vector<[1]xi8>
// CHECK:         %[[TAG0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[BLKIMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VL0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[VEC0]], %[[TAG0]], %[[BLKIMM0]], %[[VL0]])
// CHECK:         %[[IDX1:.*]] = llvm.mlir.constant(1 : i8) : i8
// CHECK:         %[[UNDEF1:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[LANE1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VEC1:.*]] = llvm.insertelement %[[IDX1]], %[[UNDEF1]][%[[LANE1]] : i32] : vector<[1]xi8>
// CHECK:         %[[TAG1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[BLKIMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VL1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"(%[[VEC1]], %[[TAG1]], %[[BLKIMM1]], %[[VL1]])

// ------------------------------------------------------------------
// CX gate lowers to QVPairIntrinsic:
// (vs1: vector<[1]xi8>, vs2: vector<[1]xi8>, block_imm: i32, vl: i32) -> void
// ------------------------------------------------------------------

llvm.func @cx_gate() attributes { passthrough = ["entry_point"] } {
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %ctrl = llvm.inttoptr %c2 : i64 to !llvm.ptr
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %tgt  = llvm.inttoptr %c3 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__cx__body(%ctrl, %tgt) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @cx_gate()
// CHECK-NOT:     llvm.call @__quantum__qis__cx__body
// CHECK:         %[[CIDX:.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK:         %[[CVEC:.*]] = llvm.insertelement %[[CIDX]]
// CHECK:         %[[TIDX:.*]] = llvm.mlir.constant(3 : i8) : i8
// CHECK:         %[[TVEC:.*]] = llvm.insertelement %[[TIDX]]
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"(%[[CVEC]], %[[TVEC]], {{.*}}, {{.*}})

// ------------------------------------------------------------------
// Measurement lowers to QVSingleIntrinsic; read_result becomes undef.
// ------------------------------------------------------------------

llvm.func @measurement() -> i1 attributes { passthrough = ["entry_point"] } {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %qptr = llvm.inttoptr %c0 : i64 to !llvm.ptr
  %rptr = llvm.inttoptr %c0 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__mz__body(%qptr, %rptr) : (!llvm.ptr, !llvm.ptr) -> ()
  %res = llvm.call @__quantum__rt__read_result(%rptr) : (!llvm.ptr) -> i1
  llvm.return %res : i1
}

// CHECK-LABEL: llvm.func @measurement()
// CHECK-NOT:     llvm.call @__quantum__qis__mz__body
// CHECK-NOT:     llvm.call @__quantum__rt__read_result
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"
// CHECK:         %[[UNDEF:.*]] = llvm.mlir.undef : i1
// CHECK:         llvm.return %[[UNDEF]] : i1

// ------------------------------------------------------------------
// Runtime init and record-output calls are erased.
// ------------------------------------------------------------------

llvm.func @rt_calls_erased() attributes { passthrough = ["entry_point"] } {
  %null = llvm.mlir.zero : !llvm.ptr
  llvm.call @__quantum__rt__initialize(%null) : (!llvm.ptr) -> ()

  %c0 = llvm.mlir.constant(0 : i64) : i64
  %qptr = llvm.inttoptr %c0 : i64 to !llvm.ptr
  llvm.call @__quantum__qis__x__body(%qptr) : (!llvm.ptr) -> ()

  %false = llvm.mlir.constant(0 : i1) : i1
  %label = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
  llvm.call @__quantum__rt__bool_record_output(%false, %label) : (i1, !llvm.ptr) -> ()

  llvm.return
}

// CHECK-LABEL: llvm.func @rt_calls_erased()
// CHECK-NOT:     llvm.call @__quantum__rt__initialize
// CHECK-NOT:     llvm.call @__quantum__rt__bool_record_output
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"

// ------------------------------------------------------------------
// QIR function declarations are removed from the module.
// ------------------------------------------------------------------

// CHECK-NOT: llvm.func @__quantum__qis__h__body
// CHECK-NOT: llvm.func @__quantum__qis__x__body
// CHECK-NOT: llvm.func @__quantum__qis__cx__body
// CHECK-NOT: llvm.func @__quantum__qis__mz__body
// CHECK-NOT: llvm.func @__quantum__rt__initialize
// CHECK-NOT: llvm.func @__quantum__rt__read_result
// CHECK-NOT: llvm.func @__quantum__rt__bool_record_output
