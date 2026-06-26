// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// RUN: qcc-opt %s -convert-qir-to-intrinsics | FileCheck %s

// Verifies batching behaviour for both the single-qubit (v1) and 8-qubit (v8)
// cases.  A single QIS call becomes a vector<[1]xi8> intrinsic call; N
// consecutive calls to the same gate are fused into a single vector<[N]xi8>
// intrinsic call.

llvm.func @__quantum__rt__initialize(!llvm.ptr) -> ()
llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1 attributes {arg_attrs = [{llvm.readonly}]}
llvm.func @__quantum__qis__h__body(!llvm.ptr) -> ()
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr) -> ()
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) -> ()

// -----------------------------------------------------------------------
// v1: a lone H call → vector<[1]xi8> carrying the one qubit index
// -----------------------------------------------------------------------

// CHECK-LABEL: llvm.func @single_h()
// CHECK-NOT:     llvm.call @__quantum__qis__h__body
// CHECK-DAG:     %[[IDX0:.*]] = llvm.mlir.constant(0 : i8)  : i8
// CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[ONE:.*]]  = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:     %[[UNDEF1:.*]] = llvm.mlir.undef : vector<[1]xi8>
// CHECK:         %[[VEC:.*]] = llvm.insertelement %[[IDX0]], %[[UNDEF1]][%[[ZERO]] : i32] : vector<[1]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[VEC]], %[[ZERO]], %[[ZERO]], %[[ONE]])
llvm.func @single_h() {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
  llvm.return
}

// -----------------------------------------------------------------------
// v1: a lone CX call → two vector<[1]xi8>
// -----------------------------------------------------------------------

// CHECK-LABEL: llvm.func @single_cx()
// CHECK-NOT:     llvm.call @__quantum__qis__cx__body
// CHECK-DAG:     %[[CIDX:.*]] = llvm.mlir.constant(0 : i8)  : i8
// CHECK-DAG:     %[[TIDX:.*]] = llvm.mlir.constant(1 : i8)  : i8
// CHECK-DAG:     %[[Z32:.*]]  = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[O32:.*]]  = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"({{.*}}, {{.*}}, %[[Z32]], %[[O32]])
llvm.func @single_cx() {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %ctrl = llvm.inttoptr %c0 : i64 to !llvm.ptr
  %tgt  = llvm.inttoptr %c1 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__cx__body(%ctrl, %tgt) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----------------------------------------------------------------------
// v8: 8 consecutive H calls → one vector<[8]xi8> intrinsic call
// -----------------------------------------------------------------------

// CHECK-LABEL: llvm.func @batch_h_8()
// CHECK-NOT:     llvm.call @__quantum__qis__h__body
// CHECK-DAG:     %[[EIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-DAG:     %[[ZERO8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[UNDEF8:.*]] = llvm.mlir.undef : vector<[8]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"({{.*}}, %[[ZERO8]], %[[ZERO8]], %[[EIGHT]])
llvm.func @batch_h_8() {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %c5 = llvm.mlir.constant(5 : i64) : i64
  %c6 = llvm.mlir.constant(6 : i64) : i64
  %c7 = llvm.mlir.constant(7 : i64) : i64

  %q0 = llvm.inttoptr %c0 : i64 to !llvm.ptr
  %q1 = llvm.inttoptr %c1 : i64 to !llvm.ptr
  %q2 = llvm.inttoptr %c2 : i64 to !llvm.ptr
  %q3 = llvm.inttoptr %c3 : i64 to !llvm.ptr
  %q4 = llvm.inttoptr %c4 : i64 to !llvm.ptr
  %q5 = llvm.inttoptr %c5 : i64 to !llvm.ptr
  %q6 = llvm.inttoptr %c6 : i64 to !llvm.ptr
  %q7 = llvm.inttoptr %c7 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__h__body(%q0) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q1) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q2) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q3) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q4) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q5) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q6) : (!llvm.ptr) -> ()
  llvm.call @__quantum__qis__h__body(%q7) : (!llvm.ptr) -> ()

  llvm.return
}

// -----------------------------------------------------------------------
// v8: 8 consecutive CX pairs → one vector<[8]xi8> intrinsic call
// -----------------------------------------------------------------------

// CHECK-LABEL: llvm.func @batch_cx_8()
// CHECK-NOT:     llvm.call @__quantum__qis__cx__body
// CHECK-DAG:     %[[EIGHT_CX:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-DAG:     %[[ZERO_CX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[UV8:.*]] = llvm.mlir.undef : vector<[8]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"({{.*}}, {{.*}}, %[[ZERO_CX]], %[[EIGHT_CX]])
llvm.func @batch_cx_8() {
  %c0  = llvm.mlir.constant(0  : i64) : i64
  %c1  = llvm.mlir.constant(1  : i64) : i64
  %c2  = llvm.mlir.constant(2  : i64) : i64
  %c3  = llvm.mlir.constant(3  : i64) : i64
  %c4  = llvm.mlir.constant(4  : i64) : i64
  %c5  = llvm.mlir.constant(5  : i64) : i64
  %c6  = llvm.mlir.constant(6  : i64) : i64
  %c7  = llvm.mlir.constant(7  : i64) : i64
  %c8  = llvm.mlir.constant(8  : i64) : i64
  %c9  = llvm.mlir.constant(9  : i64) : i64
  %c10 = llvm.mlir.constant(10 : i64) : i64
  %c11 = llvm.mlir.constant(11 : i64) : i64
  %c12 = llvm.mlir.constant(12 : i64) : i64
  %c13 = llvm.mlir.constant(13 : i64) : i64
  %c14 = llvm.mlir.constant(14 : i64) : i64
  %c15 = llvm.mlir.constant(15 : i64) : i64

  // ctrl = even indices (0,2,...,14), tgt = odd indices (1,3,...,15)
  %ctrl0 = llvm.inttoptr %c0  : i64 to !llvm.ptr
  %tgt0  = llvm.inttoptr %c1  : i64 to !llvm.ptr
  %ctrl1 = llvm.inttoptr %c2  : i64 to !llvm.ptr
  %tgt1  = llvm.inttoptr %c3  : i64 to !llvm.ptr
  %ctrl2 = llvm.inttoptr %c4  : i64 to !llvm.ptr
  %tgt2  = llvm.inttoptr %c5  : i64 to !llvm.ptr
  %ctrl3 = llvm.inttoptr %c6  : i64 to !llvm.ptr
  %tgt3  = llvm.inttoptr %c7  : i64 to !llvm.ptr
  %ctrl4 = llvm.inttoptr %c8  : i64 to !llvm.ptr
  %tgt4  = llvm.inttoptr %c9  : i64 to !llvm.ptr
  %ctrl5 = llvm.inttoptr %c10 : i64 to !llvm.ptr
  %tgt5  = llvm.inttoptr %c11 : i64 to !llvm.ptr
  %ctrl6 = llvm.inttoptr %c12 : i64 to !llvm.ptr
  %tgt6  = llvm.inttoptr %c13 : i64 to !llvm.ptr
  %ctrl7 = llvm.inttoptr %c14 : i64 to !llvm.ptr
  %tgt7  = llvm.inttoptr %c15 : i64 to !llvm.ptr

  llvm.call @__quantum__qis__cx__body(%ctrl0, %tgt0) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl1, %tgt1) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl2, %tgt2) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl3, %tgt3) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl4, %tgt4) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl5, %tgt5) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl6, %tgt6) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @__quantum__qis__cx__body(%ctrl7, %tgt7) : (!llvm.ptr, !llvm.ptr) -> ()

  llvm.return
}

// Declarations for all handled symbols are removed from the module.
// CHECK-NOT: llvm.func @__quantum__qis__h__body
// CHECK-NOT: llvm.func @__quantum__qis__cx__body
// CHECK-NOT: llvm.func @__quantum__qis__mz__body
// CHECK-NOT: llvm.func @__quantum__rt__read_result
// CHECK-NOT: llvm.func @__quantum__rt__initialize
