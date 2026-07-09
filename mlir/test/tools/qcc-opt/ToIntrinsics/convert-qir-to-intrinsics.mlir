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
// with the qubit encoded as a `vector<[4]xi8>` scalable vector, loaded from a
// dedicated 4-byte global holding that index in its first byte (HiSEP-Q's
// vector unit does not implement `vmv.s.x`/`vmv.v.i`, so the index cannot be
// synthesized in-register; it's 4 elements wide, not 1, because this target's
// ELEN=32 makes a 1-element vector legalize to the illegal LMUL=mf8).
// One global is created per distinct index and shared across all call sites in
// the module.

llvm.func @__quantum__rt__initialize(!llvm.ptr) -> ()
llvm.func @__quantum__rt__read_result(!llvm.ptr) -> i1 attributes {arg_attrs = [{llvm.readonly}]}
llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr) -> ()
llvm.func @__quantum__qis__h__body(!llvm.ptr) -> ()
llvm.func @__quantum__qis__x__body(!llvm.ptr) -> ()
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr) -> ()
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) -> ()

llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}


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
// CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:     %[[ADDR0:.*]] = llvm.mlir.addressof @".qcc_qv_idx_0" : !llvm.ptr
// CHECK-DAG:     %[[ADDR1:.*]] = llvm.mlir.addressof @".qcc_qv_idx_1" : !llvm.ptr
// CHECK:         %[[VEC0:.*]] = llvm.load %[[ADDR0]] : !llvm.ptr -> vector<[4]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"(%[[VEC0]], %[[ZERO]], %[[ZERO]], %[[ONE]])
// CHECK:         %[[VEC1:.*]] = llvm.load %[[ADDR1]] : !llvm.ptr -> vector<[4]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.x"(%[[VEC1]], %[[ZERO]], %[[ZERO]], %[[ONE]])


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
// CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:     %[[CADDR:.*]] = llvm.mlir.addressof @".qcc_qv_idx_2" : !llvm.ptr
// CHECK-DAG:     %[[TADDR:.*]] = llvm.mlir.addressof @".qcc_qv_idx_3" : !llvm.ptr
// CHECK-DAG:     %[[CVEC:.*]] = llvm.load %[[CADDR]] : !llvm.ptr -> vector<[4]xi8>
// CHECK-DAG:     %[[TVEC:.*]] = llvm.load %[[TADDR]] : !llvm.ptr -> vector<[4]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"(%[[CVEC]], %[[TVEC]], %[[ZERO]], %[[ONE]])


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
// CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:     %[[UNDEF_I1:.*]] = llvm.mlir.undef : i1
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @".qcc_qv_idx_0" : !llvm.ptr
// CHECK:         %[[VEC:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> vector<[4]xi8>
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"(%[[VEC]], %[[ZERO]], %[[ZERO]], %[[ONE]])
// The entry point has no caller, so its `llvm.return` is replaced with an infinite
// self-branch "halt here" idiom rather than an actual `ret` (see `haltEntryPoint`).
// CHECK-NOT:     llvm.return
// CHECK:         llvm.br ^[[HALT:.*]]
// CHECK:       ^[[HALT]]:
// CHECK:         llvm.br ^[[HALT]]


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

// CHECK-NOT: llvm.func @__quantum__qis__h__body
// CHECK-NOT: llvm.func @__quantum__qis__x__body
// CHECK-NOT: llvm.func @__quantum__qis__cx__body
// CHECK-NOT: llvm.func @__quantum__qis__mz__body
// CHECK-NOT: llvm.func @__quantum__rt__initialize
// CHECK-NOT: llvm.func @__quantum__rt__read_result
// CHECK-NOT: llvm.func @__quantum__rt__bool_record_output

// One shared internal-linkage global per distinct qubit index, reused across
// call sites (single_qubit_gates and measurement/rt_calls_erased both use
// index 0, so only four globals total exist for indices 0-3).
// CHECK-DAG: llvm.mlir.global internal constant @".qcc_qv_idx_0"("\00\00\00\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global internal constant @".qcc_qv_idx_1"("\01\00\00\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global internal constant @".qcc_qv_idx_2"("\02\00\00\00") {addr_space = 0 : i32}
// CHECK-DAG: llvm.mlir.global internal constant @".qcc_qv_idx_3"("\03\00\00\00") {addr_space = 0 : i32}
