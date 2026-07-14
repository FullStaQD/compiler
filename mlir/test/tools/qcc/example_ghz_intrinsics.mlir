// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// RUN: qcc --target=hisep-q --compile-to=mlir -o - %s | FileCheck %s

/// Same GHZ circuit as example_ghz.mlir, compiled with --target=hisep-q to
/// verify the full pipeline replaces QIS calls with RISC-V QV intrinsics.

func.func @main() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    %1 = qc.static 1 : !qc.qubit
    %2 = qc.static 2 : !qc.qubit

    qc.h %0 : !qc.qubit

    qc.ctrl(%0) { qc.x %1 : !qc.qubit } : !qc.qubit
    qc.ctrl(%1) { qc.x %2 : !qc.qubit } : !qc.qubit

    %m0 = qc.measure %0 : !qc.qubit -> i1
    %m1 = qc.measure %1 : !qc.qubit -> i1
    %m2 = qc.measure %2 : !qc.qubit -> i1

    %a = arith.constant 42 : i64
    aux.record_int %a : i64

    aux.record_int %m0 : i1
    aux.record_int %m1 : i1
    aux.record_int %m2 : i1

    return
}

// CHECK-LABEL: llvm.func @main()
// CHECK-NOT:     llvm.call @__quantum__qis
// CHECK-NOT:     llvm.call @__quantum__rt
// CHECK-DAG:     llvm.mlir.addressof @".qcc_qv_idx_0" : !llvm.ptr
// CHECK-DAG:     llvm.mlir.addressof @".qcc_qv_idx_1" : !llvm.ptr
// CHECK-DAG:     llvm.mlir.addressof @".qcc_qv_idx_2" : !llvm.ptr
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.h"({{.*}})
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"({{.*}})
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.cx"({{.*}})
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"({{.*}})
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"({{.*}})
// CHECK:         llvm.call_intrinsic "llvm.riscv.qv.mz"({{.*}})
// `main` is the entry point of this module. It is called by `_start` (see below) and keeps an
// ordinary return.
// CHECK:         llvm.return

// The declarations of the lowered gates and rt functions are removed. The ones PrepToQIR declares
// but the GHZ circuit does not use (s, sdg, t, tdg, rz) have no intrinsic and stay.
// CHECK-NOT: llvm.func @__quantum__qis__h__body
// CHECK-NOT: llvm.func @__quantum__qis__x__body
// CHECK-NOT: llvm.func @__quantum__qis__cx__body
// CHECK-NOT: llvm.func @__quantum__qis__mz__body
// CHECK-NOT: llvm.func @__quantum__rt__initialize
// CHECK-NOT: llvm.func @__quantum__rt__read_result
// CHECK-NOT: llvm.func @__quantum__rt__bool_record_output
// CHECK-NOT: llvm.func @__quantum__rt__int_record_output

// `_start` supersedes `main` as the entry point of the hardware: it sets up the stack, calls
// `main`, and halts in an infinite loop if that call returns (see emitStartFunc).
// CHECK: llvm.mlir.global external constant @__stack_top()
// CHECK: llvm.func @_start()
// CHECK:   llvm.mlir.addressof @__stack_top : !llvm.ptr
// CHECK:   llvm.mlir.addressof @main : !llvm.ptr
// CHECK:   llvm.inline_asm{{.*}}"mv sp, $0\0Ajalr ra, 0($1)\0A1:\0Aj 1b"
// CHECK:   llvm.unreachable
