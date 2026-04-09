// RUN: qcc-opt %s -to-qir-prep | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// FIXME: update checks
// CHECK-LABEL:   func.func @test() -> i64 attributes {qcc.entry_point} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
// CHECK:           return %[[CONSTANT_0]] : i64
// CHECK:         }
// CHECK-DAG:     llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK-DAG:     llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
// CHECK-DAG:     llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr) attributes {passthrough = ["irreversible"]}
// CHECK-DAG:     llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK-DAG:     llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK-DAG:     llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
