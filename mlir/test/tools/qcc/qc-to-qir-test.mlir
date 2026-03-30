// RUN: qcc -o - %s | FileCheck %s

// CHECK-LABEL: llvm.func @main()
// CHECK-SAME: -> i64
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @__quantum__rt__initialize(%[[NULL]])

// CHECK: ^bb1:
// CHECK: %[[QUBIT:.*]] = llvm.inttoptr %[[ZERO]] : i64 to !llvm.ptr
// CHECK: llvm.call @__quantum__qis__x__body(%[[QUBIT]])

// CHECK: llvm.return %[[ZERO]] : i64

// CHECK: llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK: llvm.func @__quantum__qis__x__body(!llvm.ptr)
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %0 = qc.alloc("q", 1, 0) : !qc.qubit
    qc.x %0 : !qc.qubit
    qc.dealloc %0 : !qc.qubit
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
}
