// RUN: qcc-opt %s -finalize-to-qir | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %0 = llvm.mlir.constant(0 : i64) : i64
    return %0 : i64
}

// FIXME: update checks
// CHECK-LABEL:   llvm.func
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           llvm.return %[[MLIR_0]] : i64
// CHECK:         }
