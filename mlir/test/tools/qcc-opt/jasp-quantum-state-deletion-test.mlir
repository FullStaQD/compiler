// RUN: qcc-opt %s --jasp-to-qc | FileCheck %s

// Test that !jasp.QuantumState types are correctly removed from function signatures and return ops.

func.func @test(%1: i32, %state: !jasp.QuantumState, %2: i32) -> (i32, !jasp.QuantumState) {
    %3 = arith.addi %1, %2 : i32
    return %3, %state : i32, !jasp.QuantumState
}

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:                    %[[ARG0:.*]]: i32,
// CHECK-NOT:                     !jasp.QuantumState,
// CHECK-SAME:                    %[[ARG1:.*]]: i32) -> i32 {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK:           return %[[ADDI_0]] : i32
// CHECK:         }
