// RUN: qcc-opt %s -std-to-qir | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// FIXME: update checks
// CHECK-LABEL:   func.func @test
