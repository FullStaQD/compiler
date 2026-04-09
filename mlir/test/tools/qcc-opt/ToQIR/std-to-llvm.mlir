// RUN: qcc-opt %s -std-to-llvm | FileCheck %s

func.func @main() -> i64 attributes { qcc.entry_point } {
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

func.func @aux(%i : i64) -> i64 attributes { qcc.entry_point } {
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// FIXME: update checks
// CHECK-LABEL:   func.func @main
