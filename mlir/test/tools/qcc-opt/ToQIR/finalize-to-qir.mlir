// finalize-to-qir.mlir
//
// Copyright (c) 2026 FullStaQD Project
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: qcc-opt %s -finalize-to-qir | FileCheck %s

func.func @main() -> i64 attributes { qcc.entry_point, passthrough = [ "some_attr" ] } {
    %0 = call @get_zero() : () -> i64
    return %0 : i64
}

func.func @get_zero() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    return %0 : i64
}

// CHECK-LABEL:   llvm.func @main() -> i64 attributes {passthrough = ["some_attr"]} {
// CHECK:           %[[CALL_0:.*]] = llvm.call @get_zero() : () -> i64
// CHECK:           llvm.return %[[CALL_0]] : i64
// CHECK:         }

// CHECK-LABEL:   llvm.func @get_zero() -> i64 {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           llvm.return %[[MLIR_0]] : i64
// CHECK:         }
