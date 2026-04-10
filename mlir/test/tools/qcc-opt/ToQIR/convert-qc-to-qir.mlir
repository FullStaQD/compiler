// RUN: qcc-opt %s -convert-qc-to-qir | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %0 = qc.static 1 : !qc.qubit
    // FIXME: add more stuff

    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// The pass -qc-to-qir assumes that these decls already exist.
llvm.func @__quantum__rt__initialize(!llvm.ptr)

// FIXME: add checks
// CHECK-LABEL: @test
