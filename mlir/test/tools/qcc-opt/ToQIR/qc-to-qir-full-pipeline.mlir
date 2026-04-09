// RUN: qcc-opt %s -pass-pipeline="builtin.module(to-qir-prep,func.func(std-to-qir,qc-to-qir),to-qir-finalize)" | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %0 = qc.static 1 : !qc.qubit
    // FIXME: add more stuff

    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// CHECK-LABEL: @test
