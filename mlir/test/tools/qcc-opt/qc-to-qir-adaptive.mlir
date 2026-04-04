// RUN: qcc-opt %s -qc-to-qir-adaptive | FileCheck %s

func.func @test() -> i64 {
    // FIXME: add better and more tests

    %0 = qc.alloc : !qc.qubit
    qc.h %0 : !qc.qubit
    %m0 = qc.measure %0 : !qc.qubit -> i1
    // recording

    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

func.func @other_func() { return }

// CHECK-LABEL: @test
