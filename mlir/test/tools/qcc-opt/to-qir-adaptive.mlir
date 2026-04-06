// RUN: qcc-opt %s -std-to-qir | FileCheck %s --check-prefix=CHECK_STD
// RUN: qcc-opt %s -qc-to-qir | FileCheck %s --check-prefix=CHECK_QC
// RUN: qcc-opt %s -pass-pipeline="builtin.module(func.func(std-to-qir,qc-to-qir),to-qir-finalize)" | FileCheck %s --check-prefix=CHECK_FULL

func.func @test() -> i64 attributes { qcc.entry_point } {
    // FIXME: add better and more tests

    %0 = qc.alloc : !qc.qubit
    qc.h %0 : !qc.qubit
    %m0 = qc.measure %0 : !qc.qubit -> i1

    cf.cond_br %m0, ^bb_true, ^bb_exit
    ^bb_true:
      qc.x %0 : !qc.qubit
      cf.br ^bb_exit
    ^bb_exit:

    // recording

    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

func.func @other_func() { return }

// CHECK_STD-LABEL: @test

// CHECK_QC-LABEL: @test

// CHECK_FULL-LABEL: @test
