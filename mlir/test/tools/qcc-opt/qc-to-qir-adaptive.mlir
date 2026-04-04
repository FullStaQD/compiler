// RUN: qcc-opt %s -qc-to-qir-adaptive | FileCheck %s

func.func @test() {
    // FIXME: add something here.
    return
}

// CHECK-LABEL: @test
