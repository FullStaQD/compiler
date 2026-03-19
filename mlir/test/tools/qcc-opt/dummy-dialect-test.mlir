// RUN: qcc-opt %s | FileCheck %s

func.func @test(%0 : !dummy.qubit) -> () {
    %1 = dummy.x %0 : !dummy.qubit
    // CHECK: dummy.x
    return
}
