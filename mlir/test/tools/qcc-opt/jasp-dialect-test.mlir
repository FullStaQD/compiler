// RUN: qcc-opt %s | FileCheck %s

func.func @test(%q0 : !jasp.Qubit, %state0: !jasp.QuantumState) -> () {
    %i1, %state1 = jasp.measure %q0, %state0 : !jasp.Qubit, !jasp.QuantumState -> i1
    // CHECK: jasp.measure
    return
}
