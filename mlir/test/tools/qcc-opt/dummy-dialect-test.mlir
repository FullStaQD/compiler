// RUN: qcc-opt %s
func.func @test(%0 : !dummy.qubit) -> () {
    %1 = dummy.x %0 : !dummy.qubit
    return
}
