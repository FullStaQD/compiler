// RUN: qcc %s | FileCheck %s

builtin.module @jasp_module {
  func.func public @main(%arg0 : !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState) {
    %0 = arith.constant dense<3> : tensor<i64>
    %1, %2 = jasp.create_qubits %0, %arg0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %3 = arith.constant dense<0> : tensor<i64>
    %4 = jasp.get_qubit %1, %3 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %5 = jasp.quantum_gate "h" (%4) , %2 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %6 = arith.constant dense<1> : tensor<i64>
    %7, %8, %9, %10 = scf.while (%arg9 = %1, %arg10 = %6, %arg11 = %0, %arg12 = %5) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %11 = tensor.extract %arg10[] : tensor<i64>
      %12 = tensor.extract %arg11[] : tensor<i64>
      %13 = arith.cmpi slt, %11, %12 : i64
      scf.condition(%13) %arg9, %arg10, %arg11, %arg12 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg1 : !jasp.QubitArray, %arg2 : tensor<i64>, %arg3 : tensor<i64>, %arg4 : !jasp.QuantumState):
      %14 = arith.constant 1 : i64
      %15 = tensor.extract %arg2[] : tensor<i64>
      %16 = arith.subi %15, %14 : i64
      %17 = tensor.from_elements %16 : tensor<i64>
      %18 = jasp.get_qubit %arg1, %17 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %19 = jasp.get_qubit %arg1, %arg2 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %20 = jasp.quantum_gate "cx" (%18, %19) , %arg4 : (!jasp.Qubit, !jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      %21 = arith.constant 1 : i64
      %22 = tensor.extract %arg2[] : tensor<i64>
      %23 = arith.addi %22, %21 : i64
      %24 = tensor.from_elements %23 : tensor<i64>
      scf.yield %arg1, %24, %arg3, %20 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }

    %measure_qubit = jasp.get_qubit %7, %3 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %25, %26 = jasp.measure %measure_qubit, %10 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    func.return %25, %26 : tensor<i1>, !jasp.QuantumState
  }
}

// TODO: Once loop unrolling is in place, add proper checks
// CHECK: qubit index must be a constant; unroll loops before this pass
