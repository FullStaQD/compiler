// RUN: qcc-opt %s | FileCheck %s

module @jasp_module {
  func.func private @main(%arg0: tensor<i64>, %arg1: !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %result, %qst_out = jasp.create_qubits %arg0, %arg1 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = jasp.get_qubit %result, %c : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %1 = jasp.quantum_gate "h"(%0), %qst_out : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    %2 = call @tracerizer() : () -> tensor<i64>
    %3 = stablehlo.convert %arg0 : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %4 = stablehlo.subtract %3, %c_0 : tensor<i64>
    %5 = stablehlo.add %2, %c : tensor<i64>
    %6:4 = scf.while (%arg2 = %result, %arg3 = %4, %arg4 = %5, %arg5 = %1) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %7 = stablehlo.compare  LE, %arg4, %arg3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %extracted = tensor.extract %7[] : tensor<i1>
      scf.condition(%extracted) %arg2, %arg3, %arg4, %arg5 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg2: !jasp.QubitArray, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: !jasp.QuantumState):
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %7 = stablehlo.add %arg3, %c_1 : tensor<i64>
      %c_2 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.subtract %arg4, %c_2 : tensor<i64>
      %9 = jasp.get_qubit %arg2, %8 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %10 = jasp.get_qubit %arg2, %arg4 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %11 = jasp.quantum_gate "cx"(%9, %10), %arg5 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
      %12 = stablehlo.add %arg4, %c_2 : tensor<i64>
      scf.yield %arg2, %arg3, %12, %11 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    return %result, %6#3 : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @jasp.create_qubits(%arg0: tensor<i64>, %arg1: !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %result, %qst_out = jasp.create_qubits %arg0, %arg1 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    return %result, %qst_out : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @jasp.get_qubit(%arg0: !jasp.QubitArray, %arg1: tensor<i64>) -> !jasp.Qubit {
    %0 = jasp.get_qubit %arg0, %arg1 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    return %0 : !jasp.Qubit
  }
  func.func private @jasp.quantum_gate(%arg0: !jasp.Qubit, %arg1: !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = jasp.quantum_gate "h"(%arg0), %arg1 : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    return %0 : !jasp.QuantumState
  }
  func.func private @tracerizer() -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    return %c : tensor<i64>
  }
  func.func private @convert_element_type(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.convert %arg0 : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @sub(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @add(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @while(%arg0: !jasp.QubitArray, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: !jasp.QuantumState) -> (tensor<i64>, tensor<i64>, !jasp.QuantumState) {
    %0:4 = scf.while (%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %1 = stablehlo.compare  LE, %arg6, %arg5,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %extracted = tensor.extract %1[] : tensor<i1>
      scf.condition(%extracted) %arg4, %arg5, %arg6, %arg7 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg4: !jasp.QubitArray, %arg5: tensor<i64>, %arg6: tensor<i64>, %arg7: !jasp.QuantumState):
      %c = stablehlo.constant dense<0> : tensor<i64>
      %1 = stablehlo.add %arg5, %c : tensor<i64>
      %c_0 = stablehlo.constant dense<1> : tensor<i64>
      %2 = stablehlo.subtract %arg6, %c_0 : tensor<i64>
      %3 = jasp.get_qubit %arg4, %2 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %4 = jasp.get_qubit %arg4, %arg6 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %5 = jasp.quantum_gate "cx"(%3, %4), %arg7 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
      %6 = stablehlo.add %arg6, %c_0 : tensor<i64>
      scf.yield %arg4, %arg5, %6, %5 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    return %0#1, %0#2, %0#3 : tensor<i64>, tensor<i64>, !jasp.QuantumState
  }
  func.func private @le(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1> {
    %0 = stablehlo.compare  LE, %arg0, %arg1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    return %0 : tensor<i1>
  }
  func.func private @jasp.quantum_gate_0(%arg0: !jasp.Qubit, %arg1: !jasp.Qubit, %arg2: !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = jasp.quantum_gate "cx"(%arg0, %arg1), %arg2 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    return %0 : !jasp.QuantumState
  }
}

// CHECK: module
