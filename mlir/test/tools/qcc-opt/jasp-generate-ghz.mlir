// RUN: qcc-opt %s | FileCheck %s

builtin.module @jasp_module {
  func.func private @main(%arg28 : tensor<i64>, %arg29 : !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %0, %1 = "jasp.create_qubits"(%arg28, %arg29) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3 = "jasp.get_qubit"(%0, %2) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    %4 = "jasp.quantum_gate"(%3, %1) {gate_type = "h"} : (!jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    %5 = func.call @tracerizer() : () -> tensor<i64>
    %6 = "stablehlo.convert"(%arg28) : (tensor<i64>) -> tensor<i64>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %8 = "stablehlo.subtract"(%6, %7) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %9 = "stablehlo.add"(%5, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %10, %11, %12, %13 = scf.while (%arg34 = %0, %arg35 = %8, %arg36 = %9, %arg37 = %4) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %14 = "stablehlo.compare"(%arg36, %arg35) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %15 = tensor.extract %14[] : tensor<i1>
      scf.condition(%15) %arg34, %arg35, %arg36, %arg37 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg30 : !jasp.QubitArray, %arg31 : tensor<i64>, %arg32 : tensor<i64>, %arg33 : !jasp.QuantumState):
      %16 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %17 = "stablehlo.add"(%arg31, %16) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %18 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %19 = "stablehlo.subtract"(%arg32, %18) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %20 = "jasp.get_qubit"(%arg30, %19) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
      %21 = "jasp.get_qubit"(%arg30, %arg32) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
      %22 = "jasp.quantum_gate"(%20, %21, %arg33) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
      %23 = "stablehlo.add"(%arg32, %18) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      scf.yield %arg30, %arg31, %23, %22 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    func.return %0, %13 : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @jasp.create_qubits(%arg26 : tensor<i64>, %arg27 : !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %0, %1 = "jasp.create_qubits"(%arg26, %arg27) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
    func.return %0, %1 : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @jasp.get_qubit(%arg24 : !jasp.QubitArray, %arg25 : tensor<i64>) -> !jasp.Qubit {
    %0 = "jasp.get_qubit"(%arg24, %arg25) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    func.return %0 : !jasp.Qubit
  }
  func.func private @jasp.quantum_gate(%arg22 : !jasp.Qubit, %arg23 : !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = "jasp.quantum_gate"(%arg22, %arg23) {gate_type = "h"} : (!jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    func.return %0 : !jasp.QuantumState
  }
  func.func private @tracerizer() -> (tensor<i64>) {
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    func.return %0 : tensor<i64>
  }
  func.func private @convert_element_type(%arg21 : tensor<i64>) -> tensor<i64> {
    %0 = "stablehlo.convert"(%arg21) : (tensor<i64>) -> tensor<i64>
    func.return %0 : tensor<i64>
  }
  func.func private @sub(%arg19 : tensor<i64>, %arg20 : tensor<i64>) -> tensor<i64> {
    %0 = "stablehlo.subtract"(%arg19, %arg20) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    func.return %0 : tensor<i64>
  }
  func.func private @add(%arg17 : tensor<i64>, %arg18 : tensor<i64>) -> tensor<i64> {
    %0 = "stablehlo.add"(%arg17, %arg18) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    func.return %0 : tensor<i64>
  }
  func.func private @while(%arg5 : !jasp.QubitArray, %arg6 : tensor<i64>, %arg7 : tensor<i64>, %arg8 : !jasp.QuantumState) -> (tensor<i64>, tensor<i64>, !jasp.QuantumState) {
    %0, %1, %2, %3 = scf.while (%arg13 = %arg5, %arg14 = %arg6, %arg15 = %arg7, %arg16 = %arg8) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %4 = "stablehlo.compare"(%arg15, %arg14) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %5 = tensor.extract %4[] : tensor<i1>
      scf.condition(%5) %arg13, %arg14, %arg15, %arg16 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg9 : !jasp.QubitArray, %arg10 : tensor<i64>, %arg11 : tensor<i64>, %arg12 : !jasp.QuantumState):
      %6 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %7 = "stablehlo.add"(%arg10, %6) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %8 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %9 = "stablehlo.subtract"(%arg11, %8) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %10 = "jasp.get_qubit"(%arg9, %9) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
      %11 = "jasp.get_qubit"(%arg9, %arg11) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
      %12 = "jasp.quantum_gate"(%10, %11, %arg12) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
      %13 = "stablehlo.add"(%arg11, %8) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      scf.yield %arg9, %arg10, %13, %12 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    func.return %1, %2, %3 : tensor<i64>, tensor<i64>, !jasp.QuantumState
  }
  func.func private @le(%arg3 : tensor<i64>, %arg4 : tensor<i64>) -> tensor<i1> {
    %0 = "stablehlo.compare"(%arg3, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    func.return %0 : tensor<i1>
  }
  func.func private @jasp.quantum_gate_0(%arg0 : !jasp.Qubit, %arg1 : !jasp.Qubit, %arg2 : !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = "jasp.quantum_gate"(%arg0, %arg1, %arg2) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    func.return %0 : !jasp.QuantumState
  }
}

// CHECK: module
