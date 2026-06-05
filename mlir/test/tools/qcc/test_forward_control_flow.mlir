// RUN: qcc -o - %s | FileCheck %s
// #####################################
//            QRISP PROGRAM
// #####################################

// def example():
//     # n = 5
//     t = 0

//     q_var = QuantumVariable(n)
//     h(q_var[0])
//     m = measure(q_var)

//     with control(m == 1):
//         t += 1

//     return t

// if __name__ == "__main__":
//     x = make_jaspr(example)().to_mlir(lower_stablehlo=True)
//     print(x)

// #####################################
//                  END
// #####################################

// #TODO: remove all this manual transformations
// The snippet of python code will produce most of the `mlir` code you find pasted below here
// The steps you need to do to go from the python output to the mlir you find here are the following.
// - Replace all the stablehlo occurrences with arith. I simply did a find + replace
// - Add the `tensor.extract` operations to obtain plain values from the measurements resulting tensors.
// - Add the `aux.record_bool` instructions
// - Add the ` attributes { qcc.entry_point } ` annotation in the function header

func.func public @main(%arg14: !jasp.QuantumState) attributes { qcc.entry_point } {
    %0 = "arith.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %1, %2 = "jasp.create_qubits"(%0, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
    %3 = "arith.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %4 = "jasp.get_qubit"(%1, %3) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    %5 = "jasp.quantum_gate"(%4, %2) {gate_type = "h"} : (!jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    %6 = "arith.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %7 = "jasp.get_qubit"(%1, %6) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    %8 = "jasp.quantum_gate"(%4, %7, %5) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    %9 = "arith.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %10 = "jasp.get_qubit"(%1, %9) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    %11 = "jasp.quantum_gate"(%7, %10, %8) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    %12, %13 = "jasp.measure"(%4, %11) : (!jasp.Qubit, !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState)
    %14, %15 = "jasp.measure"(%7, %13) : (!jasp.Qubit, !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState)
    %16, %17 = "jasp.measure"(%10, %15) : (!jasp.Qubit, !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState)

    %m0 = tensor.extract %12[] : tensor<i1>
    %m1 = tensor.extract %14[] : tensor<i1>
    %m2 = tensor.extract %16[] : tensor<i1>

    %18 = arith.cmpi eq, %m0, %m1 : i1
    %19 = arith.constant true
    %20 = arith.xori %18, %19 : i1
    %21 = scf.if %20 -> (i1) {
      scf.yield %m2 : i1
    } else {
      scf.yield %m2 : i1
    }

    aux.record_bool %m0
    aux.record_bool %m1
    aux.record_bool %m2

    func.return
  }
  func.func private @jasp.create_qubits(%arg12: tensor<i64>, %arg13: !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %0, %1 = "jasp.create_qubits"(%arg12, %arg13) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
    func.return %0, %1 : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @jasp.get_qubit(%arg10: !jasp.QubitArray, %arg11: tensor<i64>) -> !jasp.Qubit {
    %0 = "jasp.get_qubit"(%arg10, %arg11) : (!jasp.QubitArray, tensor<i64>) -> !jasp.Qubit
    func.return %0 : !jasp.Qubit
  }
  func.func private @jasp.quantum_gate(%arg8: !jasp.Qubit, %arg9: !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = "jasp.quantum_gate"(%arg8, %arg9) {gate_type = "h"} : (!jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    func.return %0 : !jasp.QuantumState
  }
  func.func private @jasp.quantum_gate_0(%arg5: !jasp.Qubit, %arg6: !jasp.Qubit, %arg7: !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = "jasp.quantum_gate"(%arg5, %arg6, %arg7) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    func.return %0 : !jasp.QuantumState
  }
  func.func private @jasp.quantum_gate_1(%arg2: !jasp.Qubit, %arg3: !jasp.Qubit, %arg4: !jasp.QuantumState) -> !jasp.QuantumState {
    %0 = "jasp.quantum_gate"(%arg2, %arg3, %arg4) {gate_type = "cx"} : (!jasp.Qubit, !jasp.Qubit, !jasp.QuantumState) -> !jasp.QuantumState
    func.return %0 : !jasp.QuantumState
  }
  func.func private @jasp.measure(%arg0: !jasp.Qubit, %arg1: !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState) {
    %0, %1 = "jasp.measure"(%arg0, %arg1) : (!jasp.Qubit, !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState)
    func.return %0, %1 : tensor<i1>, !jasp.QuantumState
  }
