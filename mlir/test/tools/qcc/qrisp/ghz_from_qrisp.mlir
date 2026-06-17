// RUN: qcc -o - %s | FileCheck %s
// #####################################
//            QRISP PROGRAM
// #####################################

// def main():
//     qv = QuantumVariable(3)

//     h(qv[0])
//     cx(qv[0], qv[1])
//     cx(qv[1], qv[2])

//     mes0 = measure(qv[0])
//     mes1 = measure(qv[1])
//     mes2 = measure(qv[2])

//     return


// jaspr = make_jaspr(main)()

// print(jaspr.to_mlir())

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

    aux.record_int %m0 : i1
    aux.record_int %m1 : i1
    aux.record_int %m2 : i1

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

// CHECK-LABEL:   llvm.func @main() attributes
// CHECK:           %[[LABEL_ADDR:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr

// CHECK-DAG:       %[[STATIC_2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       %[[STATIC_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[STATIC_0:.*]] = llvm.mlir.constant(0 : i64) : i64

// CHECK:           %[[ZERO:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.call @__quantum__rt__initialize(%[[ZERO]]) : (!llvm.ptr) -> ()

// CHECK:           %[[INTTOPTR_0:.*]] = llvm.inttoptr %[[STATIC_0]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__h__body(%[[INTTOPTR_0]]) : (!llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_1:.*]] = llvm.inttoptr %[[STATIC_1]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_0]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_2:.*]] = llvm.inttoptr %[[STATIC_2]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_1]], %[[INTTOPTR_2]]) : (!llvm.ptr, !llvm.ptr) -> ()

// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_0]], %[[INTTOPTR_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_0:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_0]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_1]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_1:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_1]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_2]], %[[INTTOPTR_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_2:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_2]]) : (!llvm.ptr) -> i1

// CHECK:           llvm.call @__quantum__rt__bool_record_output(%[[CALL_0]], %[[LABEL_ADDR]]) : (i1, !llvm.ptr) -> ()
// CHECK:           llvm.call @__quantum__rt__bool_record_output(%[[CALL_1]], %[[LABEL_ADDR]]) : (i1, !llvm.ptr) -> ()
// CHECK:           llvm.call @__quantum__rt__bool_record_output(%[[CALL_2]], %[[LABEL_ADDR]]) : (i1, !llvm.ptr) -> ()

// CHECK:           llvm.return

// CHECK:         }

// CHECK:         llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
// CHECK:         llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
// CHECK:         llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)

// CHECK:         llvm.module_flags
// CHECK:         llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}
