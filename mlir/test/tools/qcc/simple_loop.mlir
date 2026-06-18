// RUN: qcc %s | FileCheck %s

// TODO: replace with actual GHZ test measuring the whole qubit array.

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

// CHECK-LABEL:   llvm.func @main() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "3"], ["required_num_results", "3"]]} {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[MLIR_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[MLIR_4:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.call @__quantum__rt__initialize(%[[MLIR_4]]) : (!llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_0:.*]] = llvm.inttoptr %[[MLIR_3]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__h__body(%[[INTTOPTR_0]]) : (!llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_1:.*]] = llvm.inttoptr %[[MLIR_2]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_0]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_2:.*]] = llvm.inttoptr %[[MLIR_1]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_1]], %[[INTTOPTR_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_0]], %[[INTTOPTR_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_0:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_0]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.call @__quantum__rt__bool_record_output(%[[CALL_0]], %[[MLIR_0]]) : (i1, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__int_record_output(i64, !llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
// CHECK:         llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
// CHECK:         llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__s__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__sdg__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__t__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__tdg__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
// CHECK:         llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 0 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 0 : i32>, #llvm.mlir.module_flag<error, "ir_functions", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 1 : i32>, #llvm.mlir.module_flag<error, "multiple_target_branching", 0 : i32>, #llvm.mlir.module_flag<error, "multiple_return_points", 0 : i32>]
// CHECK:         llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}
