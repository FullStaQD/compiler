// RUN: qcc -o - %s | FileCheck %s

builtin.module @jasp_module {
  func.func public @main(%arg0 : !jasp.QuantumState) -> (tensor<i1>, !jasp.QuantumState) attributes {qcc.entry_point} {
    %0 = arith.constant dense<2> : tensor<i64>
    %1, %2 = jasp.create_qubits %0, %arg0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %3 = arith.constant dense<0> : tensor<i64>
    %4 = jasp.get_qubit %1, %3 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %5 = jasp.quantum_gate "h" (%4) , %2 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %6 = jasp.quantum_gate "t" (%4) , %5 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %7 = arith.constant dense<1> : tensor<i64>
    %8 = jasp.get_qubit %1, %7 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %9 = jasp.quantum_gate "cx" (%4, %8) , %6 : (!jasp.Qubit, !jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %10, %11 = jasp.measure %4, %9 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    %12 = tensor.extract %10[] : tensor<i1>
    %13 = arith.constant true
    %14 = arith.xori %12, %13 : i1
    %15 = scf.if %14 -> (!jasp.QuantumState) {
      scf.yield %11 : !jasp.QuantumState
    } else {
      %16 = jasp.quantum_gate "s" (%8) , %11 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
      scf.yield %16 : !jasp.QuantumState
    }
    %17 = jasp.quantum_gate "t_dg" (%8) , %15 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %18, %19 = jasp.measure %8, %17 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    func.return %18, %19 : tensor<i1>, !jasp.QuantumState
  }
}

// CHECK-LABEL: llvm.func @main() attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "2"], ["required_num_results", "2"]]} {
// CHECK:    %[[ADDRESS:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
// CHECK:    %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[CONSTANT_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:    %[[ZERO:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:    llvm.call @__quantum__rt__initialize(%[[ZERO]]) : (!llvm.ptr) -> ()

// CHECK:    %[[INTTOPTR_0:.*]] = llvm.inttoptr %[[CONSTANT_0]] : i64 to !llvm.ptr
// CHECK:    llvm.call @__quantum__qis__h__body(%[[INTTOPTR_0]]) : (!llvm.ptr) -> ()
// CHECK:    llvm.call @__quantum__qis__t__body(%[[INTTOPTR_0]]) : (!llvm.ptr) -> ()
// CHECK:    %[[INTTOPTR_1:.*]] = llvm.inttoptr %[[CONSTANT_1]] : i64 to !llvm.ptr
// CHECK:    llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_0]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:    llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_0]], %[[INTTOPTR_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:    %[[READ_RESULT_0:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_0]]) : (!llvm.ptr) -> i1
// CHECK:    llvm.cond_br %[[READ_RESULT_0]], ^bb1, ^bb2
// CHECK:  ^bb1:  // pred: ^bb0
// CHECK:    llvm.call @__quantum__qis__s__body(%[[INTTOPTR_1]]) : (!llvm.ptr) -> ()
// CHECK:    llvm.br ^bb2
// CHECK:  ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:    llvm.call @__quantum__qis__tdg__body(%[[INTTOPTR_1]]) : (!llvm.ptr) -> ()
// CHECK:    llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_1]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:    %[[READ_RESULT_1:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_1]]) : (!llvm.ptr) -> i1
// CHECK:    llvm.call @__quantum__rt__bool_record_output(%[[READ_RESULT_1]], %[[ADDRESS]]) : (i1, !llvm.ptr) -> ()
// CHECK:    llvm.return
// CHECK:  }
// CHECK:  llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK:  llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
// CHECK:  llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
// CHECK:  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
// CHECK:  llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__s__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__sdg__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__t__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__tdg__body(!llvm.ptr)
// CHECK:  llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
// CHECK:  llvm.module_flags [#llvm.mlir.module_flag<error, "qir_major_version", 2 : i32>, #llvm.mlir.module_flag<max, "qir_minor_version", 1 : i32>, #llvm.mlir.module_flag<error, "dynamic_qubit_management", 0 : i32>, #llvm.mlir.module_flag<error, "dynamic_result_management", 0 : i32>, #llvm.mlir.module_flag<error, "ir_functions", 1 : i32>, #llvm.mlir.module_flag<error, "backwards_branching", 1 : i32>, #llvm.mlir.module_flag<error, "multiple_target_branching", 0 : i32>, #llvm.mlir.module_flag<error, "multiple_return_points", 0 : i32>]
// CHECK:  llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}
