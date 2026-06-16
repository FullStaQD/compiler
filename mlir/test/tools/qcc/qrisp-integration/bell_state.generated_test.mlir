
// RUN: %S/integration_run.sh -n 5 -c qcc  %s | FileCheck %s

builtin.module @jasp_module {
  func.func public @main(%arg0 : !jasp.QuantumState) -> (tensor<i64>, !jasp.QuantumState) {
    %0 = arith.constant dense<2> : tensor<i64>
    %1, %2 = jasp.create_qubits %0, %arg0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %3 = arith.constant dense<0> : tensor<i64>
    %4 = jasp.get_qubit %1, %3 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %5 = jasp.quantum_gate "h" (%4) , %2 : (!jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %6 = arith.constant dense<1> : tensor<i64>
    %7 = jasp.get_qubit %1, %6 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %8 = jasp.quantum_gate "cx" (%4, %7) , %5 : (!jasp.Qubit, !jasp.Qubit) , !jasp.QuantumState -> !jasp.QuantumState
    %9, %10 = jasp.measure %1, %8 : !jasp.QubitArray, !jasp.QuantumState -> tensor<i64>, !jasp.QuantumState
    func.return %9, %10 : tensor<i64>, !jasp.QuantumState
  }
}

// CHECK-LABEL: ---QIR---
// CHECK: @__quantum__qis__h__body(ptr null)
// CHECK: @__quantum__qis__cx__body(ptr null, ptr inttoptr (i64 1 to ptr))
// CHECK-DAG: @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK-DAG: @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))

// CHECK-DAG: [[BIT0:%.+]] = call i1 @__quantum__rt__read_result(ptr null)
// CHECK-DAG: [[BIT1:%.+]] = call i1 @__quantum__rt__read_result(ptr inttoptr (i64 1 to ptr))
// CHECK-DAG: [[INT_RESULT_0:%.+]] = zext i1 [[BIT0]] to i64
// CHECK-DAG: [[INT_RESULT_1:%.+]] = zext i1 [[BIT1]] to i64

// CHECK-DAG: [[LEFT_SHIFTED_1:%.+]] = shl i64 [[INT_RESULT_1]], 1
// CHECK-DAG: [[COMBINED_INT:%.+]] = or i64 [[INT_RESULT_0]], [[LEFT_SHIFTED_1]]
// CHECK: @__quantum__rt__int_record_output(i64 [[COMBINED_INT]], ptr @.qir_dummy_label)


// CHECK-LABEL: ---OUTPUT-RECORDING---

// CHECK: START
// CHECK: METADATA entry_point
// CHECK: METADATA output_labeling_schema
// CHECK: METADATA qir_profiles
// CHECK: METADATA required_num_qubits 2
// CHECK: METADATA required_num_results 2

// We expect values 0 (|00> state) or 3 (|11> state).
// CHECK: OUTPUT INT {{[03]}} dummy_label
// CHECK: OUTPUT INT {{[03]}} dummy_label
// CHECK: OUTPUT INT {{[03]}} dummy_label
// CHECK: OUTPUT INT {{[03]}} dummy_label
// CHECK: OUTPUT INT {{[03]}} dummy_label
