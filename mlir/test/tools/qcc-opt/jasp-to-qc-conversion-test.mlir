// RUN: qcc-opt %s -jasp-to-qc | FileCheck %s

// CHECK-LABEL: func @test_simple_circuit
func.func @test_simple_circuit() -> tensor<i1> {
    %state0 = jasp.create_quantum_kernel -> !jasp.QuantumState
    // Quantum State management leaves no trace in QC IR.

    %num_qubits = arith.constant dense<1> : tensor<i64>
    // CHECK: [[num_qubits_tensor:%.+]] = arith.constant dense<1> : tensor<i64>
    %qubit_array, %state1 = jasp.create_qubits %num_qubits, %state0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    // memref.alloc needs a size of index type, not of tensor type.
    // CHECK: [[num_qubits_i64:%.+]] = tensor.extract [[num_qubits_tensor]][] : tensor<i64>
    // CHECK: [[num_qubits:%.+]] = arith.index_cast [[num_qubits_i64]] : i64 to index
    // CHECK: [[qubit_array:%.+]] = memref.alloc([[num_qubits]]) : memref<?x!qc.qubit>

    %qubit_index = arith.constant dense<0> : tensor<i64>
    // CHECK: [[qubit_index_tensor:%.+]] = arith.constant dense<0> : tensor<i64>
    %qubit_reference = jasp.get_qubit %qubit_array, %qubit_index : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    // memref.load needs an index of index type, not of tensor type.
    // CHECK: [[qubit_index_i64:%.+]] = tensor.extract [[qubit_index_tensor]][] : tensor<i64>
    // CHECK: [[qubit_index:%.+]] = arith.index_cast [[qubit_index_i64]] : i64 to index
    // CHECK: [[qubit_reference:%.+]] = memref.load [[qubit_array]][[[qubit_index]]] : memref<?x!qc.qubit>

    %state2 = jasp.quantum_gate "h" (%qubit_reference), %state1 : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    // CHECK: qc.h [[qubit_reference]] : !qc.qubit

    %random_bit, %state3 = jasp.measure %qubit_reference, %state2 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    // CHECK: [[random_bit:%.+]] = qc.measure [[qubit_reference]] : !qc.qubit

    %state4 = jasp.delete_qubits %qubit_array, %state3 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState
    // CHECK: memref.dealloc [[qubit_array]] : memref<?x!qc.qubit>

    %success = jasp.consume_quantum_kernel %state4 : !jasp.QuantumState -> tensor<i1>
    // We do not lower the functionality of returning a success value
    // for the kernel execution. A constant value of 1 is returned,
    // assuming a successful execution.
    // CHECK: [[success:%.+]] = arith.constant true

    return %random_bit : tensor<i1>
    // CHECK: return [[random_bit]] : i1
}

// -----

/// Test that measuring a qubit array returns a packed i64 with individual
/// bits unrolled to static indices when the array size is a compile-time
/// constant.
// CHECK-LABEL: func @test_arr_measure
func.func @test_arr_measure() -> tensor<i64> {
    %state0 = jasp.create_quantum_kernel -> !jasp.QuantumState
    %num_qubits = arith.constant dense<2> : tensor<i64>
    // CHECK: [[c0_i64:%.+]] = arith.constant 0 : i64

    %q_arr, %state1 = jasp.create_qubits %num_qubits, %state0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    // The qubit array measurement is unrolled: each qubit is loaded,
    // measured, zero-extended to i64, shifted by its index, and OR-ed
    // into the accumulator.
    %result, %state2 = jasp.measure %q_arr, %state1 : !jasp.QubitArray, !jasp.QuantumState -> tensor<i64>, !jasp.QuantumState
    // CHECK: [[q0:%.+]] = memref.load %{{.+}}[%c0]
    // CHECK: [[b0:%.+]] = qc.measure [[q0]]
    // CHECK: [[x0:%.+]] = arith.extui [[b0]] : i1 to i64
    // CHECK: arith.shli [[x0]], %c0_i64_0
    // CHECK: arith.ori [[c0_i64]], {{%.+}}

    // CHECK: [[q1:%.+]] = memref.load %{{.+}}[%c1]
    // CHECK: [[b1:%.+]] = qc.measure [[q1]]
    // CHECK: [[x1:%.+]] = arith.extui [[b1]] : i1 to i64
    // CHECK: [[s1:%.+]] = arith.shli [[x1]], %c1_i64
    // CHECK: [[r:%.+]] = arith.ori {{%.+}}, [[s1]]

    %state3 = jasp.delete_qubits %q_arr, %state2 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState
    // CHECK: memref.dealloc

    %success = jasp.consume_quantum_kernel %state3 : !jasp.QuantumState -> tensor<i1>
    // CHECK: arith.constant true

    return %result : tensor<i64>
    // CHECK: return [[r]] : i64
}
