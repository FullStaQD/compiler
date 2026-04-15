// RUN: qcc-opt %s -jasp-to-qc | FileCheck %s

func.func @test() -> tensor<i1> {
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
