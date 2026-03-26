// RUN: qcc-opt %s --jasp-to-qc | FileCheck %s

func.func @test() -> tensor<i1> {
    // Create Quantum State
    %state0 = jasp.create_quantum_kernel -> !jasp.QuantumState

    // Create QubitArray with one Qubit
    %num_qubits = arith.constant dense<1> : tensor<i64>
    %qubit_array, %state1 = jasp.create_qubits %num_qubits, %state0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState

    // Extract a reference to the zeroth Qubit
    %qubit_index = arith.constant dense<0> : tensor<i64>
    %qubit_reference = jasp.get_qubit %qubit_array, %qubit_index : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit

    // Perform a Hadamard gate
    %state2 = jasp.quantum_gate "h" (%qubit_reference), %state1 : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState

    // Measure
    %random_bit, %state3 = jasp.measure %qubit_reference, %state2 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState

    // Deallocate QubitArray
    %state4 = jasp.delete_qubits %qubit_array, %state3 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState

    // Terminate Quantum State
    %success = jasp.consume_quantum_kernel %state4 : !jasp.QuantumState -> tensor<i1>

    // Return measured bit
    return %random_bit : tensor<i1>
}

// CHECK-LABEL: func.func @test() -> tensor<i1>
