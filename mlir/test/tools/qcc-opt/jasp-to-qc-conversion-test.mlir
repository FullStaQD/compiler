// RUN: qcc-opt %s --jasp-to-qc | FileCheck %s

func.func @test() -> tensor<i1> {
    %state0 = jasp.create_quantum_kernel -> !jasp.QuantumState
    // Quantum State management leaves no trace in QC IR.

    %num_qubits = arith.constant dense<1> : tensor<i64>
    %qubit_array, %state1 = jasp.create_qubits %num_qubits, %state0 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    // For now, we require the number of qubits to be a constant.
    // From these lines, we obtain the information that
    //  - There is a qubit array %qubit_array, which we need to give a name in qc.
    //  - The qubit array has one qubit.
    // Alternatively, we could convert the array allocation to the allocation
    // of all member qubits in QC.

    %qubit_index = arith.constant dense<0> : tensor<i64>
    %qubit_reference = jasp.get_qubit %qubit_array, %qubit_index : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    // The extraction of the qubit reference from the array corresponds to the
    // actual allocation in QC.
    // CHECK: [[qubit_reference:%.+]] = qc.alloc("qreg_[[register_hash:.+]]", 1, 0) : !qc.qubit

    %state2 = jasp.quantum_gate "h" (%qubit_reference), %state1 : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    // CHECK: qc.h [[qubit_reference]] : !qc.qubit

    %random_bit, %state3 = jasp.measure %qubit_reference, %state2 : !jasp.Qubit, !jasp.QuantumState -> tensor<i1>, !jasp.QuantumState
    // qc.measure returns i1 values, which need to be converted to tensor<i1>
    // to connect to the remaining code.
    // CHECK: [[random_bit:%.+]] = qc.measure [[qubit_reference]] : !qc.qubit
    // CHECK: [[random_bit_tensor:%.+]] = tensor.from_elements [[random_bit]] : tensor<i1>

    %state4 = jasp.delete_qubits %qubit_array, %state3 : !jasp.QubitArray, !jasp.QuantumState -> !jasp.QuantumState
    // CHECK: qc.dealloc [[qubit_reference]] : !qc.qubit

    %success = jasp.consume_quantum_kernel %state4 : !jasp.QuantumState -> tensor<i1>
    // We do not lower the functionality of returning a success value
    // for the kernel execution. A constant value of 1 is returned,
    // assuming a successful execution.
    // CHECK: [[success:%.+]] = arith.constant dense<1> : tensor<i1>

    return %random_bit : tensor<i1>
    // CHECK: return [[random_bit_tensor]] : tensor<i1>
}
