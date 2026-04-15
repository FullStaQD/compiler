// RUN: qcc -o - %s | FileCheck %s

// This test corresponds to the following qrisp program:
//
// def generate_ghz(n):
//     # 1. Create a QuantumVariable with n qubits
//     q_var = QuantumVariable(n)
//
//     # 2. Apply Hadamard to the first qubit to start the superposition
//     h(q_var[0])
//
//     # 3. Use a for loop to entangle the rest of the qubits
//     # We start from index 1 and use the previous qubit as a control
//     for i in jrange(1, n):
//         cx(q_var[i-1], q_var[i])
//
//     return q_var

#map = affine_map<() -> ()>
module @jasp_module {
  func.func @main(%arg0: tensor<i64>, %arg1: !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState) {
    %result, %qst_out = jasp.create_qubits %arg0, %arg1 : !jasp.QuantumState, tensor<i64> -> !jasp.QubitArray, !jasp.QuantumState
    %cst = arith.constant dense<0> : tensor<i64>
    %0 = jasp.get_qubit %result, %cst : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
    %1 = jasp.quantum_gate "h"(%0), %qst_out : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
    %2 = call @tracerizer() : () -> tensor<i64>
    %3 = tensor.empty() : tensor<i64>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<i64>) outs(%3 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<i64>
    %cst_0 = arith.constant dense<1> : tensor<i64>
    %5 = tensor.empty() : tensor<i64>
    %c1_i64 = arith.constant 1 : i64
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%4 : tensor<i64>) outs(%5 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %10 = arith.subi %in, %c1_i64 : i64
      linalg.yield %10 : i64
    } -> tensor<i64>
    %7 = tensor.empty() : tensor<i64>
    %c0_i64 = arith.constant 0 : i64
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%2 : tensor<i64>) outs(%7 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %10 = arith.addi %in, %c0_i64 : i64
      linalg.yield %10 : i64
    } -> tensor<i64>
    %9:4 = scf.while (%arg2 = %result, %arg3 = %6, %arg4 = %8, %arg5 = %1) : (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState) {
      %10 = tensor.empty() : tensor<i1>
      %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg4, %arg3 : tensor<i64>, tensor<i64>) outs(%10 : tensor<i1>) {
      ^bb0(%in: i64, %in_1: i64, %out: i1):
        %12 = arith.cmpi sle, %in, %in_1 : i64
        linalg.yield %12 : i1
      } -> tensor<i1>
      %extracted = tensor.extract %11[] : tensor<i1>
      scf.condition(%extracted) %arg2, %arg3, %arg4, %arg5 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    } do {
    ^bb0(%arg2: !jasp.QubitArray, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: !jasp.QuantumState):
      %cst_1 = arith.constant dense<0> : tensor<i64>
      %10 = tensor.empty() : tensor<i64>
      %c0_i64_2 = arith.constant 0 : i64
      %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg3 : tensor<i64>) outs(%10 : tensor<i64>) {
      ^bb0(%in: i64, %out: i64):
        %19 = arith.addi %in, %c0_i64_2 : i64
        linalg.yield %19 : i64
      } -> tensor<i64>
      %cst_3 = arith.constant dense<1> : tensor<i64>
      %12 = tensor.empty() : tensor<i64>
      %c1_i64_4 = arith.constant 1 : i64
      %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg4 : tensor<i64>) outs(%12 : tensor<i64>) {
      ^bb0(%in: i64, %out: i64):
        %19 = arith.subi %in, %c1_i64_4 : i64
        linalg.yield %19 : i64
      } -> tensor<i64>
      %14 = jasp.get_qubit %arg2, %13 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %15 = jasp.get_qubit %arg2, %arg4 : !jasp.QubitArray, tensor<i64> -> !jasp.Qubit
      %16 = jasp.quantum_gate "cx"(%14, %15), %arg5 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState
      %17 = tensor.empty() : tensor<i64>
      %c1_i64_5 = arith.constant 1 : i64
      %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg4 : tensor<i64>) outs(%17 : tensor<i64>) {
      ^bb0(%in: i64, %out: i64):
        %19 = arith.addi %in, %c1_i64_5 : i64
        linalg.yield %19 : i64
      } -> tensor<i64>
      scf.yield %arg2, %arg3, %18, %16 : !jasp.QubitArray, tensor<i64>, tensor<i64>, !jasp.QuantumState
    }
    return %result, %9#3 : !jasp.QubitArray, !jasp.QuantumState
  }
  func.func private @tracerizer() -> tensor<i64> {
    %cst = arith.constant dense<1> : tensor<i64>
    return %cst : tensor<i64>
  }
}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[ARG0:.*]]: i64) -> memref<?x!qc.qubit> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG0]] : i64 to index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc(%[[INDEX_CAST_0]]) : memref<?x!qc.qubit>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]]] : memref<?x!qc.qubit>
// CHECK:           qc.h %[[LOAD_0]] : !qc.qubit
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ARG0]], %[[CONSTANT_1]] : i64
// CHECK:           %[[WHILE_0:.*]] = scf.while (%[[VAL_0:.*]] = %[[CONSTANT_1]]) : (i64) -> i64 {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi sle, %[[VAL_0]], %[[SUBI_0]] : i64
// CHECK:             scf.condition(%[[CMPI_0]]) %[[VAL_0]] : i64
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i64):
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_1]] : i64
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[SUBI_1]] : i64 to index
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[INDEX_CAST_1]]] : memref<?x!qc.qubit>
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:             %[[LOAD_2:.*]] = memref.load %[[ALLOC_0]]{{\[}}%[[INDEX_CAST_2]]] : memref<?x!qc.qubit>
// CHECK:             qc.ctrl(%[[LOAD_1]]) {
// CHECK:               qc.x %[[LOAD_2]] : !qc.qubit
// CHECK:             } : !qc.qubit
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_1]] : i64
// CHECK:             scf.yield %[[ADDI_0]] : i64
// CHECK:           }
// CHECK:           return %[[ALLOC_0]] : memref<?x!qc.qubit>
// CHECK:         }
