// RUN: qcc-opt %s --check-static-qubit-allocation | FileCheck %s

// Test that all the amount of memories we need to allocate for qubits are compile-time constants
// The input is the result of a Canonicalization step
func.func public @main() attributes {qcc.entry_point} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<3x!qc.qubit>
    %0 = memref.load %alloc[%c0] : memref<3x!qc.qubit>
    qc.h %0 : !qc.qubit
    %1 = memref.load %alloc[%c1] : memref<3x!qc.qubit>
    qc.ctrl(%0) {
      qc.x %1 : !qc.qubit
    } : !qc.qubit
    %2 = memref.load %alloc[%c2] : memref<3x!qc.qubit>
    qc.ctrl(%1) {
      qc.x %2 : !qc.qubit
    } : !qc.qubit
    %3 = qc.measure %0 : !qc.qubit -> i1
    %4 = qc.measure %1 : !qc.qubit -> i1
    %5 = qc.measure %2 : !qc.qubit -> i1
    aux.record_bool %3
    aux.record_bool %4
    aux.record_bool %5
    return
  }

// CHECK-LABEL: func.func public @main()
// CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<3x!qc.qubit>
// CHECK:   %[[Q0:.*]] = memref.load %[[ALLOC]][%{{.*}}] : memref<3x!qc.qubit>
// CHECK:   qc.h %[[Q0]] : !qc.qubit
// CHECK:   %[[Q1:.*]] = memref.load %[[ALLOC]][%{{.*}}] : memref<3x!qc.qubit>
// CHECK:   qc.ctrl(%[[Q0]]) {
// CHECK:     qc.x %[[Q1]] : !qc.qubit
// CHECK:   } : !qc.qubit
// CHECK:   %[[Q2:.*]] = memref.load %[[ALLOC]][%{{.*}}] : memref<3x!qc.qubit>
// CHECK:   qc.ctrl(%[[Q1]]) {
// CHECK:     qc.x %[[Q2]] : !qc.qubit
// CHECK:   } : !qc.qubit
// CHECK:   %[[M0:.*]] = qc.measure %[[Q0]] : !qc.qubit -> i1
// CHECK:   %[[M1:.*]] = qc.measure %[[Q1]] : !qc.qubit -> i1
// CHECK:   %[[M2:.*]] = qc.measure %[[Q2]] : !qc.qubit -> i1
// CHECK:   aux.record_bool %[[M0]]
// CHECK:   aux.record_bool %[[M1]]
// CHECK:   aux.record_bool %[[M2]]
// CHECK:   return
// CHECK: }

// An example where the pass fails
// func.func public @main(%arg0: i64) attributes {qcc.entry_point} {
//     %c2 = arith.constant 2 : index
//     %c1 = arith.constant 1 : index
//     %c0 = arith.constant 0 : index
//     %0 = arith.index_cast %arg0 : i64 to index
//     %alloc = memref.alloc(%0) : memref<?x!qc.qubit>
//     %1 = memref.load %alloc[%c0] : memref<?x!qc.qubit>
//     qc.h %1 : !qc.qubit
//     %2 = memref.load %alloc[%c1] : memref<?x!qc.qubit>
//     qc.ctrl(%1) {
//       qc.x %2 : !qc.qubit
//     } : !qc.qubit
//     %3 = memref.load %alloc[%c2] : memref<?x!qc.qubit>
//     qc.ctrl(%2) {
//       qc.x %3 : !qc.qubit
//     } : !qc.qubit
//     %4 = qc.measure %1 : !qc.qubit -> i1
//     %5 = qc.measure %2 : !qc.qubit -> i1
//     %6 = qc.measure %3 : !qc.qubit -> i1
//     aux.record_bool %4
//     aux.record_bool %5
//     aux.record_bool %6
//     return
//   }
