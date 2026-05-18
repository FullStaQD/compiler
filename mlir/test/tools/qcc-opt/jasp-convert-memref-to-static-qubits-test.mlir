// RUN: qcc-opt %s --convert-memref-to-static-qubits | FileCheck %s

// Test that the constant size `memref.alloc` are successfully converted to `qc.static` calls.
// `memref.load` are substituted with direct accesses to the `qc.static` results.
// The input is the result of a Canonicalization step which has successfully passed the `--check-static-qubit-allocation` test
// After this stage, the IR is still in a intermediate step. It needs to be standardized using a Canonicalize step that remove all the `memref` dialect occurrences.
func.func public @test(){
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

// CHECK: module {
// CHECK-LABEL:   func.func public @test() {
// CHECK:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[Q0:.*]] = qc.static 0 : !qc.qubit
// CHECK:     %[[Q1:.*]] = qc.static 1 : !qc.qubit
// CHECK:     %[[Q2:.*]] = qc.static 2 : !qc.qubit
// CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<3x!qc.qubit>
// CHECK:     %{{.*}} = memref.load %[[ALLOC]][%[[C0]]] : memref<3x!qc.qubit>
// CHECK:     qc.h %[[Q0]] : !qc.qubit
// CHECK:     %{{.*}} = memref.load %[[ALLOC]][%[[C1]]] : memref<3x!qc.qubit>
// CHECK:     qc.ctrl(%[[Q0]]) {
// CHECK:       qc.x %[[Q1]] : !qc.qubit
// CHECK:     } : !qc.qubit
// CHECK:     %{{.*}} = memref.load %[[ALLOC]][%[[C2]]] : memref<3x!qc.qubit>
// CHECK:     qc.ctrl(%[[Q1]]) {
// CHECK:       qc.x %[[Q2]] : !qc.qubit
// CHECK:     } : !qc.qubit
// CHECK:     %[[M0:.*]] = qc.measure %[[Q0]] : !qc.qubit -> i1
// CHECK:     %[[M1:.*]] = qc.measure %[[Q1]] : !qc.qubit -> i1
// CHECK:     %[[M2:.*]] = qc.measure %[[Q2]] : !qc.qubit -> i1
// CHECK:     aux.record_bool %[[M0]]
// CHECK:     aux.record_bool %[[M1]]
// CHECK:     aux.record_bool %[[M2]]
// CHECK:     return
// CHECK:   }
// CHECK: }
