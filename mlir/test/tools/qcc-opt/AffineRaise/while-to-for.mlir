// RUN: qcc-opt %s -while-to-for | FileCheck %s

// CHECK-LABEL: @main
// CHECK-NOT: scf.while
// CHECK: scf.for
// CHECK: qc.ctrl
func.func public @main() -> memref<?x!qc.qubit> {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %alloc = memref.alloc() : memref<5x!qc.qubit>
  %cast = memref.cast %alloc : memref<5x!qc.qubit> to memref<?x!qc.qubit>
  %0 = memref.load %alloc[%c0] : memref<5x!qc.qubit>
  qc.h %0 : !qc.qubit
  %1 = scf.while (%arg0 = %c1_i64) : (i64) -> i64 {
    %2 = arith.cmpi sle, %arg0, %c4_i64 : i64
    scf.condition(%2) %arg0 : i64
  } do {
  ^bb0(%arg0: i64):
    %2 = arith.subi %arg0, %c1_i64 : i64
    %3 = arith.index_cast %2 : i64 to index
    %4 = memref.load %alloc[%3] : memref<5x!qc.qubit>
    %5 = arith.index_cast %arg0 : i64 to index
    %6 = memref.load %alloc[%5] : memref<5x!qc.qubit>
    qc.ctrl(%4) {
      qc.x %6 : !qc.qubit
    } : !qc.qubit
    %7 = arith.addi %arg0, %c1_i64 : i64
    scf.yield %7 : i64
  }
  return %cast : memref<?x!qc.qubit>
}
