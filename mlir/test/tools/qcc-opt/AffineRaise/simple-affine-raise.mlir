// RUN: qcc-opt %s -affine-raise-from-scf | FileCheck %s

// FIXME: slightly better test coverage (not necessarily just this particular pass)
// - Iterate with non index type (e.g. i32, i64)
// - Non-const LB, UB, STEP.
// - arith.select of cmpi to affine.max rewrite (dedicated pass) - actually: when does this make sense?

// CHECK-LABEL: func.func @simple_loop
func.func @simple_loop(%mem: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  %val = arith.constant 42 : i32

  // CHECK: affine.for %[[IV:.*]] = 0 to 10 {
  // CHECK-NOT: scf.for
  scf.for %i = %c0 to %c10 step %c1 {
    // CHECK: memref.store
    memref.store %val, %mem[%i] : memref<10xi32>
    scf.yield
  }
  return
}
