// RUN: qcc-opt %s -affine-raise-from-scf -split-input-file | FileCheck %s

// FIXME: things to consider to test:
// - arith.select of cmpi to affine.max rewrite (dedicated pass) - actually: when does this make sense?

func.func private @some_func_index(%arg: index)
func.func private @some_func_i32(%arg: i32)

// CHECK-LABEL: func.func @simple
func.func @simple() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  // CHECK: affine.for %[[IV:.*]] = 0 to 10 {
  // CHECK-NOT: scf.for
  scf.for %i = %c0 to %c10 step %c1 {
    // CHECK: func.call @some_func_index(%[[IV]])
    func.call @some_func_index(%i) : (index) -> ()
    scf.yield
  }
  return
}

// FIXME: the same example with i32 (for lb, ub, step - *all* of them) instead of index fails!
// CHECK-LABEL:   func.func @step_normalization(
// CHECK-SAME:      %[[LB:[a-z0-9]+]]: index,
// CHECK-SAME:      %[[UB:[a-z0-9]+]]: index,
// CHECK-SAME:      %[[STEP:[a-z0-9]+]]: index) {
func.func @step_normalization(%lb: index, %ub: index, %step: index) {

// Computes UB_1 = ceil(UB - LB) / STEP) without using ceil:
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[STEP]], %[[C1]] : index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[UB]], %[[LB]] : index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_0]], %[[SUBI_1]] : index
// CHECK:           %[[UB_1:.*]] = arith.divui %[[ADDI_0]], %[[STEP]] : index

// CHECK:           affine.for %[[IV:.*]] = 0 to %[[UB_1]] {
  scf.for %i = %lb to %ub step %step {
// Rescales IV: IV_1 = IV * STEP + LB
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[IV]], %[[STEP]] : index
// CHECK:             %[[IV_1:.*]] = arith.addi %[[LB]], %[[MULI_0]] : index
// CHECK:             func.call @some_func_index(%[[IV_1]]) : (index) -> ()
    func.call @some_func_index(%i) : (index) -> ()
    scf.yield
  }

  return
}

func.func @iv_is_not_index(%lb: i32, %ub: i32) {
  %step = arith.constant 1 : i32 // step = 1 avoids (full) step normalization.

  scf.for %i = %lb to %ub step %step : i32 {
    func.call @some_func_i32(%i) : (i32) -> ()
    scf.yield
  }

  return
}

// Need to cast to index (LB, UB, IV).
// CHECK-LABEL:   func.func @iv_is_not_index(
// CHECK-SAME:      %[[LB:.*]]: i32,
// CHECK-SAME:      %[[UB:.*]]: i32) {
// CHECK:           %[[UB_1:.*]] = arith.subi %[[UB]], %[[LB]] : i32
// CHECK:           %[[UB_2:.*]] = arith.index_cast %[[UB_1]] : i32 to index
// CHECK:           affine.for %[[IV:.*]] = 0 to %[[UB_2]] {
// CHECK:             %[[IV_1:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:             %[[IV_2:.*]] = arith.addi %[[LB]], %[[IV_1]] : i32
// CHECK:             func.call @some_func_i32(%[[IV_2]]) : (i32) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----
// NOTE: split input file because affine maps are hoisted to the top of module.

func.func private @some_func_index(%arg: index)

// FIXME: add @min_ub?
//
// FIXME: This pattern is a bit special, should the pass really be responsible
// for this behaviour. Moreover, should it behave *exactly* like this or would
// it be better to always do the max?
//
// The select((cmpi sge, a, b), a, b) pattern fires here because %i is an
// affine.for IV: not a valid symbol, so isValidIndex fails on the select
// result and the worklist splits it into two LBs: %i and %lb_min.
func.func @max_lb(%ub: index, %lb_min: index) {
  %c1 = arith.constant 1 : index

  affine.for %i = 0 to %ub {
    %cond = arith.cmpi sge, %i, %lb_min : index
    %lb = arith.select %cond, %i, %lb_min : index  // lb = max(%i, 0)

    scf.for %j = %lb to %ub step %c1 {
      func.call @some_func_index(%j) : (index) -> ()
      scf.yield
    }
  }

  return
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (s0, d0)>
// CHECK-LABEL:   func.func @max_lb(
// CHECK-SAME:                      %[[UB:.*]]: index,
// CHECK-SAME:                      %[[LB_MIN:.*]]: index) {
// CHECK:           affine.for %[[IV_OUTER:.*]] = 0 to %[[UB]] {
// CHECK:             affine.for %[[IV_INNER:.*]] = max #[[$MAP]](%[[IV_OUTER]]){{\[}}%[[LB_MIN]]] to %[[UB]] {
// CHECK:               func.call @some_func_index(%[[IV_INNER]]) : (index) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
