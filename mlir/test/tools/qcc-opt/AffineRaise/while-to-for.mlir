// RUN: qcc-opt %s -while-to-for | FileCheck %s

func.func public @main() -> i64 {
  %step = arith.constant 1 : i64
  %lb = arith.constant 1 : i64
  %zero = arith.constant 0 : i64
  %ub_incl = arith.constant 4 : i64
  %loop_out_iv, %loop_out_sum = scf.while (%induction = %lb, %sum = %zero) : (i64, i64) -> (i64, i64) {
    %continue_condition = arith.cmpi sle, %induction, %ub_incl : i64
    scf.condition(%continue_condition) %induction, %sum : i64, i64
  } do {
  ^bb0(%induction: i64, %sum: i64):
    %next_induction = arith.addi %induction, %step : i64
    %next_sum = arith.addi %sum, %induction : i64
    scf.yield %next_induction, %next_sum : i64, i64
  }
  return %loop_out_sum : i64
}

// CHECK-LABEL:   func.func public @main() -> i64 {
// CHECK-DAG:       %[[UB:.*]] = arith.constant 5 : i64
// CHECK-DAG:       %[[STEP:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[LB:.*]] = arith.constant 0 : i64
// CHECK:           %[[SUM:.*]] = scf.for %[[INDUCTION:.*]] = %[[STEP]] to %[[UB]] step %[[STEP]] iter_args(%[[PREVIOUS_SUM:.*]] = %[[LB]]) -> (i64)  : i64 {
// CHECK:             %[[NEXT_SUM:.*]] = arith.addi %[[PREVIOUS_SUM]], %[[INDUCTION]] : i64
// CHECK:             scf.yield %[[NEXT_SUM]] : i64
// CHECK:           return %[[SUM]] : i64
