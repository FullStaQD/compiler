// RUN: qcc-opt %s -prepare-affine-raising | FileCheck %s

func.func private @some_func_index(%arg: index)
func.func private @some_func_i32(%arg: i32)

// Test basic behavior for replacing a lower bound with a maximum
func.func @prepare_lower_bound(%ub: index, %lb1: index, %lb2: index) {
  %cmp = arith.cmpi sge, %lb1, %lb2 : index
  %lb = arith.select %cmp, %lb1, %lb2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    func.call @some_func_index(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @prepare_lower_bound
// CHECK-SAME: ([[UB:%.+]]: index, [[LB1:%.+]]: index, [[LB2:%.+]]: index)
// CHECK-NOT: arith.select
// CHECK: [[LB:%.*]] = arith.maxsi [[LB1]], [[LB2]] : index
// CHECK: scf.for [[IV:%.+]] = [[LB]] to [[UB]] step [[STEP:%.+]] {

// Test basic behavior for replacing an upper bound with a minimum
func.func @prepare_upper_bound(%lb: index, %ub1: index, %ub2: index) {
  %cmp = arith.cmpi sle, %ub1, %ub2 : index
  %ub = arith.select %cmp, %ub1, %ub2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    func.call @some_func_index(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @prepare_upper_bound
// CHECK-SAME: ([[LB:%.+]]: index, [[UB1:%.+]]: index, [[UB2:%.+]]: index)
// CHECK-NOT: arith.select
// CHECK: [[UB:%.*]] = arith.minsi [[UB1]], [[UB2]] : index
// CHECK: scf.for [[IV:%.+]] = [[LB]] to [[UB]] step [[STEP:%.+]] {

// Test nested min/max patterns are converted correctly
func.func @prepare_nested_bounds(%ub1: index, %ub2: index, %ub3: index, %lb1: index, %lb2: index, %lb3: index) {
  %cmp1 = arith.cmpi sle, %ub1, %ub2 : index
  %ub12 = arith.select %cmp1, %ub1, %ub2 : index
  %cmp2 = arith.cmpi sle, %ub12, %ub3 : index
  %ub = arith.select %cmp2, %ub12, %ub3 : index
  %cmp3 = arith.cmpi sge, %lb1, %lb2 : index
  %lb12 = arith.select %cmp3, %lb1, %lb2 : index
  %cmp4 = arith.cmpi sge, %lb12, %lb3 : index
  %lb = arith.select %cmp4, %lb12, %lb3 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    func.call @some_func_index(%i) : (index) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @prepare_nested_bounds
// CHECK-SAME: ([[UB1:%.+]]: index, [[UB2:%.+]]: index, [[UB3:%.+]]: index, [[LB1:%.+]]: index, [[LB2:%.+]]: index, [[LB3:%.+]]: index)
// CHECK-NOT: arith.select
// CHECK-DAG: [[UB12:%.*]] = arith.minsi [[UB1]], [[UB2]] : index
// CHECK-DAG: [[UB:%.*]] = arith.minsi [[UB12]], [[UB3]] : index
// CHECK-DAG: [[LB12:%.*]] = arith.maxsi [[LB1]], [[LB2]] : index
// CHECK-DAG: [[LB:%.*]] = arith.maxsi [[LB12]], [[LB3]] : index
// CHECK: scf.for [[IV:%.+]] = [[LB]] to [[UB]] step [[STEP:%.+]] {

// Test that i32 bounds are converted to index when used as bounds, without min/max patterns
func.func @prepare_non_index_bounds_no_pattern(%ub: i32, %lb: i32) {
  %step = arith.constant 1 : i32
  scf.for unsigned %i = %lb to %ub step %step : i32 {
    func.call @some_func_i32(%i) : (i32) -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func.func @prepare_non_index_bounds_no_pattern
// CHECK-SAME: ([[UB:%.+]]: i32, [[LB:%.+]]: i32)
// CHECK-DAG: [[STEP:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[UB_C:%.*]] = arith.index_castui [[UB]] : i32 to index
// CHECK-DAG: [[LB_C:%.*]] = arith.index_castui [[LB]] : i32 to index
// CHECK: scf.for unsigned [[IV:%.+]] = [[LB_C]] to [[UB_C]] step [[STEP]] {
// CHECK: [[IV_I32:%.*]] = arith.index_castui [[IV]] : index to i32
// CHECK: func.call @some_func_i32([[IV_I32]]) : (i32) -> ()
// CHECK: return


// Test that i32 bounds are converted to index when used as bounds, including min/max patterns
func.func @prepare_non_index_bounds(%ub1: i32, %ub2: i32, %lb1: i32, %lb2: i32) {
  %cmp1 = arith.cmpi sle, %ub1, %ub2 : i32
  %ub = arith.select %cmp1, %ub1, %ub2 : i32
  %cmp2 = arith.cmpi sge, %lb1, %lb2 : i32
  %lb = arith.select %cmp2, %lb1, %lb2 : i32
  %step = arith.constant 1 : i32
  scf.for %i = %lb to %ub step %step : i32 {
    func.call @some_func_i32(%i) : (i32) -> ()
    scf.yield
  }
  return
}

// FIXME: Currently we have the order: max -> index_cast -> for, but it would be
//  more comfortable to have the max tree directly as the bounds.
// CHECK-LABEL: func.func @prepare_non_index_bounds
// CHECK-SAME: ([[UB1:%.+]]: i32, [[UB2:%.+]]: i32, [[LB1:%.+]]: i32, [[LB2:%.+]]: i32)
// CHECK-DAG: [[STEP:%.*]] = arith.constant 1 : index
// CHECK-DAG: [[LBMAX:%.*]] = arith.maxsi [[LB1]], [[LB2]] : i32
// CHECK-DAG: [[UBMIN:%.*]] = arith.minsi [[UB1]], [[UB2]] : i32
// CHECK-DAG: [[LB_C:%.*]] = arith.index_cast [[LBMAX]] : i32 to index
// CHECK-DAG: [[UB_C:%.*]] = arith.index_cast [[UBMIN]] : i32 to index
// CHECK: scf.for [[IV:%.+]] = [[LB_C]] to [[UB_C]] step [[STEP]] {
// CHECK: [[IV_I32:%.*]] = arith.index_cast [[IV]] : index to i32
// CHECK: func.call @some_func_i32([[IV_I32]]) : (i32) -> ()
// CHECK: return
