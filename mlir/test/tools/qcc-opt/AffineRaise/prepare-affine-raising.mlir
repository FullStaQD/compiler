// RUN: qcc-opt %s -prepare-affine-raising | FileCheck %s

func.func private @some_func_index(%arg: index)
func.func private @some_func_i32(%arg: i32)

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
// CHECK: [[LB:%.*]] = arith.maxsi [[LB1]], [[LB2]] : index
// CHECK: scf.for [[IV:%.+]] = [[LB]] to [[UB]] step [[STEP:%.+]] {
// CHECK-NOT: arith.select

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
// CHECK: [[UB:%.*]] = arith.minsi [[UB1]], [[UB2]] : index
// CHECK: scf.for [[IV:%.+]] = [[LB]] to [[UB]] step [[STEP:%.+]] {
// CHECK-NOT: arith.select
