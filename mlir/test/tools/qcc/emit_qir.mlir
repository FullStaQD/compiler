// RUN: qcc --emit-qir %s | FileCheck %s

func.func @main() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    %1 = qc.static 1 : !qc.qubit
    %2 = qc.static 2 : !qc.qubit

    qc.h %0 : !qc.qubit

    qc.ctrl(%0) { qc.x %1 : !qc.qubit } : !qc.qubit
    qc.ctrl(%1) { qc.x %2 : !qc.qubit } : !qc.qubit

    %m0 = qc.measure %0 : !qc.qubit -> i1
    %m1 = qc.measure %1 : !qc.qubit -> i1
    %m2 = qc.measure %2 : !qc.qubit -> i1

    %a = arith.constant 42 : i64
    aux.record_int %a : i64

    aux.record_int %m0 : i1
    aux.record_int %m1 : i1
    aux.record_int %m2 : i1

    return
}

// CHECK:         @.qir_dummy_label = internal constant [12 x i8] c"dummy_label\00"

// CHECK:         define void @main() #0 {
// CHECK:           call void @__quantum__rt__initialize(ptr null)
// CHECK:           call void @__quantum__qis__h__body(ptr null)
// CHECK:           call void @__quantum__qis__cx__body(ptr null, ptr inttoptr (i64 1 to ptr))
// CHECK:           call void @__quantum__qis__cx__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 2 to ptr))
// CHECK:           call void @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK:           %[[R0:.*]] = call i1 @__quantum__rt__read_result(ptr null)
// CHECK:           call void @__quantum__rt__int_record_output(i64 42, ptr @.qir_dummy_label)
// CHECK:           ret void
// CHECK:         }

// CHECK:         declare void @__quantum__rt__initialize(ptr)
