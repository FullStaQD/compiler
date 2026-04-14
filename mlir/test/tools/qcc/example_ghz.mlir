// RUN: qcc -o - %s | FileCheck %s

/// Prepare GHZ state (without loop so far).
///
/// TODO: This test only tracks what can be done so far with `qcc`. If more functionality
/// evolves this test can go and be replaced by a better test.
func.func @ghz_manual() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    %1 = qc.static 1 : !qc.qubit
    %2 = qc.static 2 : !qc.qubit

    qc.h %0 : !qc.qubit

    qc.ctrl(%0) { qc.x %1 : !qc.qubit } : !qc.qubit
    qc.ctrl(%1) { qc.x %2 : !qc.qubit } : !qc.qubit

    %m0 = qc.measure %0 : !qc.qubit -> i1
    %m1 = qc.measure %1 : !qc.qubit -> i1
    %m2 = qc.measure %2 : !qc.qubit -> i1

    return
}

// CHECK-LABEL:   llvm.func @ghz_manual() attributes
// CHECK:           %[[STATIC_2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           %[[STATIC_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[STATIC_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[ZERO:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.call @__quantum__rt__initialize(%[[ZERO]]) : (!llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_0:.*]] = llvm.inttoptr %[[STATIC_0]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__h__body(%[[INTTOPTR_0]]) : (!llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_1:.*]] = llvm.inttoptr %[[STATIC_1]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_0]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[INTTOPTR_2:.*]] = llvm.inttoptr %[[STATIC_2]] : i64 to !llvm.ptr
// CHECK:           llvm.call @__quantum__qis__cx__body(%[[INTTOPTR_1]], %[[INTTOPTR_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_0]], %[[INTTOPTR_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_0:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_0]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_1]], %[[INTTOPTR_1]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_1:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_1]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.call @__quantum__qis__mz__body(%[[INTTOPTR_2]], %[[INTTOPTR_2]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[CALL_2:.*]] = llvm.call @__quantum__rt__read_result(%[[INTTOPTR_2]]) : (!llvm.ptr) -> i1
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @__quantum__rt__initialize(!llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
// CHECK:         llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
// CHECK:         llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
// CHECK:         llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK:         llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
// CHECK:         llvm.module_flags
// CHECK:         llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}
