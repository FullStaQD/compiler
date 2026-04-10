// RUN: qcc-opt %s -convert-qc-to-qir | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    // CHECK-LABEL:   func.func @test() -> i64 attributes {qcc.entry_point} {
    // CHECK:           %[[ZERO_PTR:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK:           llvm.call @__quantum__rt__initialize(%[[ZERO_PTR]]) : (!llvm.ptr) -> ()

    // Physical qubits.
    %q5 = qc.static 5 : !qc.qubit
    %q7 = qc.static 7 : !qc.qubit
    // NOTE: static allocations are not directly translated to QIR.

    // Supported unitary operations.
    qc.h %q5 : !qc.qubit
    // CHECK:           %[[QC5:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK:           %[[QP5:.*]] = llvm.inttoptr %[[QC5]] : i64 to !llvm.ptr
    // CHECK:           llvm.call @__quantum__qis__h__body(%[[QP5]]) : (!llvm.ptr) -> ()
    qc.x %q5 : !qc.qubit
    // CHECK:           %[[QC5:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK:           %[[QP5:.*]] = llvm.inttoptr %[[QC5]] : i64 to !llvm.ptr
    // CHECK:           llvm.call @__quantum__qis__x__body(%[[QP5]]) : (!llvm.ptr) -> ()
    qc.ctrl(%q5) { qc.x %q7 : !qc.qubit } : !qc.qubit
    // CHECK-DAG:       %[[QC3:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK-DAG:       %[[QP3:.*]] = llvm.inttoptr %[[QC3]] : i64 to !llvm.ptr
    // CHECK-DAG:       %[[QC7:.*]] = llvm.mlir.constant(7 : i64) : i64
    // CHECK-DAG:       %[[QP7:.*]] = llvm.inttoptr %[[QC7]] : i64 to !llvm.ptr
    // CHECK:           llvm.call @__quantum__qis__cx__body(%[[QP3]], %[[QP7]]) : (!llvm.ptr, !llvm.ptr) -> ()

    // Supported measurement operations.
    %m5 = qc.measure %q5 : !qc.qubit -> i1
    // CHECK-DAG:       %[[QC5:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK-DAG:       %[[QP5:.*]] = llvm.inttoptr %[[QC5]] : i64 to !llvm.ptr
    // CHECK-DAG:       %[[RC5:.*]] = llvm.mlir.constant(5 : i64) : i64
    // CHECK-DAG:       %[[RP5:.*]] = llvm.inttoptr %[[RC5]] : i64 to !llvm.ptr
    // CHECK:           llvm.call @__quantum__qis__mz__body(%[[QP5]], %[[RP5]]) : (!llvm.ptr, !llvm.ptr) -> ()
    %m7 = qc.measure %q7 : !qc.qubit -> i1
    // CHECK:           llvm.call @__quantum__qis__mz__body

    // FIXME: treat results explicitly! The pass must be fixed! Probably dedicated test.
    // FIXME: record results.

    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
    // CHECK:           %[[EXIT_CODE:.*]] = arith.constant 0 : i64
    // CHECK:           return %[[EXIT_CODE]] : i64
}

// The pass assumes that these decls already exist.
llvm.func @__quantum__rt__initialize(!llvm.ptr)
llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
llvm.func @__quantum__qis__h__body(!llvm.ptr)
llvm.func @__quantum__qis__x__body(!llvm.ptr)
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
