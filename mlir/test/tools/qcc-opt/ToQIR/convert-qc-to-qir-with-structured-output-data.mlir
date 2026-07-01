// RUN: qcc-opt %s -convert-qc-to-qir | FileCheck %s

func.func @test(%ref: memref<3xi64>) -> i64 attributes { qcc.entry_point } {
    // CHECK-LABEL:   func.func @test(%arg0: memref<3xi64>) -> i64 attributes {
    // CHECK-SAME:        passthrough = [
    // CHECK-SAME:          "entry_point",
    // CHECK-SAME:          ["output_labeling_schema", "schema_id"],
    // CHECK-SAME:          ["qir_profiles", "adaptive_profile"],
    // CHECK-SAME:          ["required_num_qubits", "8"],
    // CHECK-SAME:          ["required_num_results", "8"]
    // CHECK-SAME:        ],
    // CHECK-SAME:        qcc.entry_point
    // CHECK-SAME:      } {

    // CHECK:           %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<3xi64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
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
    // CHECK:           %[[MR5:.*]] = llvm.call @__quantum__rt__read_result(%[[RP5]]) : (!llvm.ptr) -> i1
    %m7 = qc.measure %q7 : !qc.qubit -> i1
    // CHECK:           llvm.call @__quantum__qis__mz__body
    // CHECK:           llvm.call @__quantum__rt__read_result


    %c = arith.constant 4 : i64
    // CHECK:          %[[CONST:.*]] = arith.constant 4 : i64
    aux.record_tuple %c : i64
    // CHECK:           %[[LABEL_PTR:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
    // CHECK:           llvm.call @__quantum__rt__tuple_record_output(%[[CONST]], %[[LABEL_PTR]]) : (i64, !llvm.ptr) -> ()
    aux.record_int %m5 : i1
    // CHECK:           %[[LABEL_PTR:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
    // CHECK:           llvm.call @__quantum__rt__bool_record_output(%[[MR5]], %[[LABEL_PTR]]) : (i1, !llvm.ptr) -> ()
    aux.record_int %m7 : i1
    // CHECK:           llvm.call @__quantum__rt__bool_record_output
    aux.record_memref %ref : memref<3xi64>
        %24 = llvm.mlir.constant(3 : i64) : i64
    // CHECK: llvm.call @__quantum__rt__array_record_output(%24, %23) : (i64, !llvm.ptr) -> ()
    // CHECK: %[[value:.*]] = llvm.extractvalue %[[CAST]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[CONST_1:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[ELEM_PTR:.*]] = llvm.getelementptr %[[value]][%[[CONST_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK: %[[LOAD:.*]] = llvm.load %[[ELEM_PTR]] : !llvm.ptr -> i64
    // CHECK: llvm.call @__quantum__rt__int_record_output(%[[LOAD]], %23) : (i64, !llvm.ptr) -> ()
    // CHECK: %[[CONST2:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[ELEM_PTR2:.*]] = llvm.getelementptr %[[value]][%[[CONST2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK: %[[LOAD2:.*]] = llvm.load %[[ELEM_PTR2]] : !llvm.ptr -> i64
    // CHECK: llvm.call @__quantum__rt__int_record_output(%[[LOAD2]], %23) : (i64, !llvm.ptr) -> ()
    // CHECK: %[[CONST3:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[ELEM_PTR3:.*]] = llvm.getelementptr %[[value]][%[[CONST3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK: %[[LOAD3:.*]] = llvm.load %[[ELEM_PTR3]] : !llvm.ptr -> i64
    // CHECK: llvm.call @__quantum__rt__int_record_output(%[[LOAD3]], %23) : (i64, !llvm.ptr) -> ()
    %record_int = arith.constant 42 : i64
    aux.record_int %record_int : i64
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// The pass assumes that these decls already exist.
llvm.func @__quantum__rt__initialize(!llvm.ptr)
llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
llvm.func @__quantum__rt__int_record_output(i64, !llvm.ptr)
llvm.func @__quantum__rt__array_record_output(i64, !llvm.ptr)
llvm.func @__quantum__rt__tuple_record_output(i64, !llvm.ptr)
llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
llvm.func @__quantum__qis__h__body(!llvm.ptr)
llvm.func @__quantum__qis__x__body(!llvm.ptr)
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)

// The label needed to lower `aux.record_int` to its corresponding runtime function:
llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}
