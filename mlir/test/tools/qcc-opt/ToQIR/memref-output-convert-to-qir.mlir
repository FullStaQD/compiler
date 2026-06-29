// RUN: qcc-opt %s -convert-qc-to-qir | FileCheck %s

func.func @test() -> i64 attributes { qcc.entry_point } {
    %t = memref.alloc() : memref<3xi64>
    aux.record_memref %t : memref<3xi64>
    %exit_code = arith.constant 0 : i64
    return %exit_code : i64
}

// The pass assumes that these decls already exist.
llvm.func @__quantum__rt__initialize(!llvm.ptr)
llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
llvm.func @__quantum__rt__int_record_output(i64, !llvm.ptr)
llvm.func @__quantum__rt__array_record_output(i64, !llvm.ptr)
llvm.func @__quantum__rt__read_result(!llvm.ptr {llvm.readonly}) -> i1
llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr {llvm.writeonly}) attributes {passthrough = ["irreversible"]}
llvm.func @__quantum__qis__h__body(!llvm.ptr)
llvm.func @__quantum__qis__x__body(!llvm.ptr)
llvm.func @__quantum__qis__cx__body(!llvm.ptr, !llvm.ptr)
llvm.mlir.global internal constant @".qir_dummy_label"("dummy_label\00") {addr_space = 0 : i32}


// CHECK-LABEL:  func.func @test() -> i64 attributes {passthrough = ["entry_point", ["output_labeling_schema", "schema_id"], ["qir_profiles", "adaptive_profile"], ["required_num_qubits", "0"], ["required_num_results", "0"]], qcc.entry_point} {
// CHECK:     %[[zero:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:     llvm.call @__quantum__rt__initialize(%[[zero]]) : (!llvm.ptr) -> ()
// CHECK:     %[[alloc:.*]] = memref.alloc() : memref<3xi64>
// CHECK:     %[[cast:.*]] = builtin.unrealized_conversion_cast %[[alloc]] : memref<3xi64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:     %[[label:.*]] = llvm.mlir.addressof @".qir_dummy_label" : !llvm.ptr
// CHECK:     %[[count:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK:     llvm.call @__quantum__rt__array_record_output(%[[count]], %[[label]]) : (i64, !llvm.ptr) -> ()
// CHECK:     %[[extract:.*]] = llvm.extractvalue %[[cast]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:     %[[zero_idx:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:     %[[getelem:.*]] = llvm.getelementptr %[[extract]][%[[zero_idx]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:     %[[load:.*]] = llvm.load %[[getelem]] : !llvm.ptr -> i64
// CHECK:     llvm.call @__quantum__rt__int_record_output(%[[load]], %[[label]]) : (i64, !llvm.ptr) -> ()
// CHECK:     %[[one:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:     %[[getelem2:.*]] = llvm.getelementptr %[[extract]][%[[one]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:     %[[load2:.*]] = llvm.load %[[getelem2]] : !llvm.ptr -> i64
// CHECK:     llvm.call @__quantum__rt__int_record_output(%[[load2]], %[[label]]) : (i64, !llvm.ptr) -> ()
// CHECK:     %[[two:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:     %[[getelem3:.*]] = llvm.getelementptr %[[extract]][%[[two]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:     %[[load3:.*]] = llvm.load %[[getelem3]] : !llvm.ptr -> i64
// CHECK:     llvm.call @__quantum__rt__int_record_output(%[[load3]], %[[label]]) : (i64, !llvm.ptr) -> ()
// CHECK:     %[[const:.*]] = arith.constant 0 : i64
// CHECK:     return %[[const]] : i64
// CHECK:  }
