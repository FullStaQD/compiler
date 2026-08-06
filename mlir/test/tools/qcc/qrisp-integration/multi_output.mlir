// RUN: qcc %s | FileCheck %s

// GENERATED FROM QRISP VERSION 0.9.5 git+https://github.com/eclipse-qrisp/Qrisp.git@b81ea2f979d21cd8d600e79d8b0c7066fe7cbe1b

builtin.module @jasp_module {
  func.func public @main(%arg0: !jasp.QuantumState) -> (tensor<i64>, tensor<i64>, !jasp.QuantumState) {
    %0 = arith.constant dense<1> : tensor<i64>
    %1 = arith.constant dense<2> : tensor<i64>
    func.return %0, %1, %arg0 : tensor<i64>, tensor<i64>, !jasp.QuantumState
  }
}

//CHECK-LABEL:    define void @main() #0 {
//CHECK:        call void @__quantum__rt__initialize(ptr null)
//CHECK:        call void @__quantum__rt__tuple_record_output(i64 2, ptr @.qir_dummy_label)
//CHECK:        call void @__quantum__rt__int_record_output(i64 1, ptr @.qir_dummy_label)
//CHECK:        call void @__quantum__rt__int_record_output(i64 2, ptr @.qir_dummy_label)
//CHECK:        ret void
//CHECK:    }
