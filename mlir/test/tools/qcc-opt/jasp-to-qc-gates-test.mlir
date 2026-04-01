// RUN: qcc-opt %s --jasp-to-qc | FileCheck %s

module @jasp_module {
  func.func private @main(%state_in: !jasp.QuantumState, %q0: !jasp.Qubit, %q1: !jasp.Qubit) -> () {

    // Single qubit gates
    %s1 = jasp.quantum_gate "h"(%q0), %state_in : (!jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState

    // Single qubit gates with parameters
    %angle_0 = arith.constant dense<1.000000e+00> : tensor<f64>
    %s2 = jasp.quantum_gate "rx"(%q0, %angle_0), %s1 : (!jasp.Qubit, tensor<f64>), !jasp.QuantumState -> !jasp.QuantumState

    // Two qubit gates
    %s3 = jasp.quantum_gate "cx"(%q0, %q1), %s2 : (!jasp.Qubit, !jasp.Qubit), !jasp.QuantumState -> !jasp.QuantumState

    // Two qubit gates with parameters
    %angle_1 = arith.constant dense<2.000000e+00> : tensor<f64>
    %state_out = jasp.quantum_gate "rxx"(%q0, %q1, %angle_1), %s3 : (!jasp.Qubit, !jasp.Qubit, tensor<f64>), !jasp.QuantumState -> !jasp.QuantumState

    return
  }
}

// CHECK-LABEL:   @main
// CHECK-SAME:      !jasp.QuantumState
// CHECK-SAME:      [[Q0:%.+]]: !qc.qubit,
// CHECK-SAME:      [[Q1:%.+]]: !qc.qubit
// CHECK:           qc.h [[Q0]] : !qc.qubit
// CHECK:           [[ANGLE_0:%.*]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK:           [[ANGLE_F64_0:%.*]] = tensor.extract [[ANGLE_0]][] : tensor<f64>
// CHECK:           qc.rx([[ANGLE_F64_0]]) [[Q0]] : !qc.qubit
// CHECK:           qc.ctrl([[Q0]]) {
// CHECK:             qc.x [[Q1]] : !qc.qubit
// CHECK:           } : !qc.qubit
// CHECK:           [[ANGLE_1:%.*]] = arith.constant dense<2.000000e+00> : tensor<f64>
// CHECK:           [[ANGLE_F64_1:%.*]] = tensor.extract [[ANGLE_1]][] : tensor<f64>
// CHECK:           qc.rxx([[ANGLE_F64_1]]) [[Q0]], [[Q1]] : !qc.qubit, !qc.qubit
// CHECK:           return
// CHECK:         }
