#  ===----------------------------------------------------------------------===//

#  Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
#  Exceptions.
#  See <repo-root>/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  ===----------------------------------------------------------------------===//

from qrisp import QuantumVariable, h, cx, measure
from qrisp.jasp import make_jaspr

def prepare_bell_state():
    n = 2
    qv = QuantumVariable(n)

    h(qv[0])
    cx(qv[0], qv[1])

    m = measure(qv)

    return m

mlir = str(make_jaspr(prepare_bell_state)().to_mlir(lower_stablehlo=True))
print(
f"""
// RUN: %S/integration_run.sh -n 5 %s | FileCheck %s

{mlir}

// CHECK-LABEL: ---QIR---
// CHECK: @__quantum__qis__h__body(ptr null)
// CHECK: @__quantum__qis__cx__body(ptr null, ptr inttoptr (i64 1 to ptr))
// CHECK-DAG: @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK-DAG: @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))

// CHECK-DAG: [[BIT0:%.+]] = call i1 @__quantum__rt__read_result(ptr null)
// CHECK-DAG: [[BIT1:%.+]] = call i1 @__quantum__rt__read_result(ptr inttoptr (i64 1 to ptr))
// CHECK-DAG: [[INT_RESULT_0:%.+]] = zext i1 [[BIT0]] to i64
// CHECK-DAG: [[INT_RESULT_1:%.+]] = zext i1 [[BIT1]] to i64

// CHECK-DAG: [[LEFT_SHIFTED_1:%.+]] = shl i64 [[INT_RESULT_1]], 1
// CHECK-DAG: [[COMBINED_INT:%.+]] = or i64 [[INT_RESULT_0]], [[LEFT_SHIFTED_1]]
// CHECK: @__quantum__rt__int_record_output(i64 [[COMBINED_INT]], ptr @.qir_dummy_label)


// CHECK-LABEL: ---OUTPUT-RECORDING---

// CHECK: START
// CHECK: METADATA entry_point
// CHECK: METADATA output_labeling_schema
// CHECK: METADATA qir_profiles
// CHECK: METADATA required_num_qubits 2
// CHECK: METADATA required_num_results 2

// We expect values 0 (|00> state) or 3 (|11> state).
// CHECK: OUTPUT INT {{{{[03]}}}} dummy_label
// CHECK: OUTPUT INT {{{{[03]}}}} dummy_label
// CHECK: OUTPUT INT {{{{[03]}}}} dummy_label
// CHECK: OUTPUT INT {{{{[03]}}}} dummy_label
// CHECK: OUTPUT INT {{{{[03]}}}} dummy_label
"""
)
