#  ===----------------------------------------------------------------------===//

#  Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
#  Exceptions.
#  See <repo-root>/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  ===----------------------------------------------------------------------===//
# RUN: %S/integration_run.sh -r 1 %s | FileCheck %s

from qrisp import QuantumVariable, h, cx, measure
from qrisp.jasp import make_jaspr

def prepare_bell_state():
    n = 2
    qv = QuantumVariable(n)

    h(qv[0])
    cx(qv[0], qv[1])

    m = measure(qv)

    return m

jaspr = make_jaspr(prepare_bell_state)()
print(jaspr.to_mlir(lower_stablehlo=True))


# A seed of 1 returns integer value 0 (|00> state) for the first shot, and 3 (|11> state) for the second shot, which is consistent with a Bell state.

# CHECK: START
# CHECK: METADATA entry_point
# CHECK: METADATA output_labeling_schema
# CHECK: METADATA qir_profiles
# CHECK: METADATA required_num_qubits 2
# CHECK: METADATA required_num_results 2
# CHECK: OUTPUT INT 0 dummy_label
# CHECK: END 0
# CHECK: START
# CHECK: OUTPUT INT 3 dummy_label
# CHECK: END 0
