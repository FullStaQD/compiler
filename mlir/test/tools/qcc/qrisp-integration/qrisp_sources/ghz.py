#  ===----------------------------------------------------------------------===//

#  Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
#  Exceptions.
#  See <repo-root>/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  ===----------------------------------------------------------------------===//

from qrisp import QuantumVariable, h, cx, measure, q_fori_loop
from qrisp.jasp import make_jaspr

def prepare_ghz():
    n = 3
    qv = QuantumVariable(n)

    h(qv[0])

    # For now OK to just unroll this loop:
    q_fori_loop(1, n, lambda i, qv: (cx(qv[i-1], qv[i]), qv)[1], qv)

    m = measure(qv)

    return m

mlir = str(make_jaspr(prepare_ghz)().to_mlir(lower_stablehlo=True))

print(
f"""
// RUN: not %python %S/integration_run.py -r 42 -n 1 -c qcc %s | FileCheck %s

{mlir}

// TODO: Once loop unrolling is in place, add proper checks
// CHECK: qubit index must be a constant; unroll loops before this pass
"""
)
