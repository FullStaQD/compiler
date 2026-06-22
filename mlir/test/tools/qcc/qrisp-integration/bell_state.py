#  ===----------------------------------------------------------------------===//

#  Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
#  Exceptions.
#  See <repo-root>/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#  ===----------------------------------------------------------------------===//

from qrisp import QuantumVariable, h, cx, measure

def prepare_bell_state():
    n = 2
    qv = QuantumVariable(n)

    h(qv[0])
    cx(qv[0], qv[1])

    m = measure(qv)

    return m
