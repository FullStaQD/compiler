# RUN: not %S/integration_run.sh -r 42 %s | FileCheck %s

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

jaspr = make_jaspr(prepare_ghz)()
print(jaspr.to_mlir(lower_stablehlo=True))

# TODO: Once loop unrolling is in place, add proper checks
# CHECK: qubit index must be a constant; unroll loops before this pass
