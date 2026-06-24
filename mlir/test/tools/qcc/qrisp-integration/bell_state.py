from qrisp import QuantumVariable, h, cx, measure

def prepare_bell_state():
    n = 2
    qv = QuantumVariable(n)

    h(qv[0])
    cx(qv[0], qv[1])

    m = measure(qv)

    return m
