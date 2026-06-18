# From Qrisp to MLIR: Step-by-Step Guide

This guide documents the process of converting a [Qrisp](https://qrisp.eu/) quantum program into an MLIR test case for `qcc`.

---

## Prerequisites

This project requires a Python environment set up by the `requirements.txt` in this folder. You can install it via pip:

```bash
pip install -r requirements.txt
```

## Step 1 — Write Your Qrisp Program

Start by writing your quantum algorithm as a standard Python function using the Qrisp API. For example:

```python
from qrisp import QuantumVariable, h, cx, measure
from qrisp.jasp import make_jaspr

def bell():
    qv = QuantumVariable(2)
    h(qv[0])
    cx(qv[0], qv[1])
    mes0 = measure(qv[0])
    mes1 = measure(qv[1])
    return

jaspr = make_jaspr(bell)()
print(jaspr.to_mlir(lower_stablehlo=True))
```

This example creates a bell state state and measures all the qubits.
Run the script:

```bash
python my_program.test.py > my_program.generated_test.mlir
```

---

## Step 2 - Add lit RUN commands

For example, these `RUN` commands enable checking against `QIR` and `qirrunner` simulation output:

```
// Test 1: Check QIR output.
// RUN: qcc  %s | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=CHECK-QIR
// Test 2: Check output recording from qir-runner simulation.
// qir-runner requires a .ll file (it checks the extension and does not accept
// stdin), so we first emit the LLVM IR to a temporary .ll file and then invoke
// qir-runner on it.
// RUN: qcc %s | mlir-translate -mlir-to-llvmir > %t.ll
// RUN: uvx --from qirrunner qir-runner --file %t.ll -s 5 | FileCheck %s --check-prefix=CHECK-SIM
```

---

## Step 3 - Add FileCheck directives

Add checks for the different outputs:

```
// CHECK-QIR-DAG: ... for QIR output ...
// CHECK-SIM:     ... for qirrunner output ...
```

## Remark

Despite this guide, remember to always document all manual tweaks you have to do to run any test correctly within `qcc` in a TODO comment within the tests themselves.
