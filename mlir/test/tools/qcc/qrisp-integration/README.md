# Generation of Test Cases from Qrisp

This guide documents the process of converting a [Qrisp](https://qrisp.eu/) quantum program into an MLIR test case for `qcc`.

---

## Instructions

- Setup a python environment and run:

  ```bash
  pip install -r requirements.txt
  ```

- Generate MLIR in the `jasp` dialect:

  ```bash
  python mlir/utils/generate_qrisp_mlir.py my_program.py > my_program.mlir
  ```

- Add `lit` and `FileCheck` directives as usual.

## Notes on qrisp programs

- Qrisp programs used to generate test cases should contain a single function definition. For example:

  ```python
  from qrisp import QuantumVariable, h, cx, measure

  def bell():
      qv = QuantumVariable(2)
      h(qv[0])
      cx(qv[0], qv[1])
      mes0 = measure(qv[0])
      mes1 = measure(qv[1])
      return
  ```

- Do not use `jrange` or `with control(classical condition)`. Instead use `q_cond` or `q_fori_loop`.

## Notes on checks

- You may want to check both `QIR` and `qirrunner` simulation output:
  ```
  // RUN: qcc  %s | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=CHECK-QIR
  // RUN: qcc %s | mlir-translate -mlir-to-llvmir > %t.ll
  // RUN: qir-runner --file %t.ll -s 5 | FileCheck %s --check-prefix=CHECK-SIM
  ```
- Do not run simulations if the program involves more than **4** qubits.
