# From Qrisp to MLIR: Step-by-Step Guide

This guide documents the process of converting a [Qrisp](https://qrisp.eu/) quantum program into an MLIR input file suitable for FullStaQD compilation (e.g. with `qcc`).

---

## Prerequisites

- Python environment with `qrisp` installed

---

## Step 1 — Write Your Qrisp Program

Start by writing your quantum algorithm as a standard Python function using the Qrisp API. For example:

```python
from qrisp import QuantumVariable, h, cx, measure
from qrisp.jasp import make_jaspr

def main():
    qv = QuantumVariable(3)
    h(qv[0])
    cx(qv[0], qv[1])
    cx(qv[1], qv[2])
    mes0 = measure(qv[0])
    mes1 = measure(qv[1])
    mes2 = measure(qv[2])
    return
```

This example creates a 3-qubit GHZ state and measures all three qubits.

---

## Step 2 — Generate the MLIR via `make_jaspr`

Append the following two lines at the end of your script to compile the function into a JASPR representation and print the resulting MLIR:

```python
jaspr = make_jaspr(main)()
print(jaspr.to_mlir())
```

Run the script:

```bash
python your_script.py
```

The output will be a block of MLIR text. Copy it — this is your **raw MLIR**, which will require a few manual transformations before it is ready to use.

> **Note:** At this stage the output uses the `stablehlo` dialect for arithmetic and tensor operations. The next steps normalize it into a more standard form.

---

## Step 3 — Replace `stablehlo` with `arith`

The MLIR produced by `jaspr.to_mlir()` uses `stablehlo` ops in places where standard `arith` dialect ops are more appropriate. You need to replace every occurrence of the `stablehlo` namespace with `arith`.

### Option A — `stablehlo-opt` tool

The most reliable approach — and the one that does not rely on text manipulation tools — is to use `stablehlo-opt`.

This requires building the tool locally by following the instructions on the [stablehlo](https://example.com) main page. At the time of writing, the relevant section is **Build instructions**, steps 1 through 6. Once those steps complete successfully, the `stablehlo-opt` binary will be available at `path/to/your/stablehlo/download/build/bin/`.

You can then pipe your Qrisp script's output directly into the tool:

```bash
python path/to/your/python.py | path/to/your/stablehlo/download/build/bin/stablehlo-opt --stablehlo-legalize-to-linalg -allow-unregistered-dialect
```

Two flags are used here:

- `--stablehlo-legalize-to-linalg` — performs the dialect conversion from `stablehlo` to `arith`
- `-allow-unregistered-dialect` — prevents runtime errors caused by unregistered dialects encountered during the conversion

The result of the operation will be the mlir you can use as an input to `qcc` or `qcc-test`

### Option B — Manual find & replace (text editor)

Open the MLIR file in any editor and use **Find & Replace**:

| Find        | Replace |
| ----------- | ------- |
| `stablehlo` | `arith` |

This is the simplest approach and works well for one-off files.

---

## Step 4 — Add the `qcc.entry_point` Annotation

The compiler needs to know which function is the entry point of the quantum program. Locate the main function header in the MLIR output — it will look roughly like:

```mlir
func.func @main(...) -> ... {
```

Add the `attributes { qcc.entry_point }` annotation immediately after the function signature, before the opening brace:

```mlir
func.func @main(...) -> ... attributes { qcc.entry_point } {
```

### Example (before and after)

**Before:**

```mlir
func.func @main(%arg0: i64) -> i1 {
  ...
}
```

**After:**

```mlir
func.func @main(%arg0: i64) -> i1 attributes { qcc.entry_point } {
  ...
}
```

> **Why is this needed?** `qcc.entry_point` tells the compiler which function serves as the top-level entry point of the quantum circuit, similarly to how `main` works in classical compilers.

---

## Summary Checklist

| Step | Action                                                                        |
| ---- | ----------------------------------------------------------------------------- |
| 1    | Write your Qrisp program as a Python function                                 |
| 2    | Append `make_jaspr(main)()` and `print(jaspr.to_mlir())`, then run the script |
| 3    | Replace all `stablehlo` occurrences with `arith`                              |
| 4    | Add `attributes { qcc.entry_point }` to the main function header              |

---

## Known Limitations / TODOs

- The `stablehlo` → `arith` replacement and the `qcc.entry_point` annotation are currently **manual steps**. Future versions of the toolchain aim to automate these transformations directly within `jaspr.to_mlir()`.
