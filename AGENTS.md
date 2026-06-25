# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

**qcc** is an MLIR-based compiler that lowers JASP dialect IR (produced by
[Qrisp](https://qrisp.eu)) to QIR. Generated tools: `qcc-opt` (an
`mlir-opt`-style pass explorer) and `qcc` (the end-to-end driver).

## Quick commands

Details and alternatives are in [README.md](README.md).

```shell
# Build (beware that the user might use a different preset)
cmake --preset dev                                  # configure
cmake --build build/dev                             # build

# Test
cmake --build build/dev --target test-qcc-project   # build + run all tests
lit build/dev/mlir/test/ -v --filter NAME           # run a single/filtered test
```

## Code style

As an LLVM/MLIR base project we mostly follow their guidelines.
We also follow Google's guidelines but LLVM guidelines take precedence.
For details see [CONTRIBUTING.md](CONTRIBUTING.md).
