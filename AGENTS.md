# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

**qcc** (quantum compiler collection) — an MLIR-based compiler that lowers JASP dialect
IR (produced by [Qrisp](https://qrisp.eu)) to QIR. Generated tools: `qcc-opt` (an
`mlir-opt`-style pass explorer) and `qcc` (the end-to-end driver).

## Quick commands

Details and alternatives are in [README.md](README.md).

```shell
cmake --preset dev                                  # configure
cmake --build build/dev                             # build
cmake --build build/dev --target test-qcc-project   # build + run all tests
lit build/dev/mlir/test/ -v --filter NAME           # run a single/filtered test
```

## Architecture

The full lowering pipeline lives in `mlir/lib/Compiler/Pipeline.cpp`
(`buildQuantumPipeline`).

## Where things live

- Build / test / run details → [README.md](README.md)
- Coding style, include order, the MLIR `const` convention, PR workflow →
  [CONTRIBUTING.md](CONTRIBUTING.md)
