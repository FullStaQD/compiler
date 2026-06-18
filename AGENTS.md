# Julian's agent instructions for FullStaQD

## Dev environment knowledge

- This project is located in `/workspaces/compiler`.
- The LLVM/MLIR dependency is located in `/home/vscode/external/llvm-project`.
- When looking for LLVM/MLIR source files, keep in mind that some of them are generated from tablegen files and live in the build directory `/home/vscode/external/llvm-project/build/mlir-debug`.

## Building and Testing instructions

- Use the CMAKE installation `/usr/local/bin/cmake`.
- The default for building and/or testing should always be `cmake --build --preset julian-dev --target test-qcc-project`. This will output compilation errors, linker errors, and failed tests.

## MLIR Coding instructions

- The pattern `rewriter.create<SomeOp>(...)` is deprecated in favor of `SomeOp::create(rewriter, ...)`.
- Always use braces in if-blocks, even for single lines.
- Prefer explicit variable names over abbreviations.
