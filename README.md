<div align="center">
  <img src="https://img.shields.io/badge/Status-Early_Development-red?style=for-the-badge" alt="Early Development">
</div>

---

### **⚠️ Notice: Early Development Phase**

This project is currently in its **infant stages**, so we are keeping things "in-house" for now while we find our footing. 

* **No External PRs:** We aren't accepting outside contributions just yet. 
* **Coming Soon:** We plan to open the gates to the community once the foundation is solid.

**Stay tuned—we'd love your help later!**

---

# Compiler

The structure of the project follows that of standard MLIR compilers. 
The root directory is `mlir`, which contains all the source code. 
The two tools that are generated are called `qcc` and `qcc-opt`. 

---
## Getting Started

### Option 1: Dev Container (Recommended)
The easiest way to get started is via the provided dev container, which automatically installs all dependencies including LLVM/MLIR 22.1.0.

**Requirements:** [Docker](https://www.docker.com/) and the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VSCode extension.

1. Open the project in VSCode
2. When prompted, click **"Reopen in Container"** — or use `Cmd+Shift+P` → `Dev Containers: Reopen in Container`
3. Wait for the container to build and install dependencies (first time only)
4. Run the build commands below

### Option 2: Manual Setup
If you prefer to build outside the container, you need LLVM/MLIR 22.1.0 installed. If you have no previous installation, check the guide [here](https://mqt.readthedocs.io/projects/core/en/latest/installation.html#setting-up-mlir).

---
## Building
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMLIR_DIR=path/to/mlir/installation/dir -DLLVM_DIR=/path/to/llvm/installation/dir
```

> **Note:** If using the dev container, LLVM/MLIR is installed at `/opt/llvm`, so use `-DMLIR_DIR=/opt/llvm/lib/cmake/mlir` and `-DLLVM_DIR=/opt/llvm/lib/cmake/llvm`.

followed by:
```
cmake --build build
```
> **Note:** this command is going to Compile `MQTCore` as well. So it may take some time to complete. _However_, way less than compiling the whole LLVM/MLIR project. 


---
## Testing

Once the build has completed, verify that everything works using:
```
./build/mlir/tools/qcc-opt/qcc-opt mlir/test/tools/qcc-opt/test.mlir 
```

Expected output:
```
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %0 = qc.alloc("q", 1, 0) : !qc.qubit
    qc.x %0 : !qc.qubit
    qc.dealloc %0 : !qc.qubit
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
}
```