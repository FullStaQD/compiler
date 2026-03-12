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

To compile, use the following command: 
```
  cmake -S . -B build -G Ninja \                                      
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DMLIR_DIR=path/to/mlir/installation/dir \
      -DLLVM_DIR=/path/to/llvm/installation/dir
```
followed by:
```
  cmake --build build
```

If you have no previous installed version of LLVM/MLIR, check the guide you can find [here](https://mqt.readthedocs.io/projects/core/en/latest/installation.html#setting-up-mlir).

Once the process has ended, you can test that everything works properly by calling:
```
./build/mlir/tools/qcc/qcc mlir/test/tools/qcc-opt/test.mlir 
```

and the output is supposed to look like this:
```
module {
  func.func @test() {
    return
  }
}
```