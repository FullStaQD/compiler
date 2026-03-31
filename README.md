<div align="center">
  <img src="https://img.shields.io/badge/Status-Early_Development-red?style=for-the-badge" alt="Early Development">
</div>

---

### **⚠️ Notice: Early Development Phase**

This project is currently in its **infant stages**, so we are keeping things "in-house" for now while we find our footing.

- **No External PRs:** We aren't accepting outside contributions just yet.
- **Coming Soon:** We plan to open the gates to the community once the foundation is solid.

**Stay tuned—we'd love your help later!**

---

# Compiler

The structure of the project follows that of standard MLIR compilers.
The root directory is `mlir`, which contains all the source code.
The tool that is generated is called `qcc-opt`.

---

## Getting Started

### Option 1: Dev Container (Recommended)

The easiest way to get started is via the provided devcontainer, which automatically installs all dependencies including LLVM/MLIR.

**Requirements:** [Docker](https://www.docker.com/) and the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VSCode extension.

1. Open the project in VSCode
2. When prompted, click **"Reopen in Container"** — or use `Cmd+Shift+P` → `Dev Containers: Reopen in Container`
3. Wait for the container to build and install dependencies (first time only)
4. Run the build commands below

For advanced users: Note that the devcontainer allows for a custom mount.
Use it for whatever you want.
You could for example put a checkout of LLVM in there, build it, and link against it.

### Option 2: Manual Setup

If you prefer to build outside the container, you need LLVM/MLIR version >= [VERSION_USED_BY_DEVCONTAINER_IMAGE](.devcontainer/Dockerfile) installed. If you have no previous installation, check the guide [here](https://mqt.readthedocs.io/projects/core/en/latest/installation.html#setting-up-mlir).

---

## Building

If you are using a devcontainer, configure the project like so:

```shell
 cmake --preset dev
```

This uses [cmake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html).
You might want to create your own `CMakeUserPresets.json` (it is under gitignore).
On the other hand, if you manually installed LLVM and MLIR, run:

```shell
cmake -S . -B build/dev -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_DIR=path/to/mlir/installation/dir \
  -DLLVM_DIR=/path/to/llvm/installation/dir \
  -DLLVM_EXTERNAL_LIT=$(which lit)
```

Then build the project like so:

```shell
# With presets:
cmake --build --preset dev
# Or if you do not like presets:
cmake --build build/dev
```

> [!NOTE]
>
> If you are not using a single-config generator (such as when using MSVC on Windows), you need to specify the build configuration explicitly for the build command:

### Post build verification

To make sure that the build procedure worked correctly, run the following:

```shell
./build/dev/bin/qcc-opt mlir/test/tools/qcc-opt/mqt-core-integration-test.mlir
```

and it should output some plain MLIR source.

---

## Testing

Once the project has been built, it is possible to run the tests using the following command:

```shell
cmake --build build/dev --target test-qcc-project
```

If everything works correctly, it should print a 100% success rate.
