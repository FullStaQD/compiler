# LLVM Dependency

The file `base.rev` pins the exact revision of llvm we depend on.

The sub-directory `patches/qisa/` hosts patches to LLVM which enable quantum ISA
support (e.g. HiSEP-Q via cmake flag `QCC_ENABLE_HISEPQ=ON`). Either apply them
directly via `git am` or use the script we provide to clone a fresh LLVM
checkout:

```shell
# Make sure you understand what the script is doing:
cd <repo-root>
./utils/provision-llvm --help

./utils/provision-llvm /path/where/to/put/llvm-project-patched
```

Note that developing an LLVM backend is hard to do out-of-tree. Hence we do it
in-tree and this patching mechanism is required. The backends are developed on
[our own llvm fork](https://github.com/FullStaQD/llvm-project). To update the
patches open a PR and do:

```shell
rm third-party/llvm/patches/qisa/*.patch # always remove the old ones
BASE_REV=$(cat third-party/llvm/base.rev)
git -C /path/to/llvm-fork format-patch $BASE_REV..fullstaqd-qisa \
  -o /path/to/qcc/third-party/llvm/patches/qisa
```
