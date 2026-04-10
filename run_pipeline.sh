 /workspaces/compiler/build/julian-dev/bin/qcc-opt --pass-pipeline=" \
   builtin.module( \
    inline, \
    jasp-to-qc, \
    func.func(canonicalize), \
    func.func(cse), \
    empty-tensor-to-alloc-tensor, \
    func.func(linalg-detensorize{aggressive-mode=true}), \
    one-shot-bufferize{allow-unknown-ops=true allow-return-allocs-from-loops=true bufferize-function-boundaries=true}, \
    func.func(buffer-loop-hoisting), \
    func.func(convert-linalg-to-loops), \
    func.func(cse), \
    func.func(promote-buffers-to-stack), \
    mem2reg, \
    sccp, \
    func.func(canonicalize), \
    func.func(cse) \
  )" /workspaces/compiler/mlir/test/tools/qcc-opt/jasp-generate-ghz.mlir
