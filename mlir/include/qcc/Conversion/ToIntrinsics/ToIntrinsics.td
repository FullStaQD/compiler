// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertQIRToIntrinsics : Pass<"convert-qir-to-intrinsics", "mlir::ModuleOp"> {
  let summary = "Replaces QIR QIS call ops with RISC-V quantum vector intrinsics.";

  let description = [{
      Lowers a QIR module for the HiSEP-Q target:

      - `__quantum__qis__*__body` calls become `llvm.call_intrinsic` ops of the
        RISC-V QV extension (`llvm.riscv.qv.h`, `llvm.riscv.qv.cx`, ...). Each
        qubit ptr operand is re-encoded as the `vector<[4]xi8>` scalable vector
        that those intrinsics take, loaded from a 4-byte global holding the
        qubit index.
      - Calls to the QIR runtime (`__quantum__rt__*`) are dropped, as the
        bare-metal intrinsic path has no runtime, and so are the QIR
        declarations left without call sites.
      - A `_start` function is emitted that sets up the stack and calls the
        entry point, superseding it as the entry point of the hardware.

      The pass expects a module lowered by the ToQIR pipeline (`prep-to-qir`,
      `convert-qc-to-qir`, `finalize-to-qir`): gates appear as `llvm.call` ops
      with a statically known callee, and qubits as
      `llvm.inttoptr(llvm.mlir.constant(N : i64))`.
   }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}
