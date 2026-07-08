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
      Takes a module that has already been lowered to QIR (i.e., the body
      contains `llvm.call` ops targeting `__quantum__qis__*__body` functions)
      and replaces those calls with `llvm.call_intrinsic` ops targeting the
      RISC-V QV extension (e.g. `llvm.riscv.qv.h`, `llvm.riscv.qv.cx`).

      Each qubit pointer argument (produced by `llvm.inttoptr` of a constant
      integer index) is re-encoded as a `vector<[1]xi8>` scalable-vector value
      matching the `anyvector` operand that the QVSingle / QVPair intrinsics
      expect. That vector is materialized by loading it from a dedicated
      internal-linkage `i8` global holding the index (one global per distinct
      index, shared across call sites), rather than synthesizing it in a
      register with `vmv.s.x`/`vmv.v.i` -- HiSEP-Q's vector unit does not
      implement those instructions.

      Additionally:
      - `__quantum__rt__initialize` calls are erased (no-op on the intrinsic path).
      - `__quantum__rt__read_result` calls are replaced with `undef : i1`
        (TODO: a proper `qv.read_result` intrinsic is not yet available).
      - `__quantum__rt__bool_record_output` and `__quantum__rt__int_record_output`
        calls are erased.
      - Leftover QIR function declarations are removed from the module.

      Assumptions:
      - The module has been fully lowered by the ToQIR pipeline
        (`prep-to-qir`, `convert-qc-to-qir`, `finalize-to-qir`) so all gate
        ops appear as `llvm.call` with a statically-known callee symbol.
      - Qubit pointers were produced by `llvm.inttoptr(llvm.mlir.constant(N : i64))`,
        i.e. the static qubit management scheme used by `convert-qc-to-qir`.
   }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}
