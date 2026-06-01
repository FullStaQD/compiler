// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Constants.h"
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

namespace qcc {

#define GEN_PASS_DEF_ADDENTRYPOINTTOMAIN
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

using namespace mlir;

namespace {

struct AddEntrypointToMain final : public impl::AddEntrypointToMainBase<AddEntrypointToMain> {
  using AddEntrypointToMainBase<AddEntrypointToMain>::AddEntrypointToMainBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // If any function already carries the entry-point attribute, there is
    // nothing to do -- regardless of whether it is `@main` or some other
    // function annotated elsewhere (e.g. by hand or a previous pass).
    auto walkResult = moduleOp.walk([](func::FuncOp funcOp) {
      return funcOp->hasAttr(qcc::entryPointAttrName) ? WalkResult::interrupt() : WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return;
    }

    // No entry-point yet: mark the function with the configured name.
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(entryPointName);
    if (!funcOp) {
      moduleOp.emitError("could not find entry-point function '") << entryPointName << "'";
      return signalPassFailure();
    }

    OpBuilder builder(funcOp.getContext());
    funcOp->setAttr(qcc::entryPointAttrName, builder.getUnitAttr());
  }
};

} // namespace
} // namespace qcc
