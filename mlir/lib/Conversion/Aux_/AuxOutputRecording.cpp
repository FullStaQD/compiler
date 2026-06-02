// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/Aux_/AuxOutputRecording.h"

#include "qcc/Dialect/Aux_/IR/Aux_.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include <llvm/Support/Casting.h>

namespace qcc {

#define GEN_PASS_DEF_AUXOUTPUTRECORDING
#include "qcc/Conversion/Aux_/AuxOutputRecording.h.inc"

using namespace mlir;

namespace {

struct AuxOutputRecording final : public impl::AuxOutputRecordingBase<AuxOutputRecording> {

  using AuxOutputRecordingBase<AuxOutputRecording>::AuxOutputRecordingBase;

protected:
  void runOnOperation() override {
    // Load the aux dialect into the context
    auto* context = &getContext();
    context->loadDialect<::qcc::aux::AuxDialect>();

    auto module = cast<mlir::ModuleOp>(getOperation());

    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp) {
        return;
      }
      // Only transform functions marked as entry points
      if (!funcOp->hasAttr("qcc.entry_point")) {
        return;
      }

      // We only care if there are results to record
      if (funcOp.getNumResults() == 0) {
        return;
      }

      // Build a new function type with void return type
      // Keep the original argument types, drop results.
      auto oldType = funcOp.getFunctionType();
      auto newType = FunctionType::get(funcOp->getContext(), oldType.getInputs(), {});

      // Update the function type
      funcOp.setType(newType);

      // Work on the first block
      auto& body = funcOp.getBody().front();
      auto* returnOp = body.getTerminator();

      auto retOp = cast<func::ReturnOp>(returnOp);
      auto oldReturnOperands = retOp.getOperands();

      OpBuilder builder(retOp);
      // If there are no return operands, nothing to record
      if (oldReturnOperands.empty()) {
        retOp.erase();
        builder.setInsertionPointToEnd(&body);
        func::ReturnOp::create(builder, retOp.getLoc());
        return;
      }

      auto loc = retOp.getLoc();
      // Record each return value according to its type
      for (Value v : oldReturnOperands) {
        Type ty = v.getType();

        if (ty.isInteger(64)) {
          // aux.record_integer %a : i64
          ::qcc::aux::RecordIntOp::create(builder, loc, v);
        } else if (ty.isInteger(1)) {
          // aux.record_bool %b : i1
          ::qcc::aux::RecordBoolOp::create(builder, loc, v);
        } else {
          // TODO: Handle other types as needed. For now, we only support i64 and i1.
          return signalPassFailure();
        }
      }
      // Remove the old return with values and replace with a void return
      retOp.erase();
      builder.setInsertionPointToEnd(&body);
      func::ReturnOp::create(builder, loc);
    });
  };
};
} // namespace
} // namespace qcc
