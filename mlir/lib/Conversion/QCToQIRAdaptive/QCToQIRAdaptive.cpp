#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"

#include <llvm/Support/raw_ostream.h>

namespace qcc {

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

struct QCToQIRAdaptive final : impl::QCToQIRAdaptiveBase<QCToQIRAdaptive> {
  using QCToQIRAdaptiveBase::QCToQIRAdaptiveBase;

protected:
  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::OpBuilder builder(funcOp->getContext());

    // FIXME: finish implementation

    funcOp.eraseBody();

    mlir::Block* entryBlock = funcOp.addEntryBlock();
    mlir::Block* exitBlock = funcOp.addBlock();

    builder.setInsertionPointToEnd(entryBlock);
    mlir::LLVM::BrOp::create(builder, funcOp->getLoc(), exitBlock);

    builder.setInsertionPointToEnd(exitBlock);
    if (funcOp.getNumResults() > 0) {
      // FIXME: make this robust
      mlir::Type retType = funcOp.getResultTypes()[0];
      mlir::Value zeroVal = mlir::LLVM::ZeroOp::create(builder, funcOp.getLoc(), retType);
      mlir::func::ReturnOp::create(builder, funcOp.getLoc(), zeroVal);

    } else {
      mlir::func::ReturnOp::create(builder, funcOp.getLoc());
    }
  }
};

} // namespace qcc
