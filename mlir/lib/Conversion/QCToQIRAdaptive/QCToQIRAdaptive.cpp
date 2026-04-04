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
    builder.create<mlir::LLVM::BrOp>(funcOp.getLoc(), mlir::ValueRange{},
                                     exitBlock); // FIXME: builder.create is deprecated

    builder.setInsertionPointToEnd(exitBlock);
    if (funcOp.getNumResults() > 0) {
      // FIXME: make this robust
      mlir::Type retType = funcOp.getResultTypes()[0];
      mlir::Value zeroVal = builder.create<mlir::LLVM::ZeroOp>(funcOp.getLoc(), retType);
      builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), zeroVal);
    } else {
      builder.create<mlir::func::ReturnOp>(funcOp.getLoc());
    }
  }
};

} // namespace qcc
