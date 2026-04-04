#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"

#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

struct QCToQIRAdaptive final : impl::QCToQIRAdaptiveBase<QCToQIRAdaptive> {
  using QCToQIRAdaptiveBase::QCToQIRAdaptiveBase;

protected:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp->getContext());

    // FIXME: finish implementation

    funcOp.eraseBody();

    Block* entryBlock = funcOp.addEntryBlock();
    Block* exitBlock = funcOp.addBlock();

    builder.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(builder, funcOp->getLoc(), exitBlock);

    builder.setInsertionPointToEnd(exitBlock);
    if (funcOp.getNumResults() > 0) {
      // FIXME: make this robust
      Type retType = funcOp.getResultTypes()[0];
      Value zeroVal = LLVM::ZeroOp::create(builder, funcOp.getLoc(), retType);
      func::ReturnOp::create(builder, funcOp.getLoc(), zeroVal);

    } else {
      func::ReturnOp::create(builder, funcOp.getLoc());
    }
  }
};

} // namespace qcc
