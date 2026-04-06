#include "llvm/IR/Type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Conversion/ToQIR/constants.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace {} // namespace

namespace qcc {

#define GEN_PASS_DEF_QCTOQIR
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct QCToQIR final : impl::QCToQIRBase<QCToQIR> {
  using QCToQIRBase::QCToQIRBase;

protected:
  /// FIXME: implement
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto context = funcOp.getContext();

    if (failed(insertRtInit()))
      return signalPassFailure();
  }

private:
  LogicalResult insertRtInit() {
    func::FuncOp funcOp = getOperation();
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto context = funcOp.getContext();

    auto initFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_RT_INIT);

    if (!initFnDecl)
      return moduleOp.emitError() << "missing required declaration of QIR runtime function: " << qcc::QIR_RT_INIT;

    auto loc = funcOp.getLoc();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&funcOp.front());

    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto nullPtr = LLVM::ZeroOp::create(builder, loc, ptrType);
    LLVM::CallOp::create(builder, loc, initFnDecl, ValueRange{nullPtr});

    return llvm::success();
  }
};

} // namespace qcc
