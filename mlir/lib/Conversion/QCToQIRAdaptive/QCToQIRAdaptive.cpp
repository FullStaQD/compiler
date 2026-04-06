#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace {

// FIXME: move into dedicated "constants" file
/// A unit attribute to mark a func.funcOp as the starting point of a quantum program.
static constexpr llvm::StringLiteral QCC_ENTRY_POINT_ATTR_NAME = "qcc.entry_point";

} // namespace

namespace qcc {

#define GEN_PASS_DEF_QCTOQIRADAPTIVE
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

struct QCToQIRAdaptive final : impl::QCToQIRAdaptiveBase<QCToQIRAdaptive> {
  using QCToQIRAdaptiveBase::QCToQIRAdaptiveBase;

protected:
  // FIXME: finish implementation
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (funcOp->hasAttr(QCC_ENTRY_POINT_ATTR_NAME)) {
      handleEntryPoint();
    } else {
      handleIRDefinedFunc();
    }

    auto context = funcOp->getContext();
    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp>(); // FIXME: check which are needed
    target.addLegalOp<func::FuncOp>();
    target.addLegalOp<func::ReturnOp>();
    target.addLegalOp<func::CallOp>();

    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      signalPassFailure();
  }

private:
  /// FIXME: docstring, implement
  void handleEntryPoint() {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp->getContext());

    llvm::errs() << "found entry point: " << funcOp.getName() << "\n";
    funcOp->removeAttr(QCC_ENTRY_POINT_ATTR_NAME);
    funcOp->setAttr("passthrough", builder.getArrayAttr({builder.getStringAttr("entry_point")}));
  }

  /// FIXME: docstring, implement
  /// FIXME: this requires capability (6)!
  void handleIRDefinedFunc() {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp->getContext());

    llvm::errs() << "found IR defined function: " << funcOp.getName() << "\n";
  }
};

} // namespace qcc
