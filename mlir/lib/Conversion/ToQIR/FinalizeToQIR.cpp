#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h" // FIXME: remove
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace {

struct PrepareFuncAttrs : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter& /*rewriter*/) const override {
    OpBuilder builder(funcOp->getContext());

    if (funcOp->hasAttr(qcc::entryPointAttrName)) {
      llvm::errs() << "!!! found entry point: " << funcOp.getName() << "\n";
      funcOp->removeAttr(qcc::entryPointAttrName);
      // FIXME:
      // funcOp->setAttr("passthrough", builder.getArrayAttr({builder.getStringAttr("entry_point")}));
    } else {
      /// FIXME: requires capability (6)
      llvm::errs() << "!!! found IR defined function: " << funcOp.getName() << "\n";
    }

    return failure();
  }
};

} // namespace

namespace qcc {

#define GEN_PASS_DEF_FINALIZETOQIR
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct FinalizeToQIR final : public impl::FinalizeToQIRBase<FinalizeToQIR> {
  using FinalizeToQIRBase::FinalizeToQIRBase;

protected:
  void runOnOperation() override {
    // FIXME: finish impl

    ModuleOp moduleOp = getOperation();
    auto* ctx = moduleOp.getContext();

    // Prepare func attrs.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<PrepareFuncAttrs>(ctx);

      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    // Full conversion to LLVM.
    {
      LLVMConversionTarget target(*ctx);
      target.addLegalOp<ModuleOp>();

      target.addLegalDialect<qc::QCDialect>(); // FIXME: remove

      LLVMTypeConverter typeConverter(ctx);
      RewritePatternSet patterns(ctx);

      arith::populateArithToLLVMConversionPatterns(typeConverter, patterns); // FIXME: maybe not needed here
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);

      if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace qcc
