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

// FIXME: move into dedicated "constants" file
/// A unit attribute to mark a func.funcOp as the starting point of a quantum program.
static constexpr llvm::StringLiteral QCC_ENTRY_POINT_ATTR_NAME = "qcc.entry_point";

struct PrepareFuncAttrs : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter& rewriter) const override {
    OpBuilder builder(funcOp->getContext());

    if (funcOp->hasAttr(QCC_ENTRY_POINT_ATTR_NAME)) {
      llvm::errs() << "!!! found entry point: " << funcOp.getName() << "\n";
      funcOp->removeAttr(QCC_ENTRY_POINT_ATTR_NAME);
      funcOp->setAttr("passthrough", builder.getArrayAttr({builder.getStringAttr("entry_point")}));
    } else {
      /// FIXME: requires capability (6)
      llvm::errs() << "!!! found IR defined function: " << funcOp.getName() << "\n";
    }

    return failure();
  }
};

} // namespace

namespace qcc {

#define GEN_PASS_DEF_TOQIRFINALIZE
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct ToQIRFinalize final : public impl::ToQIRFinalizeBase<ToQIRFinalize> {
  using ToQIRFinalizeBase::ToQIRFinalizeBase;

protected:
  void runOnOperation() override {
    // FIXME: finish impl

    ModuleOp moduleOp = getOperation();
    auto context = moduleOp.getContext();

    // Prepare func attrs.
    {
      RewritePatternSet patterns(context);
      patterns.add<PrepareFuncAttrs>(context);

      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
        signalPassFailure();
    }

    // Full conversion to LLVM.
    {
      LLVMConversionTarget target(*context);
      target.addLegalOp<ModuleOp>();

      target.addLegalDialect<qc::QCDialect>(); // FIXME: remove

      LLVMTypeConverter typeConverter(context);
      RewritePatternSet patterns(context);

      arith::populateArithToLLVMConversionPatterns(typeConverter, patterns); // FIXME: maybe not needed here
      populateFuncToLLVMConversionPatterns(typeConverter, patterns);

      if (failed(applyFullConversion(moduleOp, target, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace qcc
