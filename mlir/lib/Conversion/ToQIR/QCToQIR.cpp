#include "llvm/IR/Type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Conversion/ToQIR/constants.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace {

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](qc::QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

/// FIXME: implement this in a better way (robust for all gates)
struct XGateLowering : public OpConversionPattern<qc::XOp> {
  using OpConversionPattern<qc::XOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::XOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto xFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_QIS_X);
    if (!xFnDecl)
      return op->emitError() << "QIR QIS declaration not found: " << qcc::QIR_QIS_X;

    auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), xFnDecl, adaptor.getOperands());
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

} // namespace

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
    auto ctx = funcOp.getContext();

    /// FIXME: only on entry_point functions!
    if (failed(insertRtInit()))
      return signalPassFailure();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    // target.addIllegalDialect<QCDialect>(); // FIXME:

    QCToQIRTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    patterns.add<XGateLowering>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
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
