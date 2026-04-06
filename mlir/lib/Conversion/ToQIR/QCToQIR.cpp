#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Conversion/ToQIR/constants.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
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

/// Map any of the (unitary) gates to their QIR QIS function declaration.
StringRef mapQCGateToQIS(Operation* op) {
  return llvm::TypeSwitch<Operation*, StringRef>(op)
      .Case<qc::XOp>([](auto) { return qcc::QIR_QIS_X; })
      .Case<qc::HOp>([](auto) { return qcc::QIR_QIS_H; })
      .Default([](auto) { return ""; });
}

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](qc::QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

/// FIXME: we need to read the measurement result too!
struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_QIS_MZ);
    if (!fnDecl)
      return op->emitError() << "QIR QIS declaration not found: " << qcc::QIR_QIS_MZ;

    // FIXME: measurement op takes qubit as arg, measure fn takes qubit and result as arg.
    auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), fnDecl, adaptor.getOperands());
    // FIXME: add read_result

    rewriter.replaceOp(op, callOp);
    return success();
  }
};

/// We rely on the fact that the signature of qc gates and the corresponding QIR QIS function fits.
struct UnitaryLowering : public ConversionPattern {
  UnitaryLowering(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getDialect()->getNamespace() != "qc") // FIXME: do not hardcode the name
      return failure();

    auto moduleOp = op->getParentOfType<ModuleOp>();

    // NOTE: if the function declaration is not found we report and error and
    // leave it to the pass to observe that some ops could not be converted (qc
    // dialect is illegal).
    auto qis_name = mapQCGateToQIS(op);
    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qis_name);
    if (!fnDecl)
      return op->emitError() << "QIR QIS declaration not found: " << qis_name;

    auto callOp = LLVM::CallOp::create(rewriter, op->getLoc(), fnDecl, operands);
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
    // target.addIllegalDialect<qc::QCDialect>(); // FIXME:
    // target.addIllegalOp<qc::XOp>();
    // target.addIllegalOp<qc::HOp>();
    // target.addIllegalOp<qc::MeasureOp>();

    QCToQIRTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    patterns.add<UnitaryLowering>(typeConverter, ctx);

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
