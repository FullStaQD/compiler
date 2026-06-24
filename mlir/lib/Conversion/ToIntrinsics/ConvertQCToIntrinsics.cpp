// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/ToIntrinsics/ToIntrinsics.h"
#include "qcc/Dialect/Aux_/IR/Aux_.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>

using namespace mlir;
using namespace qcc;

static StringRef mapUnitaryToIntrinsic(qc::UnitaryOpInterface unitaryOp) {
  if (unitaryOp.getNumControls() == 0) {
    return llvm::TypeSwitch<Operation*, StringRef>(unitaryOp)
        .Case<qc::XOp>([](auto) { return "llvm.riscv.qv.x"; })
        .Case<qc::HOp>([](auto) { return "llvm.riscv.qv.h"; })
        //.Case<qc::TOp>([](auto) { return "llvm.riscv.qv.t"; })
        //.Case<qc::TdgOp>([](auto) { return "llvm.riscv.qv.tdg"; })
        //.Case<qc::SOp>([](auto) { return "llvm.riscv.qv.s"; })
        //.Case<qc::SdgOp>([](auto) { return "llvm.riscv.qv.sdg"; })
        .Default([](auto) { return ""; });
  }

  if (unitaryOp.getNumControls() == 1) {
    auto ctrlOp = cast<qc::CtrlOp>(unitaryOp);
    auto bodyOp = ctrlOp.getBodyUnitary();

    return llvm::TypeSwitch<Operation*, StringRef>(bodyOp)
        .Case<qc::XOp>([](auto) { return "llvm.riscv.qv.cx"; })
        .Default([](auto) { return ""; });
  }

  return "";
}

/// Converts a qubit value (from a `qc.static` op) to an `!llvm.ptr` at the current
/// insertion point, leaving the `qc.static` op in place for later removal.
static Value qubitToPtr(OpBuilder& builder, Value qubitValue) {
  auto* defOp = qubitValue.getDefiningOp();
  assert(defOp && isa<qc::StaticOp>(defOp) &&
         "The pass assumes that all qubits come from static allocations (no function args).");
  auto alloc = cast<qc::StaticOp>(defOp);

  auto index = static_cast<int64_t>(alloc.getIndex());
  auto i64Type = builder.getI64Type();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

  auto constantOp = LLVM::ConstantOp::create(builder, defOp->getLoc(), i64Type, builder.getI64IntegerAttr(index));
  auto ptrOp = LLVM::IntToPtrOp::create(builder, defOp->getLoc(), ptrType, {constantOp});

  return ptrOp;
}

static SmallVector<Value> qubitsToPtrs(OpBuilder& builder, ValueRange qubitValues) {
  SmallVector<Value> ptrValues;
  ptrValues.reserve(qubitValues.size());
  for (auto qubitValue : qubitValues)
    ptrValues.push_back(qubitToPtr(builder, qubitValue));
  return ptrValues;
}

namespace {

struct QCToIntrinsicsTypeConverter final : LLVMTypeConverter {
  explicit QCToIntrinsicsTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](qc::QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {
    auto qubit = op.getQubit();
    auto qubitPtr = qubitToPtr(rewriter, qubit);
    // On HiSEP-Q, qubit and result share the same index.
    auto resultPtr = qubitToPtr(rewriter, qubit);

    LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(), rewriter.getStringAttr("llvm.riscv.qv.mz"),
                                  ValueRange{qubitPtr, resultPtr});
    auto readOp =
        LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(), rewriter.getI1Type(),
                                      rewriter.getStringAttr("llvm.riscv.qv.read_result"), ValueRange{resultPtr});

    rewriter.replaceOp(op, readOp.getResults());
    return success();
  }
};

// TODO: Implement intrinsic-based output recording for HiSEP-Q.
struct RecordIntLowering : public OpConversionPattern<aux::RecordIntOp> {
  using OpConversionPattern<aux::RecordIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(aux::RecordIntOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct UnitaryLowering : public ConversionPattern {
  UnitaryLowering(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> /*operands*/,
                                ConversionPatternRewriter& rewriter) const override {
    auto unitaryOp = dyn_cast<qc::UnitaryOpInterface>(op);
    if (!unitaryOp || !isa<qc::QCDialect>(op->getDialect()))
      return failure();

    auto intrName = mapUnitaryToIntrinsic(unitaryOp);
    if (intrName.empty())
      return op->emitError() << "no intrinsic mapping for gate op";

    auto allPtrs = qubitsToPtrs(rewriter, unitaryOp.getControls());
    auto targetPtrs = qubitsToPtrs(rewriter, unitaryOp.getTargets());
    allPtrs.append(targetPtrs);

    // No module-level declaration needed: LLVM intrinsics are self-contained.
    LLVM::CallIntrinsicOp::create(rewriter, op->getLoc(), rewriter.getStringAttr(intrName), allPtrs);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace qcc {

#define GEN_PASS_DEF_CONVERTQCTOINTRINSICS
#include "qcc/Conversion/ToIntrinsics/ToIntrinsics.h.inc"

namespace {

struct ConvertQCToIntrinsics final : impl::ConvertQCToIntrinsicsBase<ConvertQCToIntrinsics> {
  using ConvertQCToIntrinsicsBase::ConvertQCToIntrinsicsBase;

protected:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto* ctx = funcOp.getContext();

    if (!funcOp->hasAttr("qcc.entry_point"))
      return;

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<qc::QCDialect>();
    target.addIllegalDialect<qcc::aux::AuxDialect>();
    target.addLegalOp<qc::StaticOp>();

    QCToIntrinsicsTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    patterns.add<UnitaryLowering, MeasureLowering, RecordIntLowering>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();

    removeQCStaticOps();
  }

private:
  void removeQCStaticOps() {
    getOperation()->walk([](qc::StaticOp op) { op.erase(); });
  }
};

} // namespace
} // namespace qcc
