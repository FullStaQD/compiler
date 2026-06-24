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
#include "mlir/IR/BuiltinTypes.h"
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

/// Encodes a qubit (from a `qc.static` op) as a `vector<1xi64>` holding the qubit
/// index, matching the `anyvector` operand the QVSingle/QVPair intrinsics expect.
static Value qubitToVec(OpBuilder& builder, Value qubitValue) {
  auto* defOp = qubitValue.getDefiningOp();
  assert(defOp && isa<qc::StaticOp>(defOp) &&
         "The pass assumes that all qubits come from static allocations (no function args).");
  auto alloc = cast<qc::StaticOp>(defOp);

  auto index = static_cast<int64_t>(alloc.getIndex());
  auto loc = defOp->getLoc();
  auto i64Type = builder.getI64Type();
  auto i32Type = builder.getI32Type();
  auto vecType = VectorType::get(llvm::ArrayRef<int64_t>{1}, i64Type);

  Value indexConst = LLVM::ConstantOp::create(builder, loc, i64Type, builder.getI64IntegerAttr(index));
  Value undef = LLVM::UndefOp::create(builder, loc, vecType);
  Value lane = LLVM::ConstantOp::create(builder, loc, i32Type, builder.getI32IntegerAttr(0));
  return LLVM::InsertElementOp::create(builder, loc, undef, indexConst, lane);
}

namespace {

struct QCToIntrinsicsTypeConverter final : LLVMTypeConverter {
  explicit QCToIntrinsicsTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](qc::QubitType) -> mlir::Type {
      mlir::Type i64 = IntegerType::get(ctx, 64);
      return VectorType::get(llvm::ArrayRef<int64_t>{1}, i64);
    });
  }
};

struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto i64Type = rewriter.getI64Type();

    Value qubitVec = qubitToVec(rewriter, op.getQubit());
    Value tag = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(0));
    Value blockImm = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(0));
    Value vl = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));

    // QVSingleIntrinsic: (vs1: vector<1xi64>, rs2: i64, block_imm: i64, vl: i64) -> void
    LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr("llvm.riscv.qv.mz"),
                                  ValueRange{qubitVec, tag, blockImm, vl});

    // TODO: qv.read_result is not yet defined in IntrinsicsRISCVXQV.td.
    Value result = LLVM::UndefOp::create(rewriter, loc, rewriter.getI1Type());
    rewriter.replaceOp(op, result);
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

    auto loc = op->getLoc();
    auto i64Type = rewriter.getI64Type();

    if (unitaryOp.getNumControls() == 0) {
      // QVSingleIntrinsic: (vs1: vector<1xi64>, rs2: i64, block_imm: i64, vl: i64)
      Value vs1 = qubitToVec(rewriter, unitaryOp.getTargets()[0]);
      Value tag = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(0));
      Value blockImm = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(0));
      Value vl = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
      LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr(intrName),
                                    ValueRange{vs1, tag, blockImm, vl});
    } else {
      // QVPairIntrinsic: (vs1: vector<1xi64>, vs2: vector<1xi64>, block_imm: i64, vl: i64)
      Value vs1 = qubitToVec(rewriter, unitaryOp.getControls()[0]);
      Value vs2 = qubitToVec(rewriter, unitaryOp.getTargets()[0]);
      Value blockImm = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(0));
      Value vl = LLVM::ConstantOp::create(rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
      LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr(intrName),
                                    ValueRange{vs1, vs2, blockImm, vl});
    }

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
