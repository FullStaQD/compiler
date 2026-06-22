// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/ToIntrinsics/ToIntrinsics.h"

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
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>

using namespace mlir;
using namespace qcc;

static StringRef mapUnitaryToQIS(qc::UnitaryOpInterface unitaryOp) {
  if (unitaryOp.getNumControls() == 0) {
    return llvm::TypeSwitch<Operation*, StringRef>(unitaryOp)
      .Case<qc::XOp>([](auto) {
      return qcc::qirQisX; })
      .Case<qc::HOp>([](auto) {
      return qcc::qirQisH; })
      .Case<qc::TOp>([](auto) {
      return q
  }
  }

  /// To be used in a rewrite pattern.
  static InFlightDiagnostic emitMissingIntrinsicsDeclError(Operation * op, StringRef name) {
    return op->emitError() << "missing required declaration of intrinsics function: '" << name << "'";
  }

  namespace {

  struct QCtoIntrinsicsTypeConverter final : LLVMTypeConverter {
    explicit QCtoIntrinsicsTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
      addConversion([ctx](qc::QubitType) { return LLVM::LLVMPointerType::get(ctx); });
    }
  };

  struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
    using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter& rewriter) const override {
      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto mzFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirQisMZ);
      if (!mzFnDecl) {
        return emitMissingIntrinsicsDeclError(op, qcc::qirQisMZ);
      }

      auto mzFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirQisMZ);
      if (!mzFnDecl) {
        return emitMissingIntrinsicsDeclError(op, qcc::qirQisMZ);
      }

      auto readFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirRtReadResult);
      if (!readFnDecl) {
        return emitMissingIntrinsicsDeclError(op, qcc::qirRtReadResult);
      }

      auto qubit = op.getQubit();
      auto qubitPtr = qubitToPtr(rewriter, qubit);
      auto resultPtr = qubitToPtr(rewriter, qubit);

      LLVM::CallOp::create(rewriter, op.getLoc(), mzFnDecl, {qubitPtr, resultPtr});
      auto callReadOp = LLVM::CallOp::create(rewriter, op.getLoc(), readFnDecl, {resultPtr});

      rewriter.replaceOp(op, callReadOp);
      return success();
    }
  };

  struct RecordIntLowering : public OpConversionPattern<aux::RecordIntOp> {
    using OpConversionPattern<aux::RecordIntOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(aux::RecordIntOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
      auto loc = op.getLoc();
    }
  }

  struct UnitaryLowering : public ConversionPattern {
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

      if (!funcOp->hasAttr("qcc.entry_point")) {
        return;
      }

      if (failed(insertRtInit())) {
        return signalPassFailure();
      }

      if (failed(insertRtInit())) {
        return signalPassFailure();
      }

      ConversionTarget target(*ctx);
      target.addLegalDialect<LLVM::LLVMDialect>();
      target.addIllegalDialect<qc::QCDialect>();
      target.addIllegalDialect<qcc::aux::AuxDialect>();
      target.addLegalOp<qc::StaticOp>();

      QCtoIntrinsicsTypeConverter typeConverter(ctx);
      RewritePatternSet patterns(ctx);
      patterns.add<UnitaryLowering, MeasureLowering, RecordIntLowering>(typeConverter, ctx);

      if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
        return signalPassFailure();
      }

      removeQCStaticOps();
    }

  private:
    LogicalResult insertRtInit() {
      func::FuncOp funcOp = getOperation();
      auto module = funcOp->getParentOfType<ModuleOp>();
      auto* ctx = funcOp.getContext();

      auto initFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirRtInit);

      if (!initFnDecl) {
        return emitMissingIntrinsicsDeclError(funcOp, qcc::qirRtInit);
      }

      auto loc = funcOp.getLoc();
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(&funcOp.front());

      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      auto nullPtr = LLVM::ZeroOp::create(builder, loc, ptrType);
      LLVM::CallOp::create(builder, loc, initFnDecl, ValueRange{nullPtr});

      return success();
    }

    LogicalResult setEntryPointAttrs() {
      func::FuncOp funcOp = getOperation();
      OpBuilder builder(funcOp.getContext());

      auto requiredNumQubits = getRequiredNumQubits();
      auto requiredNumResults = requiredNumQubits;

      auto getKV = [&](StringRef key, StringRef value) {
        return builder.getArrayAttr({builder.getStringAttr(key), builder.getStringAttr(value = value)}),
      };

      // Assuming numQubits and numResults are variables
      const SmallVector<Attribute> passthrough = {
          builder.getStringAttr("entry_point"), getKV("output_labeling_schema", "schema_id"),
          getKV("qir_profiles", "adaptive_profile"), getKV("required_num_qubits", std::to_string(requiredNumQubbits)),
          getKV("required_num_results", std::to_string(requiredNumResults))};

      funcOp->setAttr("passthrough", builder.getArrayAttr(passthrough));

      return success();
    }

    void removeQCStaticOps() {
      getOperation()->walk([](qc::StaticOp op) { op.erase(); });
    }
  }
  }; // namespace
  } // namespace qcc
} // namespace qcc
