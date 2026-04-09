#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Type.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

using namespace mlir;

namespace {

/// Map any of the (unitary) gate-ops to their QIR QIS function declaration.
StringRef mapQCGateToQIS(Operation* op) {
  return llvm::TypeSwitch<Operation*, StringRef>(op)
      .Case<qc::XOp>([](auto) { return qcc::qirQisX; })
      .Case<qc::HOp>([](auto) { return qcc::qirQisH; })
      .Default([](auto) { return ""; });
}

/// Map any of the (unitary) gate-ops to the controlled version of their QIS function (if it exists).
StringRef mapQCGateToQISControlled(Operation* op) {
  return llvm::TypeSwitch<Operation*, StringRef>(op)
      .Case<qc::XOp>([](auto) { return qcc::qirQisCX; })
      .Default([](auto) { return ""; });
}

/// FIXME: docstring
SmallVector<Value> qubitsToPtrs(OpBuilder& builder, ValueRange qubitValues) {
  SmallVector<Value> ptrValues;
  ptrValues.reserve(qubitValues.size());

  for (auto qubitValue : qubitValues) {
    auto* defOp = qubitValue.getDefiningOp();
    assert(defOp && isa<qc::StaticOp>(defOp) &&
           "The pass assumes that all qubits come from static allocations (in particular no function args).");
    auto alloc = cast<qc::StaticOp>(defOp);

    int64_t index = alloc.getIndex();

    auto i64Type = builder.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

    auto constantOp = LLVM::ConstantOp::create(builder, defOp->getLoc(), i64Type, builder.getI64IntegerAttr(index));
    auto ptrOp = LLVM::IntToPtrOp::create(builder, defOp->getLoc(), ptrType, {constantOp});

    ptrValues.push_back(ptrOp);
  }

  return ptrValues;
}

struct QCToQIRTypeConverter final : LLVMTypeConverter {
  explicit QCToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    addConversion([ctx](qc::QubitType) { return LLVM::LLVMPointerType::get(ctx); });
  }
};

struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter& rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto mzFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirQisMZ);
    auto readFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirRtReadResult);

    if (!mzFnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qcc::qirQisMZ;
    }

    if (!readFnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qcc::qirRtReadResult;
    }

    // NOTE: Qubit and result pointer share the same index.
    auto qubit = op.getQubit();
    auto ptrs = qubitsToPtrs(rewriter, {qubit, qubit});

    LLVM::CallOp::create(rewriter, op.getLoc(), mzFnDecl, ptrs);
    auto callReadOp = LLVM::CallOp::create(rewriter, op.getLoc(), readFnDecl, {ptrs[1]});

    rewriter.replaceOp(op, callReadOp);
    return success();
  }
};

/// We rely on the fact that the signature of qc gates and the corresponding QIR QIS function fits.
struct UnitaryLowering : public ConversionPattern {
  UnitaryLowering(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> /*operands*/,
                                ConversionPatternRewriter& rewriter) const override {
    if (!isa<qc::QCDialect>(op->getDialect()) || !isa<qc::UnitaryOpInterface>(op)) {
      return failure();
    }

    if (isa<qc::CtrlOp>(op)) {
      return rewriteControlledGate(op, rewriter);
    }

    if (!isa<qc::CtrlOp>(op->getParentOp())) {
      return rewriteNonControlledGate(op, rewriter);
    }

    return failure();
  }

private:
  /// For simple gates like `qc.h %0 : !qc.qubit` which are not inside ctrlOp.
  static LogicalResult rewriteNonControlledGate(Operation* op, ConversionPatternRewriter& rewriter) {
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto qisName = mapQCGateToQIS(op);
    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qisName);
    if (!fnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qisName;
    }

    auto ptrs = qubitsToPtrs(rewriter, op->getOperands());

    auto callOp = LLVM::CallOp::create(rewriter, op->getLoc(), fnDecl, ptrs);
    rewriter.eraseOp(op);

    return success();
  }

  /// For controlled gates like `qc.ctrl(%0) { qc.x %1 : !qc.qubit } : !qc.qubit`. Multiple control is not supported.
  static LogicalResult rewriteControlledGate(Operation* op, ConversionPatternRewriter& rewriter) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto ctrlOp = cast<qc::CtrlOp>(op);

    if (ctrlOp.getNumControls() != 1) {
      return ctrlOp->emitError() << "Expected exactly one control qubit. Found " << ctrlOp.getNumControls() << ".";
    }

    auto* body = ctrlOp.getBody();
    Operation* innerOp = nullptr;

    for (auto& inner : body->without_terminator()) {
      if (innerOp != nullptr) {
        return op->emitError() << "Expected exactly one qc op inside ctrlOp. Found more than one.";
      }

      innerOp = &inner;
    }

    if (innerOp == nullptr) {
      return op->emitError() << "Expected exactly one qc op inside ctrlOp. Found none.";
    }

    auto qisName = mapQCGateToQISControlled(innerOp);
    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qisName);
    if (!fnDecl) {
      return op->emitError() << "QIR QIS declaration (controlled version) not found: " << qisName;
    }

    auto allPtrs = qubitsToPtrs(rewriter, ctrlOp.getControl(0));
    auto targetPtrs = qubitsToPtrs(rewriter, innerOp->getOperands());
    allPtrs.append(targetPtrs);

    auto callOp = LLVM::CallOp::create(rewriter, op->getLoc(), fnDecl, allPtrs);
    rewriter.eraseOp(op);

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
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto* ctx = funcOp.getContext();

    // TODO: For simplicity we assume that only entry_point functions have
    // quantum operations. Before we extend support we have to figure out how to
    // do qubit mapping across function calls.
    if (!funcOp->hasAttr("qcc.entry_point")) {
      return;
    }

    if (failed(setRequiredNumQubits())) {
      return signalPassFailure();
    }

    if (failed(insertRtInit())) {
      return signalPassFailure();
    }

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<qc::QCDialect>();
    target.addLegalOp<qc::StaticOp>(); // take care of slightly later.

    QCToQIRTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    patterns.add<UnitaryLowering, MeasureLowering>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeQCStaticOps();
  }

private:
  LogicalResult insertRtInit() {
    func::FuncOp funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto* context = funcOp.getContext();

    auto initFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirRtInit);

    if (!initFnDecl) {
      return moduleOp.emitError() << "missing required declaration of QIR runtime function: " << qcc::qirRtInit;
    }

    auto loc = funcOp.getLoc();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&funcOp.front());

    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto nullPtr = LLVM::ZeroOp::create(builder, loc, ptrType);
    LLVM::CallOp::create(builder, loc, initFnDecl, ValueRange{nullPtr});

    return llvm::success();
  }

  /// Determine the number of qubits from the size attribute of the alloc
  /// operations. Returns failure iff they disagree. In case of success the
  /// corresponding attribute is written to the current op (function).
  LogicalResult setRequiredNumQubits() {
    func::FuncOp funcOp = getOperation();
    uint64_t numQubits = 0;

    WalkResult result = funcOp->walk([&](qc::AllocOp allocOp) -> WalkResult {
      auto maybeIndex = allocOp.getRegisterIndex();
      auto maybeSize = allocOp.getRegisterSize();

      if (!maybeIndex.has_value()) {
        return allocOp.emitError("allocation missing register index");
      }

      if (!maybeSize.has_value()) {
        return allocOp.emitError("allocation missing register size");
      }

      uint64_t currentSize = maybeSize.value();

      if (numQubits == 0) {
        numQubits = currentSize;
      } else if (numQubits != currentSize) {
        return allocOp.emitError() << "conflicting register size: expected " << numQubits << " but found "
                                   << currentSize;
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return llvm::failure();
    }

    // FIXME: set required_num_qubits.

    return llvm::success();
  }

  void removeQCStaticOps() {
    SmallVector<Operation*> toErase;

    getOperation()->walk([&](Operation* op) {
      if (isa<qc::StaticOp>(op)) {
        toErase.push_back(op);
      }
    });

    for (Operation* op : toErase) {
      op->erase();
    }
  }
};

} // namespace qcc
