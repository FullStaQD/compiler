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
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

using namespace mlir;

namespace {

/// Map any of the (unitary) gates to their QIR QIS function declaration.
StringRef mapQCGateToQIS(Operation* op) {
  return llvm::TypeSwitch<Operation*, StringRef>(op)
      .Case<qc::XOp>([](auto) { return qcc::QIR_QIS_X; })
      .Case<qc::HOp>([](auto) { return qcc::QIR_QIS_H; })
      .Default([](auto) { return ""; });
}

/// FIXME: docstring
SmallVector<Value> qubitsToPtrs(OpBuilder& builder, ValueRange qubitValues) {
  SmallVector<Value> ptrValues;
  ptrValues.reserve(qubitValues.size());

  for (auto qubitValue : qubitValues) {
    auto* defOp = qubitValue.getDefiningOp();
    assert(defOp && isa<qc::AllocOp>(defOp) && "The pass assumes that all qubits come from allocations.");
    auto alloc = cast<qc::AllocOp>(defOp);
    assert(alloc.getRegisterIndex().has_value() &&
           "Early in the pass we made sure that qc.alloc has a register index.");
    int64_t index = alloc.getRegisterIndex().value();

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

/// FIXME: we need to read the measurement result too!
struct MeasureLowering : public OpConversionPattern<qc::MeasureOp> {
  using OpConversionPattern<qc::MeasureOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(qc::MeasureOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto mzFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_QIS_MZ);
    auto readFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_RT_READ_RESULT);

    if (!mzFnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qcc::QIR_QIS_MZ;
    }

    if (!readFnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qcc::QIR_RT_READ_RESULT;
    }

    // NOTE: Qubit and result pointer share the same index.
    auto qubit = op.getQubit();
    auto ptrs = qubitsToPtrs(rewriter, {qubit, qubit});

    auto callMZOp = LLVM::CallOp::create(rewriter, op.getLoc(), mzFnDecl, ptrs);
    auto callReadOp = LLVM::CallOp::create(rewriter, op.getLoc(), readFnDecl, {ptrs[1]});

    rewriter.replaceOp(op, callReadOp);
    return success();
  }
};

/// We rely on the fact that the signature of qc gates and the corresponding QIR QIS function fits.
struct UnitaryLowering : public ConversionPattern {
  UnitaryLowering(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getDialect()->getNamespace() != "qc") { // FIXME: do not hardcode the name
      return failure();
    }

    // FIXME: not great, if there is a trait "unitary" use that one instead!
    if (llvm::isa<qc::AllocOp>(op) || llvm::isa<qc::MeasureOp>(op)) {
      return failure();
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();

    // NOTE: if the function declaration is not found we report and error and
    // leave it to the pass to observe that some ops could not be converted (qc
    // dialect is illegal).
    auto qisName = mapQCGateToQIS(op);
    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qisName);
    if (!fnDecl) {
      return op->emitError() << "QIR QIS declaration not found: " << qisName;
    }

    auto ptrs = qubitsToPtrs(rewriter, op->getOperands());

    auto callOp = LLVM::CallOp::create(rewriter, op->getLoc(), fnDecl, ptrs);
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
    // target.addIllegalDialect<qc::QCDialect>(); // FIXME:
    // target.addIllegalOp<qc::XOp>();
    // target.addIllegalOp<qc::HOp>();
    // target.addIllegalOp<qc::MeasureOp>();

    QCToQIRTypeConverter typeConverter(ctx);
    RewritePatternSet patterns(ctx);
    patterns.add<UnitaryLowering, MeasureLowering>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeQubitAllocsAndDeallocs();
  }

private:
  LogicalResult insertRtInit() {
    func::FuncOp funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto* context = funcOp.getContext();

    auto initFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::QIR_RT_INIT);

    if (!initFnDecl) {
      return moduleOp.emitError() << "missing required declaration of QIR runtime function: " << qcc::QIR_RT_INIT;
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

  // FIXME: dealloc must happen earlier
  void removeQubitAllocsAndDeallocs() {
    SmallVector<Operation*> toErase;

    getOperation()->walk([&](Operation* op) {
      if (isa<qc::AllocOp, qc::DeallocOp>(op)) {
        toErase.push_back(op);
      }
    });

    for (Operation* op : toErase) {
      op->erase();
    }
  }
};

} // namespace qcc
