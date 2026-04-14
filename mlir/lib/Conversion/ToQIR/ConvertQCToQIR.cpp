#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Dialect/Aux/IR/Aux.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Type.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace qcc;

namespace {

/// Map any of the (unitary) gate-ops to their QIR QIS function declaration if possible.
///
/// Returns an empty string upon failure. Succeeds iff the gate is among the supported native gate set.
///
/// TODO: The native gate set is currently hardcoded.
StringRef mapUnitaryToQIS(qc::UnitaryOpInterface unitaryOp) {
  if (unitaryOp.getNumControls() == 0) {
    return llvm::TypeSwitch<Operation*, StringRef>(unitaryOp)
        .Case<qc::XOp>([](auto) { return qcc::qirQisX; })
        .Case<qc::HOp>([](auto) { return qcc::qirQisH; })
        .Default([](auto) { return ""; });
  }

  if (unitaryOp.getNumControls() == 1) {
    auto ctrlOp = cast<qc::CtrlOp>(unitaryOp);
    auto bodyOp = ctrlOp.getBodyUnitary();

    return llvm::TypeSwitch<Operation*, StringRef>(bodyOp)
        .Case<qc::XOp>([](auto) { return qcc::qirQisCX; })
        .Default([](auto) { return ""; });
  }

  return "";
}

/// For each of the qubit values it assumes that it was created by a static op, like this:
///
/// ```mlir
/// %0 = qc.static 42 : !qc.qubit
/// ```
///
/// it then adds the following for each qubit at the current insertion point (and leaves the `qc.static` as is):
///
/// ```mlir
/// %1 = llvm.mlir.constant(42 : i64) : i64
/// %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
/// ```
///
/// Finally it returns the list of all created `ptr` values (`%2` in the above example is one of these ptr).
SmallVector<Value> qubitsToPtrs(OpBuilder& builder, ValueRange qubitValues) {
  SmallVector<Value> ptrValues;
  ptrValues.reserve(qubitValues.size());

  for (auto qubitValue : qubitValues) {
    auto* defOp = qubitValue.getDefiningOp();
    assert(defOp && isa<qc::StaticOp>(defOp) &&
           "The pass assumes that all qubits come from static allocations (in particular no function args).");
    auto alloc = cast<qc::StaticOp>(defOp);

    auto index = static_cast<int64_t>(alloc.getIndex());

    auto i64Type = builder.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());

    auto constantOp = LLVM::ConstantOp::create(builder, defOp->getLoc(), i64Type, builder.getI64IntegerAttr(index));
    auto ptrOp = LLVM::IntToPtrOp::create(builder, defOp->getLoc(), ptrType, {constantOp});

    ptrValues.push_back(ptrOp);
  }

  return ptrValues;
}

/// To be used in a rewrite pattern.
InFlightDiagnostic emitMissingDeclError(Operation* op, StringRef name) {
  return op->emitError() << "QIR QIS declaration not found: '" << name << "'";
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
      return emitMissingDeclError(op, qcc::qirQisMZ);
    }

    if (!readFnDecl) {
      return emitMissingDeclError(op, qcc::qirRtReadResult);
    }

    // TODO: This holds only for HiSEP-Q.
    // NOTE: Qubit and result pointer share the same index.
    auto qubit = op.getQubit();
    auto ptrs = qubitsToPtrs(rewriter, {qubit, qubit});

    LLVM::CallOp::create(rewriter, op.getLoc(), mzFnDecl, ptrs);
    auto callReadOp = LLVM::CallOp::create(rewriter, op.getLoc(), readFnDecl, {ptrs[1]});

    rewriter.replaceOp(op, callReadOp);
    return success();
  }
};

struct RecordBoolLowering : public OpConversionPattern<aux::RecordBoolOp> {
  using OpConversionPattern<aux::RecordBoolOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(aux::RecordBoolOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    StringRef labelName = op.getLabel(); // Returns the string from FlatSymbolRefAttr

    auto addressOf =
        LLVM::AddressOfOp::create(rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()), labelName);
    LLVM::CallOp::create(rewriter, loc, TypeRange(), qirRtBoolRecordOutput, ValueRange{adaptor.getValue(), addressOf});

    rewriter.eraseOp(op);
    return success();
  }
};

/// We rely on the fact that the signature of qc gates and the corresponding QIR QIS function fits.
struct UnitaryLowering : public ConversionPattern {
  UnitaryLowering(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> /*operands*/,
                                ConversionPatternRewriter& rewriter) const override {
    auto unitaryOp = dyn_cast<qc::UnitaryOpInterface>(op);
    if (!unitaryOp || !isa<qc::QCDialect>(op->getDialect())) {
      return failure();
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto qisName = mapUnitaryToQIS(unitaryOp);
    auto fnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qisName);
    if (!fnDecl) {
      return emitMissingDeclError(unitaryOp, qisName);
    }

    auto allPtrs = qubitsToPtrs(rewriter, unitaryOp.getControls());
    auto targetPtrs = qubitsToPtrs(rewriter, unitaryOp.getTargets());
    allPtrs.append(targetPtrs);

    LLVM::CallOp::create(rewriter, op->getLoc(), fnDecl, allPtrs);
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

namespace qcc {

#define GEN_PASS_DEF_CONVERTQCTOQIR
#include "qcc/Conversion/ToQIR/ToQIR.h.inc"

struct ConvertQCToQIR final : impl::ConvertQCToQIRBase<ConvertQCToQIR> {
  using ConvertQCToQIRBase::ConvertQCToQIRBase;

protected:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto* ctx = funcOp.getContext();

    // TODO: assume that only entrypoints contain quantum ops.
    if (!funcOp->hasAttr("qcc.entry_point")) {
      return;
    }

    if (failed(setEntryPointAttrs())) {
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
    patterns.add<UnitaryLowering, MeasureLowering, RecordBoolLowering>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeQCStaticOps();
  }

private:
  LogicalResult insertRtInit() {
    func::FuncOp funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    auto* ctx = funcOp.getContext();

    auto initFnDecl = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(qcc::qirRtInit);

    if (!initFnDecl) {
      return moduleOp.emitError() << "missing required declaration of QIR runtime function: " << qcc::qirRtInit;
    }

    auto loc = funcOp.getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&funcOp.front());

    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto nullPtr = LLVM::ZeroOp::create(builder, loc, ptrType);
    LLVM::CallOp::create(builder, loc, initFnDecl, ValueRange{nullPtr});

    return llvm::success();
  }

  uint64_t getRequiredNumQubits() {
    func::FuncOp funcOp = getOperation();
    uint64_t numQubits = 0;
    funcOp->walk([&](qc::StaticOp op) -> void {
      auto index = op.getIndex();
      numQubits = std::max(numQubits, index + 1);
    });
    return numQubits;
  }

  /// Attach all the relevant QIR attributes (like `required_num_qubits`) to the function.
  LogicalResult setEntryPointAttrs() {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    auto requiredNumQubits = getRequiredNumQubits();
    auto requiredNumResults = requiredNumQubits; // TODO: holds only for HiSEP-Q!

    auto getKV = [&](StringRef key, StringRef value) {
      return builder.getArrayAttr({builder.getStringAttr(key), builder.getStringAttr(value)});
    };

    // Assuming numQubits and numResults are variables
    SmallVector<Attribute> passthrough = {
        builder.getStringAttr("entry_point"), getKV("output_labeling_schema", "schema_id"),
        getKV("qir_profiles", "adaptive_profile"), getKV("required_num_qubits", std::to_string(requiredNumQubits)),
        getKV("required_num_results", std::to_string(requiredNumResults))};

    funcOp->setAttr("passthrough", builder.getArrayAttr(passthrough));

    return success();
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
