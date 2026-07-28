// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/ToQIR/Constants.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <mlir/Pass/Pass.h>
#include <optional>

using namespace mlir;

/// Maps a QIR QIS function name to its RISC-V QV intrinsic counterpart.
/// Returns an empty string for unrecognized / unsupported names.
static StringRef mapQISToIntrinsic(StringRef qisName) {
  return llvm::StringSwitch<StringRef>(qisName)
      .Case(qcc::qirQisH, "llvm.riscv.qv.h")
      .Case(qcc::qirQisX, "llvm.riscv.qv.x")
      .Case(qcc::qirQisCX, "llvm.riscv.qv.cx")
      .Case(qcc::qirQisMZ, "llvm.riscv.qv.mz")
      .Default("");
}

/// Returns true when `name` is a QIR runtime / QIS symbol that this pass
/// handles (and therefore must not appear in the output).
static bool isHandledQIRSymbol(StringRef name) {
  return name == qcc::qirRtInit || name == qcc::qirRtReadResult || name == qcc::qirRtBoolRecordOutput ||
         name == qcc::qirRtIntRecordOutput || !mapQISToIntrinsic(name).empty();
}

/// Tries to extract the qubit index encoded in a ptr obtained via:
///   `llvm.inttoptr (llvm.mlir.constant N : i64) : !llvm.ptr`
static std::optional<int64_t> getQubitIndexFromPtr(Value ptrValue) {
  auto intToPtrOp = ptrValue.getDefiningOp<LLVM::IntToPtrOp>();
  if (!intToPtrOp) {
    return std::nullopt;
  }

  auto constOp = intToPtrOp.getArg().getDefiningOp<LLVM::ConstantOp>();
  if (!constOp) {
    return std::nullopt;
  }

  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr) {
    return std::nullopt;
  }

  return intAttr.getInt();
}

/// Encodes a qubit index as a `vector<[8]xi8>` scalable vector for QV intrinsics.
/// The index is inserted into lane 0 of an undef vector. The `nxv8i8` element
/// type matches the HiSEP-Q RISC-V backend's QV instruction-selection patterns.
static Value qubitIndexToVec(OpBuilder& builder, Location loc, int64_t index) {
  auto i8Type = builder.getIntegerType(8);
  auto i32Type = builder.getI32Type();
  auto vecType = VectorType::get({8}, i8Type, /*scalableDims=*/{true});

  Value indexConst = LLVM::ConstantOp::create(builder, loc, i8Type, builder.getIntegerAttr(i8Type, index));
  Value undef = LLVM::UndefOp::create(builder, loc, vecType);
  Value lane = LLVM::ConstantOp::create(builder, loc, i32Type, builder.getI32IntegerAttr(0));
  return {LLVM::InsertElementOp::create(builder, loc, undef, indexConst, lane)};
}

namespace {

/// Rewrites `llvm.call @__quantum__qis__*__body(qubit_ptr, ...)` into the
/// corresponding `llvm.call_intrinsic "llvm.riscv.qv.*"(vec, ...)`.
///
/// Qubit pointer arguments (produced by `llvm.inttoptr` of a constant index)
/// are re-encoded as `vector<[8]xi8>` scalable vectors.
struct QISCallLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee) {
      return failure();
    }

    StringRef intrName = mapQISToIntrinsic(*callee);
    if (intrName.empty()) {
      return failure();
    }

    auto loc = callOp.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto operands = callOp.getArgOperands();

    Value blockImm = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(0));
    Value vl = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> args;

    if (*callee == qcc::qirQisCX) {
      // QVPairIntrinsic: (vs1: vec<[8]xi8>, vs2: vec<[8]xi8>, block_imm: i32, vl: i32)
      auto ctrlIdx = getQubitIndexFromPtr(operands[0]);
      auto tgtIdx = getQubitIndexFromPtr(operands[1]);
      if (!ctrlIdx || !tgtIdx) {
        return callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                                "for '__quantum__qis__cx__body'");
      }

      args = {qubitIndexToVec(rewriter, loc, *ctrlIdx), qubitIndexToVec(rewriter, loc, *tgtIdx), blockImm, vl};
    } else {
      // QVSingleIntrinsic: (vs1: vec<[8]xi8>, rs2: i32, block_imm: i32, vl: i32)
      // For mz__body: operands[0] = qubit_ptr, operands[1] = result_ptr (discarded).
      auto qubitIdx = getQubitIndexFromPtr(operands[0]);
      if (!qubitIdx) {
        return callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                                "for '")
               << *callee << "'";
      }

      Value tag = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(0));
      args = {qubitIndexToVec(rewriter, loc, *qubitIdx), tag, blockImm, vl};
    }

    LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr(intrName), args);
    rewriter.eraseOp(callOp);
    return success();
  }
};

/// Replaces `llvm.call @__quantum__rt__read_result(%result_ptr)` with `undef : i1`.
///
/// TODO: A proper `qv.read_result` intrinsic is not yet defined in IntrinsicsRISCVXQV.td.
struct ReadResultLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || *callee != qcc::qirRtReadResult) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(callOp, rewriter.getI1Type());
    return success();
  }
};

/// Erases `llvm.call @__quantum__rt__initialize(ptr)`.
/// The runtime initialization step is not needed on the bare-metal intrinsic path.
struct RtInitLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || *callee != qcc::qirRtInit) {
      return failure();
    }

    rewriter.eraseOp(callOp);
    return success();
  }
};

/// Erases `llvm.call @__quantum__rt__bool_record_output` and
/// `llvm.call @__quantum__rt__int_record_output`.
///
/// TODO: No intrinsic equivalent for output recording exists yet.
struct RecordOutputLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee) {
      return failure();
    }

    if (*callee != qcc::qirRtBoolRecordOutput && *callee != qcc::qirRtIntRecordOutput) {
      return failure();
    }

    rewriter.eraseOp(callOp);
    return success();
  }
};

} // namespace

namespace qcc {

#define GEN_PASS_DEF_CONVERTQIRTOINTRINSICS
#include "qcc/Conversion/ToIntrinsics/ToIntrinsics.h.inc"

namespace {

/// Whether `funcOp` carries the `entry_point` passthrough attribute set by
/// `ConvertQCToQIR::setEntryPointAttrs`.
bool isEntryPointFunc(LLVM::LLVMFuncOp funcOp) {
  auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
  if (!passthrough) {
    return false;
  }

  return llvm::any_of(passthrough, [](Attribute attr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    return strAttr && strAttr.getValue() == "entry_point";
  });
}

struct ConvertQIRToIntrinsics final : impl::ConvertQIRToIntrinsicsBase<ConvertQIRToIntrinsics> {
  using ConvertQIRToIntrinsicsBase::ConvertQIRToIntrinsicsBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto* ctx = moduleOp.getContext();

    FailureOr<LLVM::LLVMFuncOp> entryPoint = findEntryPoint(moduleOp);
    if (failed(entryPoint)) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    patterns.add<QISCallLowering, ReadResultLowering, RtInitLowering, RecordOutputLowering>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeQIRDeclarations();

    if (*entryPoint) {
      emitStartFunc(moduleOp, *entryPoint);
    }
  }

private:
  /// Returns the entry point of the module, or a null func if it has none. At most one function may
  /// be tagged, as the hardware boots at a single address.
  static FailureOr<LLVM::LLVMFuncOp> findEntryPoint(ModuleOp moduleOp) {
    LLVM::LLVMFuncOp entryPoint;
    for (auto funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
      if (!isEntryPointFunc(funcOp)) {
        continue;
      }
      if (entryPoint) {
        funcOp.emitError("expected at most one function tagged as the entry point, but found '")
            << entryPoint.getName() << "' and '" << funcOp.getName() << "'";
        return failure();
      }
      entryPoint = funcOp;
    }
    return entryPoint;
  }

  /// Removes leftover QIR function declarations whose call sites were erased.
  void removeQIRDeclarations() {
    SmallVector<LLVM::LLVMFuncOp> toErase;
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      if (isHandledQIRSymbol(funcOp.getName())) {
        toErase.push_back(funcOp);
      }
    });
    for (auto funcOp : toErase) {
      funcOp.erase();
    }
  }

  /// Emits `_start`, which supersedes `entryPoint` as the entry point of the hardware.
  ///
  /// HiSEP-Q jumps to the fixed boot address (see hisepq.ld) at reset. There is no caller, and `sp`
  /// holds whatever the core reset with. `_start` sets `sp` to the linker-provided `__stack_top`,
  /// calls `entryPoint` with a `jalr` so that it can return normally, and halts in an infinite loop
  /// if it does.
  ///
  /// The sequence must be a single inline-asm block: with a `llvm.call`, the backend saves `ra` in
  /// a prologue, which is emitted ahead of the `sp` setup.
  static void emitStartFunc(ModuleOp moduleOp, LLVM::LLVMFuncOp entryPoint) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToEnd(moduleOp.getBody());
    Location loc = entryPoint.getLoc();

    // `extern char __stack_top[];`, defined by hisepq.ld. Only its address is used.
    auto stackTopType = LLVM::LLVMArrayType::get(builder.getI8Type(), 0);
    auto stackTop = LLVM::GlobalOp::create(builder, loc, stackTopType, /*isConstant=*/true, LLVM::Linkage::External,
                                           "__stack_top", /*value=*/Attribute());

    auto startFuncType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {});
    auto startFunc = LLVM::LLVMFuncOp::create(builder, loc, "_start", startFuncType);
    builder.setInsertionPointToStart(startFunc.addEntryBlock(builder));

    Value stackTopAddr = LLVM::AddressOfOp::create(builder, loc, stackTop);
    Value entryAddr = LLVM::AddressOfOp::create(builder, loc, entryPoint);

    auto asmDialect = LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT);
    LLVM::InlineAsmOp::create(builder, loc, /*resultTypes=*/TypeRange(),
                              /*operands=*/ValueRange{stackTopAddr, entryAddr},
                              /*asm_string=*/"mv sp, $0\njalr ra, 0($1)\n1:\nj 1b",
                              /*constraints=*/"r,r", /*has_side_effects=*/true,
                              /*is_align_stack=*/false, LLVM::TailCallKind::None,
                              /*asm_dialect=*/asmDialect, /*operand_attrs=*/ArrayAttr());

    LLVM::UnreachableOp::create(builder, loc);
  }
};

} // namespace
} // namespace qcc
