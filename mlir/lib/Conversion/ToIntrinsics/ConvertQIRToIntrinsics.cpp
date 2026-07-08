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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <mlir/Pass/Pass.h>
#include <optional>

using namespace mlir;

/// Element count used for the scalable `vector<[N]xi8>` type that carries a qubit index into
/// the QV intrinsics (see `loadQubitIndexVec`).
///
/// This target declares `zve32x` (ELEN=32), and the RVV spec requires LMUL >= SEW/ELEN, i.e.
/// LMUL >= 8/32 = mf4 at SEW=8. A single-element vector (`vector<[1]xi8>`) legalizes to the
/// smallest fractional LMUL that can hold it, mf8, which is below that minimum and traps as an
/// illegal instruction on this hardware. `vector<[4]xi8>` legalizes to mf2 (comfortably legal,
/// and the exact LMUL used by HiSEP-Q's own reference-compiler output), while the actual runtime
/// element count processed by the QV intrinsic call is still 1 (see the `vl` argument in
/// `QISCallLowering`) -- only the container's minimum legal size changes, not the semantics.
static constexpr int64_t kQubitIndexVecNumElements = 4;

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

/// Returns the (possibly newly-created) internal-linkage global constant holding `index` in its
/// first byte (remaining `kQubitIndexVecNumElements - 1` bytes are zero padding), reusing one
/// global per distinct index within the module.
///
/// HiSEP-Q's vector unit does not implement `vmv.s.x`/`vmv.v.i` (the instructions LLVM's
/// generic vector legalizer would otherwise select for `insertelement(undef, ..., 0)`), so
/// qubit-index vectors must instead be materialized via a memory load (`vle8.v`), which
/// requires the index to live in addressable memory first. The padding bytes exist solely so
/// that a `vector<[kQubitIndexVecNumElements]xi8>` load (see `loadQubitIndexVec`) doesn't read
/// past the global.
static LLVM::GlobalOp getOrCreateIndexGlobal(OpBuilder& builder, ModuleOp moduleOp,
                                             llvm::DenseMap<int64_t, LLVM::GlobalOp>& indexGlobals, int64_t index) {
  auto [it, inserted] = indexGlobals.try_emplace(index);
  if (!inserted) {
    return it->second;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto i8Type = builder.getIntegerType(8);
  auto i8ArrayType = LLVM::LLVMArrayType::get(i8Type, kQubitIndexVecNumElements);
  std::string name = (".qcc_qv_idx_" + llvm::Twine(index)).str();

  std::string bytes(1, static_cast<char>(index));
  bytes.resize(kQubitIndexVecNumElements, '\0');

  auto global = LLVM::GlobalOp::create(builder, moduleOp.getLoc(), i8ArrayType, /*isConstant=*/true,
                                       LLVM::Linkage::Internal, name, builder.getStringAttr(bytes));
  it->second = global;
  return global;
}

/// Encodes a qubit index as a `vector<[kQubitIndexVecNumElements]xi8>` scalable vector for QV
/// intrinsics, by loading it from a dedicated global rather than synthesizing it in-register
/// (see `getOrCreateIndexGlobal`).
static Value loadQubitIndexVec(OpBuilder& builder, Location loc, ModuleOp moduleOp,
                               llvm::DenseMap<int64_t, LLVM::GlobalOp>& indexGlobals, int64_t index) {
  auto i8Type = builder.getIntegerType(8);
  auto vecType = VectorType::get({kQubitIndexVecNumElements}, i8Type, /*scalableDims=*/{true});

  LLVM::GlobalOp global = getOrCreateIndexGlobal(builder, moduleOp, indexGlobals, index);
  Value addr = LLVM::AddressOfOp::create(builder, loc, global);
  return LLVM::LoadOp::create(builder, loc, vecType, addr);
}

namespace {

/// Rewrites `llvm.call @__quantum__qis__*__body(qubit_ptr, ...)` into the
/// corresponding `llvm.call_intrinsic "llvm.riscv.qv.*"(vec, ...)`.
///
/// Qubit pointer arguments (produced by `llvm.inttoptr` of a constant index)
/// are re-encoded as `vector<[kQubitIndexVecNumElements]xi8>` scalable vectors.
struct QISCallLowering : public OpRewritePattern<LLVM::CallOp> {
  QISCallLowering(MLIRContext* context, llvm::DenseMap<int64_t, LLVM::GlobalOp>& indexGlobals)
      : OpRewritePattern(context), indexGlobals(indexGlobals) {}

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
    auto moduleOp = callOp->getParentOfType<ModuleOp>();

    Value blockImm = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(0));
    Value vl = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> args;

    if (*callee == qcc::qirQisCX) {
      // QVPairIntrinsic: (vs1: vec<[1]xi8>, vs2: vec<[1]xi8>, block_imm: i32, vl: i32)
      auto ctrlIdx = getQubitIndexFromPtr(operands[0]);
      auto tgtIdx = getQubitIndexFromPtr(operands[1]);
      if (!ctrlIdx || !tgtIdx) {
        return callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                                "for '__quantum__qis__cx__body'");
      }

      args = {loadQubitIndexVec(rewriter, loc, moduleOp, indexGlobals, *ctrlIdx),
              loadQubitIndexVec(rewriter, loc, moduleOp, indexGlobals, *tgtIdx), blockImm, vl};
    } else {
      // QVSingleIntrinsic: (vs1: vec<[1]xi8>, rs2: i32, block_imm: i32, vl: i32)
      // For mz__body: operands[0] = qubit_ptr, operands[1] = result_ptr (discarded).
      auto qubitIdx = getQubitIndexFromPtr(operands[0]);
      if (!qubitIdx) {
        return callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                                "for '")
               << *callee << "'";
      }

      Value tag = LLVM::ConstantOp::create(rewriter, loc, i32Type, rewriter.getI32IntegerAttr(0));
      args = {loadQubitIndexVec(rewriter, loc, moduleOp, indexGlobals, *qubitIdx), tag, blockImm, vl};
    }

    LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr(intrName), args);
    rewriter.eraseOp(callOp);
    return success();
  }

private:
  llvm::DenseMap<int64_t, LLVM::GlobalOp>& indexGlobals;
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

struct ConvertQIRToIntrinsics final : impl::ConvertQIRToIntrinsicsBase<ConvertQIRToIntrinsics> {
  using ConvertQIRToIntrinsicsBase::ConvertQIRToIntrinsicsBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto* ctx = moduleOp.getContext();

    llvm::DenseMap<int64_t, LLVM::GlobalOp> indexGlobals;

    RewritePatternSet patterns(ctx);
    patterns.add<QISCallLowering>(ctx, indexGlobals);
    patterns.add<ReadResultLowering, RtInitLowering, RecordOutputLowering>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    removeQIRDeclarations();
  }

private:
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
};

} // namespace
} // namespace qcc
