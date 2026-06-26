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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

// ---------------------------------------------------------------------------
// Per-block lowering context
//
// Caches shared values (undef vectors, zero constant) so consecutive batches
// that produce the same types/constants reuse existing SSA values rather than
// creating duplicates.  This matches the CSE behaviour that the old single-op
// pattern rewriter produced implicitly, and keeps the MLIR output clean.
// ---------------------------------------------------------------------------

struct BlockContext {
  OpBuilder builder;
  // VectorType → undef value; linear scan is fine (≤2 sizes per block).
  SmallVector<std::pair<Type, Value>> undefCache;
  // N → constant(N : i32); caches vl and lane-0 zero so repeated same-size
  // batches share SSA values (matching old pass CSE behaviour).
  SmallVector<std::pair<int64_t, Value>> i32Cache;

  explicit BlockContext(MLIRContext* ctx) : builder(ctx) {}

  /// Returns (or lazily creates) `llvm.mlir.undef : vecType` at `loc`.
  Value getUndef(Location loc, VectorType vecType) {
    for (auto& [type, val] : undefCache) {
      if (type == vecType)
        return val;
    }
    auto val = LLVM::UndefOp::create(builder, loc, vecType);
    undefCache.push_back({vecType, val});
    return val;
  }

  /// Returns (or lazily creates) `llvm.mlir.constant(N : i32) : i32` at `loc`.
  Value getI32(Location loc, int64_t N) {
    for (auto& [n, val] : i32Cache) {
      if (n == N)
        return val;
    }
    auto val = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(N));
    i32Cache.push_back({N, val});
    return val;
  }
};

/// Builds a `vector<[N]xi8>` scalable vector whose lane `i` holds `indices[i]`.
/// Lane-0 index reuses `ctx.getZeroI32()` to share the SSA value with other
/// zero-valued i32 constants (tag, block_imm) in the same batch.
static Value buildQubitVec(BlockContext& ctx, Location loc, ArrayRef<int64_t> indices) {
  int64_t N = (int64_t)indices.size();
  auto i8Type = ctx.builder.getIntegerType(8);
  auto vecType = VectorType::get({N}, i8Type, /*scalableDims=*/{true});

  Value vec = ctx.getUndef(loc, vecType);

  for (int64_t i = 0; i < N; ++i) {
    Value idx = LLVM::ConstantOp::create(ctx.builder, loc, i8Type, ctx.builder.getIntegerAttr(i8Type, indices[i]));
    // Use the shared i32 cache for all lane indices so that lane-0, tag, and
    // block_imm all share one SSA value (matching old pass CSE behaviour).
    Value lane = ctx.getI32(loc, i);
    vec = LLVM::InsertElementOp::create(ctx.builder, loc, vec, idx, lane);
  }
  return vec;
}

// ---------------------------------------------------------------------------
// Batch definition and emission
// ---------------------------------------------------------------------------

/// One run of consecutive same-gate QIS calls that can be lowered together.
struct GateBatch {
  StringRef intrinsicName;
  SmallVector<int64_t> vs1;    // lane values for the first (or only) vector arg
  SmallVector<int64_t> vs2;    // lane values for the second vector arg (CX only)
  SmallVector<Operation*> ops; // constituent call ops (will be erased after replacement)
  bool isCX = false;

  bool empty() const { return ops.empty(); }
  void clear() { *this = GateBatch{}; }
};

/// Lowers a collected batch to a single `llvm.call_intrinsic` and erases the
/// original call ops.  All ops are inserted before `batch.ops.front()`.
static void emitBatch(GateBatch& batch, BlockContext& ctx) {
  if (batch.empty())
    return;

  Location loc = batch.ops.front()->getLoc();
  ctx.builder.setInsertionPoint(batch.ops.front());

  int64_t N = (int64_t)batch.vs1.size();

  Value blockImm = ctx.getI32(loc, 0);
  Value vl = ctx.getI32(loc, N);

  SmallVector<Value> args;
  if (batch.isCX) {
    // QVPairIntrinsic: (vs1: ctrl vec, vs2: tgt vec, block_imm, vl)
    args = {buildQubitVec(ctx, loc, batch.vs1), buildQubitVec(ctx, loc, batch.vs2), blockImm, vl};
  } else {
    // QVSingleIntrinsic: (vs1: qubit vec, rs2/tag, block_imm, vl)
    Value tag = ctx.getI32(loc, 0);
    args = {buildQubitVec(ctx, loc, batch.vs1), tag, blockImm, vl};
  }

  LLVM::CallIntrinsicOp::create(ctx.builder, loc, ctx.builder.getStringAttr(batch.intrinsicName), args);
  for (auto* op : batch.ops)
    op->erase();
  batch.clear();
}

// ---------------------------------------------------------------------------
// Block-level lowering
// ---------------------------------------------------------------------------

/// Lowers all QIS gate calls in `block` to batched vector intrinsics.
/// Consecutive calls to the same QIS gate are fused into a single intrinsic
/// call whose qubit vector carries all their indices; a gate-change or
/// non-QIS operation flushes the active batch.
static LogicalResult lowerBlock(Block& block) {
  // Snapshot ops so we can safely erase while iterating.
  SmallVector<Operation*> ops;
  for (auto& op : block)
    ops.push_back(&op);

  BlockContext ctx(block.getParentOp()->getContext());
  GateBatch current;
  bool failed = false;

  auto flush = [&]() { emitBatch(current, ctx); };

  for (auto* op : ops) {
    auto callOp = dyn_cast<LLVM::CallOp>(op);
    if (!callOp) {
      flush();
      continue;
    }

    auto callee = callOp.getCallee();
    if (!callee) {
      flush();
      continue;
    }

    StringRef intrName = mapQISToIntrinsic(*callee);
    if (intrName.empty()) {
      flush();
      continue;
    }

    // Different gate from the active batch → flush first.
    if (!current.empty() && current.intrinsicName != intrName) {
      flush();
    }

    auto operands = callOp.getArgOperands();
    bool isCX = (*callee == qcc::qirQisCX);

    if (isCX) {
      auto ctrlIdx = getQubitIndexFromPtr(operands[0]);
      auto tgtIdx = getQubitIndexFromPtr(operands[1]);
      if (!ctrlIdx || !tgtIdx) {
        flush();
        callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                         "for '__quantum__qis__cx__body'");
        failed = true;
        continue;
      }
      current.intrinsicName = intrName;
      current.isCX = true;
      current.vs1.push_back(*ctrlIdx);
      current.vs2.push_back(*tgtIdx);
      current.ops.push_back(op);
    } else {
      // For mz__body, operands[1] is the result ptr; we only need operands[0].
      auto qubitIdx = getQubitIndexFromPtr(operands[0]);
      if (!qubitIdx) {
        flush();
        callOp.emitError("convert-qir-to-intrinsics: cannot extract qubit index from ptr "
                         "for '")
            << *callee << "'";
        failed = true;
        continue;
      }
      current.intrinsicName = intrName;
      current.isCX = false;
      current.vs1.push_back(*qubitIdx);
      current.ops.push_back(op);
    }
  }

  flush();
  return failed ? failure() : success();
}

namespace {

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

    // Phase 1: erase/replace runtime helper calls (single-op patterns).
    RewritePatternSet rtPatterns(ctx);
    rtPatterns.add<ReadResultLowering, RtInitLowering, RecordOutputLowering>(ctx);
    if (failed(applyPatternsGreedily(moduleOp, std::move(rtPatterns)))) {
      return signalPassFailure();
    }

    // Phase 2: lower and batch QIS gate calls to vector intrinsics.
    // Consecutive calls to the same gate are fused into one vector intrinsic.
    bool anyFailed = false;
    moduleOp->walk([&](LLVM::LLVMFuncOp func) {
      for (Block& block : func.getBody()) {
        if (failed(lowerBlock(block))) {
          anyFailed = true;
        }
      }
    });
    if (anyFailed) {
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
