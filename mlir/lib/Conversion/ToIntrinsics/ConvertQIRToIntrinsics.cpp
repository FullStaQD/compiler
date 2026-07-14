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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;

//===----------------------------------------------------------------------===//
// The QIR symbols this pass lowers
//===----------------------------------------------------------------------===//

namespace {

/// A QIS gate supported by this target, and the QV intrinsic it lowers to.
///
/// The QIS names come from `Constants.h`, the intrinsics from IntrinsicsRISCVXQV.td.
struct QVGate {
  llvm::StringLiteral qisName;
  llvm::StringLiteral intrinsic;
  unsigned numQubits;
};

constexpr std::array kQVGates = {
    QVGate{qcc::qirQisH, "llvm.riscv.qv.h", /*numQubits=*/1},
    QVGate{qcc::qirQisX, "llvm.riscv.qv.x", /*numQubits=*/1},
    QVGate{qcc::qirQisCX, "llvm.riscv.qv.cx", /*numQubits=*/2},
    QVGate{qcc::qirQisMZ, "llvm.riscv.qv.mz", /*numQubits=*/1},
};

/// The QIR runtime functions with no counterpart on the bare-metal intrinsic path.
///
/// TODO: `read_result` and the output recording functions have no intrinsic in
/// IntrinsicsRISCVXQV.td yet. Until then their calls are dropped and their results are `undef`.
constexpr std::array kDroppedQIRRuntimeFuncs = {
    qcc::qirRtInit,
    qcc::qirRtReadResult,
    qcc::qirRtBoolRecordOutput,
    qcc::qirRtIntRecordOutput,
};

/// Returns the lowering of `qisName`, or nullptr if this target does not support that gate.
const QVGate* findQVGate(StringRef qisName) {
  const auto* gate = llvm::find_if(kQVGates, [&](const QVGate& gate) { return gate.qisName == qisName; });
  return gate == kQVGates.end() ? nullptr : gate;
}

bool isDroppedQIRRuntimeFunc(StringRef name) { return llvm::is_contained(kDroppedQIRRuntimeFuncs, name); }

/// Whether `name` is a QIR function that this pass rewrites away.
bool isLoweredQIRFunc(StringRef name) { return findQVGate(name) != nullptr || isDroppedQIRRuntimeFunc(name); }

//===----------------------------------------------------------------------===//
// Qubit index vectors
//===----------------------------------------------------------------------===//

/// Width of the scalable `vector<[N]xi8>` that carries a qubit index into the QV intrinsics.
///
/// A `vector<[1]xi8>` is illegal on this target: at SEW=8 and ELEN=32 it legalizes to LMUL=mf8,
/// below the RVV minimum of LMUL >= SEW/ELEN = mf4. Four elements legalize to mf2. Only the first
/// element is processed, as the intrinsics are called with `vl == 1`.
constexpr int64_t kQubitIndexVecNumElements = 4;

/// Materializes qubit indices as the scalable vectors that the QV intrinsics take, using one
/// internal-linkage global per distinct index, shared across the module.
///
/// HiSEP-Q's vector unit implements neither `vmv.s.x` nor `vmv.v.i`, so the vector cannot be built
/// in a register and is loaded (`vle8.v`) from a global instead. Each global holds its index in the
/// first byte, zero-padded to the width of the vector.
class QubitIndexVecBuilder {
public:
  explicit QubitIndexVecBuilder(ModuleOp moduleOp) : moduleOp(moduleOp) {}

  Value build(OpBuilder& builder, Location loc, int64_t index) {
    auto vecType = VectorType::get({kQubitIndexVecNumElements}, builder.getI8Type(), /*scalableDims=*/{true});

    Value addr = LLVM::AddressOfOp::create(builder, loc, getOrCreateGlobal(builder, index));
    return LLVM::LoadOp::create(builder, loc, vecType, addr);
  }

private:
  LLVM::GlobalOp getOrCreateGlobal(OpBuilder& builder, int64_t index) {
    auto [it, inserted] = globals.try_emplace(index);
    if (!inserted) {
      return it->second;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    std::string bytes(kQubitIndexVecNumElements, '\0');
    bytes[0] = static_cast<char>(index);
    auto arrayType = LLVM::LLVMArrayType::get(builder.getI8Type(), kQubitIndexVecNumElements);

    it->second =
        LLVM::GlobalOp::create(builder, moduleOp.getLoc(), arrayType, /*isConstant=*/true, LLVM::Linkage::Internal,
                               (".qcc_qv_idx_" + llvm::Twine(index)).str(), builder.getStringAttr(bytes));
    return it->second;
  }

  ModuleOp moduleOp;
  llvm::DenseMap<int64_t, LLVM::GlobalOp> globals;
};

/// Extracts the index of a statically allocated qubit, which `convert-qc-to-qir` materializes as
/// `llvm.inttoptr (llvm.mlir.constant N : i64) : !llvm.ptr`.
std::optional<int64_t> getQubitIndexFromPtr(Value ptrValue) {
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

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Rewrites `llvm.call @__quantum__qis__*__body(qubit_ptr, ...)` into the corresponding
/// `llvm.call_intrinsic "llvm.riscv.qv.*"(qubit_vec, ...)`.
struct QVGateLowering : public OpRewritePattern<LLVM::CallOp> {
  QVGateLowering(MLIRContext* context, QubitIndexVecBuilder& qubitVecs)
      : OpRewritePattern(context), qubitVecs(qubitVecs) {}

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee) {
      return failure();
    }

    const QVGate* gate = findQVGate(*callee);
    if (gate == nullptr) {
      return failure();
    }

    Location loc = callOp.getLoc();

    // Only the leading operands are qubits. `mz__body(qubit_ptr, result_ptr)` also takes a result
    // ptr, which the hardware does not use.
    SmallVector<Value> args;
    for (Value qubitPtr : callOp.getArgOperands().take_front(gate->numQubits)) {
      std::optional<int64_t> index = getQubitIndexFromPtr(qubitPtr);
      if (!index) {
        return callOp.emitError("expected a statically allocated qubit, i.e. an `llvm.inttoptr` of a "
                                "constant, as operand of '")
               << *callee << "'";
      }
      args.push_back(qubitVecs.build(rewriter, loc, *index));
    }

    auto i32Const = [&](int32_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
    };

    //   QVSingleIntrinsic: (vs1: vec, rs2: i32, block_imm: i32, vl: i32)
    //   QVPairIntrinsic:   (vs1: vec, vs2: vec, block_imm: i32, vl: i32)
    //
    // None of the gates above use `rs2`.
    if (gate->numQubits == 1) {
      args.push_back(i32Const(0)); // rs2
    }
    args.push_back(i32Const(0)); // block_imm
    /// TODO: support vl > 1 to batch several qubits into one call.
    args.push_back(i32Const(1)); // vl

    LLVM::CallIntrinsicOp::create(rewriter, loc, rewriter.getStringAttr(gate->intrinsic), args);
    rewriter.eraseOp(callOp);
    return success();
  }

private:
  QubitIndexVecBuilder& qubitVecs;
};

/// Drops the calls to `kDroppedQIRRuntimeFuncs` and replaces their results, if any, with `undef`.
struct QIRRuntimeCallLowering : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp callOp, PatternRewriter& rewriter) const override {
    auto callee = callOp.getCallee();
    if (!callee || !isDroppedQIRRuntimeFunc(*callee)) {
      return failure();
    }

    SmallVector<Value> undefResults;
    for (Type resultType : callOp.getResultTypes()) {
      undefResults.push_back(LLVM::UndefOp::create(rewriter, callOp.getLoc(), resultType));
    }

    rewriter.replaceOp(callOp, undefResults);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

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

    FailureOr<LLVM::LLVMFuncOp> entryPoint = findEntryPoint(moduleOp);
    if (failed(entryPoint)) {
      return signalPassFailure();
    }

    QubitIndexVecBuilder qubitVecs(moduleOp);

    RewritePatternSet patterns(moduleOp.getContext());
    patterns.add<QVGateLowering>(moduleOp.getContext(), qubitVecs);
    patterns.add<QIRRuntimeCallLowering>(moduleOp.getContext());

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    eraseQIRDeclarations(moduleOp);

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

  /// Erases the QIR declarations whose call sites are all gone.
  static void eraseQIRDeclarations(ModuleOp moduleOp) {
    SmallVector<LLVM::LLVMFuncOp> deadFuncs;
    for (auto funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
      if (isLoweredQIRFunc(funcOp.getName())) {
        deadFuncs.push_back(funcOp);
      }
    }

    for (auto funcOp : deadFuncs) {
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
