
#include "qcc/Conversion/JaspToQC/JaspToQC.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h" // DenseMap
#include "llvm/Support/Casting.h"

#include <cassert>
#include <utility> // std::pair

namespace qcc {

#define GEN_PASS_DEF_STATICIZEQUBITREFS
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

using namespace jasp;
using namespace mlir;
using namespace mlir::qc;

namespace {

struct StaticizeQubitRefs final : public impl::StaticizeQubitRefsBase<StaticizeQubitRefs> {
  using StaticizeQubitRefsBase<StaticizeQubitRefs>::StaticizeQubitRefsBase;

protected:
  void runOnOperation() override {
    Operation* op = getOperation();
    OpBuilder builder(op->getContext());

    // Tracks the mapping from a (MemRef, Index) pair to a specific physical qubit Value.
    // This essentially "flattens" the memory model into a lookup table.
    DenseMap<std::pair<Value, int64_t>, Value> qubitMap;

    // A global counter to ensure every physical qubit in the circuit gets a unique hardware ID.
    int64_t nextGlobalQubitIdx = 0;
    bool failed = false;

    // Helper: Identifies if a type is a MemRef containing Qubits.
    auto isQubitMemref = [](Type type) {
      if (auto mType = dyn_cast<MemRefType>(type)) {
        return isa<qc::QubitType>(mType.getElementType());
      }

      return false;
    };

    // --- Step 1: Lower Allocations to Static Hardware Qubits ---
    // We treat every 'alloc' as a request for N physical qubits.
    // We generate these qubits immediately and store them in our map.
    op->walk([&](memref::AllocOp allocOp) {
      if (!isQubitMemref(allocOp.getType())) {
        return;
      }

      auto memrefType = cast<MemRefType>(allocOp.getType());
      if (memrefType.isDynamicDim(0)) {
        allocOp->emitError("found dynamic qubit allocation; expected static size from previous pass");
        failed = true;
        return;
      }

      int64_t size = memrefType.getDimSize(0);
      builder.setInsertionPoint(allocOp);

      // Create a unique 'static' op for every slot in the memref.
      for (int64_t i = 0; i < size; i++) {
        auto staticQubit = qc::StaticOp::create(builder, allocOp->getLoc(), qc::QubitType::get(op->getContext()),
                                                nextGlobalQubitIdx++);
        qubitMap[{allocOp.getResult(), i}] = staticQubit.getResult();
      }
    });

    if (failed) {
      return signalPassFailure();
    }

    // --- Step 2: Resolve Loads to Static Values ---
    // We replace any 'load' operation with a direct reference to the physical qubit
    // created in Step 1. This effectively eliminates the need for the memref.
    op->walk([&](memref::LoadOp loadOp) {
      if (!isa<qc::QubitType>(loadOp.getType())) {
        return;
      }

      Value baseMemref = loadOp.getMemRef();
      Value indexVal = loadOp.getIndices()[0];

      // Qubit indexing must be constant because quantum hardware
      // cannot "dynamically" route wires at runtime.
      auto constantIdx = getConstantIntValue(indexVal);
      if (!constantIdx) {
        loadOp->emitError("qubit index must be a constant; unroll loops before this pass");
        failed = true;
        return;
      }

      int64_t idx = *constantIdx;
      auto memrefType = cast<MemRefType>(baseMemref.getType());

      if (idx < 0 || idx >= memrefType.getDimSize(0)) {
        loadOp->emitError("qubit index out of bounds");
        failed = true;
        return;
      }

      // Retrieve the pre-allocated physical qubit from our map.
      auto it = qubitMap.find({baseMemref, idx});
      if (it == qubitMap.end()) {
        loadOp->emitError("internal error: static qubit not found for this allocation");
        failed = true;
        return;
      }

      // Replace all uses of the 'loaded' qubit with the 'static' hardware qubit.
      loadOp.getResult().replaceAllUsesWith(it->second);
      loadOp.erase(); // The load is now redundant.
    });

    if (failed) {
      return signalPassFailure();
    }

    // Note: memref.alloc and dealloc ops are typically cleaned up by
    // subsequent standard MLIR buffer-deallocation or Canonicalization passes.
  }
};
} // namespace
} // namespace qcc
