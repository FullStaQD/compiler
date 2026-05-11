
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

#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

namespace qcc {

#define GEN_PASS_DEF_CHECKSTATICQUBITALLOCATION
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

using namespace jasp;
using namespace mlir;
using namespace mlir::qc;

namespace {

struct CheckStaticQubitAllocation final : public impl::CheckStaticQubitAllocationBase<CheckStaticQubitAllocation> {
  using CheckStaticQubitAllocationBase<CheckStaticQubitAllocation>::CheckStaticQubitAllocationBase;

protected:
  void runOnOperation() override {
    Operation* op = getOperation();
    bool failed = false;

    WalkResult result = op->walk([&](memref::AllocOp allocOp) {
      auto memrefType = dyn_cast<MemRefType>(allocOp.getType());

      if (!memrefType || !isa<qc::QubitType>(memrefType.getElementType())) {
        return WalkResult::advance();
      }

      if (memrefType.isDynamicDim(0)) {
        allocOp.emitError("qubit array allocation size must be a compile-time constant; "
                          "dynamic qubit array sizes are not supported");
        return WalkResult::interrupt();
      }

      auto size = memrefType.getDimSize(0);
      if (size <= 0) {
        allocOp.emitError("qubit array size must be positive, got ") << size;
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace qcc
