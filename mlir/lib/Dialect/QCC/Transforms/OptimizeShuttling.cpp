// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCC.h"
#include "qcc/Dialect/QCC/Transforms/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

namespace qcc {

#define GEN_PASS_DEF_OPTIMIZESHUTTLING
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// Whether `op` observes the configuration.
///
/// Gates are checked against the configuration in force. Measurement and reset
/// observe it too, since a qubit must be somewhere its readout can reach.
bool observesConfiguration(Operation* op) { return llvm::isa<qc::UnitaryOpInterface, qc::MeasureOp, qc::ResetOp>(op); }

/// The last operation in `block` that observes the configuration, or nullptr.
Operation* lastObserver(Block& block) {
  Operation* last = nullptr;
  for (Operation& op : block) {
    if (observesConfiguration(&op)) {
      last = &op;
    }
  }
  return last;
}

struct OptimizeShuttling final : public impl::OptimizeShuttlingBase<OptimizeShuttling> {
  using OptimizeShuttlingBase<OptimizeShuttling>::OptimizeShuttlingBase;

protected:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal()) {
      return;
    }

    unsigned removed = 0;
    double saved = 0.0;
    for (Block& block : function.getBody()) {
      for (Operation* op : findUnobservedTail(block, saved)) {
        op->erase();
        ++removed;
      }
    }

    if (removed > 0) {
      function.emitRemark() << "dropped " << removed << " unobserved rewrite(s), saving " << saved
                            << " us of machine time";
    }
  }

private:
  /// The maximal suffix of rewrites that no observer reads, latest first.
  ///
  /// A schedule routinely ends by returning ions to where they started; when
  /// nothing is executed afterwards, that motion costs machine time for no
  /// observable effect. Only removal is attempted: two shuttles commute when
  /// their footprints are disjoint, but acting on that needs the concurrency
  /// analysis, so reordering is left to `--qcc-schedule-concurrency`.
  llvm::SmallVector<Operation*> findUnobservedTail(Block& block, double& saved) {
    Operation* observer = lastObserver(block);
    llvm::SmallVector<Operation*> doomed;

    for (Operation& op : llvm::reverse(block)) {
      if (observer != nullptr && op.isBeforeInBlock(observer)) {
        break;
      }

      auto rule = llvm::dyn_cast<TopologyRewriteOpInterface>(&op);
      if (!rule) {
        // A terminator or unrelated operation can be scanned past, but anything
        // consuming a configuration ends the safe suffix.
        if (llvm::any_of(op.getOperands(), [](Value v) { return llvm::isa<ConfigType>(v.getType()); })) {
          break;
        }
        continue;
      }

      const bool observed = llvm::any_of(rule.getOutputConfig().getUsers(),
                                         [&](Operation* user) { return !llvm::is_contained(doomed, user); });
      if (observed) {
        break;
      }

      doomed.push_back(&op);
      saved += rule.getLatencyUs();
    }
    return doomed;
  }
};

} // namespace
} // namespace qcc
