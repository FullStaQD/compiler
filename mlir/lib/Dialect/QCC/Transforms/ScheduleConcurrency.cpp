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
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include <algorithm>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace qcc {

#define GEN_PASS_DEF_SCHEDULECONCURRENCY
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// One rewrite and its placement in the schedule.
struct Slot {
  TopologyRewriteOpInterface rule;
  Footprint footprint;
  double latency;
  unsigned step;
};

/// What the schedule costs and why.
struct Schedule {
  unsigned depth = 0;
  double criticalPath = 0.0;
  /// Pairs held apart by a shared control resource alone.
  unsigned controlBound = 0;
};

struct ScheduleConcurrency final : public impl::ScheduleConcurrencyBase<ScheduleConcurrency> {
  using ScheduleConcurrencyBase<ScheduleConcurrency>::ScheduleConcurrencyBase;

protected:
  void runOnOperation() override {
    func::FuncOp function = getOperation();

    llvm::SmallVector<Slot> slots;
    function.walk(
        [&](TopologyRewriteOpInterface rule) { slots.push_back({rule, rule.getFootprint(), rule.getLatencyUs(), 0}); });
    if (slots.empty()) {
      return markAllAnalysesPreserved();
    }

    const Schedule schedule = assignSteps(slots);

    if (annotate) {
      OpBuilder builder(&getContext());
      for (const auto& slot : slots) {
        slot.rule->setAttr("qcc.step", builder.getI64IntegerAttr(slot.step));
      }
    }

    auto report = function.emitRemark() << "scheduled " << slots.size() << " rewrite(s) into " << schedule.depth
                                        << " step(s); critical path " << schedule.criticalPath << " us";
    if (schedule.controlBound > 0) {
      report.attachNote() << schedule.controlBound
                          << " pair(s) were serialised only by a shared control resource, not by the "
                             "configuration; they would run together on a machine with one more";
    }
  }

private:
  /// Greedy list scheduling: a rule goes one step after the latest earlier rule
  /// it cannot share a step with, and in step 0 if it can share with all of
  /// them. This is the concurrency question, not the reordering one: rules held
  /// apart only by a control resource still commute.
  ///
  /// The SSA chain is not consulted. Threading one configuration value through
  /// every rule totally orders them in the IR, which is a property of the
  /// representation rather than of the machine; the footprints are what bind.
  static Schedule assignSteps(llvm::SmallVectorImpl<Slot>& slots) {
    Schedule schedule;

    for (const auto& [index, slot] : llvm::enumerate(slots)) {
      for (size_t earlier = 0; earlier < index; ++earlier) {
        if (slots[earlier].footprint.concurrentWith(slot.footprint)) {
          continue;
        }
        slots[index].step = std::max(slots[index].step, slots[earlier].step + 1);
        if (slots[earlier].footprint.serialisedOnlyByControl(slot.footprint)) {
          ++schedule.controlBound;
        }
      }
      schedule.depth = std::max(schedule.depth, slots[index].step + 1);
    }

    // The critical path is the sum over steps of the slowest rule in each.
    llvm::SmallVector<double> stepCost(schedule.depth, 0.0);
    for (const auto& slot : slots) {
      stepCost[slot.step] = std::max(stepCost[slot.step], slot.latency);
    }
    for (const double cost : stepCost) {
      schedule.criticalPath += cost;
    }
    return schedule;
  }
};

} // namespace
} // namespace qcc
