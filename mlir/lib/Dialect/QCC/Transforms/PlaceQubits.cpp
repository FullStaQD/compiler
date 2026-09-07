// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/Devices/DeviceLibrary.h"
#include "qcc/Dialect/QCC/IR/QCC.h"
#include "qcc/Dialect/QCC/Transforms/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Casting.h>

namespace qcc {

#define GEN_PASS_DEF_PLACEQUBITS
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// Whether `function` already establishes a configuration of its own.
bool isHardwareAware(func::FuncOp function) {
  bool found = false;
  function.walk([&](Operation* op) {
    found |= llvm::any_of(op->getResults(), [](Value result) { return llvm::isa<ConfigType>(result.getType()); });
  });
  return found;
}

/// Every hardware qubit the function names, in first-mention order.
llvm::SetVector<int64_t> collectQubits(func::FuncOp function) {
  llvm::SetVector<int64_t> qubits;
  function.walk([&](qc::StaticOp op) { qubits.insert(static_cast<int64_t>(op.getIndex())); });
  return qubits;
}

struct PlaceQubits final : public impl::PlaceQubitsBase<PlaceQubits> {
  using PlaceQubitsBase<PlaceQubits>::PlaceQubitsBase;

protected:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal()) {
      return;
    }

    auto moduleOp = function->getParentOfType<ModuleOp>();
    const auto device = moduleOp ? moduleOp->getAttrOfType<DeviceAttr>(kDeviceAttrName) : DeviceAttr{};
    // With no machine attached there is nothing to place against, and the pass
    // has to be harmless in a pipeline compiling programs with none.
    if (!device) {
      return;
    }

    // A function establishing its own configuration is hardware-aware already,
    // and its author's placement is not this pass's to replace.
    if (isHardwareAware(function)) {
      return;
    }

    const auto qubits = collectQubits(function);
    if (qubits.empty()) {
      return;
    }

    const auto config = layOut(device, qubits);
    if (failed(config)) {
      return signalPassFailure();
    }

    Block& entry = function.getBody().front();
    OpBuilder builder(&entry, entry.begin());
    ConfigInitOp::create(builder, function.getLoc(), ConfigType::get(&getContext(), device),
                         PlacementAttr::get(&getContext(), *config));
  }

private:
  /// Assigns qubits to sites in declaration order, filling each to capacity.
  ///
  /// On a machine of capacity-1 sites this is the identity mapping. It respects
  /// capacity and never places a qubit twice, so what it produces verifies, but
  /// it does not read the gate graph and so hands the router more work than a
  /// placement heuristic would. Replacing this function is the intended way to
  /// supply one: everything downstream consumes the `qcc.config.init`.
  FailureOr<Configuration> layOut(DeviceAttr device, const llvm::SetVector<int64_t>& qubits) {
    Configuration config;
    auto next = qubits.begin();

    for (const auto site : device.getSubstrate().getSites()) {
      Word occupants;
      for (uint64_t taken = 0; taken < site.getCapacity() && next != qubits.end(); ++taken, ++next) {
        occupants.push_back(*next);
      }
      // Empty sites are listed too, so the initial configuration describes the
      // whole machine rather than only its occupied part.
      SiteContents groups;
      if (!occupants.empty()) {
        groups.push_back(std::move(occupants));
      }
      config.sites.push_back({site.getSymName().str(), std::move(groups)});
    }

    if (next != qubits.end()) {
      const auto placed = static_cast<size_t>(std::distance(qubits.begin(), next));
      getOperation().emitError() << "machine has room for " << placed << " of the " << qubits.size()
                                 << " qubit(s) this function uses";
      return failure();
    }
    return config;
  }
};

} // namespace
} // namespace qcc
