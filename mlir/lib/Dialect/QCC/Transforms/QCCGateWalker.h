// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// Helpers shared by the passes that check `qc` gates against a configuration.
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Dialect/QCC/Analysis/ConnectivityAnalysis.h"

#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/IR/Operation.h"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <optional>

namespace qcc::conn {

/// Whether connectivity constrains `op`.
///
/// A barrier is a scheduling fence rather than an interaction. A modifier
/// (`qc.ctrl`, `qc.inv`, `qc.pow`) reports the composite qubit set of its whole
/// body, and its body addresses qubits through aliased block arguments carrying
/// no index, so the modifier is both the complete check and the only one
/// possible.
inline bool isConstrainedByConnectivity(mlir::Operation* op) {
  return !llvm::isa<mlir::qc::BarrierOp>(op) && !llvm::isa_and_present<mlir::qc::UnitaryOpInterface>(op->getParentOp());
}

/// The qubits `gate` acts on, or nullopt if any carries no program index.
///
/// Controls are included: a controlled gate is an interaction between every
/// qubit it names, whatever the roles are in the unitary.
inline std::optional<llvm::SmallVector<int64_t>> collectQubits(mlir::qc::UnitaryOpInterface gate) {
  llvm::SmallVector<int64_t> qubits;
  for (size_t i = 0; i < gate.getNumQubits(); ++i) {
    const auto index = getQubitIndex(gate.getQubit(i));
    if (!index) {
      return std::nullopt;
    }
    qubits.push_back(*index);
  }
  return qubits;
}

} // namespace qcc::conn
