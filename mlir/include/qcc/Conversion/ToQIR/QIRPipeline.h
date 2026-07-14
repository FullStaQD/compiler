// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class PassManager;
} // namespace mlir

namespace qcc {

/// Converts a circuit in the qc dialect into QIR: the LLVM dialect calling the QIS functions of the
/// QIR spec.
///
/// QIR is one way down from a circuit, not the only one, so this is a lowering that a control target
/// calls if it speaks QIR (see `ControlTarget::addLoweringPasses`) rather than a phase of the
/// pipeline. A control target with its own path from the qc dialect does not call it.
///
/// `nativeGates` are the QIS functions the device implements (see `QuantumTarget::getNativeGates`):
/// only those are declared, and a circuit using any other gate is rejected.
void buildQIRPipeline(mlir::PassManager& pm, llvm::ArrayRef<llvm::StringRef> nativeGates);

} // namespace qcc
