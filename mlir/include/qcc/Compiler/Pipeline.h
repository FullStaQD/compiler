// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Target/Target.h"

#include <mlir/Pass/PassManager.h>

namespace qcc {

/// A compilation pipeline taking the qc dialect as input and producing code for `platform`.
///
/// It runs in three phases: a hardware-agnostic one that lowers JASP to the qc dialect and unrolls
/// the circuit, a device-aware one that adapts the circuit to the qubits of the platform (see
/// `QuantumTarget::addDevicePasses`), and a controller-aware one that turns the resulting QIR into
/// code for its control hardware (see `ControlTarget::addLoweringPasses`).
///
/// NOTE: The exact shape of the pipeline is still under construction.
void buildQuantumPipeline(mlir::PassManager& pm, const Platform& platform);

} // namespace qcc
