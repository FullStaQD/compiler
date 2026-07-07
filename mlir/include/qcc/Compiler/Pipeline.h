// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once
#include <cstdint>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>

namespace qcc {

enum class Target : uint8_t;
enum class Stage : uint8_t;

/// A compilation pipeline assuming qc dialect as input and providing QIR-MLIR as output.
///
/// NOTE: The exact shape of the pipeline is still under construction.
void buildQuantumPipeline(mlir::PassManager& pm, const Target& target);

} // namespace qcc
