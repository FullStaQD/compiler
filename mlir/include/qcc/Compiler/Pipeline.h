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

/// The target determines the backend to compile for, the actual passes
/// (pipeline), and the runtime.
enum class Target : uint8_t { Qir, HisepQ };

/// The stage to compile to and emit.
enum class Stage : uint8_t { Mlir, LlvmIr, Assembly, Object };

struct PipelineOptions {
  /// The backend to compile for. For `Target::HisepQ`, the final
  /// QIR-to-intrinsics lowering pass is appended to the pipeline, producing
  /// RISC-V QV intrinsic calls instead of QIS call ops.
  Target target = Target::Qir;
  Stage stage = Stage::LlvmIr;
};

/// A compilation pipeline assuming qc dialect as input and providing QIR-MLIR as output.
///
/// NOTE: The exact shape of the pipeline is still under construction.
void buildQuantumPipeline(mlir::PassManager& pm, const PipelineOptions& options = {});

} // namespace qcc
