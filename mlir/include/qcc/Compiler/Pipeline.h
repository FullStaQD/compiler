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

/// The stage to compile to and emit. `Elf` and `Mem` require --target=hisep-q: qcc links the
/// object code (via hisepq.ld, embedded into the binary) and, for `Mem`, converts the resulting
/// ELF into a Verilog $readmemh memory file (see qcc/Support/Elf2Mem.h).
enum class Stage : uint8_t { Mlir, LlvmIr, Assembly, Object, Elf, Mem };

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
