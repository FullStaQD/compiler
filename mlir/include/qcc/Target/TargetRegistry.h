// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Pass/PassManager.h>

namespace qcc {
// Simplified form of the descriptor-plus-factory pattern in LLVM's
// `llvm/MC/TargetRegistry.h`: `Target` is a flat, non-polymorphic registry
// entry carrying metadata plus a factory (`addLoweringPasses`) for the target's
// behavior. If our implementation must be augmented follow LLVM's lead.

/// Describes a compilation target selectable via `qcc --target=<name>`.
struct Target {
  /// The `--target` value, e.g. "qir".
  llvm::StringRef name;
  /// Human-readable description shown by `--list-targets`.
  llvm::StringRef description;
  /// Whether this backend is compiled into the current build.
  bool available;
  /// Assembles the lowering pipeline for this target. Only valid (non-null)
  /// when `available` is true.
  std::function<void(mlir::PassManager&)> addLoweringPasses;
};

/// Returns all known backends, available or not.
llvm::ArrayRef<Target> getTargets();

/// Looks up a backend by its `--target` name, or returns nullptr if no backend
/// with that name is known.
const Target* lookupTarget(llvm::StringRef name);

} // namespace qcc
