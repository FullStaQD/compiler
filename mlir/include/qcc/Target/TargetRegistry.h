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
// FIXME: how about const-correctness here?
/// Describes a compilation target selectable via `qcc --target=<name>`.
struct BackendInfo {
  /// The `--target` value, e.g. "qir".
  llvm::StringRef name;
  /// Human-readable description shown by `--list-targets`.
  llvm::StringRef description;
  /// Whether this backend is compiled into the current build.
  bool available;
  /// Assembles the lowering pipeline for this backend. Only valid (non-null)
  /// when `available` is true.
  std::function<void(mlir::PassManager&)> buildPipeline;
};

/// Returns all known backends, available or not.
llvm::ArrayRef<BackendInfo> getBackends();

// FIXME: better return type (currently *raw* pointer)?
/// Looks up a backend by its `--target` name, or returns nullptr if no backend
/// with that name is known.
const BackendInfo* lookupBackend(llvm::StringRef name);

} // namespace qcc
