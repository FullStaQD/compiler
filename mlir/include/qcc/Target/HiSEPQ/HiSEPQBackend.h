// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// Public entry point for the HiSEP-Q target.
//
// ===----------------------------------------------------------------------===//

// FIXME: rename the file? (backend vs target)

#pragma once

#include <mlir/Pass/PassManager.h>

namespace qcc {

/// Assembles the lowering pipeline for the HiSEP-Q target.
void buildHiSEPQPipeline(mlir::PassManager& pm); // FIXME: more systematic name? buildPipelineForXXX ?

} // namespace qcc
