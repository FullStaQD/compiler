// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Dialect/QCC/IR/QCC.h" // IWYU pragma: keep

#include "mlir/Dialect/Func/IR/FuncOps.h" // IWYU pragma: keep

#include <mlir/Pass/Pass.h>

namespace qcc {
#define GEN_PASS_DECL
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"
} // namespace qcc
