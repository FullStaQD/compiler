// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

namespace qcc {

// FIXME: impl of this contains alot of code.
/// Whether this is a valid affine index, but slightly relaxed. E.g. allows non-index types.
bool isValidIndex(mlir::Value val, mlir::Region* scope);

/// FIXME: there might be a builtin for this
mlir::Region* getLocalAffineScope(mlir::Operation* op);

} // namespace qcc
