// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

/// A unit attribute to mark a `func.func` as the starting point of a quantum program.
static constexpr llvm::StringLiteral entryPointAttrName = "qcc.entry_point";

} // namespace qcc
