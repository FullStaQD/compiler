// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace qcc {

/// The QIS functions that `convert-qir-to-intrinsics` turns into QV instructions.
///
/// This is what the control hardware of HiSEP-Q can execute; a gate outside this set has no
/// instruction to lower to (see `ControlTarget::canExecuteGate`).
llvm::ArrayRef<llvm::StringRef> getQVGateSet();

} // namespace qcc
