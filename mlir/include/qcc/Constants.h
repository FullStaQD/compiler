// Constants.h
//
// Copyright (c) 2026 FullStaQD Project
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

/// A unit attribute to mark a `func.func` as the starting point of a quantum program.
static constexpr llvm::StringLiteral entryPointAttrName = "qcc.entry_point";

} // namespace qcc
