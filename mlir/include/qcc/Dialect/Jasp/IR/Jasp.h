// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>

//===----------------------------------------------------------------------===//
// Jasp Dialect
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/Jasp/IR/JaspDialect.h.inc"

//===----------------------------------------------------------------------===//
// Jasp Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "qcc/Dialect/Jasp/IR/JaspTypes.h.inc"

//===----------------------------------------------------------------------===//
// Jasp Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qcc/Dialect/Jasp/IR/JaspOps.h.inc"
