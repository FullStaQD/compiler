// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

//===----------------------------------------------------------------------===//
// Aux Dialect
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/Aux_/IR/AuxDialect.h.inc"

//===----------------------------------------------------------------------===//
// Aux Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qcc/Dialect/Aux_/IR/AuxOps.h.inc"
