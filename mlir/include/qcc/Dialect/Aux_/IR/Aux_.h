// Aux_.h
//
// Copyright (c) 2026 FullStaQD Project
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
