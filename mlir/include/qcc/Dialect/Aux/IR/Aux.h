#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

//===----------------------------------------------------------------------===//
// Aux Dialect
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/Aux/IR/AuxDialect.h.inc"

//===----------------------------------------------------------------------===//
// Aux Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qcc/Dialect/Aux/IR/AuxOps.h.inc"
