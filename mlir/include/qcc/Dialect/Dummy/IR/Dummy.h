#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// Dummy Dialect
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/Dummy/IR/DummyDialect.h.inc"

//===----------------------------------------------------------------------===//
// Dummy Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "qcc/Dialect/Dummy/IR/DummyTypes.h.inc"

//===----------------------------------------------------------------------===//
// Dummy Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qcc/Dialect/Dummy/IR/DummyOps.h.inc"
