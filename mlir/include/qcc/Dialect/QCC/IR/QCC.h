// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Dialect/QCC/IR/Configuration.h" // IWYU pragma: export

#include "mlir/IR/BuiltinAttributeInterfaces.h" // IWYU pragma: keep  (OpAsmAttrInterface)
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h" // IWYU pragma: keep  (OpAsmTypeInterface)
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// QCC Enums
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCCEnums.h.inc"

//===----------------------------------------------------------------------===//
// QCC Dialect
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCCDialect.h.inc"

//===----------------------------------------------------------------------===//
// QCC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "qcc/Dialect/QCC/IR/QCCAttributes.h.inc"

//===----------------------------------------------------------------------===//
// QCC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "qcc/Dialect/QCC/IR/QCCTypes.h.inc"

//===----------------------------------------------------------------------===//
// QCC Interfaces
//===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCCInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// QCC Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qcc/Dialect/QCC/IR/QCCOps.h.inc"
