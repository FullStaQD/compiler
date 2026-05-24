// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

namespace qcc {

bool isDisjoint(mlir::Value v);
bool isValidSymbolInt(mlir::Operation* defOp, bool recur, mlir::Region* scope);
bool isValidSymbolInt(mlir::Value value, bool recur, mlir::Region* scope);
bool isValidIndex(mlir::Value val, mlir::Region* scope);
bool legalCondition(mlir::Value en, bool dim, mlir::Region* scope);
bool need(mlir::AffineMap* map, llvm::SmallVectorImpl<mlir::Value>* operands, mlir::Region* scope);
bool need(mlir::IntegerSet* map, llvm::SmallVectorImpl<mlir::Value>* operands, mlir::Region* scope);
mlir::Region* getLocalAffineScope(mlir::Operation* op);

} // namespace qcc
