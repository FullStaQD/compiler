// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "AffineCheck.h"

#include "Match.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-raise-from-scf"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

// FIXME: remove NOLINT stuff
// NOLINTBEGIN
namespace qcc {

bool isDisjoint(Value v) {
  if (auto op = v.getDefiningOp()) {
    return op->hasAttr("isDisjoint");
  }
  return false;
}

// forward declaration
bool isValidSymbolInt(Value value, bool recur, Region* scope);

bool isValidSymbolInt(Operation* defOp, bool recur, Region* scope) {
  Attribute operandCst;
  if (matchPattern(defOp, m_Constant(&operandCst)))
    return true;

  if (recur) {
    if (isa<arith::SelectOp, IndexCastOp, IndexCastUIOp, AddIOp, MulIOp, DivSIOp, DivUIOp, RemSIOp, RemUIOp, SubIOp,
            CmpIOp, TruncIOp, ExtUIOp, ExtSIOp>(defOp))
      if (llvm::all_of(defOp->getOperands(), [&](Value v) { return isValidSymbolInt(v, recur, scope); }))
        return true;
    if (auto orOp = dyn_cast<OrIOp>(defOp)) {
      if (isDisjoint(orOp) && isValidSymbolInt(orOp.getLhs(), recur, scope) &&
          isValidSymbolInt(orOp.getRhs(), recur, scope))
        return true;
    }
    if (auto shiftOp = dyn_cast<ShLIOp>(defOp)) {
      APInt intValue;
      if (isValidSymbolInt(shiftOp.getLhs(), recur, scope) && matchPattern(shiftOp.getRhs(), m_ConstantInt(&intValue)))
        return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      if (isValidSymbolInt(ifOp.getCondition(), recur, scope)) {
        if (llvm::all_of(ifOp.thenBlock()->without_terminator(),
                         [&](Operation& o) { return isValidSymbolInt(&o, recur, scope); }) &&
            llvm::all_of(ifOp.elseBlock()->without_terminator(),
                         [&](Operation& o) { return isValidSymbolInt(&o, recur, scope); }))
          return true;
      }
    }
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(defOp)) {
      if (llvm::all_of(ifOp.getOperands(), [&](Value o) { return isValidSymbolInt(o, recur, scope); }))
        if (llvm::all_of(ifOp.getThenBlock()->without_terminator(),
                         [&](Operation& o) { return isValidSymbolInt(&o, recur, scope); }) &&
            llvm::all_of(ifOp.getElseBlock()->without_terminator(),
                         [&](Operation& o) { return isValidSymbolInt(&o, recur, scope); }))
          return true;
    }
  }
  return false;
}

// isValidSymbol, even if not index
bool isValidSymbolInt(Value value, bool recur, Region* scope) {
  // Check that the value is a top level value, reimplemented from
  // affine::isTopLevelValue to check ancestry.
  assert(scope);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getParentRegion()->isAncestor(scope))
      return true;
  } else {
    if (value.getDefiningOp()->getParentRegion()->isAncestor(scope))
      return true;
  }

  if (auto* defOp = value.getDefiningOp()) {
    if (isValidSymbolInt(defOp, recur, scope))
      return true;
    return affine::isValidSymbol(value, scope);
  }

  return false;
}

bool isValidIndex(Value val, Region* scope) {
  if (val.getDefiningOp<affine::AffineApplyOp>())
    return true;

  if (isValidSymbolInt(val, /*recur*/ true, scope))
    return true;

  if (auto cast = val.getDefiningOp<IndexCastOp>())
    return isValidIndex(cast.getOperand(), scope);

  if (auto cast = val.getDefiningOp<IndexCastUIOp>())
    return isValidIndex(cast.getOperand(), scope);

  if (auto cast = val.getDefiningOp<TruncIOp>())
    return isValidIndex(cast.getOperand(), scope);

  if (auto cast = val.getDefiningOp<ExtSIOp>())
    return isValidIndex(cast.getOperand(), scope);

  if (auto cast = val.getDefiningOp<ExtUIOp>())
    return isValidIndex(cast.getOperand(), scope);

  if (auto bop = val.getDefiningOp<AddIOp>())
    return isValidIndex(bop.getOperand(0), scope) && isValidIndex(bop.getOperand(1), scope);

  if (auto bop = val.getDefiningOp<OrIOp>())
    return isDisjoint(bop) && isValidIndex(bop.getOperand(0), scope) && isValidIndex(bop.getOperand(1), scope);

  if (auto bop = val.getDefiningOp<MulIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && isValidSymbolInt(bop.getOperand(1), /*recur*/ true, scope)) ||
           (isValidIndex(bop.getOperand(1), scope) && isValidSymbolInt(bop.getOperand(0), /*recur*/ true, scope));

  if (auto bop = val.getDefiningOp<DivSIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && isValidSymbolInt(bop.getOperand(1), /*recur*/ true, scope));

  if (auto bop = val.getDefiningOp<DivUIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && isValidSymbolInt(bop.getOperand(1), /*recur*/ true, scope));

  if (auto bop = val.getDefiningOp<RemSIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (auto bop = val.getDefiningOp<RemUIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (auto bop = val.getDefiningOp<SubIOp>())
    return isValidIndex(bop.getOperand(0), scope) && isValidIndex(bop.getOperand(1), scope);

  if (auto bop = val.getDefiningOp<ShRUIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (auto bop = val.getDefiningOp<ShLIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (val.getDefiningOp<ConstantIndexOp>())
    return true;

  if (val.getDefiningOp<ConstantIntOp>())
    return true;

  if (auto ba = dyn_cast<BlockArgument>(val)) {
    auto* owner = ba.getOwner();
    assert(owner);

    auto* parentOp = owner->getParentOp();
    if (!parentOp) {
      owner->dump();
      llvm::errs() << " ba: " << ba << "\n";
    }
    assert(parentOp);
    if (isa<FunctionOpInterface>(parentOp))
      return true;
    if (auto af = dyn_cast<affine::AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<affine::AffineParallelOp>(parentOp))
      return true;

    if (isa<FunctionOpInterface>(parentOp))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

bool legalCondition(Value en, bool dim, Region* scope) {
  if (en.getDefiningOp<affine::AffineApplyOp>())
    return true;

  if (!dim && !isValidSymbolInt(en, /*recur*/ false, scope)) {
    if (isValidIndex(en, scope) || isValidSymbolInt(en, /*recur*/ true, scope)) {
      return true;
    }
  }

  while (auto ic = en.getDefiningOp<IndexCastOp>())
    en = ic.getIn();

  while (auto ic = en.getDefiningOp<IndexCastUIOp>())
    en = ic.getIn();

  APInt intValue;
  if ((en.getDefiningOp<AddIOp>() || en.getDefiningOp<SubIOp>() || en.getDefiningOp<MulIOp>() ||
       en.getDefiningOp<RemUIOp>() || en.getDefiningOp<RemSIOp>() || en.getDefiningOp<ShLIOp>()) &&
      matchPattern(en.getDefiningOp()->getOperand(1), m_ConstantInt(&intValue)))
    return true;

  if (auto orOp = en.getDefiningOp<OrIOp>()) {
    if (isDisjoint(orOp) && matchPattern(orOp.getRhs(), m_ConstantInt(&intValue)))
      return true;
  }

  if (!dim)
    if (auto BA = dyn_cast<BlockArgument>(en)) {
      if (isa<affine::AffineForOp, affine::AffineParallelOp>(BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

bool need(AffineMap* map, llvm::SmallVectorImpl<Value>* operands, Region* scope) {
  assert(map->getNumInputs() == operands->size());
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims(), scope))
      return true;
  }
  return false;
}

bool need(IntegerSet* map, llvm::SmallVectorImpl<Value>* operands, Region* scope) {
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims(), scope))
      return true;
  }
  return false;
}

Region* getLocalAffineScope(Operation* op) {
  auto curOp = op;
  while (auto parentOp = curOp->getParentOp()) {
    if (parentOp->hasTrait<OpTrait::AffineScope>()) {
      return curOp->getParentRegion();
    }
    curOp = parentOp;
  }
  return nullptr;
}

} // namespace qcc
// NOLINTEND
