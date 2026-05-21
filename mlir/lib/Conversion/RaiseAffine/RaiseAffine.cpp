// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/RaiseAffine/RaiseAffine.h" // IWYU pragma: keep

#include "SimplifyAffEx.h"
#include "fromEnzyme.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h" // IWYU pragma: keep
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

// FIXME: check this debug type stuff
#define DEBUG_TYPE "affine-raise-from-scf"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace qcc {

#define GEN_PASS_DEF_AFFINERAISEFROMSCF
#include "qcc/Conversion/RaiseAffine/RaiseAffine.h.inc"

// FIXME: remove no lint stuff at the end.
// NOLINTBEGIN
namespace {

bool isDisjoint(Value v) {
  if (auto op = v.getDefiningOp()) {
    return op->hasAttr("isDisjoint");
  }
  return false;
}

bool isValidSymbolInt(Value value, bool recur, Region* scope);
bool isValidSymbolInt(Operation* defOp, bool recur, Region* scope) {
  Attribute operandCst;
  if (matchPattern(defOp, m_Constant(&operandCst)))
    return true;

  if (recur) {
    if (isa<arith::SelectOp, IndexCastOp, IndexCastUIOp, AddIOp, MulIOp, DivSIOp, DivUIOp, RemSIOp, RemUIOp, SubIOp,
            CmpIOp, TruncIOp, ExtUIOp, ExtSIOp>(defOp))
      if (llvm::all_of(defOp->getOperands(), [&](Value v) {
            bool b = isValidSymbolInt(v, recur, scope);
            // if (!b)
            //	LLVM_DEBUG(llvm::dbgs() << "illegal isValidSymbolInt: "
            //<< value << " due to " << v << "\n");
            return b;
          }))
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

  if (auto bop = val.getDefiningOp<RemSIOp>()) {
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());
  }

  if (auto bop = val.getDefiningOp<RemUIOp>())
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (auto bop = val.getDefiningOp<SubIOp>())
    return isValidIndex(bop.getOperand(0), scope) && isValidIndex(bop.getOperand(1), scope);

  if (auto bop = val.getDefiningOp<ShRUIOp>()) {
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());
  }

  if (auto bop = val.getDefiningOp<ShLIOp>()) {
    return (isValidIndex(bop.getOperand(0), scope) && bop.getOperand(1).getDefiningOp<arith::ConstantOp>());
  }

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

static bool legalCondition(Value en, bool dim, Region* scope) {
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

  // if (auto IC = dyn_cast_or_null<IndexCastOp>(en.getDefiningOp())) {
  //	if (!outer || legalCondition(IC.getOperand(), false)) return true;
  //}
  if (!dim)
    if (auto BA = dyn_cast<BlockArgument>(en)) {
      if (isa<affine::AffineForOp, affine::AffineParallelOp>(BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

bool need(AffineMap* map, SmallVectorImpl<Value>* operands, Region* scope) {
  assert(map->getNumInputs() == operands->size());
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims(), scope))
      return true;
  }
  return false;
}
bool need(IntegerSet* map, SmallVectorImpl<Value>* operands, Region* scope) {
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

// FIXME: seems to exist in LLVM already
/// Entry point for matching a pattern over a Value.
// template <typename Pattern> inline bool matchPattern(Value value, const Pattern& pattern) {
//   assert(value);
//   // TODO: handle other cases
//   if (auto* op = value.getDefiningOp())
//     return const_cast<Pattern&>(pattern).match(op);
//   return false;
// }

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <typename AttrClass,
          // Require AttrClass to be a derived class from Attribute and get its
          // value type
          typename ValueType =
              typename std::enable_if_t<std::is_base_of<Attribute, AttrClass>::value, AttrClass>::ValueType,
          // Require the ValueType is not void
          typename = std::enable_if_t<!std::is_void<ValueType>::value>>
struct attr_value_binder {
  ValueType* bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType* bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    if (auto intAttr = llvm::dyn_cast<AttrClass>(attr)) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches operations that have the `ConstantLike` trait, and
/// binds the folded attribute value.
template <typename AttrT> struct constant_op_binder {
  AttrT* bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT* bind_value) : bind_value(bind_value) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  constant_op_binder() : bind_value(nullptr) {}

  bool match(Operation* op) {
    if (!op->hasTrait<OpTrait::ConstantLike>())
      return false;

    // Fold the constant to an attribute.
    SmallVector<OpFoldResult, 1> foldedOp;
    LogicalResult result = op->fold(/*operands=*/{}, foldedOp);
    (void)result;
    assert(succeeded(result) && "expected ConstantLike op to be foldable");

    if (auto attr = llvm::dyn_cast<AttrT>(cast<Attribute>(foldedOp.front()))) {
      if (bind_value)
        *bind_value = attr;
      return true;
    }
    return false;
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer Attribute or Operation and binds the constant integer value.
struct constant_int_value_binder {
  IntegerAttr::ValueType* bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_value_binder(IntegerAttr::ValueType* bv) : bind_value(bv) {}

  bool match(Attribute attr) {
    attr_value_binder<IntegerAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<Attribute>());

    return false;
  }

  bool match(Operation* op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<IntegerType, IndexType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// Copied from Enzyme-JAX
/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline constant_int_value_binder m_ConstantInt(IntegerAttr::ValueType* bind_value) {
  return constant_int_value_binder(bind_value);
}

#if 0  // replaced by affine::fullyComposeAffineMapAndOperands
struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands, PatternRewriter* rewriter, DominanceInfo* DI,
                        Region* scope);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value, 8> getOperands() {
    SmallVector<Value, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr renumberOneDim(Value v);

  /// Maps of Value to position in `affineMap`.
  DenseMap<Value, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value, 8> reorderedDims;
  SmallVector<Value, 8> concatenatedSymbols;

  AffineMap affineMap;
};

bool isNonTopLevelPureSymbol(Value value) {
  if (auto* defOp = value.getDefiningOp()) {
    if (!isPure(defOp))
      return false;

    auto region = getLocalAffineScope(defOp);
    Attribute operandCst;
    if (!matchPattern(defOp, m_Constant(&operandCst)) && !affine::isValidSymbol(value, region))
      return false;
    if (defOp->getNumOperands() != 0)
      return false;
    if (defOp->getParentRegion() == region)
      return false;
    return true;
  }
  return false;
}

static bool isAffineForArg(Value val) {
  if (!isa<BlockArgument>(val))
    return false;
  Operation* parentOp = cast<BlockArgument>(val).getOwner()->getParentOp();
  return (isa_and_nonnull<affine::AffineForOp, affine::AffineParallelOp>(parentOp));
}

/// The AffineNormalizer composes AffineApplyOp recursively. Its purpose is to
/// keep a correspondence between the mathematical `map` and the `operands` of
/// a given affine::AffineApplyOp. This correspondence is maintained by
/// iterating over the operands and forming an `auxiliaryMap` that can be
/// composed mathematically with `map`. To keep this correspondence in cases
/// where symbols are produced by affine.apply operations, we perform a local
/// rewrite of symbols as dims.
///
/// Rationale for locally rewriting symbols as dims:
/// ================================================
/// The mathematical composition of AffineMap must always concatenate symbols
/// because it does not have enough information to do otherwise. For example,
/// composing `(d0)[s0] -> (d0 + s0)` with itself must produce
/// `(d0)[s0, s1] -> (d0 + s0 + s1)`.
///
/// The result is only equivalent to `(d0)[s0] -> (d0 + 2 * s0)` when
/// applied to the same mlir::Value for both s0 and s1.
/// As a consequence mathematical composition of AffineMap always concatenates
/// symbols.
///
/// When AffineMaps are used in affine::AffineApplyOp however, they may specify
/// composition via symbols, which is ambiguous mathematically. This corner case
/// is handled by locally rewriting such symbols that come from
/// affine::AffineApplyOp into dims and composing through dims.
/// TODO: Composition via symbols comes at a significant code
/// complexity. Alternatively we should investigate whether we want to
/// explicitly disallow symbols coming from affine.apply and instead force the
/// user to compose symbols beforehand. The annoyances may be small (i.e. 1 or 2
/// extra API calls for such uses, which haven't popped up until now) and the
/// benefit potentially big: simpler and more maintainable code for a
/// non-trivial, recursive, procedure.
AffineApplyNormalizer::AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands, PatternRewriter* rewriter,
                                             DominanceInfo* DI, Region* scope) {
  assert(map.getNumInputs() == operands.size() && "number of operands does not match the number of map inputs");

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nInput map: "));

  SmallVector<Value, 8> addedValues;

  llvm::SmallSet<unsigned, 1> symbolsToPromote;

  unsigned numDims = map.getNumDims();

  SmallVector<AffineExpr, 8> dimReplacements;
  SmallVector<AffineExpr, 8> symReplacements;

  SmallVector<SmallVectorImpl<Value>*> opsTodos;
  auto replaceOp = [&](Operation* oldOp, Operation* newOp) {
    for (auto [oldV, newV] : llvm::zip(oldOp->getResults(), newOp->getResults()))
      for (auto ops : opsTodos)
        for (auto& op : *ops)
          if (op == oldV)
            op = newV;
  };

  SmallVector<Operation**> operationContext;
  std::function<Value(Value, bool)> fix = [&](Value v, bool index) -> Value /*legal*/ {
    bool ntop = isNonTopLevelPureSymbol(v);
    if (!ntop && isValidSymbolInt(v, /*recur*/ false, scope)) {
      return v;
    }
    if (index && isAffineForArg(v)) {
      return v;
    }
    auto* op = v.getDefiningOp();
    if (!op)
      return nullptr;
    if (!op)
      llvm::errs() << v << "\n";
    assert(op);
    if (!isReadOnly(op)) { // FIXME: this was speculatively taken from Enzyme. Check!
      return nullptr;
    }
    Operation* front = nullptr;
    operationContext.push_back(&front);
    if (front)
      assert(front->getBlock());
    SmallVector<Value> ops;
    opsTodos.push_back(&ops);
    std::function<void(Operation*)> getAllOps = [&](Operation* todo) {
      assert(todo->getBlock());
      for (auto v : todo->getOperands()) {
        if (llvm::all_of(op->getRegions(), [&](Region& r) { return !r.isAncestor(v.getParentRegion()); }))
          ops.push_back(v);
      }
      for (auto& r : todo->getRegions()) {
        for (auto& b : r.getBlocks())
          for (auto& o2 : b.without_terminator())
            getAllOps(&o2);
      }
    };

    if (front)
      assert(front->getBlock());
    getAllOps(op);

    if (front)
      assert(front->getBlock());

    for (auto o : ops) {
      if (front)
        assert(front->getBlock());
      Operation* next;
      if (auto* op = o.getDefiningOp()) {
        assert(op->getBlock());
        if (front)
          assert(front->getBlock());
        if (Value nv = fix(o, index)) {
          op = nv.getDefiningOp();
        } else {
          operationContext.pop_back();
          opsTodos.pop_back();
          return nullptr;
        }
        next = op->getNextNode();
        assert(next->getBlock());
        if (front)
          assert(front->getBlock());
      } else {
        auto BA = cast<BlockArgument>(o);
        if (index && isAffineForArg(BA)) {
        } else if (!isValidSymbolInt(o, /*recur*/ false, scope)) {
          operationContext.pop_back();
          opsTodos.pop_back();
          return nullptr;
        }
        next = &BA.getOwner()->front();
        assert(next->getBlock());
        if (front)
          assert(front->getBlock());
      }
      if (front)
        assert(front->getBlock());
      if (next)
        assert(next->getBlock());
      if (front == nullptr)
        front = next;
      else if (DI && DI->dominates(front, next))
        front = next;
      if (front)
        assert(front->getBlock());
    }
    if (!front && ntop) {
      auto region = getLocalAffineScope(op);
      front = &region->front().front();
    }
    opsTodos.pop_back();
    if (!front)
      op->dump();
    assert(front);
    if (!rewriter) {
      operationContext.pop_back();
      assert(isValidSymbolInt(op->getResult(0), /*recur*/ false, scope));
      return op->getResult(0);
    } else {
      PatternRewriter::InsertionGuard B(*rewriter);
      rewriter->setInsertionPoint(front);
      if (front)
        assert(front->getBlock());
      auto cloned = rewriter->clone(*op);
      replaceOp(op, cloned);
      if (front)
        assert(front->getBlock());
      for (auto op_ptr : operationContext) {
        if (*op_ptr == op) {
          *op_ptr = cloned;
        }
      }
      rewriter->replaceOp(op, cloned->getResults());

      operationContext.pop_back();
      if (!isValidSymbolInt(cloned->getResult(0), /*recur*/ false, scope)) {
        llvm::errs() << " clonedParent: " << *cloned->getParentOfType<FunctionOpInterface>() << "\n";
        llvm::errs() << " cloned: " << *cloned << "\n";
        llvm_unreachable("busted");
      }
      return cloned->getResult(0);
    }
  };
  auto renumberOneSymbol = [&](Value v) {
    for (auto i : llvm::enumerate(addedValues)) {
      if (i.value() == v)
        return getAffineSymbolExpr(i.index(), map.getContext());
    }
    auto expr = getAffineSymbolExpr(addedValues.size(), map.getContext());
    addedValues.push_back(v);
    return expr;
  };

  // 2. Compose affine::AffineApplyOps and dispatch dims or symbols.
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    auto t = operands[i];
    auto decast = t;
    while (true) {
      if (auto idx = decast.getDefiningOp<IndexCastOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<IndexCastUIOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<TruncIOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<ExtUIOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<ExtSIOp>()) {
        decast = idx.getIn();
        continue;
      }
      break;
    }

    if (!isValidSymbolInt(t, /*recur*/ false, scope)) {
      t = decast;
    }

    // Only promote one at a time, lest we end up with two dimensions
    // multiplying each other.

    if (((!isValidSymbolInt(t, /*recur*/ false, scope) &&
          (t.getDefiningOp<AddIOp>() || t.getDefiningOp<SubIOp>() || (t.getDefiningOp<OrIOp>() && isDisjoint(t)) ||
           (t.getDefiningOp<MulIOp>() &&
            ((isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope)) ||
             (isValidIndex(t.getDefiningOp()->getOperand(1), scope) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(0), /*recur*/ true, scope))) &&
            !(fix(t.getDefiningOp()->getOperand(0), false) && fix(t.getDefiningOp()->getOperand(1), false))

                ) ||
           ((t.getDefiningOp<DivUIOp>() || t.getDefiningOp<DivSIOp>()) &&
            (isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope)) &&
            (!(fix(t.getDefiningOp()->getOperand(0), false) && fix(t.getDefiningOp()->getOperand(1), false)))) ||
           (t.getDefiningOp<DivSIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope))) ||
           (t.getDefiningOp<DivUIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope))) ||
           (t.getDefiningOp<RemUIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope))) ||
           (t.getDefiningOp<RemSIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0), scope) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1), /*recur*/ true, scope))) ||
           t.getDefiningOp<ConstantIntOp>() || t.getDefiningOp<ConstantIndexOp>())) ||
         ((decast.getDefiningOp<AddIOp>() || decast.getDefiningOp<SubIOp>() ||
           (decast.getDefiningOp<OrIOp>() && isDisjoint(decast)) || decast.getDefiningOp<MulIOp>() ||
           decast.getDefiningOp<RemUIOp>() || decast.getDefiningOp<RemSIOp>() || decast.getDefiningOp<ShRUIOp>() ||
           decast.getDefiningOp<ShLIOp>()) &&
          (decast.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIntOp>() ||
           decast.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIndexOp>())))) {
      t = decast;
      LLVM_DEBUG(llvm::dbgs() << " Replacing: " << t << "\n");

      AffineMap affineApplyMap;
      SmallVector<Value, 8> affineApplyOperands;

      // llvm::dbgs() << "\nop to start: " << t << "\n";

      if (auto op = t.getDefiningOp<AddIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) + getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(op.getLhs());
        affineApplyOperands.push_back(op.getRhs());
      } else if (auto op = t.getDefiningOp<OrIOp>()) {
        assert(isDisjoint(t));
        affineApplyMap =
            AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) + getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(op.getLhs());
        affineApplyOperands.push_back(op.getRhs());
      } else if (auto op = t.getDefiningOp<SubIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) - getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(op.getLhs());
        affineApplyOperands.push_back(op.getRhs());
      } else if (auto op = t.getDefiningOp<MulIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) * ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) * ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) * getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<DivSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap = AffineMap::get(
              0, 2, getAffineSymbolExpr(0, op.getContext()).floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<DivUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap = AffineMap::get(
              0, 2, getAffineSymbolExpr(0, op.getContext()).floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<RemSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) % getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<RemUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2, getAffineSymbolExpr(0, op.getContext()) % getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<ShRUIOp>()) {

        APInt iattr;
        if (!matchPattern(op.getRhs(), m_ConstantInt(&iattr))) {
          llvm_unreachable("shr rhs needed to be constant int");
        }

        affineApplyMap =
            AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()).floorDiv(1 << iattr.getZExtValue()));
        affineApplyOperands.push_back(op.getLhs());
      } else if (auto op = t.getDefiningOp<ShLIOp>()) {

        APInt iattr;
        if (!matchPattern(op.getRhs(), m_ConstantInt(&iattr))) {
          llvm_unreachable("shl rhs needed to be constant int");
        }

        affineApplyMap = AffineMap::get(0, 1, getAffineSymbolExpr(0, op.getContext()) * (1 << iattr.getZExtValue()));
        affineApplyOperands.push_back(op.getLhs());
      } else if (auto op = t.getDefiningOp<ConstantIntOp>()) {
        affineApplyMap = AffineMap::get(0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else if (auto op = t.getDefiningOp<ConstantIndexOp>()) {
        affineApplyMap = AffineMap::get(0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else {
        llvm_unreachable("");
      }

      SmallVector<AffineExpr, 0> dimRemapping;
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols);
      for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
        symRemapping[idx] = renumberOneSymbol(affineApplyOperands[idx]);
      }
      affineApplyMap =
          affineApplyMap.replaceDimsAndSymbols(dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(affineApplyMap.print(llvm::dbgs() << "\nRenumber into current normalizer: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));

    } else if (isAffineForArg(t)) {
      if (i >= numDims)
        symReplacements.push_back(renumberOneDim(t));
      else
        dimReplacements.push_back(renumberOneDim(t));
    } else if (t.getDefiningOp<affine::AffineApplyOp>()) {
      auto affineApply = t.getDefiningOp<affine::AffineApplyOp>();
      // a. Compose affine.apply operations.
      LLVM_DEBUG(affineApply->print(llvm::dbgs() << "\nCompose affine::AffineApplyOp recursively: "));
      AffineMap affineApplyMap = affineApply.getAffineMap();
      SmallVector<Value, 8> affineApplyOperands(affineApply.getOperands().begin(), affineApply.getOperands().end());

      SmallVector<AffineExpr, 0> dimRemapping(affineApplyMap.getNumDims());

      for (size_t i = 0; i < affineApplyMap.getNumDims(); ++i) {
        assert(i < affineApplyOperands.size());
        dimRemapping[i] = renumberOneDim(affineApplyOperands[i]);
      }
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols - affineApplyMap.getNumDims());
      for (unsigned idx = 0; idx < symRemapping.size(); ++idx) {
        symRemapping[idx] = renumberOneSymbol(affineApplyOperands[idx + affineApplyMap.getNumDims()]);
      }
      affineApplyMap =
          affineApplyMap.replaceDimsAndSymbols(dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(affineApplyMap.print(llvm::dbgs() << "\nAffine apply fixup map: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));
    } else {
      if (!isValidSymbolInt(t, /*recur*/ false, scope)) {
        if (t.getDefiningOp()) {
          if ((t = fix(t, false))) {
            if (!isValidSymbolInt(t, /*recur*/ false, scope)) {
              llvm::errs() << " op: " << *t.getDefiningOp()->getParentOfType<FunctionOpInterface>() << "\n";
              llvm::errs() << " failed to move:" << t << " to become valid symbol\n";
              llvm_unreachable("cannot move");
            }
          } else
            llvm_unreachable("cannot move");
        } else
          llvm_unreachable("cannot move2");
      }
      if (i < numDims) {
        // b. The mathematical composition of AffineMap composes dims.
        dimReplacements.push_back(renumberOneDim(t));
      } else {
        // c. The mathematical composition of AffineMap concatenates symbols.
        //    Note that the map composition will put symbols already present
        //    in the map before any symbols coming from the auxiliary map, so
        //    we insert them before any symbols that are due to renumbering,
        //    and after the proper symbols we have seen already.
        symReplacements.push_back(renumberOneSymbol(t));
      }
    }
  }
  for (auto v : addedValues)
    concatenatedSymbols.push_back(v);

  // Create the new map by replacing each symbol at pos by the next new dim.
  unsigned numNewDims = reorderedDims.size();
  unsigned numNewSymbols = addedValues.size();
  assert(dimReplacements.size() == map.getNumDims());
  assert(symReplacements.size() == map.getNumSymbols());
  auto auxiliaryMap = map.replaceDimsAndSymbols(dimReplacements, symReplacements, numNewDims, numNewSymbols);
  LLVM_DEBUG(auxiliaryMap.print(llvm::dbgs() << "\nRewritten map: "));

  affineMap = auxiliaryMap; // simplifyAffineMap(auxiliaryMap);

  LLVM_DEBUG(affineMap.print(llvm::dbgs() << "\nSimplified result: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

static void composeAffineMapAndOperands(AffineMap* map, SmallVectorImpl<Value>* operands, PatternRewriter* rewriter,
                                        DominanceInfo* DI, Region* scope) {
  AffineApplyNormalizer normalizer(*map, *operands, rewriter, DI, scope);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  affine::canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  normalizedMap = recreateExpr(normalizedMap);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}
#endif // replaced by affine::fullyComposeAffineMapAndOperands

void fully2ComposeAffineMapAndOperands(PatternRewriter* builder, AffineMap* map, SmallVectorImpl<Value>* operands,
                                       DominanceInfo* DI, Region* scope,
                                       SmallVectorImpl<Operation*>* insertedOps = nullptr) {
  IRMapping indexMap;
  if (builder)
    for (auto op : *operands) {
      SmallVector<IndexCastOp> attempt;
      auto idx0 = op.getDefiningOp<IndexCastOp>();
      attempt.push_back(idx0);
      if (!idx0)
        continue;

      for (auto& u : idx0.getIn().getUses()) {
        if (auto idx = dyn_cast<IndexCastOp>(u.getOwner()))
          if (DI->dominates((Operation*)idx, &*builder->getInsertionPoint()))
            attempt.push_back(idx);
      }

      for (auto idx : attempt) {
        if (affine::isValidSymbol(idx, scope)) {
          indexMap.map(idx.getIn(), idx);
          break;
        }
      }
    }

  // FIXME: check if this replacement works!
  // -  assert(map->getNumInputs() == operands->size());
  // -  while (need(map, operands, scope)) {
  // -    composeAffineMapAndOperands(map, operands, builder, DI, scope);
  // -    assert(map->getNumInputs() == operands->size());
  // -  }
  // +  affine::fullyComposeAffineMapAndOperands(map, operands);
  affine::fullyComposeAffineMapAndOperands(map, operands);

  *map = simplifyAffineMap(*map);
  if (builder)
    for (auto& op : *operands) {
      if (!op.getType().isIndex()) {
        Operation* toInsert;
        if (auto* o = op.getDefiningOp())
          toInsert = o->getNextNode();
        else {
          auto BA = cast<BlockArgument>(op);
          toInsert = &BA.getOwner()->front();
        }

        if (auto v = indexMap.lookupOrNull(op))
          op = v;
        else {
          if (insertedOps) {
            OpBuilder builder(toInsert);
            auto inserted = IndexCastOp::create(builder, op.getLoc(), builder.getIndexType(), op);
            op = inserted->getResult(0);
            insertedOps->push_back(inserted);
          } else {
            PatternRewriter::InsertionGuard B(*builder);
            builder->setInsertionPoint(toInsert);
            auto inserted = IndexCastOp::create(*builder, op.getLoc(), builder->getIndexType(), op);
            op = inserted->getResult(0);
          }
        }
      }
    }
}

void fully2ComposeAffineMapAndOperands(PatternRewriter& builder, AffineMap* map, SmallVectorImpl<Value>* operands,
                                       DominanceInfo& DI, Region* scope,
                                       SmallVectorImpl<Operation*>* insertedOps = nullptr) {
  fully2ComposeAffineMapAndOperands(&builder, map, operands, &DI, scope, insertedOps);
}

static void preserveDiscardableAttributes(Operation* newOp, Operation* oldOp) {
  for (auto attr : oldOp->getDiscardableAttrs()) {
    newOp->setAttr(attr.getName(), attr.getValue());
  }
}

/// Copied from Enzyme-JAX
struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // TODO: remove me or rename me.
  bool isAffine(scf::ForOp loop) const {
    // return true;
    // enforce step to be a ConstantIndexOp (maybe too restrictive).
    APInt apint;
    return affine::isValidSymbol(loop.getStep()) || matchPattern(loop.getStep(), m_ConstantInt(&apint));
  }

  int64_t getStep(mlir::Value value) const {
    APInt apint;
    if (matchPattern(value, m_ConstantInt(&apint)))
      return apint.getZExtValue();
    else
      return 1;
  }

  AffineMap getMultiSymbolIdentity(Builder& B, unsigned rank) const {
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(B.getAffineSymbolExpr(i));
    return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs, B.getContext());
  }
  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter& rewriter) const final {
    if (isAffine(loop)) {
      auto scope = getLocalAffineScope(loop);
      OpBuilder builder(loop);

      SmallVector<Value> lbs;
      {
        SmallVector<Value> todo = {loop.getLowerBound()};
        while (todo.size()) {
          auto cur = todo.back();
          todo.pop_back();
          if (isValidIndex(cur, scope)) {
            lbs.push_back(cur);
            continue;
          } else if (auto selOp = cur.getDefiningOp<arith::SelectOp>()) {
            // LB only has max of operands
            if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
              if (cmp.getLhs() == selOp.getTrueValue() && cmp.getRhs() == selOp.getFalseValue() &&
                  cmp.getPredicate() == CmpIPredicate::sge) {
                todo.push_back(cmp.getLhs());
                todo.push_back(cmp.getRhs());
                continue;
              }
            }
          }
          return failure();
        }
      }

      SmallVector<Value> ubs;
      {
        SmallVector<Value> todo = {loop.getUpperBound()};
        while (todo.size()) {
          auto cur = todo.back();
          todo.pop_back();
          if (isValidIndex(cur, scope)) {
            ubs.push_back(cur);
            continue;
          } else if (auto selOp = cur.getDefiningOp<arith::SelectOp>()) {
            // UB only has min of operands
            if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
              if (cmp.getLhs() == selOp.getTrueValue() && cmp.getRhs() == selOp.getFalseValue() &&
                  cmp.getPredicate() == CmpIPredicate::sle) {
                todo.push_back(cmp.getLhs());
                todo.push_back(cmp.getRhs());
                continue;
              }
            }
          }
          return failure();
        }
      }

      bool rewrittenStep = false;
      if (!loop.getStep().getDefiningOp<ConstantIndexOp>()) {
        if (ubs.size() != 1 || lbs.size() != 1)
          return failure();
        ubs[0] = DivUIOp::create(
            rewriter, loop.getLoc(),
            AddIOp::create(
                rewriter, loop.getLoc(),
                SubIOp::create(rewriter, loop.getLoc(), loop.getStep(),
                               isa<IndexType>(loop.getStep().getType())
                                   ? ConstantIndexOp::create(rewriter, loop.getLoc(), 1).getResult()
                                   : ConstantIntOp::create(rewriter, loop.getLoc(), loop.getStep().getType(), 1))
                    .getResult(),
                SubIOp::create(rewriter, loop.getLoc(), loop.getUpperBound(), loop.getLowerBound())),
            loop.getStep());
        lbs[0] = ConstantIndexOp::create(rewriter, loop.getLoc(), 0);
        rewrittenStep = true;
      }

      auto* parentScope = scope->getParentOp();
      DominanceInfo DI(parentScope);

      AffineMap lbMap = getMultiSymbolIdentity(builder, lbs.size());
      {
        fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbs, DI, scope);
        affine::canonicalizeMapAndOperands(&lbMap, &lbs);
        lbMap = removeDuplicateExprs(lbMap);
        lbMap = recreateExpr(lbMap);
      }
      AffineMap ubMap = getMultiSymbolIdentity(builder, ubs.size());
      {
        fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubs, DI, scope);
        affine::canonicalizeMapAndOperands(&ubMap, &ubs);
        ubMap = removeDuplicateExprs(ubMap);
        ubMap = recreateExpr(ubMap);
      }

      affine::AffineForOp affineLoop =
          affine::AffineForOp::create(rewriter, loop.getLoc(), lbs, lbMap, ubs, ubMap,
                                      rewrittenStep ? 1 : getStep(loop.getStep()), loop.getInits());
      preserveDiscardableAttributes(affineLoop, loop);

      auto mergedYieldOp = cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

      Block& newBlock = affineLoop.getRegion().front();

      // The terminator is added if the iterator args are not provided.
      // see the ::build method.
      if (affineLoop.getNumIterOperands() == 0) {
        auto* affineYieldOp = newBlock.getTerminator();
        rewriter.eraseOp(affineYieldOp);
      }

      SmallVector<Value> vals;
      rewriter.setInsertionPointToStart(&affineLoop.getRegion().front());
      for (Value arg : affineLoop.getRegion().front().getArguments()) {
        bool isInduction = arg == affineLoop.getInductionVar();
        if (isInduction && arg.getType() != loop.getInductionVar().getType()) {
          arg = arith::IndexCastOp::create(rewriter, loop.getLoc(), loop.getInductionVar().getType(), arg);
        }
        if (rewrittenStep && isInduction) {
          arg = AddIOp::create(rewriter, loop.getLoc(), loop.getLowerBound(),
                               MulIOp::create(rewriter, loop.getLoc(), arg, loop.getStep()));
        }
        vals.push_back(arg);
      }
      assert(vals.size() == loop.getRegion().front().getNumArguments());
      rewriter.mergeBlocks(&loop.getRegion().front(), &affineLoop.getRegion().front(), vals);

      rewriter.setInsertionPoint(mergedYieldOp);
      affine::AffineYieldOp::create(rewriter, mergedYieldOp.getLoc(), mergedYieldOp.getOperands());
      rewriter.eraseOp(mergedYieldOp);

      rewriter.replaceOp(loop, affineLoop.getResults());

      return success();
    }
    return failure();
  }
};

struct AffineRaiseFromSCF final : public impl::AffineRaiseFromSCFBase<AffineRaiseFromSCF> {
  using impl::AffineRaiseFromSCFBase<AffineRaiseFromSCF>::AffineRaiseFromSCFBase;

  void runOnOperation() override {}
};

} // namespace
// NOLINTEND
} // namespace qcc
