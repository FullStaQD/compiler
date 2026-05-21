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

#include "mlir/Dialect/Affine/IR/AffineOps.h" // IWYU pragma: keep
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ForOpRaising>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
// NOLINTEND
} // namespace qcc
