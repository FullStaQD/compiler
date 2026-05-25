// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// FIXME: acknowledge that we took this code from Enzyme-JAX

#include "qcc/Conversion/AffineRaise/AffineRaise.h" // IWYU pragma: keep

#include "AffineCheck.h"
#include "Match.h"
#include "SimplifyAffEx.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h" // IWYU pragma: keep
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace qcc {

#define GEN_PASS_DEF_AFFINERAISEFROMSCF
#include "qcc/Conversion/AffineRaise/AffineRaise.h.inc"

// FIXME: remove no lint stuff at the end.
// NOLINTBEGIN
namespace {

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

/// FIXME: this is what affine::composeAffineMapAndOperands does. This thing
/// here does more apparently. One job is to insert index casts (affine.for works
/// only on index types).
///
/// Takes map(operand) and *moves* affine.apply from operands into map.
///
/// E.g. this map = affine_map<()[s0] -> (s0)> operands = [affine.apply
/// affine_map<()[s0] -> (s0 + 4)>()[%n]] Becomes map = affine_map<()[s0] -> (s0
/// + 4)> operands = [%n]
void fully2ComposeAffineMapAndOperands(PatternRewriter& builder, AffineMap* map, SmallVectorImpl<Value>* operands,
                                       DominanceInfo& DI, Region* scope,
                                       SmallVectorImpl<Operation*>* insertedOps = nullptr) {
  fully2ComposeAffineMapAndOperands(&builder, map, operands, &DI, scope, insertedOps);
}

/// FIXME: is there a builtin for this?
static void preserveDiscardableAttributes(Operation* newOp, Operation* oldOp) {
  for (auto attr : oldOp->getDiscardableAttrs()) {
    newOp->setAttr(attr.getName(), attr.getValue());
  }
}

/// Copied from Enzyme-JAX
struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // FIXME: can likely be removed
  /// Whether step is a valid symbol
  bool isAffine(scf::ForOp loop) const {
    // return true;
    // enforce step to be a ConstantIndexOp (maybe too restrictive).
    APInt apint;

    llvm::errs() << " *** isAffine *** \n";
    llvm::errs() << "!!! isValidSymbol: " << affine::isValidSymbol(loop.getStep()) << "\n";
    llvm::errs() << "!!! matchPattern : " << matchPattern(loop.getStep(), m_ConstantInt(&apint)) << "\n";

    // FIXME: might be equivalent to
    // return affine::isValidSymbol(loop.getStep());
    return affine::isValidSymbol(loop.getStep()) || matchPattern(loop.getStep(), m_ConstantInt(&apint));
  }

  /// FIXME: bad name, getConstant? Is there a builtin? It is only used once refactor that.
  int64_t getStep(mlir::Value value) const {
    APInt apint;
    if (matchPattern(value, m_ConstantInt(&apint)))
      return apint.getZExtValue();
    else
      return 1; // due to step normalization
  }

  // FIXME: check if there is a builtin method.
  /// Creates affine_map<()[s_0, ..., s_{rank-1}] -> (s_0, ..., s_{rank-1})>
  AffineMap getMultiSymbolIdentity(Builder& builder, unsigned rank) const {
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(builder.getAffineSymbolExpr(i));
    return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs, builder.getContext());
  }

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter& rewriter) const final {
    if (!isAffine(loop)) // FIXME: means "if step is not a symbol"
      return failure();

    auto scope = getLocalAffineScope(loop);
    OpBuilder builder(loop);

    /// NOTE: %r = arith.select (cmpi sge, a, b), a, b   // a >= b ? a : b  (sge)

    // FIXME: This matches LB = max(a, b, c, ...) *implemented* as arith.select
    // (cmpi sge, a, b), a, b - SOMETIMES!
    //
    // SOMETIMES: It matches if the select itself is *not* provably a valid index.
    //
    // In principle this info could be used to always translate this into
    // affine.max (in a dedicated pass). In that case we could simplify the code
    // here to require that the lbs is a valid index already. But the
    // "sometimes" might hint that we do not want to do this always.
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

    // FIXME: same as for lbs.
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

    // FIXME: this is super implicit, it is not at all clear when the max/min
    // patterns fire. This code here depends on it and hence is equally
    // unpredictable. Disentangle the stuff and precisely say what should be
    // raised and what not.
    //
    // Step normalization (affine.for needs constant step): If the step is not a
    // constant it gets normalized to 1 by scaling LB and UB. This fails however
    // if LB or UB is a max/min.
    //
    // If the step size is not a constant of index type: Rewrite step as follows
    // - assert that the max or min pattern above did not match (FIXME: this is
    //   brittle)
    // - (indirectly via size of ubs / lbs)
    // - set ubs[0] = ((STEP - 1) + (UB - LB)) / STEP = ceil((UB - LB) / STEP)
    // - set lbs[0] = 0
    bool rewrittenStep = false; // NOTE: `false` implies step is constant index op.
    if (!loop.getStep().getDefiningOp<ConstantIndexOp>()) {
      if (ubs.size() != 1 || lbs.size() != 1) // FIXME: very indirect (means max/min pattern did not fire)
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

    // Best effort to bring affine maps for lb and ub into some canonical form.
    // But: also inserts index casts (necessary correctness).
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

    affine::AffineForOp affineLoop = affine::AffineForOp::create(
        rewriter, loop.getLoc(), lbs, lbMap, ubs, ubMap, rewrittenStep ? 1 : getStep(loop.getStep()), loop.getInits());
    preserveDiscardableAttributes(affineLoop, loop);

    auto mergedYieldOp = cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block& newBlock = affineLoop.getRegion().front();

    if (affineLoop.getNumIterOperands() == 0) {
      // AffineFor::create adds its own yield op (only) in this case. We have to delete it as we create it ourselves
      // below.
      auto* affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    SmallVector<Value> vals;
    rewriter.setInsertionPointToStart(&affineLoop.getRegion().front());
    for (Value arg : affineLoop.getRegion().front().getArguments()) {
      bool isInduction = arg == affineLoop.getInductionVar();
      // Cast IV from index to whatever integer type the loop body expects:
      if (isInduction && arg.getType() != loop.getInductionVar().getType()) {
        arg = arith::IndexCastOp::create(rewriter, loop.getLoc(), loop.getInductionVar().getType(), arg);
      }
      // In case of rewritten step: Old_IV = New_IV * Old_Step + Old_LB
      if (isInduction && rewrittenStep) {
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
