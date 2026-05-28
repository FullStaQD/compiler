//===- PrepareAffineRaising.cpp - prep for affine raising -----------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qcc/Conversion/AffineRaise/AffineRaise.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_PREPAREAFFINERAISING
#include "qcc/Conversion/AffineRaise/AffineRaise.h.inc"

namespace {

struct SelectToMinMax : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  static std::optional<Value> replaceMinMaxBound(Value bound, const bool isLower, PatternRewriter& rewriter) {
    auto selOp = bound.getDefiningOp<arith::SelectOp>();
    if (!selOp) {
      return std::nullopt;
    }

    auto cmp = selOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp) {
      return std::nullopt;
    }

    auto selectLeft = cmp.getLhs() == selOp.getTrueValue() && cmp.getRhs() == selOp.getFalseValue();
    auto selectRight = cmp.getLhs() == selOp.getFalseValue() && cmp.getRhs() == selOp.getTrueValue();

    // For this purpose, it doesn't matter whether the comparison is strict or not, or signed or not.
    // FIXME: The for loop treats its bounds as unsigned, so it doesn't really make sense to have signed comparisons
    //  beforehand. Figure out how to handle this properly.
    auto leftSmaller =
        cmp.getPredicate() == arith::CmpIPredicate::slt || cmp.getPredicate() == arith::CmpIPredicate::sle ||
        cmp.getPredicate() == arith::CmpIPredicate::ult || cmp.getPredicate() == arith::CmpIPredicate::ule;
    auto leftLarger =
        cmp.getPredicate() == arith::CmpIPredicate::sgt || cmp.getPredicate() == arith::CmpIPredicate::sge ||
        cmp.getPredicate() == arith::CmpIPredicate::ugt || cmp.getPredicate() == arith::CmpIPredicate::uge;

    auto signedComparison =
        cmp.getPredicate() == arith::CmpIPredicate::slt || cmp.getPredicate() == arith::CmpIPredicate::sle ||
        cmp.getPredicate() == arith::CmpIPredicate::sgt || cmp.getPredicate() == arith::CmpIPredicate::sge;

    auto isLegalMin = !isLower && (selectLeft && leftSmaller) || (selectRight && leftLarger);
    auto isLegalMax = isLower && (selectLeft && leftLarger) || (selectRight && leftSmaller);

    if (!isLegalMin && !isLegalMax) {
      return std::nullopt;
    }

    // Traverse the select operands to check for nested min/max patterns.
    auto leftMaxValue = replaceMinMaxBound(selOp.getTrueValue(), isLower, rewriter).value_or(selOp.getTrueValue());
    auto rightMaxValue = replaceMinMaxBound(selOp.getFalseValue(), isLower, rewriter).value_or(selOp.getFalseValue());

    if (isLegalMax) {
      if (signedComparison) {
        return arith::MaxSIOp::create(rewriter, selOp.getLoc(), selOp.getType(), leftMaxValue, rightMaxValue);
      }
      return arith::MaxUIOp::create(rewriter, selOp.getLoc(), selOp.getType(), leftMaxValue, rightMaxValue);
    }
    if (isLegalMin) {
      if (signedComparison) {
        return arith::MinSIOp::create(rewriter, selOp.getLoc(), selOp.getType(), leftMaxValue, rightMaxValue);
      }
      return arith::MinUIOp::create(rewriter, selOp.getLoc(), selOp.getType(), leftMaxValue, rightMaxValue);
    }

    llvm_unreachable("should have been legal min or max");
  }

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter& rewriter) const override {
    auto newLb = replaceMinMaxBound(loop.getLowerBound(), /*isLower=*/true, rewriter).value_or(loop.getLowerBound());
    auto newUb = replaceMinMaxBound(loop.getUpperBound(), /*isLower=*/false, rewriter).value_or(loop.getUpperBound());

    auto noChange = newLb == loop.getLowerBound() && newUb == loop.getUpperBound();
    if (noChange) {
      return rewriter.notifyMatchFailure(loop, "bounds are not min/max select patterns");
    }

    loop.setLowerBound(newLb);
    loop.setUpperBound(newUb);
    return success();
  }
};

/// Cast bounds and induction indices of `scf.for` loops to `index` type.
struct ForBoundsIndexCast : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter& rewriter) const override {
    Location loc = loop.getLoc();
    Value lb = loop.getLowerBound();
    Value ub = loop.getUpperBound();
    Value step = loop.getStep();
    Type originalBoundsType = step.getType();

    assert(lb.getType() == originalBoundsType && ub.getType() == originalBoundsType &&
           "expected all bounds and step to have the same type");
    bool needCasts = !isa<IndexType>(lb.getType());
    if (!needCasts) {
      return rewriter.notifyMatchFailure(loop, "bounds and step are already index-typed");
    }

    Value newLb = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), lb);
    Value newUb = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), ub);
    Value newStep = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), step);

    loop.setLowerBound(newLb);
    loop.setUpperBound(newUb);
    loop.setStep(newStep);

    Value iv = loop.getInductionVar();
    iv.setType(rewriter.getIndexType());
    rewriter.setInsertionPointToStart(loop.getBody());
    Value castIv = arith::IndexCastOp::create(rewriter, loc, originalBoundsType, iv);
    iv.replaceAllUsesWith(castIv);

    return success();
  }
};

} // namespace

struct PrepareAffineRaising final : public impl::PrepareAffineRaisingBase<PrepareAffineRaising> {
  using impl::PrepareAffineRaisingBase<PrepareAffineRaising>::PrepareAffineRaisingBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SelectToMinMax, ForBoundsIndexCast>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace qcc

namespace impl {
std::unique_ptr<::mlir::Pass> createPrepareAffineRaising() { return std::make_unique<qcc::PrepareAffineRaising>(); }
} // namespace impl
