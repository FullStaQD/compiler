// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
// Stripped from Enzyme-JAX SimplifyAffEx.cpp — only recreateExpr(AffineMap)
// and its transitive helpers are retained.

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// NOLINTBEGIN
namespace qcc {

AffineExpr internalAdd(AffineExpr LHS, AffineExpr RHS, bool allownegate = true);

AffineExpr commonAddWithMul(AffineExpr LHS, AffineExpr RHS, bool allownegate = true) {
  auto lhsD = llvm::DynamicAPInt(LHS.getLargestKnownDivisor());
  auto rhsD = llvm::DynamicAPInt(RHS.getLargestKnownDivisor());
  auto gcd = llvm::int64fromDynamicAPInt(llvm::gcd(abs(lhsD), abs(rhsD)));
  SmallVector<int64_t, 2> vals;

  if (gcd != 1)
    vals.push_back(gcd);
  bool negate = false;
  for (auto v : {LHS, RHS})
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(v)) {
      if (auto cst1 = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        if (cst1.getValue() < 0)
          negate = true;
      if (auto cst2 = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        if (cst2.getValue() < 0)
          negate = true;
    }
  if (negate && allownegate)
    vals.push_back(-gcd);

  for (auto val : vals) {
    auto LHSg = val == -1 ? (LHS * val) : LHS.floorDiv(val);
    auto RHSg = val == -1 ? (RHS * val) : RHS.floorDiv(val);
    auto add = internalAdd(LHSg, RHSg, val != -1);
    auto add2 = dyn_cast<AffineBinaryOpExpr>(add);
    if (!add2)
      return add * val;
    if (add2.getKind() != AffineExprKind::Add)
      return add * val;
    if (!((add2.getLHS() == LHSg && add2.getRHS() == RHSg) || (add2.getRHS() == LHSg && add2.getLHS() == RHSg)))
      return add * val;
  }

  return LHS + RHS;
}

bool affineCmp(AffineExpr lhs, AffineExpr rhs) {
  if (isa<AffineConstantExpr>(lhs) && !isa<AffineConstantExpr>(rhs))
    return true;
  if (!isa<AffineConstantExpr>(lhs) && isa<AffineConstantExpr>(rhs))
    return false;
  if (auto L = dyn_cast<AffineConstantExpr>(lhs))
    if (auto R = dyn_cast<AffineConstantExpr>(rhs))
      return L.getValue() < R.getValue();

  if (isa<AffineSymbolExpr>(lhs) && !isa<AffineSymbolExpr>(rhs))
    return true;
  if (!isa<AffineSymbolExpr>(lhs) && isa<AffineSymbolExpr>(rhs))
    return false;
  if (auto L = dyn_cast<AffineSymbolExpr>(lhs))
    if (auto R = dyn_cast<AffineSymbolExpr>(rhs))
      return L.getPosition() < R.getPosition();

  if (isa<AffineDimExpr>(lhs) && !isa<AffineDimExpr>(rhs))
    return true;
  if (!isa<AffineDimExpr>(lhs) && isa<AffineDimExpr>(rhs))
    return false;
  if (auto L = dyn_cast<AffineDimExpr>(lhs))
    if (auto R = dyn_cast<AffineDimExpr>(rhs))
      return L.getPosition() < R.getPosition();

  auto L = cast<AffineBinaryOpExpr>(lhs);
  auto R = cast<AffineBinaryOpExpr>(rhs);
  if (affineCmp(L.getLHS(), R.getLHS()))
    return true;
  if (affineCmp(R.getLHS(), L.getLHS()))
    return false;
  if (affineCmp(L.getRHS(), R.getRHS()))
    return true;
  if (affineCmp(R.getRHS(), L.getRHS()))
    return false;
  return false;
}

SmallVector<AffineExpr> getSumOperands(AffineExpr expr) {
  SmallVector<AffineExpr> todo = {expr};
  SmallVector<AffineExpr> base;
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (auto Add = dyn_cast<AffineBinaryOpExpr>(cur))
      if (Add.getKind() == AffineExprKind::Add) {
        todo.push_back(Add.getLHS());
        todo.push_back(Add.getRHS());
        continue;
      }
    base.push_back(cur);
  }
  return base;
}

AffineExpr sortSum(AffineExpr expr) {
  auto Add = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!Add)
    return expr;
  auto exprs = getSumOperands(Add);
  llvm::sort(exprs, affineCmp);
  auto res = exprs[0];
  for (int i = 1; i < (int)exprs.size(); i++)
    res = res + exprs[i];
  return res;
}

AffineExpr internalAdd(AffineExpr LHS, AffineExpr RHS, bool allownegate) {
  SmallVector<AffineExpr> base[2] = {getSumOperands(LHS), getSumOperands(RHS)};
  if (base[0].size() == 1 && base[1].size() == 1)
    return commonAddWithMul(LHS, RHS, allownegate);

  llvm::sort(base[0], affineCmp);
  llvm::sort(base[1], affineCmp);

  for (int i = 0; i < (int)base[0].size(); i++)
    for (int j = 0; j < (int)base[1].size(); j++) {
      auto fuse = commonAddWithMul(base[0][i], base[1][j]);
      bool simplified = false;
      if (auto Add = dyn_cast<AffineBinaryOpExpr>(fuse)) {
        if (Add.getLHS() == base[0][i] && Add.getRHS() == base[1][j])
          simplified = true;
        if (Add.getRHS() == base[0][i] && Add.getLHS() == base[1][j])
          simplified = true;
      }
      if (!simplified) {
        for (int i2 = 0; i2 < (int)base[0].size(); i2++) {
          if (i != i2)
            fuse = commonAddWithMul(fuse, base[0][i2]);
        }
        for (int j2 = 0; j2 < (int)base[1].size(); j2++) {
          if (j != j2)
            fuse = commonAddWithMul(fuse, base[1][j2]);
        }
        return fuse;
      }
    }
  return commonAddWithMul(LHS, RHS, allownegate);
}

AffineExpr recreateExpr(AffineExpr expr) {
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = recreateExpr(bin.getLHS());
    auto rhs = recreateExpr(bin.getRHS());

    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return internalAdd(lhs, rhs);
    case AffineExprKind::Mul:
      return sortSum(lhs) * sortSum(rhs);
    case AffineExprKind::Mod: {
      rhs = sortSum(rhs);
      SmallVector<AffineExpr> toMod;
      if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
        for (auto e : getSumOperands(lhs)) {
          if (!e.isMultipleOf(cst.getValue()))
            toMod.push_back(e);
        }
      } else {
        toMod.push_back(sortSum(lhs));
      }
      llvm::sort(toMod, affineCmp);
      AffineExpr out = getAffineConstantExpr(0, expr.getContext());
      for (auto e : toMod)
        out = out + e;
      return out % rhs;
    }
    case AffineExprKind::FloorDiv: {
      rhs = sortSum(rhs);
      SmallVector<AffineExpr> toDivide;
      SmallVector<AffineExpr> alreadyDivided;
      if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
        for (auto e : getSumOperands(lhs)) {
          if (e.isMultipleOf(cst.getValue())) {
            alreadyDivided.push_back(e.floorDiv(cst));
          } else if (auto cst2 = dyn_cast<AffineConstantExpr>(e)) {
            if (cst2.getValue() > 0 && cst.getValue() > 0 && cst2.getValue() > cst.getValue()) {
              toDivide.push_back(e % rhs);
              alreadyDivided.push_back(e.floorDiv(rhs));
            } else {
              toDivide.push_back(e);
            }
          } else {
            toDivide.push_back(e);
          }
        }
      } else {
        toDivide.push_back(sortSum(lhs));
      }
      llvm::sort(toDivide, affineCmp);
      AffineExpr out = getAffineConstantExpr(0, expr.getContext());
      for (auto e : toDivide)
        out = out + e;
      out = out.floorDiv(rhs);
      alreadyDivided.push_back(out);
      out = getAffineConstantExpr(0, expr.getContext());
      llvm::sort(alreadyDivided, affineCmp);
      for (auto e : alreadyDivided)
        out = out + e;
      return out;
    }
    default:
      return expr;
    }
  }
  return expr;
}

AffineMap recreateExpr(AffineMap map) {
  SmallVector<AffineExpr> exprs;
  for (auto expr : map.getResults())
    exprs.push_back(sortSum(recreateExpr(expr)));
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, map.getContext());
}

} // namespace qcc
// NOLINTEND
