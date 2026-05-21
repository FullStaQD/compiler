// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/RaiseAffine/RaiseAffine.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_AFFINERAISEFROMSCF
#include "qcc/Conversion/RaiseAffine/RaiseAffine.h.inc"

namespace {

struct AffineRaiseFromSCF final : public impl::AffineRaiseFromSCFBase<AffineRaiseFromSCF> {
  using impl::AffineRaiseFromSCFBase<AffineRaiseFromSCF>::AffineRaiseFromSCFBase;

  void runOnOperation() override {}
};

} // namespace
} // namespace qcc
