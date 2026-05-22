//===- WhileToFor.cpp - scf.while to scf.for pass ------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qcc/Conversion/AffineRaise/AffineRaise.h" // IWYU pragma: keep

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace qcc {

#define GEN_PASS_DEF_WHILETOFOR
#include "qcc/Conversion/AffineRaise/AffineRaise.h.inc"

struct WhileToFor final : public impl::WhileToForBase<WhileToFor> {
  using impl::WhileToForBase<WhileToFor>::WhileToForBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::scf::populateUpliftWhileToForPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace qcc

namespace impl {
std::unique_ptr<::mlir::Pass> createWhileToFor() { return std::make_unique<qcc::WhileToFor>(); }
} // namespace impl
