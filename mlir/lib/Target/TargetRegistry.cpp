// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Target/TargetRegistry.h"

#include "qcc/Config/Config.h"
#include "qcc/Target/QIR/QIRTarget.h"

#if QCC_ENABLE_HISEP_Q
#include "qcc/Target/HiSEPQ/HiSEPQTarget.h"
#endif

#include <vector>

namespace qcc {

llvm::ArrayRef<TargetInfo> getTargets() {
  static const std::vector<TargetInfo> backends = [] {
    std::vector<TargetInfo> result;
    result.push_back({.name = "qir",
                      .description = "QIR (LLVM-based) target",
                      .available = true,
                      .buildPipeline = [](mlir::PassManager& pm) { buildPipelineQIR(pm); }});

    TargetInfo hisepq{.name = "hisep-q",
                      .description = "HiSEP-Q QISA target (RISC-V based)",
                      .available = false,
                      .buildPipeline = nullptr};
#if QCC_ENABLE_HISEP_Q
    hisepq.available = true;
    hisepq.buildPipeline = [](mlir::PassManager& pm) { buildPipelineHiSEPQ(pm); };
#endif
    result.push_back(std::move(hisepq));

    return result;
  }();

  return backends;
}

const TargetInfo* lookupTarget(llvm::StringRef name) {
  for (const TargetInfo& backend : getTargets()) {
    if (backend.name == name) {
      return &backend;
    }
  }
  return nullptr;
}

} // namespace qcc
