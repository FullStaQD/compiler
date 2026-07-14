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
  // FIXME: the pre-processor directives look a bit unergonomic here.
  static const std::vector<TargetInfo> backends = [] {
    std::vector<TargetInfo> result;
    result.push_back({.name = "qir",
                      .description = "QIR (LLVM-based) target",
                      .available = true,
                      .buildPipeline = [](mlir::PassManager& pm) { buildQIRPipeline(pm); }});
#if QCC_ENABLE_HISEP_Q
    result.push_back({.name = "hisep-q",
                      .description = "HiSEP-Q QISA target (RISC-V based)",
                      .available = true,
                      .buildPipeline = [](mlir::PassManager& pm) { buildHiSEPQPipeline(pm); }});
#else
    result.push_back({.name = "hisep-q",
                      .description = "HiSEP-Q QISA target (RISC-V based)",
                      .available = false,
                      .buildPipeline = nullptr});
#endif
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
