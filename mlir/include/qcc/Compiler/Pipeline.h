#pragma once
#include <mlir/Pass/PassManager.h>

namespace qcc {

// Helper function to add standard cleanups, keeping code DRY
void addCleanupPasses(mlir::PassManager& pm);

// The main entry point to build your specific compilation flow
void buildQuantumPipeline(mlir::PassManager& pm);

} // namespace qcc
