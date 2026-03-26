#pragma once
#include <mlir/Pass/PassManager.h>

namespace mlir {
namespace qcc {

// Helper function to add standard cleanups, keeping code DRY
void addCleanupPasses(PassManager& pm);

// The main entry point to build your specific compilation flow
void buildQuantumPipeline(PassManager& pm);

} // namespace qcc
} // namespace mlir
