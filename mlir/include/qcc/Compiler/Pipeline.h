#pragma once
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>

namespace qcc {

/// A compilation pipeline assuming qc dialect as input and providing QIR-MLIR as output.
///
/// NOTE: The exact shape of the pipeline is still under construction.
void buildQuantumPipeline(mlir::PassManager& pm);

} // namespace qcc
