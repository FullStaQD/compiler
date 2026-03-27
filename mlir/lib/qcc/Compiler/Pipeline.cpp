
#include "qcc/Compiler/Pipeline.h"

#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Transforms/Passes.h"

#include <llvm/Support/raw_ostream.h>

namespace qcc {

void addCleanupPasses(mlir::PassManager& pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

void buildQuantumPipeline(mlir::PassManager& pm) {
  addCleanupPasses(pm); // QC Canonicalization

  // QC → QIR Conversion
  pm.addPass(mlir::createQCToQIR());
  addCleanupPasses(pm); // QIR Canonicalization
}

} // namespace qcc
