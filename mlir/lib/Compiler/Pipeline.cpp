
#include "qcc/Compiler/Pipeline.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/QCToQIR/QCToQIR.h>
#include <mlir/Transforms/Passes.h>

namespace qcc {

void buildQuantumPipeline(mlir::PassManager& pm) {
  // Cleanup QC
  pm.addPass(mlir::createCanonicalizerPass());

  // QC → QIR Conversion
  pm.addPass(mlir::createQCToQIR());

  // Cleanup QIR
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

} // namespace qcc
