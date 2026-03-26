
#include "qcc/Compiler/Pipeline.h"

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Transforms/Passes.h"

#include <llvm/Support/raw_ostream.h>

namespace mlir {
namespace qcc {

void addCleanupPasses(PassManager& pm) {
  pm.addPass(createCanonicalizerPass());
  // Note: 'createRemoveDeadValuesPass' isn't standard MLIR,
  // assuming it's your custom pass or SymbolDCE
  pm.addPass(createRemoveDeadValuesPass());
}

void buildQuantumPipeline(PassManager& pm) {
  llvm::errs() << "test\n";
  // Initial QC Canonicalization
  pm.addPass(createCanonicalizerPass());

  // QC → QCO Conversion
  pm.addPass(createQCToQCO());
  addCleanupPasses(pm); // QCO Canonicalization

  // QCO → QC Conversion
  pm.addPass(createQCOToQC());
  addCleanupPasses(pm); // Final QC Canonicalization

  // QC → QIR Conversion
  pm.addPass(createQCToQIR());
  addCleanupPasses(pm); // QIR Canonicalization
}

} // namespace qcc
} // namespace mlir
