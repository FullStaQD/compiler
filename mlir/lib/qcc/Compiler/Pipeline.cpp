
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
  pm.addPass(createRemoveDeadValuesPass());
}

void buildQuantumPipeline(PassManager& pm) {
  // Initial QC Canonicalization
  pm.addPass(createCanonicalizerPass());

  // QC → QIR Conversion
  pm.addPass(createQCToQIR());
  addCleanupPasses(pm); // QIR Canonicalization
}

} // namespace qcc
} // namespace mlir
