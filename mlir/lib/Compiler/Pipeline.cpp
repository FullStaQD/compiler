
#include "qcc/Compiler/Pipeline.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "qcc/Conversion/JaspToQC/JaspToQC.h"

namespace qcc {

void buildQuantumPipeline(mlir::PassManager& pm) {

  // Lowering from JASP to QC
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(qcc::createJaspToQC());

  // Initial cleanup
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Bufferization prep
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

  mlir::LinalgDetensorizePassOptions detensorizeOptions;
  detensorizeOptions.aggressiveMode = true;
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass(detensorizeOptions));

  // One-shot Bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.allowUnknownOps = true;
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

  // Post-bufferization optimization
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createPromoteBuffersToStackPass());

  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::createSCCPPass());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

} // namespace qcc
