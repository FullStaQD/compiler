
#include "qcc/Compiler/Pipeline.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace qcc {

void buildQuantumPipeline(mlir::PassManager& pm) {
  // Cleanup QC
  pm.addPass(mlir::createCanonicalizerPass());

  // QC → QIR Conversion
  pm.addPass(qcc::createToQIRPrep());

  mlir::OpPassManager& fpm = pm.nest<mlir::func::FuncOp>();
  fpm.addPass(mlir::createArithToLLVMConversionPass());
  fpm.addPass(qcc::createConvertQCToQIR());

  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(qcc::createToQIRFinalize());

  // Cleanup QIR
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

} // namespace qcc
