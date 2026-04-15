
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

  // Qrisp output contains a lot of functions that can be trivially inlined.
  pm.addPass(mlir::createInlinerPass());

  // Lowering from JASP to QC
  // In addition to the obvious conversions, the rank-0 tensors
  // are converted to plain values (e.g. tensor<i64> becomes i64)
  // whenever possible.
  pm.addPass(qcc::createJaspToQC());

  // Initial cleanup
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Convert `tensor.empty` to `bufferization.alloc_tensor`, which is expected by
  // bufferization.
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

  // Detensorization attempts to convert those rank-0-tensors to plain values
  // which have not been eliminated in the JaspToQC pass.
  // Aggressive mode ensures that some trivial `linalg.generic` ops are
  // unwrapped.
  mlir::LinalgDetensorizePassOptions detensorizeOptions;
  detensorizeOptions.aggressiveMode = true;
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass(detensorizeOptions));

  // Via Bufferization, we go from value semantics (tensor types) to
  // memory semantics (memref types). We want to make these changes in function signatures
  // as well, and allow for unknown (quantum) operations in the IR.
  mlir::bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.allowUnknownOps = true;
  bufferizeOptions.allowReturnAllocsFromLoops = true;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));

  // To facilitate data flow analysis, memory allocation is "hoisted out of loops" whenever possible.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());

  // Leftover `linalg` operations are converted to `scf` loops.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());

  // Second cleanup
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Whenever it makes sense (according to internal heuristics), promote a heap allocation
  // (`memref.alloc`) to a stack allocation (`memref.alloca`).
  pm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createPromoteBuffersToStackPass());

  // Whenever it makes sense, promote stack-allocated variables (e.g. `memref<i64>`) to
  // plain register-values (e.g. `i64`).
  pm.addPass(mlir::createMem2Reg());

  // Final cleanup
  pm.addPass(mlir::createSCCPPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

} // namespace qcc
