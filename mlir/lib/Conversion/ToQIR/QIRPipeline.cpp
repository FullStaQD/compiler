// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/ToQIR/QIRPipeline.h"

#include "qcc/Conversion/Aux_/AuxOutputRecording.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace qcc {

void buildQIRPipeline(mlir::PassManager& pm, llvm::ArrayRef<llvm::StringRef> nativeGates) {
  // The device tells these passes which QIS functions it implements: PrepToQIR declares only those,
  // and ConvertQCToQIR rejects a gate that is not among them.
  const llvm::SmallVector<std::string> gates(nativeGates.begin(), nativeGates.end());

  PrepToQIROptions prepOptions;
  prepOptions.nativeGates = gates;
  ConvertQCToQIROptions convertOptions;
  convertOptions.nativeGates = gates;

  pm.addPass(qcc::createAuxOutputRecording());
  pm.addPass(qcc::createPrepToQIR(prepOptions));
  mlir::OpPassManager& fpm = pm.nest<mlir::func::FuncOp>();
  fpm.addPass(mlir::createArithToLLVMConversionPass());
  fpm.addPass(qcc::createConvertQCToQIR(convertOptions));
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(qcc::createFinalizeToQIR());

  // cleanup QIR
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

} // namespace qcc
