#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "qcc/Conversion/JaspToQC/JaspToQC.h"
#include "qcc/Conversion/ToQIR/ToQIR.h"
#include "qcc/Dialect/Aux/IR/Aux.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include <stablehlo/dialect/Register.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<
      // clang-format off
    mlir::func::FuncDialect,
    mlir::arith::ArithDialect,
    mlir::tensor::TensorDialect,
    mlir::cf::ControlFlowDialect,
    mlir::scf::SCFDialect,
    mlir::LLVM::LLVMDialect,
    jasp::JaspDialect,
    mlir::qc::QCDialect,
    qcc::aux::AuxDialect
      // clang-format on
      >();

  // 3rd party dialects
  mlir::stablehlo::registerAllDialects(registry);

  // Builtin passes:
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();

  // Our passes
  qcc::registerJaspToQC();
  qcc::registerConvertQCToQIR();
  qcc::registerStdToLLVM();
  qcc::registerPrepToQIR();
  qcc::registerFinalizeToQIR(); // FIXME: do we really want to register them individually?

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "qcc optimizer", registry));
}
