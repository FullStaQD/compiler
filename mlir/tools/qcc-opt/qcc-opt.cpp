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
#include "qcc/Dialect/Jasp/IR/Jasp.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <stablehlo/dialect/Register.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect, jasp::JaspDialect, mlir::qc::QCDialect>();
  qcc::registerJaspToQC();
  qcc::registerQCToQIR();
  qcc::registerStdToQIR();
  qcc::registerToQIRPrep();
  qcc::registerToQIRFinalize(); // FIXME: do we really want to register them individually?
  mlir::stablehlo::registerAllDialects(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "qcc optimizer", registry));
}
