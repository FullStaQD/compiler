#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "qcc/Compiler/Pipeline.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0] << " <input_file.mlir>\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::qc::QCDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string errorMessage;
  auto file = mlir::openInputFile(argv[1], &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to open file: " << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Parser failed! Your MLIR file has a syntax or dialect error.\n";
    return 1;
  }

  mlir::PassManager pm(&context);
  qcc::buildQuantumPipeline(pm);

  llvm::errs() << "Running pipeline...\n";
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pipeline execution failed!\n";
    return 1;
  }

  module->print(llvm::outs());
  return 0;
}
