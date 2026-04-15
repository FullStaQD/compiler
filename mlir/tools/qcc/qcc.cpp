#include "qcc/Compiler/Pipeline.h"
#include "qcc/Dialect/Aux_/IR/Aux_.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

namespace cl = llvm::cl;

static cl::OptionCategory qccCategory("QCC options");

int main(int argc, char** argv) {
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("Input-file"), cl::Required, cl::cat(qccCategory));
  cl::opt<std::string> outputFilename("o", cl::desc("Output-file"), cl::value_desc("filename"), cl::cat(qccCategory));

  cl::HideUnrelatedOptions(qccCategory);
  cl::ParseCommandLineOptions(argc, argv, "qcc - quantum compiler collection\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::qc::QCDialect, qcc::aux::AuxDialect>();

  mlir::MLIRContext context(registry);

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Enable nice diagnostic printing for parser and pass errors
  mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(&context);
  qcc::buildQuantumPipeline(pm);

  if (mlir::failed(pm.run(*module))) {
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep(); // otherwise file gets deleted

  return 0;
}
