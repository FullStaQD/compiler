#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "qcc/Compiler/Pipeline.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

namespace cl = llvm::cl;

static cl::OptionCategory QccCategory("QCC options");

int main(int argc, char** argv) {
  llvm::cl::opt<std::string> inputFilename(cl::Positional, cl::desc("Input-file"), cl::Required, cl::cat(QccCategory));
  llvm::cl::opt<std::string> outputFilename("o", cl::desc("Output-file"), cl::value_desc("filename"),
                                            cl::cat(QccCategory));

  llvm::cl::HideUnrelatedOptions(QccCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv, "qcc - quantum compiler collection\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect, mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  jasp::JaspDialect, mlir::qc::QCDialect>();

  // Extension registration
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);

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
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  qcc::buildQuantumPipeline(pm);

  if (mlir::failed(pm.run(*module)))
    return 1;

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep(); // otherwise file gets deleted

  return 0;
}
