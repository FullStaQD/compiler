// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Compiler/Pipeline.h"
#include "qcc/Dialect/Aux_/IR/Aux_.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"
#include "qcc/Target/Target.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include <optional>
#include <string>

namespace cl = llvm::cl;

using qcc::ControlTarget;
using qcc::Platform;
using qcc::Stage;

static cl::OptionCategory qccCategory("QCC options");

namespace {

/// Registers everything the pipeline needs: the builtin dialects, ours, the interfaces that
/// bufferization relies on, and the translations to LLVM IR.
mlir::DialectRegistry buildDialectRegistry() {
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  registry.insert<jasp::JaspDialect, mlir::qc::QCDialect, qcc::aux::AuxDialect>();

  // OneShotBufferize requires these for the "standard" dialects.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::func::registerInlinerExtension(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  return registry;
}

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& llvmContext) {
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "error: failed to translate the module to LLVM IR\n";
  }
  return llvmModule;
}

/// Creates a temporary file that `remover` deletes once it goes out of scope.
std::optional<llvm::SmallString<128>> createTempFile(llvm::StringRef suffix, llvm::FileRemover& remover) {
  llvm::SmallString<128> path;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile("qcc", suffix, path)) {
    llvm::errs() << "error: could not create a temporary ." << suffix << " file: " << ec.message() << "\n";
    return std::nullopt;
  }
  remover.setFile(path);
  return path;
}

/// Runs the codegen passes that emit `fileType` for `llvmModule` on `os`.
bool emitNativeFile(llvm::Module& llvmModule, llvm::TargetMachine& targetMachine, llvm::CodeGenFileType fileType,
                    llvm::raw_pwrite_stream& os) {
  llvm::legacy::PassManager codegenPM;
  if (targetMachine.addPassesToEmitFile(codegenPM, os, /*DwoOut=*/nullptr, fileType)) {
    llvm::errs() << "error: the target machine cannot emit files of this type\n";
    return false;
  }
  codegenPM.run(llvmModule);
  return true;
}

/// Writes `llvmModule` to a temporary object file, whose path is returned.
std::optional<llvm::SmallString<128>> emitObjectFile(llvm::Module& llvmModule, llvm::TargetMachine& targetMachine,
                                                     llvm::FileRemover& remover) {
  std::optional<llvm::SmallString<128>> objPath = createTempFile("o", remover);
  if (!objPath) {
    return std::nullopt;
  }

  std::error_code ec;
  llvm::raw_fd_ostream objFile(*objPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "error: could not open a temporary object file: " << ec.message() << "\n";
    return std::nullopt;
  }

  if (!emitNativeFile(llvmModule, targetMachine, llvm::CodeGenFileType::ObjectFile, objFile)) {
    return std::nullopt;
  }
  return objPath;
}

/// Links `objPath` into an ELF image at `elfPath`, laid out by `linkerScript` (which the target
/// carries, see `Target::getLinkerScript`) and using `linkerExe` (e.g. "ld.lld") from PATH.
bool linkElf(llvm::StringRef linkerExe, llvm::StringRef linkerScript, llvm::StringRef objPath,
             llvm::StringRef elfPath) {
  llvm::ErrorOr<std::string> linkerPath = llvm::sys::findProgramByName(linkerExe);
  if (!linkerPath) {
    llvm::errs() << "error: could not find the linker '" << linkerExe << "' on PATH (override with --linker)\n";
    return false;
  }

  llvm::FileRemover scriptRemover;
  std::optional<llvm::SmallString<128>> scriptPath = createTempFile("ld", scriptRemover);
  if (!scriptPath) {
    return false;
  }

  {
    std::error_code ec;
    llvm::raw_fd_ostream scriptFile(*scriptPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "error: could not write the linker script: " << ec.message() << "\n";
      return false;
    }
    scriptFile << linkerScript;
  }

  llvm::SmallVector<llvm::StringRef> args = {*linkerPath, "-T", *scriptPath, objPath, "-o", elfPath};
  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*linkerPath, args, /*Env=*/std::nullopt, /*Redirects=*/{},
                                     /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (rc != 0) {
    llvm::errs() << "error: linking failed" << (errMsg.empty() ? "" : ": " + errMsg) << "\n";
    return false;
  }
  return true;
}

/// Emits `Stage::Elf` and `Stage::Mem`: compiles `llvmModule` to an object file and links it with
/// the linker script of `controlTarget`. For `Stage::Mem` the ELF is a scratch file, which the
/// control target then converts into the memory image its hardware loads.
int emitImage(llvm::Module& llvmModule, llvm::TargetMachine& targetMachine, const ControlTarget& controlTarget,
              Stage stage, llvm::StringRef outputFilename, llvm::StringRef linkerExe) {
  llvm::FileRemover objRemover;
  std::optional<llvm::SmallString<128>> objPath = emitObjectFile(llvmModule, targetMachine, objRemover);
  if (!objPath) {
    return 1;
  }

  llvm::FileRemover elfRemover;
  llvm::SmallString<128> elfPath(outputFilename);
  if (stage == Stage::Mem) {
    std::optional<llvm::SmallString<128>> tempElfPath = createTempFile("elf", elfRemover);
    if (!tempElfPath) {
      return 1;
    }
    elfPath = *tempElfPath;
  }

  if (!linkElf(linkerExe, controlTarget.getLinkerScript(), *objPath, elfPath)) {
    return 1;
  }

  if (stage == Stage::Elf) {
    return 0;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> elfBuffer = llvm::MemoryBuffer::getFile(elfPath);
  if (std::error_code ec = elfBuffer.getError()) {
    llvm::errs() << "error: " << elfPath << ": " << ec.message() << "\n";
    return 1;
  }

  std::string errorMessage;
  auto outFile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (llvm::Error err = controlTarget.writeMemoryImage(**elfBuffer, outFile->os())) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  outFile->keep();
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  // `--target` is validated against the registry rather than being a fixed set of options, so a new
  // platform shows up here without touching the driver.
  std::string targetNames;
  for (const Platform* platform : qcc::getPlatforms()) {
    targetNames += (targetNames.empty() ? "" : ", ") + platform->getName().str();
  }
  const std::string targetHelp = "Target to compile for, one of: " + targetNames;

  const cl::opt<std::string> inputFilename(cl::Positional, cl::desc("Input-file"), cl::Required, cl::cat(qccCategory));
  const cl::opt<std::string> outputFilename("o", cl::desc("Output-file"), cl::value_desc("filename"), cl::init("-"),
                                            cl::cat(qccCategory));
  const cl::opt<std::string> targetName("target", cl::desc(targetHelp), cl::value_desc("target"),
                                        cl::init(qcc::getDefaultPlatform().getName().str()), cl::cat(qccCategory));
  const cl::opt<Stage> compileTo(
      "compile-to", cl::desc("Stage to lower to and emit; a target only supports some of them"),
      cl::init(Stage::LlvmIr),
      cl::values(clEnumValN(Stage::Mlir, "mlir", "MLIR in the LLVM dialect"),
                 clEnumValN(Stage::LlvmIr, "ll", "LLVM IR"), clEnumValN(Stage::LlvmIr, "llvmir", "alias for ll"),
                 clEnumValN(Stage::Assembly, "s", "Assembly"), clEnumValN(Stage::Assembly, "assembly", "alias for s"),
                 clEnumValN(Stage::Object, "o", "Object file"), clEnumValN(Stage::Object, "object", "alias for o"),
                 clEnumValN(Stage::Elf, "elf", "ELF image, linked with the linker script of the target"),
                 clEnumValN(Stage::Mem, "mem", "Memory image, ready for the simulator of the target")),
      cl::cat(qccCategory));
  const cl::opt<std::string> mtriple("mtriple", cl::desc("Overrides the target triple used for code generation"),
                                     cl::init(""), cl::cat(qccCategory));
  const cl::opt<std::string> mattr("mattr", cl::desc("Overrides the target attributes used for code generation"),
                                   cl::init(""), cl::cat(qccCategory));
  const cl::opt<std::string> linker(
      "linker", cl::desc("Linker to invoke for --compile-to=elf/mem (default: ld.lld, resolved via PATH)"),
      cl::init("ld.lld"), cl::cat(qccCategory));

  cl::ParseCommandLineOptions(argc, argv, "qcc - quantum compiler collection\n");

  const Platform* platform = qcc::lookupPlatform(targetName);
  if (platform == nullptr) {
    llvm::errs() << "error: unknown target '" << targetName << "', expected one of: " << targetNames << "\n";
    return 1;
  }

  if (llvm::Error err = platform->verify()) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  // How far a platform can compile is decided by its control hardware: the device shapes the
  // circuit, but only a controller turns it into code.
  const ControlTarget& controlTarget = platform->getControlTarget();
  if (!controlTarget.supportsStage(compileTo)) {
    llvm::errs() << "error: target '" << platform->getName()
                 << "' does not support --compile-to=" << qcc::getStageName(compileTo) << "\n";
    return 1;
  }

  mlir::DialectRegistry registry = buildDialectRegistry();
  mlir::MLIRContext context(registry);

  std::string errorMessage;
  auto inFile = mlir::openInputFile(inputFilename, &errorMessage);
  if (!inFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inFile), llvm::SMLoc());

  // Enable nice diagnostic printing for parser and pass errors.
  const mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(&context);
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
    return 1;
  }
  qcc::buildQuantumPipeline(pm, *platform);

  if (mlir::failed(pm.run(*module))) {
    return 1;
  }

  if (compileTo == Stage::Mlir) {
    auto outFile = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!outFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    module->print(outFile->os());
    outFile->keep();
    return 0;
  }

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    return 1;
  }

  if (compileTo == Stage::LlvmIr) {
    auto outFile = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!outFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    llvmModule->print(outFile->os(), /*AAW=*/nullptr);
    outFile->keep();
    return 0;
  }

  // Everything below here is native code, and so needs a target machine.
  const qcc::CodeGenOptions codeGenOptions = {.triple = mtriple, .features = mattr};
  std::unique_ptr<llvm::TargetMachine> targetMachine = controlTarget.createTargetMachine(*llvmModule, codeGenOptions);
  if (!targetMachine) {
    return 1;
  }

  if (compileTo == Stage::Elf || compileTo == Stage::Mem) {
    return emitImage(*llvmModule, *targetMachine, controlTarget, compileTo, outputFilename, linker);
  }

  auto outFile = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!outFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto fileType = compileTo == Stage::Object ? llvm::CodeGenFileType::ObjectFile : llvm::CodeGenFileType::AssemblyFile;
  auto& os = static_cast<llvm::raw_pwrite_stream&>(outFile->os());
  if (!emitNativeFile(*llvmModule, *targetMachine, fileType, os)) {
    return 1;
  }

  outFile->keep();
  return 0;
}
