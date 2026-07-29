// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// HiSEP-Q backend implementation. This is the ONLY place in the project allowed
// to depend on the HiSEP-Q LLVM fork (its headers and libraries); it is compiled
// only when QCC_ENABLE_HISEP_Q is enabled.
//
// ===----------------------------------------------------------------------===//

#include "qcc/Target/HiSEPQ/HiSEPQTarget.h"

#include "qcc/Conversion/ToIntrinsics/ToIntrinsics.h"
#include "qcc/Target/QIR/QIRTarget.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <optional>
#include <string>

// `llvm/Support/TargetSelect.h` only declares the `LLVMInitializeRISCV*` entry
// points when RISCV is in the toolchain's `LLVM_TARGETS_TO_BUILD`. That holds
// for the HiSEP-Q fork we build and link against, but not necessarily for the
// LLVM a clang-tidy pass parses with. Declare them explicitly so the file always
// parses; the symbols resolve at link time against the RISCV component libs
// pulled in (fork-only) via LLVM_LINK_COMPONENTS.
// NOLINTBEGIN(readability-identifier-naming): these are LLVM C-API entry points
// whose names are fixed by LLVM and cannot follow the project's naming style.
extern "C" {
void LLVMInitializeRISCVTargetInfo();
void LLVMInitializeRISCVTarget();
void LLVMInitializeRISCVTargetMC();
void LLVMInitializeRISCVAsmPrinter();
}
// NOLINTEND(readability-identifier-naming)

namespace qcc {

void addLoweringPassesHiSEPQ(mlir::PassManager& pm) {
  // HiSEP-Q shares the full QIR lowering, then replaces the QIS call ops with
  // RISC-V QV intrinsics so the module can be emitted as native QISA.
  addLoweringPassesQIR(pm);
  pm.addPass(qcc::createConvertQIRToIntrinsics());
}

bool emitNativeHiSEPQ(llvm::Module& module, llvm::raw_pwrite_stream& os, const NativeCodegenOptions& options) {
  // HiSEP-Q QISA is encoded as the experimental "xqv" RISC-V vector extension,
  // provided by the HiSEP-Q LLVM fork.
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTarget();
  LLVMInitializeRISCVTargetMC();
  LLVMInitializeRISCVAsmPrinter();

  const std::string attrsStr = "+experimental-xqv";
  llvm::Triple triple(llvm::Triple::normalize("riscv32-unknown-unknown"));

  std::string errorStr;
  const llvm::Target* theTarget = llvm::TargetRegistry::lookupTarget(/*MArch=*/"", triple, errorStr);
  if (theTarget == nullptr) {
    llvm::errs() << "could not find target '" << triple.str() << "': " << errorStr << "\n";
    return true;
  }

  const llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      theTarget->createTargetMachine(triple, /*cpu=*/"", attrsStr, targetOptions, std::nullopt));

  module.setDataLayout(targetMachine->createDataLayout());
  module.setTargetTriple(triple);

  const auto fileType = options.binary ? llvm::CodeGenFileType::ObjectFile : llvm::CodeGenFileType::AssemblyFile;

  llvm::legacy::PassManager codegenPM;
  if (targetMachine->addPassesToEmitFile(codegenPM, os, /*DwoOut=*/nullptr, fileType)) {
    llvm::errs() << "target machine cannot emit files of this type\n";
    return true;
  }
  codegenPM.run(module);
  return false;
}

} // namespace qcc
