// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Target/Target.h"

#include "Targets.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <array>

namespace qcc {

llvm::StringRef getStageName(Stage stage) {
  switch (stage) {
  case Stage::Mlir:
    return "mlir";
  case Stage::LlvmIr:
    return "ll";
  case Stage::Assembly:
    return "s";
  case Stage::Object:
    return "o";
  case Stage::Elf:
    return "elf";
  case Stage::Mem:
    return "mem";
  }
  llvm_unreachable("unknown stage");
}

bool ControlTarget::supportsStage(Stage stage) const { return llvm::is_contained(getSupportedStages(), stage); }

bool ControlTarget::canExecuteGate(llvm::StringRef /*qisName*/) const { return true; }

llvm::Error Platform::verify() const {
  const QuantumTarget& device = getQuantumTarget();
  const ControlTarget& controller = getControlTarget();

  for (llvm::StringRef gate : device.getNativeGates()) {
    if (!controller.canExecuteGate(gate)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "platform '%s': the device '%s' implements '%s', but the control hardware "
                                     "'%s' cannot execute it",
                                     getName().str().c_str(), device.getName().str().c_str(), gate.str().c_str(),
                                     controller.getName().str().c_str());
    }
  }
  return llvm::Error::success();
}

// The three hooks below are reached only when a control target lists a stage in
// `getSupportedStages()` without implementing it.

std::unique_ptr<llvm::TargetMachine> ControlTarget::createTargetMachine(llvm::Module& /*llvmModule*/,
                                                                        const CodeGenOptions& /*options*/) const {
  llvm::errs() << "error: control target '" << getName() << "' has no native code generator\n";
  return nullptr;
}

llvm::StringRef ControlTarget::getLinkerScript() const { return ""; }

llvm::Error ControlTarget::writeMemoryImage(const llvm::MemoryBuffer& /*elf*/, llvm::raw_ostream& /*os*/) const {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "control target '%s' has no memory image format",
                                 getName().str().c_str());
}

llvm::ArrayRef<const Platform*> getPlatforms() {
  static const std::array<const Platform*, 2> platforms = {&getQirPlatform(), &getHisepQPlatform()};
  return platforms;
}

const Platform* lookupPlatform(llvm::StringRef name) {
  const auto* platform =
      llvm::find_if(getPlatforms(), [&](const Platform* platform) { return platform->getName() == name; });
  return platform == getPlatforms().end() ? nullptr : *platform;
}

const Platform& getDefaultPlatform() { return getQirPlatform(); }

} // namespace qcc
