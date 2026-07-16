// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace llvm {
class MemoryBuffer;
class Module;
class TargetMachine;
class raw_ostream;
} // namespace llvm

namespace mlir {
class PassManager;
} // namespace mlir

namespace qcc {

/// The stage to compile to and emit, selected with `--compile-to`.
enum class Stage : uint8_t { Mlir, LlvmIr, Assembly, Object, Elf, Mem };

/// The primary `--compile-to` spelling of `stage`.
llvm::StringRef getStageName(Stage stage);

/// The `--mtriple` and `--mattr` overrides. An empty field keeps the default of the target.
struct CodeGenOptions {
  llvm::StringRef triple;
  llvm::StringRef features;
};

/// A quantum device: what the qubits are and which gates act on them.
///
/// This is the hardware qcc compiles *for* at the quantum level, independently of the classical
/// machine that drives it. Its passes run on the qc dialect, above QIR, and are where gate-set
/// decomposition, qubit mapping and QEC encoding belong.
class QuantumTarget {
public:
  virtual ~QuantumTarget() = default;

  [[nodiscard]] virtual llvm::StringRef getName() const = 0;

  /// The QIS functions this device implements, by the names of `ToQIR/Constants.h`.
  ///
  /// A circuit reaching QIR may only use these: a gate outside the set has no hardware to run on.
  /// The passes of the pipeline are told about them through the `native-gates` pass option.
  [[nodiscard]] virtual llvm::ArrayRef<llvm::StringRef> getNativeGates() const = 0;

  /// Appends the passes that adapt a circuit to this device.
  virtual void addDevicePasses(mlir::PassManager& pm) const = 0;
};

/// The classical control hardware that executes a compiled circuit.
///
/// This is the machine qcc generates code for: it turns QIR into its own instruction set, and owns
/// everything below that -- the LLVM target machine, the linker script, and the memory image its
/// hardware or simulator loads.
class ControlTarget {
public:
  virtual ~ControlTarget() = default;

  [[nodiscard]] virtual llvm::StringRef getName() const = 0;

  /// The stages this control hardware can emit. The driver rejects the other ones up front, so the
  /// hooks below are only called for a stage listed here.
  [[nodiscard]] virtual llvm::ArrayRef<Stage> getSupportedStages() const = 0;

  [[nodiscard]] bool supportsStage(Stage stage) const;

  /// Whether this control hardware can execute the QIS function `qisName`.
  ///
  /// The default accepts every gate: a controller that emits the QIR calls unchanged puts no
  /// constraint on the gate set. One that lowers them to instructions only executes the gates its
  /// instruction set covers.
  [[nodiscard]] virtual bool canExecuteGate(llvm::StringRef qisName) const;

  /// Appends the passes that take the circuit of `device`, in the qc dialect, down to what this
  /// control hardware executes.
  ///
  /// The path is the choice of the control hardware. One that speaks QIR calls `buildQIRPipeline`
  /// and lowers from there; one with its own route out of the qc dialect does not go through QIR at
  /// all.
  virtual void addLoweringPasses(mlir::PassManager& pm, const QuantumTarget& device) const = 0;

  /// Creates the machine that emits `Stage::Assembly` and `Stage::Object`, and sets the data layout
  /// and the triple on `llvmModule`. Prints an error and returns nullptr on failure.
  virtual std::unique_ptr<llvm::TargetMachine> createTargetMachine(llvm::Module& llvmModule,
                                                                   const CodeGenOptions& options) const;

  /// The linker script that lays out `Stage::Elf`.
  [[nodiscard]] virtual llvm::StringRef getLinkerScript() const;

  /// Converts the ELF image linked for `Stage::Elf` into the memory image of `Stage::Mem`.
  virtual llvm::Error writeMemoryImage(const llvm::MemoryBuffer& elf, llvm::raw_ostream& os) const;
};

/// A machine qcc compiles for, selected with `--target`: a quantum device plus the control hardware
/// that drives it.
///
/// The two are separate because they vary separately -- the same control hardware can drive another
/// device, and a device can be driven by another controller -- and because they own different
/// halves of the pipeline: the device shapes the circuit, the controller turns it into code.
class Platform {
public:
  virtual ~Platform() = default;

  /// The name that selects this platform on the command line, and the help text next to it.
  [[nodiscard]] virtual llvm::StringRef getName() const = 0;
  [[nodiscard]] virtual llvm::StringRef getDescription() const = 0;

  [[nodiscard]] virtual const QuantumTarget& getQuantumTarget() const = 0;
  [[nodiscard]] virtual const ControlTarget& getControlTarget() const = 0;

  /// Checks that the control hardware can execute every gate the device implements.
  ///
  /// The two halves are written independently, and a device gate that the controller cannot turn
  /// into an instruction would be compiled into a call that nothing executes.
  llvm::Error verify() const;
};

/// The platforms qcc knows about, in the order `--target` lists them. A new machine goes here.
llvm::ArrayRef<const Platform*> getPlatforms();

/// The platform named `name`, or nullptr if qcc has no such platform.
const Platform* lookupPlatform(llvm::StringRef name);

/// The platform that `--target` defaults to.
const Platform& getDefaultPlatform();

} // namespace qcc
