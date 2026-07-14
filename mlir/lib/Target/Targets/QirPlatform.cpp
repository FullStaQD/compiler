// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Conversion/ToQIR/Constants.h"
#include "qcc/Conversion/ToQIR/QIRPipeline.h"

#include "../Targets.h"

#include "mlir/Pass/PassManager.h"

#include <array>

namespace qcc {

namespace {

/// An idealized device: it implements the whole QIR gate set and its qubits are unbounded, so a
/// circuit needs no adaptation to run on it.
class IdealDevice : public QuantumTarget {
public:
  llvm::StringRef getName() const override { return "ideal"; }

  llvm::ArrayRef<llvm::StringRef> getNativeGates() const override {
    static constexpr std::array<llvm::StringRef, 10> kGates = {qirQisH,   qirQisX,  qirQisS,  qirQisSdg, qirQisT,
                                                               qirQisTdg, qirQisRZ, qirQisCX, qirQisMZ,  qirQisReset};
    return kGates;
  }

  void addDevicePasses(mlir::PassManager& /*pm*/) const override {}
};

/// No control hardware: QIR is where the compilation ends, and nothing executes it.
class NoController : public ControlTarget {
public:
  llvm::StringRef getName() const override { return "none"; }

  llvm::ArrayRef<Stage> getSupportedStages() const override {
    static constexpr Stage kStages[] = {Stage::Mlir, Stage::LlvmIr};
    return kStages;
  }

  /// QIR is the output rather than an intermediate, so the lowering stops there.
  void addLoweringPasses(mlir::PassManager& pm, const QuantumTarget& device) const override {
    buildQIRPipeline(pm, device.getNativeGates());
  }
};

/// QIR, the output of the shared pipeline: LLVM IR calling the QIS and runtime functions of the QIR
/// spec. It is an interchange format rather than a machine, so it pairs an idealized device with no
/// controller and stops at LLVM IR.
class QirPlatform : public Platform {
public:
  llvm::StringRef getName() const override { return "qir"; }
  llvm::StringRef getDescription() const override { return "QIR, as LLVM IR calling the QIS functions"; }

  const QuantumTarget& getQuantumTarget() const override { return device; }
  const ControlTarget& getControlTarget() const override { return controller; }

private:
  IdealDevice device;
  NoController controller;
};

} // namespace

const Platform& getQirPlatform() {
  static const QirPlatform platform;
  return platform;
}

} // namespace qcc
