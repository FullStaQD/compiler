// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/Analysis/ConnectivityAnalysis.h"
#include "qcc/Dialect/QCC/IR/QCC.h"
#include "qcc/Dialect/QCC/Transforms/Passes.h" // IWYU pragma: keep

#include "QCCGateWalker.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

namespace qcc {

#define GEN_PASS_DEF_VERIFYCONNECTIVITY
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// Renders a qubit set as `{a, b, c}`.
std::string formatSet(llvm::ArrayRef<int64_t> qubits) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "{";
  llvm::interleaveComma(qubits, os);
  os << "}";
  return text;
}

struct VerifyConnectivity final : public impl::VerifyConnectivityBase<VerifyConnectivity> {
  using VerifyConnectivityBase<VerifyConnectivity>::VerifyConnectivityBase;

protected:
  void runOnOperation() override {
    func::FuncOp function = getOperation();

    auto& configs = getAnalysis<DominatingConfigMap>();
    // A function mentioning no configuration has not opted in, so the pass can
    // sit in a pipeline that also compiles programs with no device attached.
    if (configs.empty()) {
      return markAllAnalysesPreserved();
    }

    ConfigurationAnalysis analysis(function);
    llvm::DenseMap<Value, InteractionComplex> complexes;

    const auto result =
        function.walk([&](qc::UnitaryOpInterface gate) { return checkGate(gate, configs, analysis, complexes); });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }

    markAllAnalysesPreserved();
  }

private:
  /// Checks one gate against the configuration in force where it stands.
  WalkResult checkGate(qc::UnitaryOpInterface gate, DominatingConfigMap& configs, ConfigurationAnalysis& analysis,
                       llvm::DenseMap<Value, InteractionComplex>& complexes) {
    Operation* op = gate.getOperation();
    if (!isConstrainedByConnectivity(op) || gate.getNumQubits() == 0) {
      return WalkResult::advance();
    }

    const auto lookup = configs.at(op);
    if (lookup.ambiguous) {
      op->emitOpError() << "two configurations reach this operation and neither dominates the other; "
                           "control flow merged them without an explicit join";
      return WalkResult::interrupt();
    }
    // Nothing establishes a configuration before this point, so no connectivity
    // is in force to check against.
    if (!lookup.config) {
      return WalkResult::advance();
    }

    const auto qubits = collectQubits(gate);
    if (!qubits) {
      if (!requireResolvableQubits) {
        return WalkResult::advance();
      }
      op->emitOpError() << "cannot be checked against connectivity: at least one of its qubits carries no "
                           "program index; connectivity is stated over hardware qubits, so the operands "
                           "have to reach a `qc.static`";
      return WalkResult::interrupt();
    }

    const auto reaching = analysis.get(lookup.config);
    // An ill-formed chain has already reported why.
    if (failed(reaching)) {
      return WalkResult::interrupt();
    }
    if (!reaching->isKnown()) {
      return reportDynamic(op, *reaching);
    }

    const Configuration& config = *reaching->config;
    // Reported before membership, since "qubit 9 is not placed" is a clearer
    // diagnostic than "no facet contains it".
    for (const int64_t qubit : *qubits) {
      if (!config.findQubit(qubit)) {
        op->emitOpError() << "acts on qubit " << qubit << ", which the configuration reaching it does not place";
        return WalkResult::interrupt();
      }
    }

    // A single-qubit gate is a local rotation. It needs the qubit placed and
    // nothing more, and on a device that is not downward closed a membership
    // test would spuriously reject it.
    if (qubits->size() < 2) {
      return WalkResult::advance();
    }

    const auto device = llvm::cast<ConfigType>(lookup.config.getType()).getDevice();
    const auto& complex = complexes.try_emplace(lookup.config, device, config).first->second;
    if (complex.isExecutable(*qubits)) {
      return WalkResult::advance();
    }

    auto diagnostic =
        op->emitOpError() << "operand set " << formatSet(*qubits)
                          << " is not executable in the configuration reaching it: " << complex.explain(*qubits);
    diagnostic.attachNote(lookup.config.getLoc()) << "connectivity in force here is defined by this value";
    return WalkResult::interrupt();
  }

  /// Reports a configuration the analysis could not fold to a constant.
  WalkResult reportDynamic(Operation* op, const ConfigurationAnalysis::Result& reaching) {
    auto diagnostic = allowDynamic ? op->emitWarning() : op->emitOpError();
    diagnostic << "the configuration reaching this operation is not statically known: it is carried by "
               << (llvm::isa<BlockArgument>(reaching.dynamicOrigin) ? "a loop or region boundary"
                                                                    : "an operation outside the rule set")
               << ", so K cannot be computed here";
    diagnostic.attachNote() << "a configuration crossing a loop boundary is statically known only if the body's "
                               "net rewrite is the identity; otherwise the choice is a fixpoint over an "
                               "ordered-partition lattice or a runtime co-location check";
    return allowDynamic ? WalkResult::advance() : WalkResult::interrupt();
  }
};

} // namespace
} // namespace qcc
