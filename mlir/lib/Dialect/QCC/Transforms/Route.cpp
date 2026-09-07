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

#define GEN_PASS_DEF_ROUTE
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// Where each of `qubits` currently sits, for a reader who has to move them.
std::string describePlacement(const Configuration& config, llvm::ArrayRef<int64_t> qubits) {
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleaveComma(qubits, os, [&](const int64_t qubit) {
    os << qubit << " at ";
    if (const auto site = config.findQubit(qubit)) {
      os << "@" << *site;
    } else {
      os << "no site";
    }
  });
  return text;
}

struct Route final : public impl::RouteBase<Route> {
  using RouteBase<Route>::RouteBase;

protected:
  void runOnOperation() override {
    func::FuncOp function = getOperation();

    auto& configs = getAnalysis<DominatingConfigMap>();
    if (configs.empty()) {
      return markAllAnalysesPreserved();
    }

    ConfigurationAnalysis analysis(function);
    llvm::DenseMap<Value, InteractionComplex> complexes;
    unsigned unroutable = 0;

    const auto result = function.walk(
        [&](qc::UnitaryOpInterface gate) { return checkGate(gate, configs, analysis, complexes, unroutable); });

    if (result.wasInterrupted() || (unroutable > 0 && !emitRemarks)) {
      signalPassFailure();
    }

    markAllAnalysesPreserved();
  }

private:
  /// Reports `gate` when the configuration in force cannot execute it. Gates
  /// this pass cannot evaluate are skipped rather than reported, since
  /// `--qcc-verify-connectivity` is the pass that judges them.
  WalkResult checkGate(qc::UnitaryOpInterface gate, DominatingConfigMap& configs, ConfigurationAnalysis& analysis,
                       llvm::DenseMap<Value, InteractionComplex>& complexes, unsigned& unroutable) {
    Operation* op = gate.getOperation();
    if (!isConstrainedByConnectivity(op) || gate.getNumQubits() < 2) {
      return WalkResult::advance();
    }

    const auto lookup = configs.at(op);
    if (!lookup.config || lookup.ambiguous) {
      return WalkResult::advance();
    }

    const auto qubits = collectQubits(gate);
    if (!qubits) {
      return WalkResult::advance();
    }

    const auto reaching = analysis.get(lookup.config);
    if (failed(reaching) || !reaching->isKnown()) {
      return WalkResult::advance();
    }

    const auto device = llvm::cast<ConfigType>(lookup.config.getType()).getDevice();
    const Configuration& config = *reaching->config;
    const auto& complex = complexes.try_emplace(lookup.config, device, config).first->second;
    if (complex.isExecutable(*qubits)) {
      return WalkResult::advance();
    }

    ++unroutable;
    report(op, device, config, *qubits);
    return emitRemarks ? WalkResult::advance() : WalkResult::interrupt();
  }

  /// States which set is unreachable and where its members are, which is what a
  /// router consumes. Finding the sequence that brings them together is pebble
  /// motion with capacities and chain order, and is not attempted here.
  void report(Operation* op, DeviceAttr device, const Configuration& config, llvm::ArrayRef<int64_t> qubits) {
    std::string operands;
    llvm::raw_string_ostream os(operands);
    llvm::interleaveComma(qubits, os);

    auto diagnostic = emitRemarks ? op->emitRemark() : op->emitOpError();
    diagnostic << "needs routing: {" << operands << "} is not executable here";
    diagnostic.attachNote(op->getLoc()) << "currently " << describePlacement(config, qubits);
    if (device.isStatic()) {
      diagnostic.attachNote() << "this machine has no rewrite rules, so no shuttle can fix it; the router has to "
                                 "insert SWAP gates into the program instead";
    }
  }
};

} // namespace
} // namespace qcc
