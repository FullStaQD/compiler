// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Dialect/QCC/IR/QCC.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <optional>
#include <string>
#include <utility>

namespace qcc::conn {

//===----------------------------------------------------------------------===//
// Read-only derivations from the IR, in the sense of MLIR's own `lib/Analysis`.
// Only `DominatingConfigMap` is registered with the pass manager; the other two
// are constructed directly by the passes that need them.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Layer 2 — the executable interaction complex
//===----------------------------------------------------------------------===//

/// `K(P, C)`: the qubit subsets that are actionable right now.
///
/// Derived on demand and never stored in the IR. The configuration supplies
/// co-location and the live links; the device supplies the `link` edges and the
/// algebraic properties. `control` edges do not contribute.
///
/// `K` need not be downward closed, and its facets need not partition the
/// qubits.
class InteractionComplex {
public:
  InteractionComplex(DeviceAttr device, const Configuration& config);

  /// The representation this complex was built under.
  [[nodiscard]] Representation getRepresentation() const { return device.getRepresentation(); }

  /// Whether an entangling operation may act on `qubits` right now. The
  /// argument need not be sorted or deduplicated.
  ///
  /// Single-qubit gates are not constrained by `K`; check placement for those.
  [[nodiscard]] bool isExecutable(llvm::ArrayRef<int64_t> qubits) const;

  /// Why `qubits` is not executable. Undefined if `isExecutable(qubits)` holds.
  [[nodiscard]] std::string explain(llvm::ArrayRef<int64_t> qubits) const;

  /// A facet together with the widest operation that may act on it. The rank
  /// is per-facet because addressability is a per-site property.
  struct Facet {
    Word qubits;
    uint64_t rank;
  };

  /// The maximal executable sets. On a partitioning device these are the groups
  /// held by sites that can host a gate.
  [[nodiscard]] llvm::ArrayRef<Facet> getFacets() const { return facets; }

private:
  /// Index the facets by qubit, when the device declares `partitioning` and the
  /// derived facets bear that out. Called once the facets are complete.
  void indexPartition();

  /// Membership by lookup, independent of the number of facets. Available only
  /// under a verified partition.
  [[nodiscard]] bool isExecutableInPartition(const Word& wanted) const;

  /// Membership by search over the facets, for a complex whose facets overlap.
  [[nodiscard]] bool isExecutableByScan(const Word& wanted) const;

  DeviceAttr device;
  /// Each facet is sorted ascending; the list holds no duplicates.
  llvm::SmallVector<Facet> facets;
  /// Qubit to the one facet containing it. Empty unless `partitioned`.
  llvm::DenseMap<int64_t, unsigned> facetOfQubit;
  /// Whether the facets were found to partition the qubits they cover.
  bool partitioned = false;
  /// Groups held by a site that cannot host an entangling gate. Not facets;
  /// kept so `explain` can name that case rather than report no facet at all.
  llvm::SmallVector<std::pair<std::string, Word>> ungated;
};

//===----------------------------------------------------------------------===//
// Layer 1 — recovering the configuration at a program point
//===----------------------------------------------------------------------===//

/// Recovers the configuration behind a `!qcc.config` value by folding its
/// definition chain: a `qcc.config.init` names one outright, and every rewrite
/// operation applies its rule to its operand's.
///
/// Being a fold rather than a fixpoint, it stops at a block argument, such as a
/// configuration carried by an `scf.for`. Those are reported as dynamic and the
/// decision is left to the caller.
class ConfigurationAnalysis {
public:
  struct Result {
    /// The configuration, when it is statically known.
    std::optional<Configuration> config;
    /// The block argument that stopped the fold, when it is not.
    mlir::Value dynamicOrigin;

    [[nodiscard]] bool isKnown() const { return config.has_value(); }
  };

  /// Shaped as an MLIR analysis but not obtained through `getAnalysis<>()`: it
  /// remembers an ill-formed chain after `applyRule` has emitted the diagnostic
  /// once, so a second pass handed a cached instance would see the failure with
  /// no diagnostic attached. Registering it requires reworking that first.
  explicit ConfigurationAnalysis(mlir::Operation* /*root*/) {}

  /// Fold the chain behind `config`, which must have `ConfigType`.
  ///
  /// Fails only when a rule in the chain does not apply to the configuration
  /// reaching it, having already emitted a diagnostic on that operation. A
  /// well-formed but unknown chain succeeds with an empty `Result::config`.
  mlir::FailureOr<Result> get(mlir::Value config);

private:
  llvm::DenseMap<mlir::Value, Result> cache;
  llvm::DenseSet<mlir::Value> invalid;
  /// Guards against a malformed chain that reaches itself.
  llvm::DenseSet<mlir::Value> inFlight;
};

//===----------------------------------------------------------------------===//
// Helpers shared with the verification pass
//===----------------------------------------------------------------------===//

/// The program-level index of a `!qc.qubit` reference, taken from `qc.static`.
///
/// A `qc.alloc` qubit and one arriving as an argument both yield `nullopt`,
/// which callers report as unresolvable rather than as an illegal gate.
std::optional<int64_t> getQubitIndex(mlir::Value qubit);

/// Resolves the connectivity in force at a program point to the dominating
/// `!qcc.config` definition, read off the def-use graph.
///
/// Registered: obtain it with `getAnalysis<DominatingConfigMap>()`. It emits no
/// diagnostics and remembers no failures, so caching it across a pipeline is
/// sound. It owns its `DominanceInfo` rather than borrowing one, which would
/// dangle whenever a pass preserved this analysis but not that one.
///
/// A pass that adds, removes or moves a `!qcc.config` definition must let this
/// be invalidated, which is the default absent `markAllAnalysesPreserved`.
class DominatingConfigMap {
public:
  explicit DominatingConfigMap(mlir::Operation* function);

  struct Lookup {
    /// The latest definition dominating the queried operation, if any.
    mlir::Value config;
    /// Set when two definitions dominate the operation and neither dominates
    /// the other, so no single configuration is in force.
    bool ambiguous = false;
  };

  [[nodiscard]] Lookup at(mlir::Operation* op);

  /// Whether the function mentions any configuration. One that does not has not
  /// opted into the dialect and is left unchanged.
  [[nodiscard]] bool empty() const { return candidates.empty(); }

private:
  llvm::SmallVector<mlir::Value> candidates;
  mlir::DominanceInfo dominance;
};

} // namespace qcc::conn
