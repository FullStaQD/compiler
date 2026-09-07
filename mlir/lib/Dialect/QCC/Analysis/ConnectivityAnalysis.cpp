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

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <algorithm>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;

namespace qcc::conn {

//===----------------------------------------------------------------------===//
// InteractionComplex
//===----------------------------------------------------------------------===//

namespace {

/// Sorted, deduplicated copy of `qubits`.
Word normalize(const llvm::ArrayRef<int64_t> qubits) {
  Word sorted(qubits.begin(), qubits.end());
  llvm::sort(sorted);
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  return sorted;
}

/// Both arguments must be sorted ascending.
bool isSubset(const llvm::ArrayRef<int64_t> subset, const llvm::ArrayRef<int64_t> superset) {
  return std::includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}

void printSet(llvm::raw_ostream& os, const llvm::ArrayRef<int64_t> qubits) {
  os << "{";
  llvm::interleaveComma(qubits, os);
  os << "}";
}

} // namespace

InteractionComplex::InteractionComplex(const DeviceAttr device, const Configuration& config) : device(device) {
  const auto add = [&](Word qubits, const uint64_t rank) {
    llvm::sort(qubits);
    qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
    if (qubits.size() < 2) {
      return;
    }
    // The same qubits may be a facet by more than one route; the widest rank
    // among them is the one that applies.
    for (auto& existing : facets) {
      if (existing.qubits == qubits) {
        existing.rank = std::max(existing.rank, rank);
        return;
      }
    }
    facets.push_back({std::move(qubits), rank});
  };

  const auto substrate = device.getSubstrate();
  const auto deviceRank = device.getMaxFacetRank();

  // Co-location is the first source of facets: the ions of one group share a
  // motional mode and can be addressed together, but only at a site the machine
  // can address. A storage trap on a QCCD device has no gate lasers, so its
  // groups share a mode and nothing more, and the ions must be shuttled to a
  // zone that can act on them.
  for (const auto& site : config.sites) {
    const auto declared = substrate.lookupSite(site.name);
    const auto rank = declared ? declared.getEffectiveFacetRank(deviceRank) : deviceRank;
    for (const auto& group : site.groups) {
      if (rank >= 2) {
        add(group, rank);
      } else if (group.size() >= 2) {
        ungated.emplace_back(site.name, group);
      }
    }
  }

  // Persistent link edges are the second source: a superconducting coupler or a
  // Rydberg blockade disc exists unconditionally, so it contributes a facet
  // drawn from whichever qubits currently occupy its sites.
  for (const auto edge : substrate.getEdgesOfKind(EdgeKind::Link)) {
    if (!edge.isPersistent()) {
      continue;
    }
    Word joined;
    for (const auto site : edge.getSites()) {
      if (const auto* contents = config.lookup(site.getValue())) {
        for (const auto& group : *contents) {
          llvm::append_range(joined, group);
        }
      }
    }
    add(std::move(joined), deviceRank);
  }

  // Live links are the third source, and the reason `K` is not a function of
  // placement alone. A non-persistent link edge contributes nothing on its own,
  // since it states only that a photon can be heralded there rather than that
  // entanglement exists. Only a link produced by `qcc.link.generate` is a facet,
  // and consuming it removes that facet.
  for (const auto& link : config.links) {
    add(link, deviceRank);
  }

  indexPartition();
}

void InteractionComplex::indexPartition() {
  if (!device.getPartitioning()) {
    return;
  }
  for (const auto& [index, facet] : llvm::enumerate(facets)) {
    for (const int64_t qubit : facet.qubits) {
      if (!facetOfQubit.try_emplace(qubit, static_cast<unsigned>(index)).second) {
        // Two facets share a qubit, so `partitioning` does not hold of the
        // facets actually derived here. Fall back to the search rather than
        // answer from an index that is not a function.
        facetOfQubit.clear();
        return;
      }
    }
  }
  partitioned = true;
}

bool InteractionComplex::isExecutable(const llvm::ArrayRef<int64_t> qubits) const {
  const Word wanted = normalize(qubits);
  if (wanted.size() < 2) {
    return true;
  }
  // The two paths answer the same question; only the cost differs. A partition
  // admits a lookup because each qubit lies in exactly one facet, so the first
  // operand names the only candidate. Overlapping facets admit no such index.
  return partitioned ? isExecutableInPartition(wanted) : isExecutableByScan(wanted);
}

bool InteractionComplex::isExecutableInPartition(const Word& wanted) const {
  const auto entry = facetOfQubit.find(wanted.front());
  if (entry == facetOfQubit.end()) {
    return false;
  }
  const Facet& facet = facets[entry->second];
  // The rank cap belongs to the facet rather than the device: a set is
  // executable only up to the width of whatever makes it a facet.
  if (wanted.size() > facet.rank) {
    return false;
  }
  for (const int64_t qubit : wanted) {
    const auto other = facetOfQubit.find(qubit);
    if (other == facetOfQubit.end() || other->second != entry->second) {
      return false;
    }
  }
  // Every operand is in this facet. Where the complex is downward closed that
  // settles it; where it is not, the facet must be acted on whole, and since
  // both words are sorted and deduplicated, equal sizes mean equal sets.
  return device.getDownwardClosed() || wanted.size() == facet.qubits.size();
}

bool InteractionComplex::isExecutableByScan(const Word& wanted) const {
  for (const auto& facet : facets) {
    if (wanted.size() > facet.rank) {
      continue;
    }
    // Where the complex is downward closed, any face of a facet is executable,
    // since individual addressing lets an MS pulse select a subset of a chain.
    // Where it is not, as on a machine offering only a global MS gate, the facet
    // must be acted on as a whole, so only an exact match counts.
    if (device.getDownwardClosed() ? isSubset(wanted, facet.qubits) : wanted == facet.qubits) {
      return true;
    }
  }
  return false;
}

std::string InteractionComplex::explain(const llvm::ArrayRef<int64_t> qubits) const {
  const Word wanted = normalize(qubits);

  std::string reason;
  llvm::raw_string_ostream os(reason);

  // Report co-location at an ungated site first: the qubits are together but
  // the site cannot act on them, and the remedy is to shuttle them elsewhere.
  for (const auto& [site, group] : ungated) {
    if (isSubset(wanted, group)) {
      os << "the qubits are co-located at @" << site
         << ", but that site cannot host an entangling gate; they have to be moved to one that can";
      return reason;
    }
  }

  if (wanted.size() > device.getMaxFacetRank()) {
    os << "it has " << wanted.size() << " qubits, above the device's maximum facet rank of "
       << device.getMaxFacetRank();
    return reason;
  }

  // A facet that contains the set but is too narrow for it is a different
  // diagnosis from no facet at all, and calls for a different remedy.
  for (const auto& facet : facets) {
    if (isSubset(wanted, facet.qubits) && wanted.size() > facet.rank) {
      os << "facet ";
      printSet(os, facet.qubits);
      os << " contains it, but that site hosts operations of at most " << facet.rank << " qubit(s)";
      return reason;
    }
  }

  if (!device.getDownwardClosed()) {
    for (const auto& facet : facets) {
      if (isSubset(wanted, facet.qubits)) {
        os << "the device is not downward closed, so facet ";
        printSet(os, facet.qubits);
        os << " must be acted on whole rather than in part";
        return reason;
      }
    }
  }

  os << "no facet of K contains it";
  if (facets.empty()) {
    os << " (K has no facet of rank 2 or more here)";
  } else {
    os << "; the facets here are ";
    llvm::interleaveComma(facets, os, [&](const Facet& facet) { printSet(os, facet.qubits); });
  }
  return reason;
}

//===----------------------------------------------------------------------===//
// ConfigurationAnalysis
//===----------------------------------------------------------------------===//

FailureOr<ConfigurationAnalysis::Result> ConfigurationAnalysis::get(const Value config) {
  if (const auto cached = cache.find(config); cached != cache.end()) {
    return cached->second;
  }
  if (invalid.contains(config)) {
    return failure();
  }

  const auto fail = [&] {
    invalid.insert(config);
    return failure();
  };
  const auto remember = [&](Result result) {
    cache.try_emplace(config, result);
    return result;
  };

  // A block argument is where the fold stops. Inside an `scf.for` the
  // configuration is loop-carried, and unless the body's net rewrite is the
  // identity, no single placement describes the loop body.
  if (llvm::isa<BlockArgument>(config)) {
    return remember(Result{std::nullopt, config});
  }

  Operation* definition = config.getDefiningOp();
  if (definition == nullptr) {
    return remember(Result{std::nullopt, config});
  }

  if (auto init = llvm::dyn_cast<ConfigInitOp>(definition)) {
    return remember(Result{init.getPlacement().toConfiguration(), Value{}});
  }

  auto rule = llvm::dyn_cast<TopologyRewriteOpInterface>(definition);
  if (!rule) {
    // The value was produced by an operation outside the rule set, such as an
    // `scf.if`, whose result cannot be folded.
    return remember(Result{std::nullopt, config});
  }

  if (!inFlight.insert(config).second) {
    definition->emitOpError() << "configuration definition is cyclic";
    return fail();
  }
  const llvm::scope_exit restore([&] { inFlight.erase(config); });

  const auto incoming = get(rule.getInputConfig());
  if (failed(incoming)) {
    return fail();
  }
  if (!incoming->isKnown()) {
    return remember(*incoming);
  }

  auto rewritten = rule.applyRule(*incoming->config);
  if (failed(rewritten)) {
    // `applyRule` has already reported why the rule does not apply here.
    return fail();
  }
  return remember(Result{std::move(*rewritten), Value{}});
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

std::optional<int64_t> getQubitIndex(const Value qubit) {
  Operation* definition = qubit.getDefiningOp();
  if (definition == nullptr) {
    return std::nullopt;
  }
  if (auto staticQubit = llvm::dyn_cast<mlir::qc::StaticOp>(definition)) {
    return static_cast<int64_t>(staticQubit.getIndex());
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// DominatingConfigMap
//===----------------------------------------------------------------------===//

namespace {

/// Whether `a` is available strictly before `b` is defined. `DominanceInfo`
/// answers value-versus-operation directly; a block argument has no defining
/// operation, so it is compared at the granularity of its block.
bool definitionDominates(DominanceInfo& dominance, const Value a, const Value b) {
  if (a == b) {
    return false;
  }
  if (Operation* device = b.getDefiningOp()) {
    return dominance.properlyDominates(a, device);
  }

  Block* targetBlock = llvm::cast<BlockArgument>(b).getOwner();
  Block* sourceBlock =
      a.getDefiningOp() != nullptr ? a.getDefiningOp()->getBlock() : llvm::cast<BlockArgument>(a).getOwner();
  return dominance.properlyDominates(sourceBlock, targetBlock);
}

} // namespace

DominatingConfigMap::DominatingConfigMap(Operation* function) : dominance(function) {
  function->walk([&](Operation* candidate) {
    for (const Value result : candidate->getResults()) {
      if (llvm::isa<ConfigType>(result.getType())) {
        candidates.push_back(result);
      }
    }
    // Loop-carried configurations enter the body as block arguments, which are
    // what dominates a gate inside the loop.
    for (Region& region : candidate->getRegions()) {
      for (Block& block : region) {
        for (const BlockArgument argument : block.getArguments()) {
          if (llvm::isa<ConfigType>(argument.getType())) {
            candidates.push_back(argument);
          }
        }
      }
    }
  });
}

DominatingConfigMap::Lookup DominatingConfigMap::at(Operation* op) {
  llvm::SmallVector<Value> dominating;
  for (const Value candidate : candidates) {
    if (dominance.properlyDominates(candidate, op)) {
      dominating.push_back(candidate);
    }
  }
  if (dominating.empty()) {
    return {};
  }

  // The connectivity in force is the latest dominating definition, that is, the
  // one every other dominating definition itself dominates.
  for (const Value candidate : dominating) {
    if (llvm::all_of(dominating, [&](const Value other) {
          return other == candidate || definitionDominates(dominance, other, candidate);
        })) {
      return {candidate, false};
    }
  }
  return {Value{}, true};
}

} // namespace qcc::conn
