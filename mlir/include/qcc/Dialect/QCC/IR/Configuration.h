// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <optional>
#include <string>

namespace qcc::conn {

/// The resource written by every configuration-rewriting operation.
///
/// Gates take no configuration operand, so without an effect a rewrite whose
/// result is unused would be dead code and the motion it describes would be
/// eliminated.
struct ConfigurationResource : public mlir::SideEffects::Resource::Base<ConfigurationResource> {
  llvm::StringRef getName() const final { return "qcc::Configuration"; }

  /// Machine state, not addressable memory, so it never aliases a pointer.
  bool isAddressable() const final { return false; }
};

/// How a configuration over a given device is represented.
///
/// The device's algebraic properties determine what a configuration can be
/// without loss of information, and hence how membership in `K` is decided.
/// The classification is derived from those properties rather than declared,
/// so a machine description states physics and the model follows from it.
enum class Representation {
  /// The rule set is empty, so no operation can produce a new configuration and
  /// every use resolves to the one `qcc.config.init` established. This is the
  /// superconducting case, and also any machine that declares no transport,
  /// rigid or consumable-link edge: with nothing to rewrite, the stronger
  /// statement applies whatever the other properties say.
  Static,
  /// The facets partition the qubits and each group is an ordered word, so the
  /// interaction complex collapses to an ordered partition with no loss.
  /// Membership is then a lookup rather than a scan. Trapped ions.
  OrderedPartition,
  /// Every site holds at most one qubit, so the placement is an injection and
  /// the facets are drawn from the substrate rather than from co-location.
  /// Neutral atoms.
  Injection,
  /// The general case: overlapping facets over multi-qubit sites, together with
  /// the live link multiset, which is consumed by use and is not a function of
  /// the placement. Modular and photonic machines.
  ExplicitIncidence,
};

/// The spelling of `representation`, for diagnostics and `--list-devices`.
[[nodiscard]] inline llvm::StringRef stringifyRepresentation(const Representation representation) {
  switch (representation) {
  case Representation::Static:
    return "static";
  case Representation::OrderedPartition:
    return "ordered-partition";
  case Representation::Injection:
    return "injection";
  case Representation::ExplicitIncidence:
    return "explicit-incidence";
  }
  return "unknown";
}

/// A group of co-located qubits, as a word over the qubit alphabet. Position 0
/// is the head. The order is physical state: on a linear target only the ends
/// of a group may be detached.
using Word = llvm::SmallVector<int64_t, 8>;

/// The groups a site holds, in spatial order. A site holds several because
/// splitting a group opens a second potential well without moving anything.
using SiteContents = llvm::SmallVector<Word, 2>;

/// One site and its contents.
struct SitePlacement {
  std::string name;
  SiteContents groups;
};

/// What a rewrite rule touches, and therefore what it conflicts over.
///
/// The two components answer two different questions and must not be merged.
/// `sites` carries the double-pushout condition: rules whose matches meet only
/// inside their interfaces are parallel-independent, so applying them in either
/// order yields the same configuration. `uses` carries the scheduling
/// condition: rules contending for one control resource cannot run in the same
/// time step however disjoint their sites are, but they still commute, because
/// a control resource is control-plane occupancy rather than a property of the
/// configuration.
struct Footprint {
  llvm::SmallVector<std::string> sites;
  llvm::SmallVector<std::string> uses;

  /// Whether the two rules may be reordered with respect to one another. This
  /// is the double-pushout question, and it reads `sites` alone.
  [[nodiscard]] bool commutesWith(const Footprint& other) const { return !shares(sites, other.sites); }

  /// Whether the two rules may occupy the same time step. Commuting is
  /// necessary and not sufficient: a shared control resource serialises rules
  /// that are otherwise independent.
  [[nodiscard]] bool concurrentWith(const Footprint& other) const {
    return commutesWith(other) && !shares(uses, other.uses);
  }

  /// Whether the two rules are held apart by a shared control resource alone,
  /// that is, whether one more of that resource would let them run together.
  [[nodiscard]] bool serialisedOnlyByControl(const Footprint& other) const {
    return commutesWith(other) && shares(uses, other.uses);
  }

private:
  static bool shares(llvm::ArrayRef<std::string> lhs, llvm::ArrayRef<std::string> rhs) {
    return llvm::any_of(lhs, [&](const std::string& name) { return llvm::is_contained(rhs, name); });
  }
};

/// The mutable form of a configuration, `C = (pi, <, Lambda)`, which the Layer 3
/// rules operate on. `PlacementAttr` is the uniqued counterpart of its
/// placement component.
///
/// `links` is what makes a configuration more than a placement. It stays empty
/// on ion and atom targets. On a photonic target it does not: a heralded Bell
/// pair is consumed by use, and no placement records whether it is live.
struct Configuration {
  /// Sites in the order the placement declared them. A placement need not
  /// mention every site of the substrate.
  llvm::SmallVector<SitePlacement> sites;

  /// The live entanglement links, each keeping the party order it was heralded
  /// with. A multiset: two links over the same parties are two resources.
  llvm::SmallVector<Word> links;

  /// Index of the site named `name`, if the configuration mentions it.
  [[nodiscard]] std::optional<unsigned> findSite(llvm::StringRef name) const {
    for (const auto& [index, site] : llvm::enumerate(sites)) {
      if (site.name == name) {
        return static_cast<unsigned>(index);
      }
    }
    return std::nullopt;
  }

  /// The groups held by site `name`, or nullptr if it is not mentioned.
  [[nodiscard]] const SiteContents* lookup(llvm::StringRef name) const {
    const auto index = findSite(name);
    return index ? &sites[*index].groups : nullptr;
  }
  SiteContents* lookup(llvm::StringRef name) {
    const auto index = findSite(name);
    return index ? &sites[*index].groups : nullptr;
  }

  /// Adds `name` if absent and returns its contents.
  SiteContents& lookupOrAdd(llvm::StringRef name) {
    if (auto* contents = lookup(name)) {
      return *contents;
    }
    sites.push_back({name.str(), {}});
    return sites.back().groups;
  }

  /// Number of qubits held by site `name`, summed over its groups.
  [[nodiscard]] uint64_t getLoad(llvm::StringRef name) const {
    const auto* contents = lookup(name);
    if (contents == nullptr) {
      return 0;
    }
    uint64_t load = 0;
    for (const auto& group : *contents) {
      load += group.size();
    }
    return load;
  }

  /// The site holding `qubit`, if it is placed.
  [[nodiscard]] std::optional<llvm::StringRef> findQubit(int64_t qubit) const {
    for (const auto& site : sites) {
      for (const auto& group : site.groups) {
        if (llvm::is_contained(group, qubit)) {
          return llvm::StringRef(site.name);
        }
      }
    }
    return std::nullopt;
  }

  /// Whether `qubit` is party to a live link.
  [[nodiscard]] bool isEntangled(int64_t qubit) const {
    return llvm::any_of(links, [&](const Word& link) { return llvm::is_contained(link, qubit); });
  }

  /// Position of a live link joining exactly `parties`, in any order. The
  /// record keeps a party order; matching one for consumption ignores it.
  [[nodiscard]] std::optional<unsigned> findLink(llvm::ArrayRef<int64_t> parties) const {
    const auto sorted = [](llvm::ArrayRef<int64_t> qubits) {
      Word copy(qubits.begin(), qubits.end());
      llvm::sort(copy);
      return copy;
    };
    const Word wanted = sorted(parties);
    for (const auto& [index, link] : llvm::enumerate(links)) {
      if (sorted(link) == wanted) {
        return static_cast<unsigned>(index);
      }
    }
    return std::nullopt;
  }
};

} // namespace qcc::conn
