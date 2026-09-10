// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/Devices/DeviceLibrary.h"

#include "qcc/Dialect/QCC/IR/QCC.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <cmath>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>
#include <optional>
#include <string>

using namespace mlir;

namespace qcc::conn {

namespace {

/// Euclidean distance between two sites, when both carry a position.
std::optional<double> distance(const SiteSnapshot& a, const SiteSnapshot& b) {
  if (!a.position || !b.position) {
    return std::nullopt;
  }
  double sum = 0.0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    const double delta = (*a.position)[axis] - (*b.position)[axis];
    sum += delta * delta;
  }
  return std::sqrt(sum);
}

/// Builds the `#qcc.site` attributes, applying any supplied overrides.
FailureOr<llvm::SmallVector<SiteAttr>> buildSites(MLIRContext* ctx, const DeviceSnapshot& snapshot,
                                                  const DeviceSupplement& supplement,
                                                  const std::function<InFlightDiagnostic()>& emitError) {
  llvm::StringSet<> seen;
  llvm::SmallVector<SiteAttr> sites;

  for (const auto& site : snapshot.sites) {
    if (!seen.insert(site.name).second) {
      return emitError() << "device '" << snapshot.name << "' reports site '" << site.name << "' twice";
    }
    const auto* override_ = supplement.find(site.name);

    // A device-management interface reports only zone or not-zone. Which kind
    // of zone a site is must come from the supplement.
    auto kind = site.isZone ? SiteKind::Trap : SiteKind::Fixed;
    if (override_ && override_->kind) {
      kind = *override_->kind;
    }

    uint64_t capacity = site.capacity;
    if (override_ && override_->capacity) {
      capacity = *override_->capacity;
    }
    if (kind == SiteKind::Fixed) {
      capacity = 1;
    }

    DenseF64ArrayAttr coords;
    if (site.position) {
      coords = DenseF64ArrayAttr::get(ctx, *site.position);
    }

    sites.push_back(SiteAttr::get(ctx, FlatSymbolRefAttr::get(ctx, site.name), kind, capacity,
                                  override_ ? override_->maxFacetRank : std::nullopt, std::nullopt, coords,
                                  override_ ? override_->props : DictionaryAttr{}));
  }
  return sites;
}

/// Builds the link edges implied by the snapshot: one per reported coupling,
/// plus one blockade disc per site when a blockade radius is given.
///
/// A reported coupling is a permanent interaction, since nothing on a machine
/// described this way moves the pair apart.
FailureOr<llvm::SmallVector<HyperedgeAttr>> buildDerivedEdges(MLIRContext* ctx, const DeviceSnapshot& snapshot,
                                                              const std::function<InFlightDiagnostic()>& emitError) {
  Builder builder(ctx);
  llvm::SmallVector<HyperedgeAttr> edges;

  const auto persistentLink = [&](const std::string& id, llvm::ArrayRef<unsigned> members) {
    llvm::SmallVector<FlatSymbolRefAttr> refs;
    for (const unsigned member : members) {
      refs.push_back(FlatSymbolRefAttr::get(ctx, snapshot.sites[member].name));
    }
    const NamedAttribute props[] = {
        builder.getNamedAttr("id", builder.getStringAttr(id)),
        builder.getNamedAttr("persistent", builder.getBoolAttr(true)),
    };
    return HyperedgeAttr::get(ctx, EdgeKind::Link, llvm::ArrayRef<FlatSymbolRefAttr>(refs),
                              builder.getDictionaryAttr(props));
  };

  for (const auto& [a, b] : snapshot.couplings) {
    if (a >= snapshot.sites.size() || b >= snapshot.sites.size()) {
      return emitError() << "device '" << snapshot.name << "' reports a coupling between sites " << a << " and " << b
                         << ", but only declares " << snapshot.sites.size() << " site(s)";
    }
    const unsigned members[] = {a, b};
    edges.push_back(persistentLink("coupler_" + snapshot.sites[a].name + "_" + snapshot.sites[b].name, members));
  }

  // A blockade disc follows from the radius and the trap positions rather than
  // being listed, so moving a trap moves its disc.
  if (snapshot.minAtomDistance) {
    for (const auto& [i, site] : llvm::enumerate(snapshot.sites)) {
      llvm::SmallVector<unsigned> disc{static_cast<unsigned>(i)};
      for (const auto& [j, other] : llvm::enumerate(snapshot.sites)) {
        const auto separation = i == j ? std::nullopt : distance(site, other);
        if (separation && *separation <= *snapshot.minAtomDistance) {
          disc.push_back(static_cast<unsigned>(j));
        }
      }
      if (disc.size() >= 2) {
        edges.push_back(persistentLink("blockade_" + site.name, disc));
      }
    }
  }

  return edges;
}

} // namespace

FailureOr<DeviceAttr> buildDevice(MLIRContext* ctx, const DeviceSnapshot& snapshot, const DeviceSupplement& supplement,
                                  const std::function<InFlightDiagnostic()>& emitError) {
  const auto sites = buildSites(ctx, snapshot, supplement, emitError);
  if (failed(sites)) {
    return failure();
  }

  auto edges = buildDerivedEdges(ctx, snapshot, emitError);
  if (failed(edges)) {
    return failure();
  }

  for (const auto edge : supplement.edges) {
    for (const auto member : edge.getSites()) {
      const bool declared =
          llvm::any_of(*sites, [&](const SiteAttr site) { return site.getSymName() == member.getValue(); });
      if (!declared) {
        return emitError() << "supplementary edge '" << edge.getId() << "' names site @" << member.getValue()
                           << ", which device '" << snapshot.name << "' does not report";
      }
    }
    edges->push_back(edge);
  }

  const auto substrate = SubstrateAttr::getChecked(emitError, ctx, llvm::ArrayRef<SiteAttr>(*sites),
                                                   llvm::ArrayRef<HyperedgeAttr>(*edges));
  if (!substrate) {
    return failure();
  }

  // The defaults describe a coupling-graph machine, which is all a snapshot on
  // its own supports. A machine with chains states its own.
  const auto device = DeviceAttr::getChecked(
      emitError, ctx, substrate, supplement.maxFacetRank.value_or(std::max<uint64_t>(snapshot.maxOperationArity, 2)),
      supplement.downwardClosed.value_or(true), supplement.partitioning.value_or(false),
      supplement.facetOrdering.value_or(FacetOrdering::None));
  if (!device) {
    return failure();
  }
  return device;
}

} // namespace qcc::conn
