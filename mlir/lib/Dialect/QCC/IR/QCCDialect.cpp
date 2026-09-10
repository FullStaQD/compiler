// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCC.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h> // IWYU pragma: keep
#include <optional>
#include <string>

using namespace mlir;
using namespace qcc::conn;

#include "qcc/Dialect/QCC/IR/QCCDialect.cpp.inc"
#include "qcc/Dialect/QCC/IR/QCCEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "qcc/Dialect/QCC/IR/QCCAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qcc/Dialect/QCC/IR/QCCTypes.cpp.inc"

void QCCDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "qcc/Dialect/QCC/IR/QCCAttributes.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "qcc/Dialect/QCC/IR/QCCTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "qcc/Dialect/QCC/IR/QCCOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SiteAttr
//===----------------------------------------------------------------------===//

bool SiteAttr::permitsTurn(const llvm::StringRef from, const llvm::StringRef to) const {
  const auto props = getProps();
  if (!props) {
    return true;
  }
  const auto turns = props.getAs<ArrayAttr>("turns");
  // No table means no restriction: a junction that declares no turns permits
  // every rotation.
  if (!turns) {
    return true;
  }
  const std::string wanted = (from + ">" + to).str();
  return llvm::any_of(turns, [&](const Attribute entry) {
    const auto name = llvm::dyn_cast<StringAttr>(entry);
    return name && name.getValue() == wanted;
  });
}

double SiteAttr::getTurnLatencyUs() const {
  const auto props = getProps();
  if (!props) {
    return 0.0;
  }
  if (const auto latency = props.getAs<FloatAttr>("turn_latency_us")) {
    return latency.getValueAsDouble();
  }
  if (const auto latency = props.getAs<IntegerAttr>("turn_latency_us")) {
    return static_cast<double>(latency.getInt());
  }
  return 0.0;
}

std::optional<std::array<double, 3>> SiteAttr::getPosition() const {
  const auto coords = getCoords();
  if (!coords || coords.size() < 2) {
    return std::nullopt;
  }
  return std::array<double, 3>{coords[0], coords[1], coords.size() > 2 ? coords[2] : 0.0};
}

std::optional<double> SiteAttr::distanceTo(const SiteAttr other) const {
  const auto here = getPosition();
  const auto there = other ? other.getPosition() : std::nullopt;
  if (!here || !there) {
    return std::nullopt;
  }
  const double dx = (*here)[0] - (*there)[0];
  const double dy = (*here)[1] - (*there)[1];
  const double dz = (*here)[2] - (*there)[2];
  return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

LogicalResult SiteAttr::verify(const llvm::function_ref<InFlightDiagnostic()> emitError, const FlatSymbolRefAttr name,
                               const SiteKind kind, const uint64_t capacity, const std::optional<uint64_t> maxFacetRank,
                               const std::optional<FacetOrdering> /*facetOrdering*/, const DenseF64ArrayAttr coords,
                               const DictionaryAttr props) {
  if (!name || name.getValue().empty()) {
    return emitError() << "site must have a non-empty name";
  }
  if (capacity == 0) {
    return emitError() << "site @" << name.getValue() << " must have a capacity of at least 1";
  }
  // A fixed site models a single qubit, such as a superconducting transmon or
  // an SLM trap, rather than a container; nothing can enter or leave it.
  if (kind == SiteKind::Fixed && capacity != 1) {
    return emitError() << "fixed site @" << name.getValue() << " must have capacity 1, but has " << capacity;
  }
  // A site cannot host an operation on more qubits than it can hold.
  if (maxFacetRank && *maxFacetRank > capacity) {
    return emitError() << "site @" << name.getValue() << " declares maxFacetRank = " << *maxFacetRank
                       << " but holds at most " << capacity << " qubit(s)";
  }
  // A position has two or three components. Accepting any other length and
  // reading the first two would mask a typo in the description.
  if (coords && (coords.size() < 2 || coords.size() > 3)) {
    return emitError() << "site @" << name.getValue() << " declares " << coords.size()
                       << " coordinate(s); a position is 2 or 3 numbers";
  }
  // A malformed turn entry would silently forbid every turn through the
  // junction, which is harder to diagnose than a rejected description.
  if (props) {
    if (const auto turns = props.getAs<ArrayAttr>("turns")) {
      for (const auto entry : turns) {
        const auto name_ = llvm::dyn_cast<StringAttr>(entry);
        if (!name_ || !name_.getValue().contains('>')) {
          return emitError() << "site @" << name.getValue()
                             << " has a malformed `turns` entry; each is \"<incoming-id>><outgoing-id>\"";
        }
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// HyperedgeAttr
//===----------------------------------------------------------------------===//

llvm::StringRef HyperedgeAttr::getId() const {
  if (const auto props = getProps()) {
    if (const auto id = props.getAs<StringAttr>("id")) {
      return id.getValue();
    }
  }
  return {};
}

std::optional<ChainEnd> HyperedgeAttr::getEndAt(const llvm::StringRef site) const {
  const auto props = getProps();
  if (!props) {
    return std::nullopt;
  }
  const auto ends = props.getAs<ArrayAttr>("ends");
  if (!ends) {
    return std::nullopt;
  }
  for (const auto& [index, member] : llvm::enumerate(getSites())) {
    if (member.getValue() != site) {
      continue;
    }
    if (index >= ends.size()) {
      return std::nullopt;
    }
    const auto name = llvm::dyn_cast<StringAttr>(ends[index]);
    return name ? symbolizeChainEnd(name.getValue()) : std::nullopt;
  }
  return std::nullopt;
}

bool HyperedgeAttr::isPersistent() const {
  const auto props = getProps();
  if (!props) {
    return false;
  }
  const auto flag = props.getAs<BoolAttr>("persistent");
  return flag && flag.getValue();
}

double HyperedgeAttr::getLatencyUs() const {
  if (const auto props = getProps()) {
    if (const auto latency = props.getAs<FloatAttr>("latency_us")) {
      return latency.getValueAsDouble();
    }
    if (const auto latency = props.getAs<IntegerAttr>("latency_us")) {
      return static_cast<double>(latency.getInt());
    }
  }
  return 0.0;
}

bool HyperedgeAttr::hasFlag(const llvm::StringRef key) const {
  const auto props = getProps();
  if (!props) {
    return false;
  }
  const auto flag = props.getAs<BoolAttr>(key);
  return flag && flag.getValue();
}

std::optional<int64_t> HyperedgeAttr::getIntProp(const llvm::StringRef key) const {
  if (const auto props = getProps()) {
    if (const auto value = props.getAs<IntegerAttr>(key)) {
      return value.getInt();
    }
  }
  return std::nullopt;
}

bool HyperedgeAttr::contains(const llvm::StringRef site) const {
  return llvm::any_of(getSites(), [&](const FlatSymbolRefAttr name) { return name.getValue() == site; });
}

LogicalResult HyperedgeAttr::verify(const llvm::function_ref<InFlightDiagnostic()> emitError, const EdgeKind kind,
                                    const llvm::ArrayRef<FlatSymbolRefAttr> sites, const DictionaryAttr props) {
  if (sites.empty()) {
    return emitError() << "hyperedge must be incident to at least one site";
  }

  llvm::StringSet<> seen;
  for (const auto site : sites) {
    if (!site || site.getValue().empty()) {
      return emitError() << "hyperedge site names must be non-empty";
    }
    if (!seen.insert(site.getValue()).second) {
      return emitError() << "hyperedge lists site @" << site.getValue() << " twice";
    }
  }

  // Displacement is a binary relation: a shuttle moves a group from one site to
  // another. There is no physical operation moving an ion out of {A, B, C}, and
  // a junction is a site of high degree rather than an edge of high arity.
  if (kind == EdgeKind::Transport && sites.size() != 2) {
    return emitError() << "transport edge must have rank 2, but is incident to " << sites.size() << " sites";
  }

  // An entanglement resource joins at least two parties; a Bell pair is rank 2
  // and a GHZ resource is rank >= 3.
  if (kind == EdgeKind::Link && sites.size() < 2) {
    return emitError() << "link edge must have rank at least 2, but is incident to " << sites.size() << " site";
  }

  // A rule addresses an edge by name, so every kind a rule can act on requires
  // one. Control edges are exempt, since no rule fires on them and only the
  // scheduler reads them.
  if (kind != EdgeKind::Control) {
    const auto id = props ? props.getAs<StringAttr>("id") : StringAttr{};
    if (!id || id.getValue().empty()) {
      return emitError() << stringifyEdgeKind(kind)
                         << " edge must carry a non-empty string `id` property, which is how a rewrite rule "
                            "addresses it";
    }
  }

  // A segment joins a trap at one of its ends, and which end determines what
  // can leave. Declaring the ends is optional, but a length mismatch would
  // silently associate the wrong end with the wrong trap.
  if (const auto ends = props ? props.getAs<ArrayAttr>("ends") : ArrayAttr{}) {
    if (kind != EdgeKind::Transport) {
      return emitError() << "`ends` is only meaningful on a transport edge: it names which end of each trap the "
                            "segment meets";
    }
    if (ends.size() != sites.size()) {
      return emitError() << "`ends` has " << ends.size() << " entr" << (ends.size() == 1 ? "y" : "ies")
                         << " but the edge is incident to " << sites.size() << " site(s); give one end per endpoint";
    }
    for (const auto entry : ends) {
      const auto name = llvm::dyn_cast<StringAttr>(entry);
      if (!name || !symbolizeChainEnd(name.getValue())) {
        return emitError() << "`ends` entries must be \"head\" or \"tail\"";
      }
    }
  }

  // Whether a link is consumed by use determines whether the edge contributes a
  // facet to K on its own or only once `qcc.link.generate` has heralded one over
  // it. A permanent coupler and a heralded Bell pair are both link edges with
  // different semantics, so the description must state which applies rather than
  // rely on a default.
  if (kind == EdgeKind::Link && (!props || !props.getAs<BoolAttr>("persistent"))) {
    return emitError() << "link edge must declare `persistent`: true for a coupler that is never consumed, false "
                          "for a resource spent by use";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SubstrateAttr
//===----------------------------------------------------------------------===//

std::optional<unsigned> SubstrateAttr::findSite(const llvm::StringRef name) const {
  for (const auto& [index, site] : llvm::enumerate(getSites())) {
    if (site.getSymName() == name) {
      return static_cast<unsigned>(index);
    }
  }
  return std::nullopt;
}

SiteAttr SubstrateAttr::lookupSite(const llvm::StringRef name) const {
  const auto index = findSite(name);
  return index ? getSites()[*index] : SiteAttr{};
}

HyperedgeAttr SubstrateAttr::lookupEdge(const EdgeKind kind, const llvm::StringRef id) const {
  for (const auto edge : getEdges()) {
    if (edge.getKind() == kind && edge.getId() == id) {
      return edge;
    }
  }
  return {};
}

llvm::SmallVector<HyperedgeAttr> SubstrateAttr::getEdgesOfKind(const EdgeKind kind) const {
  llvm::SmallVector<HyperedgeAttr> result;
  for (const auto edge : getEdges()) {
    if (edge.getKind() == kind) {
      result.push_back(edge);
    }
  }
  return result;
}

Attribute SubstrateAttr::parse(AsmParser& parser, Type /*type*/) {
  llvm::SmallVector<SiteAttr> sites;
  llvm::SmallVector<HyperedgeAttr> edges;

  const auto parseSite = [&]() -> ParseResult {
    SiteAttr site;
    if (parser.parseAttribute(site)) {
      return failure();
    }
    sites.push_back(site);
    return success();
  };
  const auto parseEdge = [&]() -> ParseResult {
    HyperedgeAttr edge;
    if (parser.parseAttribute(edge)) {
      return failure();
    }
    edges.push_back(edge);
    return success();
  };

  if (parser.parseLess() || parser.parseKeyword("sites") || parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseSite) || parser.parseComma() ||
      parser.parseKeyword("edges") || parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseEdge) || parser.parseGreater()) {
    return {};
  }

  return parser.getChecked<SubstrateAttr>(parser.getContext(), sites, edges);
}

void SubstrateAttr::print(AsmPrinter& printer) const {
  printer << "<sites = [";
  llvm::interleaveComma(getSites(), printer);
  printer << "], edges = [";
  llvm::interleaveComma(getEdges(), printer);
  printer << "]>";
}

LogicalResult SubstrateAttr::verify(const llvm::function_ref<InFlightDiagnostic()> emitError,
                                    const llvm::ArrayRef<SiteAttr> sites, const llvm::ArrayRef<HyperedgeAttr> edges) {
  llvm::StringSet<> declared;
  for (const auto site : sites) {
    if (!declared.insert(site.getSymName()).second) {
      return emitError() << "substrate declares site @" << site.getSymName() << " twice";
    }
  }

  llvm::StringSet<> edgeIds;
  for (const auto edge : edges) {
    for (const auto name : edge.getSites()) {
      if (!declared.contains(name.getValue())) {
        return emitError() << "edge refers to undeclared site @" << name.getValue();
      }
    }

    // Ids are unique across the whole substrate rather than per kind, so a
    // rule's `via` names exactly one edge irrespective of its kind.
    if (const auto id = edge.getId(); !id.empty() && !edgeIds.insert(id).second) {
      return emitError() << "substrate declares two edges with id '" << id << "'";
    }

    if (edge.getKind() != EdgeKind::Transport) {
      continue;
    }
    // Nothing enters or leaves a fixed site. Permitting an incident transport
    // edge would let a superconducting substrate express shuttling.
    for (const auto name : edge.getSites()) {
      const auto* site = llvm::find_if(sites, [&](const SiteAttr s) { return s.getSymName() == name.getValue(); });
      if (site != sites.end() && site->getKind() == SiteKind::Fixed) {
        return emitError() << "transport edge '" << edge.getId() << "' touches fixed site @" << name.getValue()
                           << "; a fixed site cannot be entered or left";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DeviceAttr
//===----------------------------------------------------------------------===//

bool DeviceAttr::isStatic() const {
  const auto substrate = getSubstrate();
  if (!substrate.getEdgesOfKind(EdgeKind::Transport).empty() || !substrate.getEdgesOfKind(EdgeKind::Rigid).empty()) {
    return false;
  }
  // A consumable link makes the device dynamic even though nothing moves, since
  // heralding and consuming one both rewrite the configuration.
  return llvm::all_of(substrate.getEdgesOfKind(EdgeKind::Link),
                      [](const HyperedgeAttr edge) { return edge.isPersistent(); });
}

Representation DeviceAttr::getRepresentation() const {
  // An empty rule set leaves nothing to represent along the chain: every use of
  // a configuration resolves to the one `qcc.config.init` established.
  if (isStatic()) {
    return Representation::Static;
  }
  // Partitioning facets over ordered words are exactly an ordered partition.
  if (getPartitioning() && getFacetOrdering() == FacetOrdering::Linear) {
    return Representation::OrderedPartition;
  }
  // One qubit per site makes the placement an injection, and the facets then
  // come from the substrate's link edges rather than from co-location.
  const auto sites = getSubstrate().getSites();
  if (!getPartitioning() && llvm::all_of(sites, [](const SiteAttr site) { return site.getCapacity() <= 1; })) {
    return Representation::Injection;
  }
  return Representation::ExplicitIncidence;
}

LogicalResult DeviceAttr::verify(const llvm::function_ref<InFlightDiagnostic()> emitError,
                                 const SubstrateAttr substrate, const uint64_t maxFacetRank, bool /*downwardClosed*/,
                                 const bool partitioning, const FacetOrdering facetOrdering) {
  if (!substrate) {
    return emitError() << "device must carry a substrate";
  }
  if (maxFacetRank < 2) {
    return emitError() << "device maxFacetRank must be at least 2, but is " << maxFacetRank
                       << "; a device that cannot entangle two qubits has nothing to route";
  }
  // `partitioning` asserts that the facets of K form a partition of the qubits,
  // which is what allows the hypergraph to collapse to an ordered partition
  // without loss of information. A link edge contradicts that by construction:
  // it forms a facet from qubits drawn from several sites, and that facet
  // overlaps every group it drew from. Superconducting couplers, Rydberg
  // blockade discs and photonic interconnects are all link edges, which is why
  // those devices are non-partitioning.
  //
  // `facetOrdering = linear` together with `partitioning = false` is not
  // checked, because it is not an error: it describes a modular ion machine
  // whose chains are ordered words and whose photonic links lay facets across
  // them.
  if (const auto links = substrate.getEdgesOfKind(EdgeKind::Link); partitioning && !links.empty()) {
    return emitError() << "partitioning = true is inconsistent with " << links.size()
                       << " link edge(s): a link forms a facet from qubits at several sites, which overlaps the "
                          "groups it draws from, so the facets are not a partition";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PlacementAttr
//===----------------------------------------------------------------------===//

Attribute PlacementAttr::parse(AsmParser& parser, Type /*type*/) {
  if (parser.parseLess()) {
    return {};
  }

  llvm::SmallVector<FlatSymbolRefAttr> sites;
  llvm::SmallVector<ArrayAttr> groups;

  if (succeeded(parser.parseOptionalGreater())) {
    return parser.getChecked<PlacementAttr>(parser.getContext(), sites, groups);
  }

  const auto parseSite = [&]() -> ParseResult {
    FlatSymbolRefAttr name;
    if (parser.parseAttribute(name) || parser.parseEqual()) {
      return failure();
    }

    llvm::SmallVector<Attribute> words;
    const auto parseWord = [&]() -> ParseResult {
      llvm::SmallVector<int64_t> word;
      const auto parseQubit = [&]() -> ParseResult {
        int64_t qubit = 0;
        if (parser.parseInteger(qubit)) {
          return failure();
        }
        word.push_back(qubit);
        return success();
      };
      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseQubit)) {
        return failure();
      }
      words.push_back(DenseI64ArrayAttr::get(parser.getContext(), word));
      return success();
    };

    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseWord)) {
      return failure();
    }

    sites.push_back(name);
    groups.push_back(ArrayAttr::get(parser.getContext(), words));
    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseSite) || parser.parseGreater()) {
    return {};
  }

  return parser.getChecked<PlacementAttr>(parser.getContext(), sites, groups);
}

void PlacementAttr::print(AsmPrinter& printer) const {
  printer << "<";
  llvm::interleaveComma(llvm::zip_equal(getSites(), getGroups()), printer, [&](const auto& entry) {
    const auto& [site, words] = entry;
    printer << site << " = [";
    llvm::interleaveComma(words, printer, [&](const Attribute word) {
      printer << "[";
      llvm::interleaveComma(llvm::cast<DenseI64ArrayAttr>(word).asArrayRef(), printer,
                            [&](const int64_t qubit) { printer << qubit; });
      printer << "]";
    });
    printer << "]";
  });
  printer << ">";
}

LogicalResult PlacementAttr::verify(const llvm::function_ref<InFlightDiagnostic()> emitError,
                                    const llvm::ArrayRef<FlatSymbolRefAttr> sites,
                                    const llvm::ArrayRef<ArrayAttr> groups) {
  if (sites.size() != groups.size()) {
    return emitError() << "placement has " << sites.size() << " sites but " << groups.size() << " group lists";
  }

  llvm::StringSet<> seenSites;
  llvm::DenseSet<int64_t> seenQubits;
  for (const auto& [site, words] : llvm::zip_equal(sites, groups)) {
    if (!site || site.getValue().empty()) {
      return emitError() << "placement site names must be non-empty";
    }
    if (!seenSites.insert(site.getValue()).second) {
      return emitError() << "placement mentions site @" << site.getValue() << " twice";
    }

    for (const auto word : words) {
      const auto group = llvm::dyn_cast<DenseI64ArrayAttr>(word);
      if (!group) {
        return emitError() << "each group must be an array of qubit indices";
      }
      if (group.empty()) {
        return emitError() << "site @" << site.getValue()
                           << " holds an empty group; an empty potential well is not a group, so drop it";
      }
      for (const int64_t qubit : group.asArrayRef()) {
        if (qubit < 0) {
          return emitError() << "qubit index must be non-negative, but got " << qubit;
        }
        // A placement is a partial injection: a qubit is in at most one place.
        if (!seenQubits.insert(qubit).second) {
          return emitError() << "qubit " << qubit << " is placed more than once";
        }
      }
    }
  }

  return success();
}

Configuration PlacementAttr::toConfiguration() const {
  Configuration config;
  for (const auto& [site, words] : llvm::zip_equal(getSites(), getGroups())) {
    SiteContents groups;
    for (const auto word : words) {
      const auto group = llvm::cast<DenseI64ArrayAttr>(word).asArrayRef();
      groups.emplace_back(group.begin(), group.end());
    }
    config.sites.push_back({site.getValue().str(), std::move(groups)});
  }
  return config;
}

PlacementAttr PlacementAttr::get(MLIRContext* ctx, const Configuration& config) {
  llvm::SmallVector<FlatSymbolRefAttr> sites;
  llvm::SmallVector<ArrayAttr> groups;
  for (const auto& site : config.sites) {
    sites.push_back(FlatSymbolRefAttr::get(ctx, site.name));
    llvm::SmallVector<Attribute> words;
    words.reserve(site.groups.size());
    for (const auto& word : site.groups) {
      words.push_back(DenseI64ArrayAttr::get(ctx, word));
    }
    groups.push_back(ArrayAttr::get(ctx, words));
  }
  return PlacementAttr::get(ctx, sites, groups);
}
