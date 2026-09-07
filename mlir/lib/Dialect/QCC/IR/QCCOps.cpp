// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/IR/QCC.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <utility>

using namespace mlir;
using namespace qcc::conn;

#include "qcc/Dialect/QCC/IR/QCCInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "qcc/Dialect/QCC/IR/QCCOps.cpp.inc"

namespace qcc::conn {

namespace {

/// The substrate `op`'s configuration operand is typed over.
SubstrateAttr getSubstrateOf(Operation* op) {
  return llvm::cast<ConfigType>(op->getResult(0).getType()).getDevice().getSubstrate();
}

/// Static check shared by every rule that names a site: the substrate must
/// declare it. Checking here catches typos at verification time rather than
/// leaving them to the analysis, which only observes the sites a placement
/// happens to mention.
LogicalResult verifyDeclaredSite(Operation* op, const FlatSymbolRefAttr site, const llvm::StringRef what) {
  if (!getSubstrateOf(op).lookupSite(site.getValue())) {
    return op->emitOpError() << "substrate declares no " << what << " site @" << site.getValue();
  }
  return success();
}

/// Static check shared by the chain rules. All of them rewrite a chain as an
/// ordered word: `split` detaches at an end, `merge` concatenates and `reorder`
/// permutes. The property that admits them is therefore `facetOrdering =
/// linear`.
///
/// This is not a check on `partitioning`. A modular ion machine has ordered
/// chains and non-partitioning K simultaneously, since its chains are words and
/// its photonic links lay facets across them; gating these rules on
/// `partitioning` would withdraw the whole rule set from any machine with an
/// optical interconnect.
LogicalResult verifyChainSite(Operation* op, const DeviceAttr device, const llvm::StringRef site,
                              const llvm::StringRef opName) {
  const auto declared = device.getSubstrate().lookupSite(site);
  const auto ordering =
      declared ? declared.getEffectiveFacetOrdering(device.getFacetOrdering()) : device.getFacetOrdering();
  if (ordering != FacetOrdering::Linear) {
    return op->emitOpError() << opName << " requires facetOrdering = linear, but @" << site << " is "
                             << stringifyFacetOrdering(ordering)
                             << ": this rule rewrites a chain as an ordered word, and a facet carrying no such "
                                "order has no ends to detach at and no order to permute";
  }
  return success();
}

/// Locate a chain in a configuration, reporting where it went missing.
FailureOr<std::pair<SiteContents*, unsigned>> findChain(Operation* op, Configuration& config,
                                                        const llvm::StringRef site, const uint64_t chain) {
  auto* contents = config.lookup(site);
  if (contents == nullptr) {
    return op->emitOpError() << "the configuration reaching this operation places nothing at site @" << site;
  }
  if (chain >= contents->size()) {
    return op->emitOpError() << "site @" << site << " holds " << contents->size() << " chain(s), so there is no "
                             << "chain " << chain << " to act on here";
  }
  return std::pair{contents, static_cast<unsigned>(chain)};
}

} // namespace

} // namespace qcc::conn

//===----------------------------------------------------------------------===//
// ConfigInitOp
//===----------------------------------------------------------------------===//

LogicalResult ConfigInitOp::verify() {
  const auto device = getDeviceAttr();
  const auto substrate = device.getSubstrate();
  const auto placement = getPlacement();
  const auto config = placement.toConfiguration();

  for (const auto& placed : config.sites) {
    const auto site = substrate.lookupSite(placed.name);
    if (!site) {
      return emitOpError() << "places qubits at @" << placed.name << ", which the substrate does not declare";
    }

    const auto load = config.getLoad(placed.name);
    if (load > site.getCapacity()) {
      return emitOpError() << "places " << load << " qubits at @" << placed.name << ", exceeding its capacity of "
                           << site.getCapacity();
    }

    // Implied by capacity 1, but reported separately because holding two groups
    // is a more precise diagnosis than exceeding capacity.
    if (site.getKind() == SiteKind::Fixed && placed.groups.size() > 1) {
      return emitOpError() << "fixed site @" << placed.name << " holds " << placed.groups.size()
                           << " groups; a fixed site holds exactly one qubit";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// IonSplitOp
//===----------------------------------------------------------------------===//

LogicalResult IonSplitOp::verify() {
  // The `end` attribute exists because only the ends of a chain can be
  // detached. An unordered device has no chain ends, so the operation is
  // undefined there rather than unrestricted.
  if (failed(verifyChainSite(*this, getDeviceAttr(), getSite(), "ion.split"))) {
    return failure();
  }
  return verifyDeclaredSite(*this, getSiteAttr(), "trap or junction");
}

FailureOr<Configuration> IonSplitOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto located = findChain(*this, out, getSite(), getChain());
  if (failed(located)) {
    return failure();
  }
  auto* contents = located->first;
  const auto index = located->second;

  const Word word = (*contents)[index];
  const auto count = static_cast<size_t>(getCount());
  if (count >= word.size()) {
    return emitOpError() << "cannot detach " << count << " ion(s) from a chain of length " << word.size()
                         << "; a split must leave a non-empty remainder on both sides";
  }

  const auto atHead = getEnd() == ChainEnd::Head;
  const Word detached = atHead ? Word(word.begin(), word.begin() + count) : Word(word.end() - count, word.end());
  const Word remainder = atHead ? Word(word.begin() + count, word.end()) : Word(word.begin(), word.end() - count);

  // The detached part remains on the side it came from: a split opens a second
  // potential well next to the first rather than rearranging the segment.
  (*contents)[index] = atHead ? detached : remainder;
  contents->insert(contents->begin() + index + 1, atHead ? remainder : detached);

  return out;
}

//===----------------------------------------------------------------------===//
// IonMergeOp
//===----------------------------------------------------------------------===//

LogicalResult IonMergeOp::verify() {
  if (failed(verifyChainSite(*this, getDeviceAttr(), getSite(), "ion.merge"))) {
    return failure();
  }
  if (getDst() == getSrc()) {
    return emitOpError() << "cannot merge chain " << getDst() << " into itself";
  }
  return verifyDeclaredSite(*this, getSiteAttr(), "trap or junction");
}

FailureOr<Configuration> IonMergeOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto dstLocated = findChain(*this, out, getSite(), getDst());
  if (failed(dstLocated)) {
    return failure();
  }
  const auto srcLocated = findChain(*this, out, getSite(), getSrc());
  if (failed(srcLocated)) {
    return failure();
  }

  auto* contents = dstLocated->first;
  const auto dst = dstLocated->second;
  const auto src = srcLocated->second;

  const Word source = (*contents)[src];
  Word merged;
  if (getAt() == ChainEnd::Head) {
    // Prepending leaves the source's ions at the head, where they remain
    // detachable without a reorder.
    merged.append(source.begin(), source.end());
    merged.append((*contents)[dst].begin(), (*contents)[dst].end());
  } else {
    merged.append((*contents)[dst].begin(), (*contents)[dst].end());
    merged.append(source.begin(), source.end());
  }

  (*contents)[dst] = std::move(merged);
  contents->erase(contents->begin() + src);

  return out;
}

//===----------------------------------------------------------------------===//
// IonTransportOp
//===----------------------------------------------------------------------===//

HyperedgeAttr IonTransportOp::getEdge() { return getSubstrateOf(*this).lookupTransportEdge(getVia()); }

llvm::StringRef IonTransportOp::getDestination() {
  const auto edge = getEdge();
  if (!edge) {
    return {};
  }
  for (const auto site : edge.getSites()) {
    if (site.getValue() != getFrom()) {
      return site.getValue();
    }
  }
  return {};
}

LogicalResult IonTransportOp::verify() {
  if (failed(verifyDeclaredSite(*this, getFromAttr(), "source"))) {
    return failure();
  }

  const auto edge = getEdge();
  if (!edge) {
    return emitOpError() << "substrate declares no transport edge with id '" << getVia() << "'";
  }
  if (!edge.contains(getFrom())) {
    return emitOpError() << "transport edge '" << getVia() << "' is not incident to @" << getFrom()
                         << ", so a chain cannot leave that site along it";
  }

  // A segment declared `bidirectional = false` runs from its first incident
  // site to its second. Absent the property, both directions are allowed.
  const auto bidirectional = edge.getProps().getAs<BoolAttr>("bidirectional");
  if (bidirectional && !bidirectional.getValue() && edge.getSites().front().getValue() != getFrom()) {
    return emitOpError() << "transport edge '" << getVia() << "' is unidirectional from @"
                         << edge.getSites().front().getValue() << " to @" << edge.getSites().back().getValue()
                         << ", but this operation traverses it the other way";
  }

  return success();
}

FailureOr<Configuration> IonTransportOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto located = findChain(*this, out, getFrom(), getChain());
  if (failed(located)) {
    return failure();
  }
  auto* source = located->first;
  const auto index = located->second;
  const Word word = (*source)[index];

  const auto edge = getEdge();

  // A segment joins a trap at one end, so only the chain at that end can leave;
  // anything further in would have to pass through the chains in front of it.
  // The chain list is maintained in spatial order, since a split leaves the
  // detached part on the side it came from, so the reachable chain is either the
  // first or the last index.
  if (const auto door = edge.getEndAt(getFrom())) {
    const auto atDoor = *door == ChainEnd::Head ? 0U : static_cast<unsigned>(source->size() - 1);
    if (index != atDoor) {
      return emitOpError() << "chain " << index << " is not at the door: segment '" << getVia() << "' meets @"
                           << getFrom() << " at its " << stringifyChainEnd(*door) << " end, so only chain " << atDoor
                           << " can leave; the " << (index > atDoor ? index - atDoor : atDoor - index)
                           << " chain(s) in between would have to be moved first";
    }
  }

  if (const auto maxLen = edge.getIntProp("max_chain_len"); maxLen && word.size() > static_cast<size_t>(*maxLen)) {
    return emitOpError() << "chain of length " << word.size() << " exceeds the max_chain_len of " << *maxLen
                         << " declared by segment '" << getVia() << "'";
  }

  const auto destination = getDestination();
  const auto destSite = getSubstrateOf(*this).lookupSite(destination);
  const auto arriving = out.getLoad(destination) + word.size();
  if (arriving > destSite.getCapacity()) {
    return emitOpError() << "would place " << arriving << " qubits at @" << destination
                         << ", exceeding its capacity of " << destSite.getCapacity();
  }

  source->erase(source->begin() + index);

  // The arrival lands at the end it entered through rather than at an arbitrary
  // position, so that it can leave again along the segment it arrived on.
  const auto arrivalDoor = edge.getEndAt(destination);
  const bool landsAtHead = arrivalDoor && *arrivalDoor == ChainEnd::Head;

  auto& arrival = out.lookupOrAdd(destination);
  if (landsAtHead) {
    arrival.insert(arrival.begin(), word);
  } else {
    arrival.push_back(word);
  }

  return out;
}

double IonTransportOp::getLatencyUs() {
  const auto edge = getEdge();
  return edge ? edge.getLatencyUs() : 0.0;
}

//===----------------------------------------------------------------------===//
// IonReorderOp
//===----------------------------------------------------------------------===//

LogicalResult IonReorderOp::verify() {
  if (failed(verifyChainSite(*this, getDeviceAttr(), getSite(), "ion.reorder"))) {
    return failure();
  }
  if (failed(verifyDeclaredSite(*this, getSiteAttr(), "trap or junction"))) {
    return failure();
  }

  const auto permutation = getPermutation();
  if (permutation.empty()) {
    return emitOpError() << "permutation must not be empty";
  }

  llvm::DenseSet<int64_t> seen;
  for (const int64_t position : permutation) {
    if (position < 0 || static_cast<size_t>(position) >= permutation.size()) {
      return emitOpError() << "permutation entry " << position << " is out of range for a chain of length "
                           << permutation.size();
    }
    if (!seen.insert(position).second) {
      return emitOpError() << "permutation names position " << position << " twice";
    }
  }

  return success();
}

FailureOr<Configuration> IonReorderOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto located = findChain(*this, out, getSite(), getChain());
  if (failed(located)) {
    return failure();
  }
  auto& word = (*located->first)[located->second];

  const auto permutation = getPermutation();
  if (word.size() != permutation.size()) {
    return emitOpError() << "permutation has " << permutation.size() << " entries but chain " << getChain() << " at @"
                         << getSite() << " holds " << word.size() << " ions";
  }

  Word reordered;
  reordered.reserve(word.size());
  for (const int64_t from : permutation) {
    reordered.push_back(word[static_cast<size_t>(from)]);
  }
  word = std::move(reordered);

  return out;
}

//===----------------------------------------------------------------------===//
// AtomAodMoveOp
//===----------------------------------------------------------------------===//

HyperedgeAttr AtomAodMoveOp::getEdge() { return getSubstrateOf(*this).lookupEdge(EdgeKind::Rigid, getVia()); }

LogicalResult AtomAodMoveOp::verify() {
  const auto substrate = getSubstrateOf(*this);

  const auto edge = getEdge();
  if (!edge) {
    return emitOpError() << "substrate declares no rigid edge with id '" << getVia() << "'";
  }

  const auto sources = edge.getSites();
  const auto destinations = getTo().getValue();
  if (sources.size() != destinations.size()) {
    return emitOpError() << "rigid edge '" << getVia() << "' carries " << sources.size() << " site(s), but "
                         << destinations.size() << " destination(s) were given; the whole axis moves as a unit";
  }

  llvm::StringSet<> seen;
  llvm::SmallVector<unsigned> destinationOrder;
  for (const auto destination : destinations) {
    const auto name = llvm::cast<FlatSymbolRefAttr>(destination).getValue();
    const auto index = substrate.findSite(name);
    if (!index) {
      return emitOpError() << "substrate declares no site @" << name;
    }
    if (!seen.insert(name).second) {
      return emitOpError() << "two atoms on '" << getVia() << "' would land in @" << name;
    }
    destinationOrder.push_back(*index);
  }

  // AOD rows cannot pass through one another, so the axis must arrive in the
  // order it left in. The substrate's declaration order is read as the geometric
  // order along the axis.
  llvm::SmallVector<unsigned> sourceOrder;
  for (const auto source : sources) {
    sourceOrder.push_back(*substrate.findSite(source.getValue()));
  }
  for (size_t i = 1; i < sourceOrder.size(); ++i) {
    const bool sourceAscends = sourceOrder[i] > sourceOrder[i - 1];
    const bool destinationAscends = destinationOrder[i] > destinationOrder[i - 1];
    if (sourceAscends != destinationAscends) {
      return emitOpError() << "move is crossing: @" << sources[i - 1].getValue() << " and @" << sources[i].getValue()
                           << " would swap order on the way to @"
                           << llvm::cast<FlatSymbolRefAttr>(destinations[i - 1]).getValue() << " and @"
                           << llvm::cast<FlatSymbolRefAttr>(destinations[i]).getValue()
                           << "; AOD traps cannot pass through one another";
    }
  }

  return success();
}

FailureOr<Configuration> AtomAodMoveOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto edge = getEdge();
  const auto destinations = getTo().getValue();

  // Every atom on the axis lifts simultaneously, so all sources are read and
  // cleared before any destination is written; otherwise an axis sliding by one
  // site would collide with itself.
  llvm::SmallVector<SiteContents> inFlight;
  for (const auto source : edge.getSites()) {
    auto* contents = out.lookup(source.getValue());
    if (contents == nullptr) {
      inFlight.emplace_back();
      continue;
    }
    inFlight.push_back(*contents);
    contents->clear();
  }

  const auto substrate = getSubstrateOf(*this);
  for (const auto& [destination, arriving] : llvm::zip_equal(destinations, inFlight)) {
    if (arriving.empty()) {
      continue;
    }
    const auto name = llvm::cast<FlatSymbolRefAttr>(destination).getValue();

    uint64_t landing = 0;
    for (const auto& word : arriving) {
      landing += word.size();
    }
    const auto load = out.getLoad(name) + landing;
    if (const auto capacity = substrate.lookupSite(name).getCapacity(); load > capacity) {
      return emitOpError() << "would place " << load << " atoms at @" << name << ", exceeding its capacity of "
                           << capacity;
    }

    llvm::append_range(out.lookupOrAdd(name), arriving);
  }

  return out;
}

double AtomAodMoveOp::getLatencyUs() {
  const auto edge = getEdge();
  return edge ? edge.getLatencyUs() : 0.0;
}

//===----------------------------------------------------------------------===//
// AtomSlmTransferOp
//===----------------------------------------------------------------------===//

LogicalResult AtomSlmTransferOp::verify() {
  if (failed(verifyDeclaredSite(*this, getFromAttr(), "source")) ||
      failed(verifyDeclaredSite(*this, getToAttr(), "destination"))) {
    return failure();
  }
  if (getFrom() == getTo()) {
    return emitOpError() << "source and destination are the same site @" << getFrom();
  }

  // The tweezer can only carry an atom along an axis the atom is on, so the
  // substrate determines how far a handoff can reach.
  const auto substrate = getSubstrateOf(*this);
  for (const auto edge : substrate.getEdgesOfKind(EdgeKind::Rigid)) {
    if (edge.contains(getFrom()) && edge.contains(getTo())) {
      return success();
    }
  }
  return emitOpError() << "no rigid edge is incident to both @" << getFrom() << " and @" << getTo()
                       << "; a tweezer can only carry an atom along an axis that reaches both traps";
}

FailureOr<Configuration> AtomSlmTransferOp::applyRule(const Configuration& in) {
  Configuration out = in;

  auto* source = out.lookup(getFrom());
  if (source == nullptr || source->empty()) {
    return emitOpError() << "the configuration reaching this operation holds no atom at @" << getFrom();
  }
  if (source->size() != 1 || (*source)[0].size() != 1) {
    return emitOpError() << "@" << getFrom() << " holds " << out.getLoad(getFrom())
                         << " atoms; a tweezer handoff carries exactly one, use `qcc.atom.aod_move` for a group";
  }

  const Word atom = (*source)[0];
  source->clear();

  const auto load = out.getLoad(getTo()) + 1;
  if (const auto capacity = getSubstrateOf(*this).lookupSite(getTo()).getCapacity(); load > capacity) {
    return emitOpError() << "would place " << load << " atoms at @" << getTo() << ", exceeding its capacity of "
                         << capacity;
  }

  out.lookupOrAdd(getTo()).push_back(atom);

  return out;
}

//===----------------------------------------------------------------------===//
// LinkGenerateOp
//===----------------------------------------------------------------------===//

HyperedgeAttr LinkGenerateOp::getEdge() { return getSubstrateOf(*this).lookupEdge(EdgeKind::Link, getVia()); }

LogicalResult LinkGenerateOp::verify() {
  const auto edge = getEdge();
  if (!edge) {
    return emitOpError() << "substrate declares no link edge with id '" << getVia() << "'";
  }

  // A permanent coupler exists unconditionally: there is nothing to herald and
  // nothing that could later be consumed.
  if (edge.isPersistent()) {
    return emitOpError() << "link edge '" << getVia()
                         << "' is persistent, so it is available without being generated; only a link declared "
                            "`persistent = false` is a resource to herald";
  }

  const auto qubits = getQubits();
  if (qubits.size() != edge.getSites().size()) {
    return emitOpError() << "link edge '" << getVia() << "' joins " << edge.getSites().size() << " site(s), but "
                         << qubits.size() << " qubit(s) were given";
  }

  llvm::DenseSet<int64_t> seen;
  for (const int64_t qubit : qubits) {
    if (!seen.insert(qubit).second) {
      return emitOpError() << "qubit " << qubit << " is named twice as a party to the link";
    }
  }

  return success();
}

FailureOr<Configuration> LinkGenerateOp::applyRule(const Configuration& in) {
  Configuration out = in;
  const auto edge = getEdge();
  const auto qubits = getQubits();

  // Party `i` must occupy site `i` of the edge, since a photon leaves the module
  // its emitter is in.
  for (const auto& [site, qubit] : llvm::zip_equal(edge.getSites(), qubits)) {
    const auto where = out.findQubit(qubit);
    if (!where) {
      return emitOpError() << "qubit " << qubit << " is not placed anywhere, so it cannot be a party to a link";
    }
    if (*where != site.getValue()) {
      return emitOpError() << "qubit " << qubit << " sits at @" << *where << ", but link edge '" << getVia()
                           << "' expects that party at @" << site.getValue();
    }
    // One communication qubit holds one link at a time.
    if (out.isEntangled(qubit)) {
      return emitOpError() << "qubit " << qubit
                           << " already holds a live link; spend it with `qcc.link.consume` before heralding another";
    }
  }

  // The party order is preserved rather than sorted. Bell and GHZ resources are
  // largely symmetric, but which party sits at which end of the fibre is a
  // property of the resource, and a graph state depends on it. Matching a link
  // for consumption remains order-insensitive; only the record is ordered.
  out.links.emplace_back(qubits.begin(), qubits.end());

  return out;
}

double LinkGenerateOp::getLatencyUs() {
  const auto edge = getEdge();
  return edge ? edge.getLatencyUs() : 0.0;
}

//===----------------------------------------------------------------------===//
// LinkConsumeOp
//===----------------------------------------------------------------------===//

LogicalResult LinkConsumeOp::verify() {
  const auto qubits = getQubits();
  if (qubits.size() < 2) {
    return emitOpError() << "a link joins at least two qubits, but " << qubits.size() << " were given";
  }

  llvm::DenseSet<int64_t> seen;
  for (const int64_t qubit : qubits) {
    if (!seen.insert(qubit).second) {
      return emitOpError() << "qubit " << qubit << " is named twice as a party to the link";
    }
  }

  return success();
}

FailureOr<Configuration> LinkConsumeOp::applyRule(const Configuration& in) {
  Configuration out = in;

  const Word parties(getQubits().begin(), getQubits().end());

  const auto live = out.findLink(parties);
  if (!live) {
    std::string wanted;
    llvm::raw_string_ostream os(wanted);
    llvm::interleaveComma(parties, os);
    return emitOpError() << "no live link joins {" << wanted
                         << "} in the configuration reaching this operation; a link is spent by use, so it has to be "
                            "heralded again before it can be spent again";
  }
  out.links.erase(out.links.begin() + *live);

  return out;
}

//===----------------------------------------------------------------------===//
// IonJunctionTurnOp
//===----------------------------------------------------------------------===//

LogicalResult IonJunctionTurnOp::verify() {
  if (failed(verifyDeclaredSite(*this, getSiteAttr(), "junction"))) {
    return failure();
  }

  const auto substrate = getSubstrateOf(*this);
  const auto junction = substrate.lookupSite(getSite());

  // A turn joins two segments, both of which must be incident to this junction.
  for (const auto& [role, id] : {std::pair{"incoming", getFrom()}, std::pair{"outgoing", getTo()}}) {
    const auto edge = substrate.lookupEdge(EdgeKind::Transport, id);
    if (!edge) {
      return emitOpError() << "substrate declares no transport edge with id '" << id << "'";
    }
    if (!edge.contains(getSite())) {
      return emitOpError() << role << " segment '" << id << "' is not incident to @" << getSite();
    }
  }

  if (getFrom() == getTo()) {
    return emitOpError() << "a turn joins two different segments, but both are '" << getFrom()
                         << "'; a crystal that leaves the way it came needs no rotation";
  }

  // The turn-restriction table is a property of the junction, in the same way
  // that a road network attaches it to the node rather than to a ternary edge.
  if (junction && !junction.permitsTurn(getFrom(), getTo())) {
    return emitOpError() << "@" << getSite() << " does not permit the turn '" << getFrom() << "' -> '" << getTo()
                         << "'; its turn table lists the rotations the hardware can actually perform";
  }

  return success();
}

FailureOr<Configuration> IonJunctionTurnOp::applyRule(const Configuration& in) {
  Configuration out = in;
  // A rotation changes which segment the crystal faces, not where it is or how
  // its ions are ordered. The configuration records position only; the facing is
  // implied by the transport the turn has made legal.
  const auto located = findChain(*this, out, getSite(), getChain());
  if (failed(located)) {
    return failure();
  }
  return out;
}

double IonJunctionTurnOp::getLatencyUs() {
  const auto junction = getSubstrateOf(*this).lookupSite(getSite());
  return junction ? junction.getTurnLatencyUs() : 0.0;
}

//===----------------------------------------------------------------------===//
// Footprints
//
// What each rule touches, which determines whether two of them may be
// reordered. The sites come from the rule's own attributes; the control
// resources come from the substrate, as the `control` edges covering those
// sites.
//===----------------------------------------------------------------------===//

namespace qcc::conn {

namespace {

/// The ids of every control edge incident to any of `sites`, that is, the
/// control resources a rule acting on those sites contends for.
llvm::SmallVector<std::string> contendedControls(Operation* op, llvm::ArrayRef<std::string> sites) {
  llvm::SmallVector<std::string> uses;
  for (const auto edge : getSubstrateOf(op).getEdgesOfKind(EdgeKind::Control)) {
    const bool touches = llvm::any_of(sites, [&](const std::string& site) { return edge.contains(site); });
    if (touches && !edge.getId().empty()) {
      uses.emplace_back(edge.getId());
    }
  }
  return uses;
}

Footprint footprintOver(Operation* op, llvm::ArrayRef<llvm::StringRef> sites) {
  Footprint footprint;
  for (const auto site : sites) {
    if (!site.empty()) {
      footprint.sites.emplace_back(site);
    }
  }
  footprint.uses = contendedControls(op, footprint.sites);
  return footprint;
}

} // namespace

} // namespace qcc::conn

Footprint IonSplitOp::getFootprint() { return footprintOver(*this, {getSite()}); }

Footprint IonMergeOp::getFootprint() { return footprintOver(*this, {getSite()}); }

Footprint IonReorderOp::getFootprint() { return footprintOver(*this, {getSite()}); }

Footprint IonJunctionTurnOp::getFootprint() { return footprintOver(*this, {getSite()}); }

// Transport touches both endpoints, since the chain leaves one site and arrives
// at the other, so a rule acting on either is not independent of it.
Footprint IonTransportOp::getFootprint() { return footprintOver(*this, {getFrom(), getDestination()}); }

Footprint AtomAodMoveOp::getFootprint() {
  llvm::SmallVector<llvm::StringRef> sites;
  if (const auto edge = getEdge()) {
    for (const auto member : edge.getSites()) {
      sites.push_back(member.getValue());
    }
  }
  for (const auto destination : getTo()) {
    sites.push_back(llvm::cast<FlatSymbolRefAttr>(destination).getValue());
  }
  return footprintOver(*this, sites);
}

Footprint AtomSlmTransferOp::getFootprint() { return footprintOver(*this, {getFrom(), getTo()}); }

Footprint LinkGenerateOp::getFootprint() {
  llvm::SmallVector<llvm::StringRef> sites;
  if (const auto edge = getEdge()) {
    for (const auto member : edge.getSites()) {
      sites.push_back(member.getValue());
    }
  }
  return footprintOver(*this, sites);
}

Footprint LinkConsumeOp::getFootprint() {
  // Consuming a link touches wherever its parties are, which the operation does
  // not name. Reporting the sites of every non-persistent link edge is
  // conservative: it can only cause the scheduler to serialise more.
  llvm::SmallVector<llvm::StringRef> sites;
  for (const auto edge : getSubstrateOf(*this).getEdgesOfKind(EdgeKind::Link)) {
    if (edge.isPersistent()) {
      continue;
    }
    for (const auto member : edge.getSites()) {
      sites.push_back(member.getValue());
    }
  }
  return footprintOver(*this, sites);
}
