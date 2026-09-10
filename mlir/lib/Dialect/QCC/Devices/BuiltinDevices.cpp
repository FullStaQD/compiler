// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// The machines compiled into this build.
//
// These describe topologies and rule sets, not calibrated devices: the
// latencies are order-of-magnitude and no error rates are claimed. They make
// `--device=<name>` usable without further setup. A real machine arrives
// through QDMI or as a hand-written `#qcc.device` attribute, and neither
// requires changes here.
//
// The names describe the topology rather than a vendor, since a vendor name
// would imply calibration data that is not present.
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/Devices/DeviceLibrary.h"
#include "qcc/Dialect/QCC/IR/QCC.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <string>

using namespace mlir;

namespace qcc::conn {

namespace {

/// An edge of `kind` over `members`, carrying `id` and any extra properties.
HyperedgeAttr edge(MLIRContext* ctx, EdgeKind kind, llvm::ArrayRef<llvm::StringRef> members, llvm::StringRef id,
                   llvm::ArrayRef<NamedAttribute> extra = {}) {
  Builder builder(ctx);
  llvm::SmallVector<FlatSymbolRefAttr> refs;
  for (const auto member : members) {
    refs.push_back(FlatSymbolRefAttr::get(ctx, member));
  }
  llvm::SmallVector<NamedAttribute> props{builder.getNamedAttr("id", builder.getStringAttr(id))};
  llvm::append_range(props, extra);
  return HyperedgeAttr::get(ctx, kind, llvm::ArrayRef<FlatSymbolRefAttr>(refs), builder.getDictionaryAttr(props));
}

/// A zone site of the given capacity, at an optional position.
SiteSnapshot zone(llvm::StringRef name, uint64_t capacity) {
  SiteSnapshot site;
  site.name = name.str();
  site.isZone = true;
  site.capacity = capacity;
  return site;
}

/// A single-qubit site on a grid.
SiteSnapshot cell(llvm::StringRef name, double x, double y) {
  SiteSnapshot site;
  site.name = name.str();
  site.isZone = false;
  site.position = std::array<double, 3>{x, y, 0.0};
  return site;
}

/// A malformed built-in is a bug in this file rather than in a user program, so
/// its diagnostics go to the context.
DeviceAttr finish(MLIRContext* ctx, const DeviceSnapshot& snapshot, const DeviceSupplement& supplement) {
  const auto device = buildDevice(ctx, snapshot, supplement, [&] { return emitError(UnknownLoc::get(ctx)); });
  return succeeded(device) ? *device : DeviceAttr{};
}

//===----------------------------------------------------------------------===//
// Superconducting: a 4x4 grid of fixed qubits.
//
// The degenerate case: no transport, no rigid axes, every link permanent. The
// rule set is empty, so a configuration threaded through such a program never
// changes.
//===----------------------------------------------------------------------===//

DeviceAttr buildScGrid4x4(MLIRContext* ctx) {
  constexpr unsigned kSide = 4;

  DeviceSnapshot snapshot;
  snapshot.name = "sc-grid-4x4";
  snapshot.maxOperationArity = 2;
  for (unsigned row = 0; row < kSide; ++row) {
    for (unsigned col = 0; col < kSide; ++col) {
      snapshot.sites.push_back(cell("q" + std::to_string((row * kSide) + col), col, row));
    }
  }
  for (unsigned row = 0; row < kSide; ++row) {
    for (unsigned col = 0; col < kSide; ++col) {
      const unsigned here = (row * kSide) + col;
      if (col + 1 < kSide) {
        snapshot.couplings.emplace_back(here, here + 1);
      }
      if (row + 1 < kSide) {
        snapshot.couplings.emplace_back(here, here + kSide);
      }
    }
  }
  // The coupling map is the whole machine, so no supplement is needed.
  return finish(ctx, snapshot, DeviceSupplement{});
}

//===----------------------------------------------------------------------===//
// QCCD ions: two storage traps either side of a junction.
//
// The smallest machine on which shuttling matters. The segments, the shared
// waveform generator and the chain order all live in the supplement, having no
// coupling-map spelling.
//===----------------------------------------------------------------------===//

DeviceAttr buildQccdLinear2(MLIRContext* ctx) {
  Builder builder(ctx);
  const auto latency = [&](double us) { return builder.getNamedAttr("latency_us", builder.getF64FloatAttr(us)); };
  const auto maxLen = builder.getNamedAttr("max_chain_len", builder.getI64IntegerAttr(10));
  const auto bidirectional = builder.getNamedAttr("bidirectional", builder.getBoolAttr(true));

  DeviceSnapshot snapshot;
  snapshot.name = "qccd-linear-2";
  snapshot.maxOperationArity = 10;
  snapshot.sites = {zone("trap_a", 20), zone("j0", 10), zone("trap_b", 20)};
  // No couplings: a gate is possible through co-location inside a trap, which
  // is a property of the configuration rather than of the substrate.

  DeviceSupplement supplement;
  supplement.sites = {
      {"trap_a", SiteKind::Trap, 20, std::nullopt, {}},
      {"j0", SiteKind::Junction, 10, std::nullopt, {}},
      {"trap_b", SiteKind::Trap, 20, std::nullopt, {}},
  };
  supplement.edges = {
      edge(ctx, EdgeKind::Transport, {"trap_a", "j0"}, "seg_aj", {latency(80.0), bidirectional, maxLen}),
      edge(ctx, EdgeKind::Transport, {"j0", "trap_b"}, "seg_jb", {latency(80.0), bidirectional, maxLen}),
      // One generator drives every shuttling electrode, so no two transports
      // may overlap in time even on disjoint segments.
      edge(ctx, EdgeKind::Control, {"trap_a", "j0", "trap_b"}, "awg_shuttle"),
      // One MS laser, switched between the traps.
      edge(ctx, EdgeKind::Control, {"trap_a", "trap_b"}, "laser_ms"),
  };
  supplement.maxFacetRank = 10;
  supplement.downwardClosed = true;
  supplement.partitioning = true;
  supplement.facetOrdering = FacetOrdering::Linear;
  return finish(ctx, snapshot, supplement);
}

//===----------------------------------------------------------------------===//
// QCCD ions, H-series shape: a storage trap and a small gate zone.
//
// Storage holds thirty ions and gates none of them, so a program shuttles a
// pair into the gate zone before acting on them. Without `maxFacetRank = 0` on
// storage the compiler would accept gates the hardware cannot perform.
//===----------------------------------------------------------------------===//

DeviceAttr buildQccdStorageGate(MLIRContext* ctx) {
  Builder builder(ctx);
  const auto latency = [&](double us) { return builder.getNamedAttr("latency_us", builder.getF64FloatAttr(us)); };
  const auto maxLen = builder.getNamedAttr("max_chain_len", builder.getI64IntegerAttr(2));
  const auto bidirectional = builder.getNamedAttr("bidirectional", builder.getBoolAttr(true));

  DeviceSnapshot snapshot;
  snapshot.name = "qccd-storage-gate";
  snapshot.maxOperationArity = 2;
  snapshot.sites = {zone("storage", 30), zone("j0", 4), zone("gate", 2)};

  DeviceSupplement supplement;
  // Storage and the junction are places to be, not places to be acted on. The
  // gate zone declares nothing and inherits the device rank of 2.
  supplement.sites = {
      {"storage", SiteKind::Trap, 30, 0, {}},
      {"j0", SiteKind::Junction, 4, 0, {}},
      {"gate", SiteKind::GateZone, 2, std::nullopt, {}},
  };
  supplement.edges = {
      edge(ctx, EdgeKind::Transport, {"storage", "j0"}, "seg_sj", {latency(100.0), bidirectional, maxLen}),
      edge(ctx, EdgeKind::Transport, {"j0", "gate"}, "seg_jg", {latency(60.0), bidirectional, maxLen}),
      edge(ctx, EdgeKind::Control, {"storage", "j0", "gate"}, "awg_shuttle"),
  };
  supplement.maxFacetRank = 2;
  supplement.downwardClosed = true;
  supplement.partitioning = true;
  supplement.facetOrdering = FacetOrdering::Linear;
  return finish(ctx, snapshot, supplement);
}

//===----------------------------------------------------------------------===//
// Neutral atoms: a 3x3 SLM array under a crossed AOD.
//
// The blockade discs are derived from the trap positions and the blockade
// radius rather than listed, so moving a trap moves its disc.
//===----------------------------------------------------------------------===//

DeviceAttr buildNeutralAtom3x3(MLIRContext* ctx) {
  constexpr unsigned kSide = 3;
  Builder builder(ctx);
  const auto axis = [&](llvm::StringRef which) { return builder.getNamedAttr("axis", builder.getStringAttr(which)); };

  DeviceSnapshot snapshot;
  snapshot.name = "neutral-atom-3x3";
  snapshot.maxOperationArity = 3;
  for (unsigned row = 0; row < kSide; ++row) {
    for (unsigned col = 0; col < kSide; ++col) {
      snapshot.sites.push_back(cell("t" + std::to_string(row) + std::to_string(col), col, row));
    }
  }
  // On a unit grid this radius reaches the four edge neighbours, so an atom in
  // the middle belongs to five overlapping discs.
  snapshot.minAtomDistance = 1.0;

  DeviceSupplement supplement;
  // An AOD row or column moves as a unit. This has no coupling-map spelling: it
  // says which atoms must move together, not which can interact.
  supplement.edges = {
      edge(ctx, EdgeKind::Rigid, {"t00", "t01", "t02"}, "aod_row_0", {axis("x")}),
      edge(ctx, EdgeKind::Rigid, {"t10", "t11", "t12"}, "aod_row_1", {axis("x")}),
      edge(ctx, EdgeKind::Rigid, {"t20", "t21", "t22"}, "aod_row_2", {axis("x")}),
      edge(ctx, EdgeKind::Rigid, {"t00", "t10", "t20"}, "aod_col_0", {axis("y")}),
      edge(ctx, EdgeKind::Rigid, {"t01", "t11", "t21"}, "aod_col_1", {axis("y")}),
      edge(ctx, EdgeKind::Rigid, {"t02", "t12", "t22"}, "aod_col_2", {axis("y")}),
  };
  supplement.maxFacetRank = 3;
  supplement.downwardClosed = true;
  supplement.partitioning = false;
  supplement.facetOrdering = FacetOrdering::None;
  return finish(ctx, snapshot, supplement);
}

//===----------------------------------------------------------------------===//
// Modular photonic: three modules on an optical switch.
//
// The interconnects are consumable, so they are supplied rather than derived
// from the coupling map: a reported coupling means the pair can interact, and a
// heralded Bell pair means it can interact once.
//===----------------------------------------------------------------------===//

DeviceAttr buildModularPhotonic3(MLIRContext* ctx) {
  Builder builder(ctx);
  const auto consumable = builder.getNamedAttr("persistent", builder.getBoolAttr(false));
  const auto latency = [&](double us) { return builder.getNamedAttr("latency_us", builder.getF64FloatAttr(us)); };

  DeviceSnapshot snapshot;
  snapshot.name = "modular-photonic-3";
  snapshot.maxOperationArity = 2;
  for (unsigned module = 0; module < 3; ++module) {
    auto site = zone("m" + std::to_string(module), 4);
    site.moduleIndex = module;
    snapshot.sites.push_back(site);
  }

  DeviceSupplement supplement;
  supplement.sites = {
      {"m0", SiteKind::Module, 4, std::nullopt, {}},
      {"m1", SiteKind::Module, 4, std::nullopt, {}},
      {"m2", SiteKind::Module, 4, std::nullopt, {}},
  };
  supplement.edges = {
      edge(ctx, EdgeKind::Link, {"m0", "m1"}, "fibre_01", {consumable, latency(5500.0)}),
      edge(ctx, EdgeKind::Link, {"m1", "m2"}, "fibre_12", {consumable, latency(6100.0)}),
      // One heralding detector array serves every module.
      edge(ctx, EdgeKind::Control, {"m0", "m1", "m2"}, "bsa_detector"),
  };
  supplement.maxFacetRank = 2;
  supplement.downwardClosed = true;
  supplement.partitioning = false;
  supplement.facetOrdering = FacetOrdering::None;
  return finish(ctx, snapshot, supplement);
}

const DeviceEntry builtinDevices[] = {
    {"sc-grid-4x4", "Superconducting 4x4 lattice; static coupling, empty rule set", buildScGrid4x4},
    {"qccd-linear-2", "Trapped-ion QCCD: two traps either side of a junction, with shuttling", buildQccdLinear2},
    {"qccd-storage-gate", "Trapped-ion QCCD: a storage trap that cannot gate, and a two-ion gate zone",
     buildQccdStorageGate},
    {"neutral-atom-3x3", "Neutral atoms: 3x3 SLM array under a crossed AOD, blockade discs derived from geometry",
     buildNeutralAtom3x3},
    {"modular-photonic-3", "Three modules on an optical switch; interconnects consumed by use", buildModularPhotonic3},
};

} // namespace

llvm::ArrayRef<DeviceEntry> getDevices() { return builtinDevices; }

const DeviceEntry* lookupDevice(const llvm::StringRef name) {
  for (const auto& entry : builtinDevices) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

} // namespace qcc::conn
