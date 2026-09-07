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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include <array>
#include <cstdint>
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <optional>
#include <string>

namespace qcc::conn {

/// The module attribute naming the machine a program is compiled for.
/// `--qcc-attach-device` writes it; the placement, routing and verification
/// passes read it.
constexpr llvm::StringLiteral kDeviceAttrName = "qcc.device";

//===----------------------------------------------------------------------===//
// Device snapshots
//===----------------------------------------------------------------------===//

/// One site as a machine description reports it. The fields are those QDMI can
/// supply, so a hand-written description and an imported one meet here.
struct SiteSnapshot {
  std::string name;
  /// A zone holds many qubits (a trap, a module); a non-zone site holds one (a
  /// transmon, an SLM trap). QDMI reports this as `ISZONE`.
  bool isZone = false;
  /// Transient peak occupancy. QDMI does not report it, so it defaults to 1 and
  /// must be supplied for a zone.
  uint64_t capacity = 1;
  /// Position, when the description gives one.
  std::optional<std::array<double, 3>> position;
  /// Module on a modular machine, from QDMI's `MODULEINDEX`.
  std::optional<uint64_t> moduleIndex;
};

/// What a machine description reports: sites, the pairs that can interact, and
/// the arity of the available operations. This is the coupling-graph view, and
/// for a superconducting chip it is the whole machine.
struct DeviceSnapshot {
  std::string name;
  llvm::SmallVector<SiteSnapshot> sites;
  /// Pairs of site indices that can interact directly, from QDMI's
  /// `COUPLINGMAP`.
  llvm::SmallVector<std::pair<unsigned, unsigned>> couplings;
  /// Widest single operation, which becomes the device `maxFacetRank`.
  uint64_t maxOperationArity = 2;
  /// Rydberg blockade radius, in the units of the site positions. When set,
  /// blockade discs are derived from geometry rather than listed.
  std::optional<double> minAtomDistance;
};

//===----------------------------------------------------------------------===//
// Device supplements
//===----------------------------------------------------------------------===//

/// Per-site facts a device-management interface does not report.
struct SiteOverride {
  std::string name;
  std::optional<SiteKind> kind;
  std::optional<uint64_t> capacity;
  /// Entangling width. Below 2 means the site cannot host a gate, which is how
  /// a QCCD description says its storage traps have no gate lasers.
  std::optional<uint64_t> maxFacetRank;
  /// Junction turn table and rotation cost.
  mlir::DictionaryAttr props;
};

/// The half of a machine description with no coupling-map spelling.
///
/// QDMI models a coupling map over qubits that do not move. It has no property
/// for a transport segment, for the zones sharing a waveform generator, or for
/// an AOD axis. Those arrive here, and `buildDevice` merges the two halves.
struct DeviceSupplement {
  /// Transport segments, control groups, rigid axes, and any link the coupling
  /// map does not imply.
  llvm::SmallVector<HyperedgeAttr> edges;
  llvm::SmallVector<SiteOverride> sites;
  /// Algebraic properties of `K`. Unset, they are inferred from the snapshot,
  /// which suits a coupling-graph machine and not one with chains.
  std::optional<bool> downwardClosed;
  std::optional<bool> partitioning;
  std::optional<FacetOrdering> facetOrdering;
  std::optional<uint64_t> maxFacetRank;

  /// The override for `name`, or nullptr.
  [[nodiscard]] const SiteOverride* find(llvm::StringRef name) const {
    for (const auto& site : sites) {
      if (site.name == name) {
        return &site;
      }
    }
    return nullptr;
  }
};

/// Folds a snapshot and its supplement into the attribute the dialect consumes.
/// Reports through `emitError` when the two disagree rather than dropping the
/// part it cannot reconcile.
mlir::FailureOr<DeviceAttr> buildDevice(mlir::MLIRContext* ctx, const DeviceSnapshot& snapshot,
                                        const DeviceSupplement& supplement,
                                        const std::function<mlir::InFlightDiagnostic()>& emitError);

//===----------------------------------------------------------------------===//
// The library
//===----------------------------------------------------------------------===//

/// A machine the compiler can describe without being told.
///
/// `qcc::Target` selects a backend: how code is emitted. A `DeviceEntry`
/// selects a machine: where the qubits are and how they move. A program is
/// compiled for one of each.
struct DeviceEntry {
  /// The `--device` value, e.g. "qccd-linear-2".
  llvm::StringRef name;
  /// Description shown by `--list-devices`.
  llvm::StringRef description;
  /// Builds the description, returning a null attribute on failure.
  std::function<DeviceAttr(mlir::MLIRContext*)> build;
};

/// The machines compiled into this build.
llvm::ArrayRef<DeviceEntry> getDevices();

/// Looks up a machine by its `--device` name, or returns nullptr.
const DeviceEntry* lookupDevice(llvm::StringRef name);

} // namespace qcc::conn
