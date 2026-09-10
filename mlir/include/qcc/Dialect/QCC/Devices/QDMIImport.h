// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#pragma once

#include "qcc/Dialect/QCC/Devices/DeviceLibrary.h"

#include "mlir/Support/LLVM.h"

#include <llvm/ADT/StringRef.h>

namespace qcc::conn {

/// Read a machine description out of a QDMI device session.
///
/// QDMI reports sites, with `ISZONE`, coordinates, extents and `MODULEINDEX`;
/// the coupling map; the operation set, whose widest member becomes
/// `maxFacetRank`; `MINATOMDISTANCE`, from which blockade discs are derived; and
/// `CHILDDEVICES` for a modular machine. For a superconducting chip that is the
/// complete description, and the resulting device needs no supplement.
///
/// QDMI models a coupling map over qubits that do not move, so it has no
/// property for a transport segment, for the zones sharing an arbitrary-waveform
/// generator, for an AOD axis, or for the order of ions in a chain. Those are
/// supplied by the `DeviceSupplement`, which `buildDevice` merges with the
/// snapshot.
///
/// `queryDeviceSnapshot` is the only part requiring a live session; everything
/// that interprets the machine description operates on the snapshot and can
/// therefore be exercised without one.
///
/// `session` is a `QDMI_Device_Session`, taken as `void*` so that this header
/// costs nothing to include when QDMI is not in the build.
mlir::FailureOr<DeviceSnapshot> queryDeviceSnapshot(void* session);

/// Whether this build was configured with QDMI support.
bool isQDMIAvailable();

} // namespace qcc::conn
