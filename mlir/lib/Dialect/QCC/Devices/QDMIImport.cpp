// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "qcc/Dialect/QCC/Devices/QDMIImport.h"

#include "qcc/Dialect/QCC/Devices/DeviceLibrary.h"

#include "mlir/Support/LLVM.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <qdmi/constants.h>
#include <qdmi/device.h>
#include <qdmi/types.h>
#include <string>
#include <vector>

using namespace mlir;

namespace qcc::conn {

namespace {

/// Query a fixed-size device property. Returns false when the device does not
/// offer it, which is normal: most properties are optional.
template <typename T> bool queryDevice(QDMI_Device_Session session, const QDMI_Device_Property prop, T& out) {
  return QDMI_device_session_query_device_property(session, prop, sizeof(T), &out, nullptr) == QDMI_SUCCESS;
}

/// Query a variable-length device property, sizing the buffer from the device.
template <typename T>
bool queryDeviceList(QDMI_Device_Session session, const QDMI_Device_Property prop, std::vector<T>& out) {
  size_t bytes = 0;
  if (QDMI_device_session_query_device_property(session, prop, 0, nullptr, &bytes) != QDMI_SUCCESS || bytes == 0) {
    return false;
  }
  out.resize(bytes / sizeof(T));
  return QDMI_device_session_query_device_property(session, prop, bytes, out.data(), nullptr) == QDMI_SUCCESS;
}

template <typename T>
bool querySite(QDMI_Device_Session session, QDMI_Site site, const QDMI_Site_Property prop, T& out) {
  return QDMI_device_session_query_site_property(session, site, prop, sizeof(T), &out, nullptr) == QDMI_SUCCESS;
}

bool querySiteName(QDMI_Device_Session session, QDMI_Site site, std::string& out) {
  size_t bytes = 0;
  if (QDMI_device_session_query_site_property(session, site, QDMI_SITE_PROPERTY_NAME, 0, nullptr, &bytes) !=
          QDMI_SUCCESS ||
      bytes == 0) {
    return false;
  }
  out.resize(bytes);
  if (QDMI_device_session_query_site_property(session, site, QDMI_SITE_PROPERTY_NAME, bytes, out.data(), nullptr) !=
      QDMI_SUCCESS) {
    return false;
  }
  // QDMI returns a NUL-terminated buffer; the terminator is not part of the name
  // used as a symbol.
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return true;
}

} // namespace

bool isQDMIAvailable() { return true; }

FailureOr<DeviceSnapshot> queryDeviceSnapshot(void* opaqueSession) {
  auto* session = static_cast<QDMI_Device_Session>(opaqueSession);
  if (session == nullptr) {
    return failure();
  }

  DeviceSnapshot snapshot;

  //--- Name ------------------------------------------------------------------
  size_t nameBytes = 0;
  if (QDMI_device_session_query_device_property(session, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &nameBytes) ==
          QDMI_SUCCESS &&
      nameBytes > 0) {
    snapshot.name.resize(nameBytes);
    if (QDMI_device_session_query_device_property(session, QDMI_DEVICE_PROPERTY_NAME, nameBytes, snapshot.name.data(),
                                                  nullptr) != QDMI_SUCCESS) {
      return failure();
    }
    if (!snapshot.name.empty() && snapshot.name.back() == '\0') {
      snapshot.name.pop_back();
    }
  }

  //--- Sites -----------------------------------------------------------------
  std::vector<QDMI_Site> sites;
  if (!queryDeviceList(session, QDMI_DEVICE_PROPERTY_SITES, sites)) {
    return failure();
  }

  llvm::SmallVector<QDMI_Site> order;
  for (const auto site : sites) {
    SiteSnapshot entry;
    if (!querySiteName(session, site, entry.name)) {
      // A site with no name of its own is addressed by index, and the dialect
      // addresses sites by symbol, so synthesise one.
      size_t index = 0;
      querySite(session, site, QDMI_SITE_PROPERTY_INDEX, index);
      entry.name = "s" + std::to_string(index);
    }

    // A zone holds many qubits; a plain site holds one. Which *kind* of zone it
    // is has no QDMI spelling, so it stays a trap until a supplement says
    // otherwise.
    if (int isZone = 0; querySite(session, site, QDMI_SITE_PROPERTY_ISZONE, isZone)) {
      entry.isZone = isZone != 0;
    }

    // A position needs at least two axes to be one; the third defaults to zero
    // for the planar layouts most devices report.
    double x = 0.0;
    double y = 0.0;
    if (querySite(session, site, QDMI_SITE_PROPERTY_XCOORDINATE, x) &&
        querySite(session, site, QDMI_SITE_PROPERTY_YCOORDINATE, y)) {
      double z = 0.0;
      querySite(session, site, QDMI_SITE_PROPERTY_ZCOORDINATE, z);
      entry.position = std::array<double, 3>{x, y, z};
    }
    if (size_t module = 0; querySite(session, site, QDMI_SITE_PROPERTY_MODULEINDEX, module)) {
      entry.moduleIndex = module;
    }

    order.push_back(site);
    snapshot.sites.push_back(std::move(entry));
  }

  //--- Coupling map ----------------------------------------------------------
  // Reported as a flat list of site pairs. Every reported pair is a permanent
  // interaction, which `buildDevice` turns into a persistent link edge.
  std::vector<QDMI_Site> couplings;
  if (queryDeviceList(session, QDMI_DEVICE_PROPERTY_COUPLINGMAP, couplings)) {
    const auto indexOf = [&](QDMI_Site site) -> unsigned {
      for (const auto& [index, candidate] : llvm::enumerate(order)) {
        if (candidate == site) {
          return static_cast<unsigned>(index);
        }
      }
      return static_cast<unsigned>(order.size());
    };
    for (size_t i = 0; i + 1 < couplings.size(); i += 2) {
      const unsigned a = indexOf(couplings[i]);
      const unsigned b = indexOf(couplings[i + 1]);
      if (a < order.size() && b < order.size()) {
        snapshot.couplings.emplace_back(a, b);
      }
    }
  }

  //--- Blockade radius -------------------------------------------------------
  if (double distance = 0.0; queryDevice(session, QDMI_DEVICE_PROPERTY_MINATOMDISTANCE, distance) && distance > 0.0) {
    snapshot.minAtomDistance = distance;
  }

  // QDMI reports operations, but their arity is a per-operation property whose
  // spelling varies by device. Two is the safe floor: a machine that could not
  // entangle a pair would have nothing to route.
  snapshot.maxOperationArity = 2;

  return snapshot;
}

} // namespace qcc::conn
