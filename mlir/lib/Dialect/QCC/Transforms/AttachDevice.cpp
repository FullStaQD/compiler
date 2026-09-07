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
#include "qcc/Dialect/QCC/Transforms/Passes.h" // IWYU pragma: keep

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

namespace qcc {

#define GEN_PASS_DEF_ATTACHDEVICE
#include "qcc/Dialect/QCC/Transforms/Passes.h.inc"

using namespace mlir;
using namespace qcc::conn;

namespace {

/// The machines this build knows, for a diagnostic the user can act on.
std::string knownDevices() {
  std::string names;
  llvm::raw_string_ostream os(names);
  llvm::interleaveComma(getDevices(), os, [&](const DeviceEntry& entry) { os << entry.name; });
  return names;
}

struct AttachDevice final : public impl::AttachDeviceBase<AttachDevice> {
  using AttachDeviceBase<AttachDevice>::AttachDeviceBase;

protected:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (device.empty()) {
      moduleOp.emitError() << "no machine named: pass `device=<name>` to say which one to compile for";
      return signalPassFailure();
    }

    const auto* entry = lookupDevice(device);
    if (entry == nullptr) {
      moduleOp.emitError() << "unknown machine '" << device << "'; this build knows " << knownDevices();
      return signalPassFailure();
    }

    const auto attribute = entry->build(&getContext());
    if (!attribute) {
      moduleOp.emitError() << "machine '" << device << "' failed to build; this is a bug in its description";
      return signalPassFailure();
    }

    // An existing attribute is the user's. Overwriting it would silently
    // recompile for a machine they did not ask for.
    if (const auto existing = moduleOp->getAttrOfType<DeviceAttr>(kDeviceAttrName)) {
      if (existing != attribute) {
        moduleOp.emitError() << "module is already attached to a different machine; remove the `" << kDeviceAttrName
                             << "` attribute to retarget it";
        return signalPassFailure();
      }
      return;
    }

    moduleOp->setAttr(kDeviceAttrName, attribute);
  }
};

} // namespace
} // namespace qcc
