#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

//===----------------------------------------------------------------------===//
// QIS
//===----------------------------------------------------------------------===//

// TODO: This is a hardcoded QIS, we need to query it from the device in the future.

/// Z-Basis measurement (irreversible).
static constexpr llvm::StringLiteral QIR_QIS_MZ = "__quantum__qis__mz__body";

} // namespace qcc
