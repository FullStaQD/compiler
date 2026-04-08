#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

//===----------------------------------------------------------------------===//
// QIR runtime functions
//===----------------------------------------------------------------------===//

/// Must be called right at the start of an entry-point.
static constexpr llvm::StringLiteral qirRtInit = "__quantum__rt__initialize";

/// Convert a measurement result to a i1.
static constexpr llvm::StringLiteral qirRtReadResult = "__quantum__rt__read_result";

//===----------------------------------------------------------------------===//
// QIR quantum instruction set (QIS)
//===----------------------------------------------------------------------===//

// TODO: This is a hardcoded QIS, we need to query it from the device in the future.

/// Z-Basis measurement (irreversible).
static constexpr llvm::StringLiteral qirQisMZ = "__quantum__qis__mz__body";

static constexpr llvm::StringLiteral qirQisH = "__quantum__qis__h__body";
static constexpr llvm::StringLiteral qirQisX = "__quantum__qis__x__body";

static constexpr llvm::StringLiteral qirQisCX = "__quantum__qis__cx__body";

} // namespace qcc
