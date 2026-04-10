#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

//===----------------------------------------------------------------------===//
// FIXME: stuff that probably belongs elsewhere
//===----------------------------------------------------------------------===//

/// A unit attribute to mark a `func.func` as the starting point of a quantum program.
static constexpr llvm::StringLiteral QCC_ENTRY_POINT_ATTR_NAME = "qcc.entry_point";

//===----------------------------------------------------------------------===//
// QIR runtime functions
//===----------------------------------------------------------------------===//

/// Must be called right at the start of an entry-point.
static constexpr llvm::StringLiteral qirRtInit = "__quantum__rt__initialize";

/// Convert a measurement result to a i1.
static constexpr llvm::StringLiteral qirRtReadResult = "__quantum__rt__read_result";

/// Record a measurement result.
static constexpr llvm::StringLiteral qirRtResultRecordOutput = "__quantum__rt__result_record_output";

//===----------------------------------------------------------------------===//
// QIR quantum instruction set (QIS)
//===----------------------------------------------------------------------===//

// TODO: This is a hardcoded QIS, we need to query it from the device in the future.

/// Z-Basis measurement (irreversible).
static constexpr llvm::StringLiteral qirQisMZ = "__quantum__qis__mz__body";

/// Single target hadamard gate.
static constexpr llvm::StringLiteral qirQisH = "__quantum__qis__h__body";

/// Single target X gate.
static constexpr llvm::StringLiteral qirQisX = "__quantum__qis__x__body";

/// CX gate controlled on first qubit/ptr.
static constexpr llvm::StringLiteral qirQisCX = "__quantum__qis__cx__body";

} // namespace qcc
