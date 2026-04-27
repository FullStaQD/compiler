#pragma once

#include <llvm/ADT/StringRef.h>

namespace qcc {

/// A unit attribute to mark a `func.func` as the starting point of a quantum program.
static constexpr llvm::StringLiteral entryPointAttrName = "qcc.entry_point";

} // namespace qcc
