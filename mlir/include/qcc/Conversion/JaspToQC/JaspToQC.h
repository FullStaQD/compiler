

#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {
#define GEN_PASS_DECL_JASPTOQC
#include "mlir/Conversion/JaspToQC/JaspToQC.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/JaspToQC/JaspToQC.h.inc"
} // namespace mlir
