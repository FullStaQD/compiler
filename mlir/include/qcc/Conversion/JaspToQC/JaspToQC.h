#pragma once

#include <mlir/Pass/Pass.h>

namespace qcc {
#define GEN_PASS_DECL_JASPTOQC
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

#define GEN_PASS_REGISTRATION
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"
} // namespace qcc
