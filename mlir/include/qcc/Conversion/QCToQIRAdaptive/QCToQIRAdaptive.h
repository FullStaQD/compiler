#pragma once

#include <mlir/Pass/Pass.h>

namespace qcc {

#define GEN_PASS_DECL
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

#define GEN_PASS_REGISTRATION
#include "qcc/Conversion/QCToQIRAdaptive/QCToQIRAdaptive.h.inc"

} // namespace qcc
