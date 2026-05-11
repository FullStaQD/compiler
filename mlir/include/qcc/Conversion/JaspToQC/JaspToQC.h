// JaspToQC.h
//
// Copyright (c) 2026 FullStaQD Project
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

namespace qcc {
#define GEN_PASS_DECL_JASPTOQC
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"

#define GEN_PASS_REGISTRATION
#include "qcc/Conversion/JaspToQC/JaspToQC.h.inc"
} // namespace qcc
