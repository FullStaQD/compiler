// AuxDialect.cpp
//
// Copyright (c) 2026 FullStaQD Project
// All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "qcc/Dialect/Aux_/IR/Aux_.h"

using namespace mlir;
using namespace qcc::aux;

#include "qcc/Dialect/Aux_/IR/AuxDialect.cpp.inc"

#define GET_OP_CLASSES
#include "qcc/Dialect/Aux_/IR/AuxOps.cpp.inc"

void AuxDialect::initialize() {
  addTypes<>();

  addOperations<
#define GET_OP_LIST
#include "qcc/Dialect/Aux_/IR/AuxOps.cpp.inc"
      >();
}
