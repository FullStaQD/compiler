// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#include "Match.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace qcc {

// NOLINTBEGIN

bool constant_int_value_binder::match(Attribute attr) {
  attr_value_binder<IntegerAttr> matcher(bind_value);
  if (matcher.match(attr))
    return true;

  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr))
    return matcher.match(splatAttr.getSplatValue<Attribute>());

  return false;
}

bool constant_int_value_binder::match(Operation* op) {
  Attribute attr;
  if (!constant_op_binder<Attribute>(&attr).match(op))
    return false;

  Type type = op->getResult(0).getType();
  if (isa<IntegerType, IndexType, VectorType, RankedTensorType>(type))
    return match(attr);

  return false;
}

// NOLINTEND

} // namespace qcc
