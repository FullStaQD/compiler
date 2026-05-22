// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

// FIXME: check if this stuff is builtin.
// FIXME: acknowledge that we took this code from Enzyme-JAX

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"

#include <type_traits>

namespace qcc {

// FIXME: fix lint errors
// NOLINTBEGIN

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <typename AttrClass,
          // Require AttrClass to be a derived class from Attribute and get its
          // value type
          typename ValueType =
              typename std::enable_if_t<std::is_base_of<mlir::Attribute, AttrClass>::value, AttrClass>::ValueType,
          // Require the ValueType is not void
          typename = std::enable_if_t<!std::is_void<ValueType>::value>>
struct attr_value_binder {
  ValueType* bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType* bv) : bind_value(bv) {}

  bool match(mlir::Attribute attr) {
    if (auto intAttr = llvm::dyn_cast<AttrClass>(attr)) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches operations that have the `ConstantLike` trait, and
/// binds the folded attribute value.
template <typename AttrT> struct constant_op_binder {
  AttrT* bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT* bind_value) : bind_value(bind_value) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  constant_op_binder() : bind_value(nullptr) {}

  bool match(mlir::Operation* op) {
    if (!op->hasTrait<mlir::OpTrait::ConstantLike>())
      return false;

    // Fold the constant to an attribute.
    llvm::SmallVector<mlir::OpFoldResult, 1> foldedOp;
    mlir::LogicalResult result = op->fold(/*operands=*/{}, foldedOp);
    (void)result;
    assert(mlir::succeeded(result) && "expected ConstantLike op to be foldable");

    if (auto attr = llvm::dyn_cast<AttrT>(mlir::cast<mlir::Attribute>(foldedOp.front()))) {
      if (bind_value)
        *bind_value = attr;
      return true;
    }
    return false;
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer Attribute or Operation and binds the constant integer value.
struct constant_int_value_binder {
  mlir::IntegerAttr::ValueType* bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_value_binder(mlir::IntegerAttr::ValueType* bv) : bind_value(bv) {}

  bool match(mlir::Attribute attr) {
    attr_value_binder<mlir::IntegerAttr> matcher(bind_value);
    if (matcher.match(attr))
      return true;

    if (auto splatAttr = mlir::dyn_cast<mlir::SplatElementsAttr>(attr))
      return matcher.match(splatAttr.getSplatValue<mlir::Attribute>());

    return false;
  }

  bool match(mlir::Operation* op) {
    mlir::Attribute attr;
    if (!constant_op_binder<mlir::Attribute>(&attr).match(op))
      return false;

    mlir::Type type = op->getResult(0).getType();
    if (mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::VectorType, mlir::RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline constant_int_value_binder m_ConstantInt(mlir::IntegerAttr::ValueType* bind_value) {
  return constant_int_value_binder(bind_value);
}

// NOLINTEND

} // namespace qcc
