#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "qcc/Dialect/Jasp/IR/Jasp.h"

using namespace mlir;
using namespace jasp;

#include "qcc/Dialect/Jasp/IR/JaspDialect.cpp.inc"

#define GET_OP_CLASSES
#include "qcc/Dialect/Jasp/IR/JaspOps.cpp.inc"

void JaspDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "qcc/Dialect/Jasp/IR/JaspTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "qcc/Dialect/Jasp/IR/JaspOps.cpp.inc"
      >();
}
