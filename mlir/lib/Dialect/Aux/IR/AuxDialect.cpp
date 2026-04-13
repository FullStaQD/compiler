#include "qcc/Dialect/Aux/IR/Aux.h"

// FIXME: what includes needed?
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace qcc::aux;

#include "qcc/Dialect/Aux/IR/AuxDialect.cpp.inc"

#define GET_OP_CLASSES
#include "qcc/Dialect/Aux/IR/AuxOps.cpp.inc"

void AuxDialect::initialize() {
  addTypes<>();

  addOperations<
#define GET_OP_LIST
#include "qcc/Dialect/Aux/IR/AuxOps.cpp.inc"
      >();
}
