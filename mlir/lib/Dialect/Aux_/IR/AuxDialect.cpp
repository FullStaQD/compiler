#include "qcc/Dialect/Aux_/IR/Aux.h"

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
