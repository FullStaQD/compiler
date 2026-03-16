#include "qcc/Dialect/Dummy/IR/Dummy.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace qcc::dummy;

#include "qcc/Dialect/Dummy/IR/DummyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qcc/Dialect/Dummy/IR/DummyTypes.cpp.inc"

#define GET_OP_CLASSES
#include "qcc/Dialect/Dummy/IR/DummyOps.cpp.inc"

void DummyDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "qcc/Dialect/Dummy/IR/DummyTypes.cpp.inc"
    >();

    addOperations<
#define GET_OP_LIST
#include "qcc/Dialect/Dummy/IR/DummyOps.cpp.inc"
    >();
}