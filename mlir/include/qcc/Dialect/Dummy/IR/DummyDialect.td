#ifndef INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYDIALECT_TD_
#define INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Dummy_Dialect : Dialect {
  let name = "dummy";
  let summary = "Basic dummy dialect.";
  let description = [{
    A dummy dialect to setup the project.
  }];
  let cppNamespace = "::qcc::dummy";

  let useDefaultTypePrinterParser = 1;
  let hasCanonicalizer = 0;
}

#endif // INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYDIALECT_TD_
