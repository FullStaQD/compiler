#ifndef INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPDIALECT_TD_
#define INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// PrelimHLEP dialect
//===----------------------------------------------------------------------===//

def PrelimHLEPDialect : Dialect {
    let summary = "PRELIMINARY High-level entry point dialect.";
    let description = [{
        See issue: https://github.com/FullStaQD/compiler/issues/34
    }];

    let name = "prelimhlep";
    let cppNamespace = "::qcc::prelimhlep";
    let useDefaultTypePrinterParser = 1;
}


#endif // INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPDIALECT_TD_
