#ifndef INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPTYPES_TD_
#define INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPTYPES_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "PrelimHLEPDialect.td"


//===----------------------------------------------------------------------===//
// PrelimHLEP types
//===----------------------------------------------------------------------===//

class PrelimHLEPType<string name, string typeMnemonic>
    : TypeDef<PrelimHLEPDialect, name> {
    let mnemonic = typeMnemonic;
}

def DummyType : PrelimHLEPType<"Dummy", "Dummy"> {
    let summary = "dummy type";
    let description = ".";
}

#endif // INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPTYPES_TD_
