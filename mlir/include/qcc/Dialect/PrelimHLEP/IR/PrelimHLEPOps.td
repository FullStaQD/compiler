#ifndef INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPOPS_TD_
#define INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "PrelimHLEPDialect.td"
include "PrelimHLEPTypes.td"


//===----------------------------------------------------------------------===//
// PrelimHLEP operations
//===----------------------------------------------------------------------===//

class PrelimHLEPOp<string mnemonic, list<Trait> traits = []> :
        Op<PrelimHLEPDialect, mnemonic, traits>;

def DummyOp : PrelimHLEPOp<"dummy_op"> {
    let summary = "dummy op.";
    let description = ".";

    let arguments = (ins DummyType:$dt);
    let results = (outs DummyType:$dt_out);

    let assemblyFormat = "$dt attr-dict  `:`  type($dt) `->` type($dt_out)";
}


#endif // INCLUDE_QCC_DIALECT_PRELIMHLEP_IR_PRELIMHLEPOPS_TD_
