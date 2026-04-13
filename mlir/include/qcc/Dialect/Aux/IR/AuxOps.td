#ifndef INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_
#define INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_

// Copied from https://github.com/eclipse-qrisp/Qrisp/tree/main/src/qrisp/jasp/mlir/dialect_definition
// TODO: Agree on a single source of truth for the dialect definition.
// See also issue https://github.com/FullStaQD/compiler/issues/16

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "AuxDialect.td"

//===----------------------------------------------------------------------===//
// Jasp operations
//===----------------------------------------------------------------------===//

class AuxOp<string mnemonic, list<Trait> traits = []> :
        Op<AuxDialect, mnemonic, traits>;

// FIXME: implement
def RecordBoolOp : AuxOp<"record_bool"> {
    let summary = "Records a boolean value for output.";
    let description = "FIXME:";

    let arguments = (ins I1:$value); // FIXME: add label
    let results = (outs);

    let assemblyFormat = "$value attr-dict";
}

#endif // INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_
