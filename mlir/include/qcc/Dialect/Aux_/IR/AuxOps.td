#ifndef INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_
#define INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "AuxDialect.td"

//===----------------------------------------------------------------------===//
// Aux operations
//===----------------------------------------------------------------------===//

class AuxOp<string mnemonic, list<Trait> traits = []> :
        Op<AuxDialect, mnemonic, traits>;

def RecordBoolOp : AuxOp<"record_bool"> {
    let summary = "Records a boolean value.";

    let description = [{
        This corresponds to the following QIR runtime function:

        ```llvm
        llvm.func @__quantum__rt__bool_record_output(i1, !llvm.ptr)
        ```

        Our principal dialects have nothing in it which can be translated to it. Hence we use this as a workaround.
        It is used like this:

        ```mlir
        func.func @main() -> i64 attributes { qcc.entry_point } {
          ...
          aux.record_bool %m7
          ...
        }
        ```

        To actually translate it to the above QIR runtime function we use a dummy label for the second argument of
        the QIR function.
    }];

    let arguments = (ins I1:$value);
    let results = (outs);

    let assemblyFormat = "$value attr-dict";
}

#endif // INCLUDE_QCC_DIALECT_AUX_IR_AUXOPS_TD_
