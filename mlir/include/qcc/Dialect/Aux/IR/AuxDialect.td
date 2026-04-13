#ifndef INCLUDE_QCC_DIALECT_AUX_IR_AUXDIALECT_TD_
#define INCLUDE_QCC_DIALECT_AUX_IR_AUXDIALECT_TD_

// Copied from https://github.com/eclipse-qrisp/Qrisp/tree/main/src/qrisp/jasp/mlir/dialect_definition
// TODO: Agree on a single source of truth for the dialect definition.
// See also issue https://github.com/FullStaQD/compiler/issues/16

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Aux dialect
//===----------------------------------------------------------------------===//

def AuxDialect : Dialect {
    let summary = "Auxiliary dialect to contain workarounds.";
    let description = [{
        The purpose of the dialect is to enable us moving forward if some
        feature needs some functionality we cannot provide at the moment.

        Please use this dialect only if truly necessary and add a TODO or issue
        at the respective usage to find a better solution.
    }];

    let name = "aux";
    let cppNamespace = "::qcc::aux";
    let useDefaultTypePrinterParser = 0; // set to 1 once custom types are added.
}

#endif // INCLUDE_QCC_DIALECT_AUX_IR_AUXDIALECT_TD_
