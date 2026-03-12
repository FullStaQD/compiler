#ifndef INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYTYPES_TD_
#define INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYTYPES_TD_

include "qcc/Dialect/Dummy/IR/DummyDialect.td"

include "mlir/IR/AttrTypeBase.td"

/// Base class for all types in the dummy dialect.
class Dummy_Type<string name, string typeMnemonic> : TypeDef<Dummy_Dialect, name> {
  let mnemonic = typeMnemonic;
}

/// Dummy operations act on Qubits as their "primary" arguments.
def Qubit : Dummy_Type<"Qubit", "qubit"> {

  let summary = "Represents a single qubit.";
}

#endif // INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYTYPES_TD_
