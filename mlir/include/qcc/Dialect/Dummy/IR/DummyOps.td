// ParityQC © 2025. See LICENSE.md in the top level directory for details.

#ifndef INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYOPS_TD_
#define INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYOPS_TD_

include "qcc/Dialect/Dummy/IR/DummyDialect.td"
include "qcc/Dialect/Dummy/IR/DummyTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Base class for all ops in the dummy dialect
///
/// All operations in this dialect have value semantics and have no side effects (are pure).
class Dummy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dummy_Dialect, mnemonic, traits#[Pure]>;

/// A gate (in our sense) is a dummy operation with the same number of in and output qubits.
class Dummy_GateOp<string mnemonic, list<Trait> traits = []>
    : Dummy_Op<mnemonic, traits>;


def Dummy_XOp : Dummy_GateOp<"x", []> {
  let summary = "Represents a Pauli-X gate.";

  let arguments = (ins Qubit:$in_qubit);
  let results = (outs Qubit:$out_qubit);
  let assemblyFormat = [{
    attr-dict $in_qubit `:` type($in_qubit)
  }];
}

#endif // INCLUDE_QCC_DIALECT_DUMMY_IR_DUMMYOPS_TD_
