// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCTYPES_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

include "qcc/Dialect/QCC/IR/QCCAttributes.td"
include "qcc/Dialect/QCC/IR/QCCDialect.td"

//===----------------------------------------------------------------------===//
// Configuration type
//===----------------------------------------------------------------------===//

def QCC_ConfigType : QCCType<"Config", "config", [OpAsmTypeInterface]> {
  let summary = "The connectivity configuration of a machine at a program point";
  let description = [{
    The configuration is threaded through the IR as an SSA value rather than
    held in a pass-managed side table, which lets existing framework machinery
    supply what the model needs:

    | requirement           | mechanism                                          |
    |-----------------------|----------------------------------------------------|
    | version tracking      | def-use chains; connectivity at a point is the dominating def |
    | reordering legality   | dependence analysis on the configuration chain     |
    | snapshot immutability | attribute uniquing                                 |
    | cost propagation      | `TopologyRewriteOpInterface::getLatencyUs`         |

    The type is parameterised by the device, which is static. The configuration
    itself is not part of the type; `ConfigurationAnalysis` recovers it from the
    definition chain, starting at a `qcc.config.init` and folding each rewrite
    along the way. Two values of the same type therefore describe two different
    machine states, in the same way that two `!qc.qubit` values describe two
    different qubits.

    Example:
    ```mlir
    %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1]]>) : !qcc.config<#ion>
    ```
  }];

  let parameters = (ins "::qcc::conn::DeviceAttr":$device);
  let assemblyFormat = "`<` qualified($device) `>`";

  let extraClassDeclaration = [{
    /// The substrate the configuration is typed over.
    SubstrateAttr getSubstrate() const { return getDevice().getSubstrate(); }

    //===------------------------------------------------------------------===//
    // OpAsmTypeInterface
    //===------------------------------------------------------------------===//

    /// Prints as `!config` rather than spelling the full machine description
    /// inline on every value in the chain. See `DeviceAttr::getAlias`.
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream& os) const {
      os << "config";
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCTYPES_TD_
