// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCDIALECT_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCDIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// QCC dialect
//===----------------------------------------------------------------------===//

def QCCDialect : Dialect {
    let summary = "Dynamic quantum connectivity as a first-class IR value.";
    let description = [{
        Most quantum dialects model connectivity as a fixed coupling graph. On a
        trapped-ion QCCD machine the compiler moves the qubits, so the subsets
        it can act on change as it does. The `qcc` dialect represents that
        change, in four layers:

        | layer | object                                           | spelling                |
        |-------|--------------------------------------------------|-------------------------|
        | 0     | substrate: a typed hypergraph of sites and edges  | `#qcc.substrate` (attr) |
        | 1     | configuration: where the qubits sit               | `!qcc.config` (value)   |
        | 2     | interaction complex `K(P, C)`: what is executable | derived, never stored   |
        | 3     | rule set: one double-pushout span per rule        | the operations          |

        Layer 0 is static, hence an attribute, and carries the `transport`,
        `control`, `rigid` and `link` edges together with the algebraic
        properties that select a representation. Layer 1 is dynamic, hence an
        SSA value: an ordered partition, produced by `qcc.config.init` and
        rewritten by the Layer 3 operations.

        `InteractionComplex` derives Layer 2 from co-location in `C` and the
        `link` edges of `P`. `transport` and `rigid` edges generate the rule set
        instead, and `control` edges constrain concurrency rather than
        connectivity.

        Layer 3 is what differs between machines: `qcc.ion.*` for QCCD,
        `qcc.atom.*` for neutral atoms, `qcc.link.*` for modular targets. A
        superconducting device instantiates the empty rule set, and the dialect
        degenerates to a coupling-graph check.

        Gates are not part of this dialect. They remain in `qc`, and
        `--qcc-verify-connectivity` checks each against the `K` of its
        dominating configuration. A function containing no `!qcc.config` is left
        untouched, so the dialect is opt-in.

        Snapshot semantics follow from the framework: attribute uniquing makes a
        substrate immutable, and def-use chains make the connectivity in force
        at a program point the dominating definition. No versioning or
        invalidation machinery is needed.
    }];

    let name = "qcc";
    let cppNamespace = "::qcc::conn";

    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class QCCAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QCCDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class QCCType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QCCDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCDIALECT_TD_
