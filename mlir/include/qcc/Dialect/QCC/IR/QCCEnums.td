// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCENUMS_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCENUMS_TD_

include "mlir/IR/EnumAttr.td"

include "qcc/Dialect/QCC/IR/QCCDialect.td"

//===----------------------------------------------------------------------===//
// Hyperedge kinds
//===----------------------------------------------------------------------===//

def QCC_EdgeKind : I32EnumAttr<"EdgeKind", "kind of a substrate hyperedge", [
    // Displacement of ions or atoms between sites. Always rank 2: displacement
    // is a binary relation, and a junction is a site of high degree rather than
    // an edge of high arity.
    I32EnumAttrCase<"Transport", 0, "transport">,
    // Sites sharing a control resource such as an AWG, a DAC, a laser path or a
    // decoder. Rank k, and not reducible to a clique of rank-2 edges, which
    // would lose which conflict originates from which resource. Constrains
    // scheduling only; does not contribute to K.
    I32EnumAttrCase<"Control", 1, "control">,
    // Sites that move as a unit, such as a neutral-atom AOD row. Rank k.
    // Contributes to both the rule set and concurrency.
    I32EnumAttrCase<"Rigid", 2, "rigid">,
    // An entanglement interconnect (Bell for rank 2, GHZ for rank 3 and above),
    // or a superconducting coupler when marked `persistent`. Contributes to K.
    I32EnumAttrCase<"Link", 3, "link">
  ]> {
  let cppNamespace = "::qcc::conn";
}

//===----------------------------------------------------------------------===//
// Site kinds
//===----------------------------------------------------------------------===//

def QCC_SiteKind : I32EnumAttr<"SiteKind", "kind of a substrate site", [
    I32EnumAttrCase<"Trap", 0, "trap">,
    I32EnumAttrCase<"Junction", 1, "junction">,
    I32EnumAttrCase<"GateZone", 2, "gatezone">,
    I32EnumAttrCase<"Module", 3, "module">,
    // A site that qubits can neither enter nor leave, such as a superconducting
    // qubit or a neutral-atom SLM trap. Its capacity is 1 and no transport edge
    // may be incident to it.
    I32EnumAttrCase<"Fixed", 4, "fixed">
  ]> {
  let cppNamespace = "::qcc::conn";
}

//===----------------------------------------------------------------------===//
// Facet ordering
//===----------------------------------------------------------------------===//

def QCC_FacetOrdering : I32EnumAttr<"FacetOrdering",
    "whether the qubits within a facet carry a physical order", [
    // Facets are sets: any member may be detached and the order carries no
    // meaning.
    I32EnumAttrCase<"None", 0, "none">,
    // Facets are words: the ions form a linear chain, and a split may only
    // detach at a chain end. The order is required to decide split legality.
    I32EnumAttrCase<"Linear", 1, "linear">,
    // Facets are planar crystals, such as a Penning-trap Coulomb crystal or a
    // chain past its zigzag transition. The members are jointly addressable and
    // arranged in two dimensions, so a word cannot describe them.
    //
    // Gates on such a site verify against K as they do anywhere else, since K
    // only queries group membership. The chain rewrite rules, however, are
    // defined in terms of chain ends, which a plane does not have; they are
    // rejected on a planar site rather than applied as though the site were
    // linear. Planar split and merge rules would correspond to operations no
    // platform currently performs, and can be added alongside the ion rules if
    // that changes.
    I32EnumAttrCase<"Planar", 2, "planar">
  ]> {
  let cppNamespace = "::qcc::conn";
}

//===----------------------------------------------------------------------===//
// Chain ends
//===----------------------------------------------------------------------===//

def QCC_ChainEnd : I32EnumAttr<"ChainEnd", "an end of an ordered chain", [
    I32EnumAttrCase<"Head", 0, "head">,
    I32EnumAttrCase<"Tail", 1, "tail">
  ]> {
  let cppNamespace = "::qcc::conn";
  let genSpecializedAttr = 0;
}

def QCC_ChainEndAttr : EnumAttr<QCCDialect, QCC_ChainEnd, "chain_end"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCENUMS_TD_
