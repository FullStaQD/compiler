// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCOPS_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "qcc/Dialect/QCC/IR/QCCAttributes.td"
include "qcc/Dialect/QCC/IR/QCCDialect.td"
include "qcc/Dialect/QCC/IR/QCCInterfaces.td"
include "qcc/Dialect/QCC/IR/QCCTypes.td"

//===----------------------------------------------------------------------===//
// Side-effect resource
//===----------------------------------------------------------------------===//

/// Rewrite operations describe physical motion and must survive dead-code
/// elimination even when no later operation takes their result as an operand,
/// which under the dominating-definition discipline gates never do. They
/// therefore read and write a dedicated resource rather than being `Pure`. The
/// effect additionally keeps them ordered with respect to one another, which is
/// what a shuttle schedule requires.
def QCC_ConfigurationResource : Resource<"::qcc::conn::ConfigurationResource">;

class QCCOp<string mnemonic, list<Trait> traits = []>
    : Op<QCCDialect, mnemonic, traits>;

/// A Layer 3 rule: consumes one configuration and produces its successor,
/// keeping the chain live and ordered.
class QCC_RewriteOp<string mnemonic, list<Trait> traits = []>
    : QCCOp<mnemonic, !listconcat(traits, [
        AllTypesMatch<["in", "out"]>,
        MemoryEffects<[MemRead<QCC_ConfigurationResource>,
                       MemWrite<QCC_ConfigurationResource>]>,
        DeclareOpInterfaceMethods<QCC_TopologyRewriteOpInterface, ["applyRule", "getFootprint"]>])> {
  let results = (outs QCC_ConfigType:$out);
  let assemblyFormat = "$in attr-dict `:` qualified(type($out))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layer 1 — establishing a configuration
//===----------------------------------------------------------------------===//

def QCC_ConfigInitOp : QCCOp<"config.init", [
    MemoryEffects<[MemWrite<QCC_ConfigurationResource>]>]> {
  let summary = "Establish the initial configuration of a machine";
  let description = [{
    The root of a configuration chain, and the only operation that states a
    placement directly. Every downstream configuration is derived from it by
    applying rules, which lets `ConfigurationAnalysis` be a fold rather than a
    fixpoint computation.

    The device is carried by the result type rather than repeated here, so a
    placement is checked against exactly one substrate.

    Example:
    ```mlir
    %c0 = qcc.config.init(#qcc.placement<
        @trap_a = [[0, 1, 2, 3, 4]],
        @trap_b = [[5, 6, 7, 8, 9]],
        @j0     = []>) : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_PlacementAttr:$placement);
  let results = (outs QCC_ConfigType:$out);

  let assemblyFormat = "`(` qualified($placement) `)` attr-dict `:` qualified(type($out))";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    DeviceAttr getDeviceAttr() {
      return ::llvm::cast<ConfigType>(getOut().getType()).getDevice();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Layer 3 — the QCCD rule set
//===----------------------------------------------------------------------===//

def QCC_IonSplitOp : QCC_RewriteOp<"ion.split"> {
  let summary = "Detach ions from one end of a chain";
  let description = [{
    Splits `count` ions off the `end` of chain `chain` at `site`, leaving two
    chains in the same site. Nothing moves between sites: a split opens a second
    potential well in the same segment, so the detached part remains in the
    site's chain list, adjacent to the remainder and on the side it came from.

    Only the ends of a chain can be detached, which is why the operation takes
    an `end` attribute rather than an arbitrary index. On a device with
    `facetOrdering = none` a chain has no ends and the operation is rejected.

    Splitting reduces `K`: after detaching ion 0 from `[0 1 2]`, the set
    `{0, 1}` is no longer executable. A gate requiring it must be ordered before
    the split; `--qcc-verify-connectivity` reports the violation otherwise.

    Example:
    ```mlir
    // @trap_a: [0 1 2 3] -> [0] [1 2 3]
    %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                             end = #qcc.chain_end<head>, count = 1 : i64}
        : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$site,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$chain,
                       QCC_ChainEndAttr:$end,
                       ConfinedAttr<I64Attr, [IntPositive]>:$count);
}

def QCC_IonMergeOp : QCC_RewriteOp<"ion.merge"> {
  let summary = "Concatenate two chains held by the same site";
  let description = [{
    Merges chain `src` into chain `dst` at `site`, attaching it to the `at`
    end of `dst`'s word; `src` then leaves the site's chain list.

    The resulting order is physical state. Merging `[0]` onto the head of
    `[10 11 12]` gives `[0 10 11 12]`, in which ion 0 is detachable without a
    reorder; merging at the tail gives `[10 11 12 0]`, in which it is not. A
    set-valued encoding would not distinguish the two.

    Example:
    ```mlir
    // @trap_b: [10 11 12] [0] -> [0 10 11 12]
    %c4 = qcc.ion.merge %c3 {site = @trap_b, dst = 0 : i64, src = 1 : i64,
                             at = #qcc.chain_end<head>}
        : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$site,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$dst,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$src,
                       QCC_ChainEndAttr:$at);
}

def QCC_IonTransportOp : QCC_RewriteOp<"ion.transport"> {
  let summary = "Move a chain across a transport segment";
  let description = [{
    Moves chain `chain` out of site `from` and across the transport edge
    whose `id` property is `via`, appending it to the chain list of the edge's
    other endpoint.

    Transport edges are rank 2 because displacement is a binary relation: there
    is no physical operation moving an ion out of `{A, B, C}`. A junction is a
    site of high degree rather than an edge of high arity, so routing through
    one is two transports rather than a single ternary step. Routing over such a
    substrate is therefore an instance of pebble motion on a graph.

    A segment joins a trap at one of its ends. When the edge declares which
    (`ends = ["head", "tail"]`), only the chain at that end can leave, since a
    chain further in would have to pass through the ones in front of it. The
    same property positions arrivals, so an ion entering a trap through its tail
    end lands on the tail and can leave again by the same route. Omitting `ends`
    treats the segment as reaching either end, which suits a gate zone holding a
    single chain.

    The operation is rejected if no such edge exists, if `from` is not one of its
    endpoints, or if the edge declares `bidirectional = false` and the traversal
    runs against the declared direction. The end, capacity and `max_chain_len`
    constraints depend on the configuration and are checked when the rule is
    applied.

    Example:
    ```mlir
    %c2 = qcc.ion.transport %c1 {from = @trap_a, chain = 0 : i64, via = "seg_aj"}
        : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$from,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$chain,
                       StrAttr:$via);

  let extraClassDeclaration = [{
    /// The transport edge named by `via`, or a null attribute if the substrate
    /// does not declare one.
    HyperedgeAttr getEdge();
    /// The endpoint the chain arrives at, or the empty string if `getEdge`
    /// is null or `from` is not an endpoint.
    ::llvm::StringRef getDestination();
    /// The `latency_us` property of the traversed segment. The in-place rules
    /// report the interface default of zero.
    double getLatencyUs();
  }];
}

def QCC_IonJunctionTurnOp : QCC_RewriteOp<"ion.junction_turn"> {
  let summary = "Rotate a chain at a junction so it can leave on another segment";
  let description = [{
    Turns the chain `chain` at junction `site`, which arrived on segment `from`,
    so that it can depart on segment `to`.

    A turn is a distinct physical action rather than bookkeeping. Carrying a
    crystal straight through a junction and turning it are different operations:
    the turn rotates the crystal, costs substantially more than a hop, and on
    much hardware is possible only between certain pairs of segments. Modelling
    it as an operation exposes the cost through `getLatencyUs` and the legality
    through the verifier.

    The permitted turns are a property of the junction, declared as `turns` in
    the site's `props`, each entry an `"in>out"` pair of segment ids. A junction
    that declares none permits every turn, which is appropriate for a machine
    where the rotation is unrestricted.

    Example:
    ```mlir
    %c2 = qcc.ion.junction_turn %c1 {site = @j0, chain = 0 : i64,
                                     from = "seg_aj", to = "seg_jb"}
        : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$site,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$chain,
                       StrAttr:$from,
                       StrAttr:$to);

  let extraClassDeclaration = [{
    /// The junction's `turn_latency_us`. A rotation is slow relative to a hop,
    /// and is typically the dominant term in a route that crosses a grid.
    double getLatencyUs();
  }];
}

def QCC_IonReorderOp : QCC_RewriteOp<"ion.reorder"> {
  let summary = "Permute the ions within a chain";
  let description = [{
    Rewrites the word of chain `chain` at `site` by `permutation`, where
    `permutation[i]` is the index in the old word of the ion that ends up at
    position `i`.

    A reorder leaves `K` unchanged, since the chain holds the same qubits, but
    it changes which ions are at the ends and therefore which splits are legal
    next. It is what makes an unfavourable merge recoverable.

    Example:
    ```mlir
    // @trap_a: [10 11 0] -> [0 10 11]
    %c1 = qcc.ion.reorder %c0 {site = @trap_a, chain = 0 : i64,
                               permutation = array<i64: 2, 0, 1>}
        : !qcc.config<#ion>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$site,
                       ConfinedAttr<I64Attr, [IntNonNegative]>:$chain,
                       DenseI64ArrayAttr:$permutation);
}

//===----------------------------------------------------------------------===//
// Layer 3 — the neutral-atom rule set
//===----------------------------------------------------------------------===//

def QCC_AtomAodMoveOp : QCC_RewriteOp<"atom.aod_move"> {
  let summary = "Displace every atom on an AOD axis at once";
  let description = [{
    Moves the contents of each site on the rigid edge `via` to the corresponding
    site in `to`, all in one step.

    A rank-2 transport relation cannot express this rule. One AOD waveform
    displaces every atom on the axis simultaneously, so the move is a single
    physical action over k sites rather than k actions over pairs. This is why
    `rigid` is a hyperedge, and why the binary nature of displacement is a QCCD
    property rather than a general one. The cost of the move is independent of
    how many atoms the axis carries.

    The map must be non-crossing. AOD rows cannot pass through one another, so
    the destinations must appear in the substrate's site order in the same
    relative order as the sources; the substrate's declaration order is read as
    the geometric order along the axis.

    Example:
    ```mlir
    // every atom in row 0 slides down into row 1
    %c1 = qcc.atom.aod_move %c0 {via = "aod_row_0", to = [@t10, @t11, @t12]}
        : !qcc.config<#atoms>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       StrAttr:$via,
                       FlatSymbolRefArrayAttr:$to);

  let extraClassDeclaration = [{
    /// The rigid edge named by `via`, or a null attribute.
    HyperedgeAttr getEdge();
    /// The `latency_us` property of the rigid edge. One waveform drives the
    /// whole axis, so the cost does not depend on how many atoms it carries.
    double getLatencyUs();
  }];
}

def QCC_AtomSlmTransferOp : QCC_RewriteOp<"atom.slm_transfer"> {
  let summary = "Hand a single atom from one trap to another";
  let description = [{
    Moves the atom held by `from` into `to`: the tweezer handoff that picks one
    atom out of the static SLM pattern, carries it, and puts it back down. It is
    the single-atom counterpart of `atom.aod_move`.

    `from` and `to` must share a rigid edge, since the AOD can only carry an
    atom along an axis the atom is on, so the substrate determines how far a
    handoff can reach. Rearranging a defective array into a filled one is a
    sequence of these operations.

    Example:
    ```mlir
    %c1 = qcc.atom.slm_transfer %c0 {from = @t00, to = @t02} : !qcc.config<#atoms>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       FlatSymbolRefAttr:$from,
                       FlatSymbolRefAttr:$to);
}

//===----------------------------------------------------------------------===//
// Layer 3 — the modular / photonic rule set
//===----------------------------------------------------------------------===//

def QCC_LinkGenerateOp : QCC_RewriteOp<"link.generate"> {
  let summary = "Herald an entanglement link between two modules";
  let description = [{
    Attempts entanglement over the non-persistent link edge `via` and, on
    success, adds a live link joining `qubits` to the configuration. Operand `i`
    of `qubits` is the party in site `i` of the edge, so the operands are given
    in the edge's own order.

    This rule introduces state a placement alone cannot hold. Elsewhere `K` is a
    function of where the qubits are; here it is a function of where they are
    and of which links are currently live. A link exists because this operation
    ran rather than because two qubits are near one another, and it ceases to
    exist when it is consumed.

    A qubit holds at most one link at a time, so a party that is already
    entangled must consume that link before heralding another.

    Example:
    ```mlir
    %c1 = qcc.link.generate %c0 {via = "fibre_01", qubits = array<i64: 0, 4>}
        : !qcc.config<#photonic>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in,
                       StrAttr:$via,
                       DenseI64ArrayAttr:$qubits);

  let extraClassDeclaration = [{
    /// The link edge named by `via`, or a null attribute.
    HyperedgeAttr getEdge();
    /// The `latency_us` property of the link edge: the cost of one heralding
    /// attempt.
    double getLatencyUs();
  }];
}

def QCC_LinkConsumeOp : QCC_RewriteOp<"link.consume"> {
  let summary = "Spend a live entanglement link";
  let description = [{
    Removes one live link joining `qubits`. Teleporting a gate across a link
    consumes the link, and this operation records that.

    Links are linear resources: the same live link cannot be consumed twice, and
    once consumed the qubits it joined are no longer a facet of `K` until
    another link is heralded. A program that entangles across a fibre twice must
    generate twice, and the verifier reports the omission otherwise.

    Example:
    ```mlir
    %c2 = qcc.link.consume %c1 {qubits = array<i64: 0, 4>} : !qcc.config<#photonic>
    ```
  }];

  let arguments = (ins QCC_ConfigType:$in, DenseI64ArrayAttr:$qubits);
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCOPS_TD_
