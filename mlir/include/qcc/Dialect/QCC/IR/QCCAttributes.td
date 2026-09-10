// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_IR_QCCATTRIBUTES_TD_
#define INCLUDE_QCC_DIALECT_QCC_IR_QCCATTRIBUTES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "qcc/Dialect/QCC/IR/QCCDialect.td"
include "qcc/Dialect/QCC/IR/QCCEnums.td"

//===----------------------------------------------------------------------===//
// Parameter helpers
//===----------------------------------------------------------------------===//

/// A plain `bool` parameter that reads and writes as `true` / `false` rather
/// than as `1` / `0`, so that a device attribute stays legible.
def QCC_BoolParam : AttrParameter<"bool", "a boolean flag"> {
  let parser = [{[&]() -> ::mlir::FailureOr<bool> {
    if (::mlir::succeeded($_parser.parseOptionalKeyword("true")))
      return true;
    if (::mlir::succeeded($_parser.parseOptionalKeyword("false")))
      return false;
    return ::mlir::FailureOr<bool>((::llvm::LogicalResult)(
        $_parser.emitError($_parser.getCurrentLocation())
        << "expected 'true' or 'false'"));
  }()}];
  let printer = [{$_printer << ($_self ? "true" : "false")}];
}

/// An optional `FacetOrdering`, spelled as a bare keyword. MLIR's stock
/// `FieldParser` covers optional attributes and optional integers but not
/// optional enums, so this supplies the two functions the generated format
/// needs.
def QCC_OptionalFacetOrdering
    : AttrParameter<"std::optional<::qcc::conn::FacetOrdering>", "an optional facet ordering"> {
  let defaultValue = "std::optional<::qcc::conn::FacetOrdering>()";
  let parser = [{[&]() -> ::mlir::FailureOr<std::optional<::qcc::conn::FacetOrdering>> {
    const auto loc = $_parser.getCurrentLocation();
    std::string keyword;
    if (::mlir::failed($_parser.parseKeywordOrString(&keyword)))
      return ::mlir::failure();
    const auto parsed = ::qcc::conn::symbolizeFacetOrdering(keyword);
    if (!parsed)
      return ::mlir::FailureOr<std::optional<::qcc::conn::FacetOrdering>>(
          (::llvm::LogicalResult)($_parser.emitError(loc)
              << "expected 'none', 'linear' or 'planar'"));
    return std::optional<::qcc::conn::FacetOrdering>(*parsed);
  }()}];
  let printer = [{$_printer << ::qcc::conn::stringifyFacetOrdering(*$_self)}];
}

//===----------------------------------------------------------------------===//
// Layer 0 — the static substrate
//===----------------------------------------------------------------------===//

def QCC_SiteAttr : QCCAttr<"Site", "site"> {
  let summary = "A node of the substrate hypergraph";
  let description = [{
    A place where qubits can be held. `capacity` is the transient maximum rather
    than the nominal load: during a merge a trap briefly holds every ion it will
    hold afterwards plus the arriving chain, and a junction must hold a chain in
    flight. A `fixed` site has capacity 1 and no transport edge may be incident
    to it, which is what distinguishes a superconducting qubit from a trap.

    `maxFacetRank` gives the width of the widest entangling operation this site
    can host, overriding the device default. Addressability is a property of the
    site as well as of the machine: on a QCCD device a storage trap has no gate
    lasers, so co-location there means only that the ions share a motional mode.
    A rank below 2 states that no group held at this site contributes a facet,
    and hence that the qubits must be shuttled elsewhere before they can be
    acted on. Omitting the field makes the site inherit the device's
    `maxFacetRank`, which is the appropriate default for a long-chain machine
    whose trap is also its gate zone.

    `facetOrdering` overrides the device default in the same way, so that one
    machine may hold linear storage chains alongside an unordered gate zone or a
    planar crystal. `coords` gives the site a position, which a router uses to
    prefer a near zone over a far one and from which blockade discs are derived.

    `props` carries the properties only a junction needs. Recognised keys:

    | key                | meaning                                            |
    |--------------------|----------------------------------------------------|
    | `turns`            | permitted turns, as `"in>out"` segment-id pairs     |
    | `turn_latency_us`  | the cost of one rotation                           |

    A junction that declares no `turns` permits every turn. Declaring them
    restricts which outgoing segment a chain may take given the segment it
    arrived on, in the manner of a turn-restriction table: the restriction is a
    property of the node rather than of a ternary edge.

    Example:
    ```mlir
    #qcc.site<name = @j0, kind = junction, capacity = 4,
              props = {turns = ["seg_aj>seg_jb", "seg_jb>seg_aj"],
                       turn_latency_us = 4.000000e+02}>
    ```

    Example:
    ```mlir
    #qcc.site<name = @storage, kind = trap,     capacity = 30, maxFacetRank = 0>
    #qcc.site<name = @gate,    kind = gatezone, capacity = 2>
    ```
  }];

  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name,
                        EnumParameter<QCC_SiteKind>:$kind,
                        "uint64_t":$capacity,
                        OptionalParameter<"std::optional<uint64_t>">:$maxFacetRank,
                        QCC_OptionalFacetOrdering:$facetOrdering,
                        OptionalParameter<"::mlir::DenseF64ArrayAttr">:$coords,
                        OptionalParameter<"::mlir::DictionaryAttr">:$props);

  // Every field is keyed. Three independently optional tails all beginning with
  // a comma are not distinguishable by the parser; `struct` resolves this and
  // prints only the fields that are set.
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The bare site name, without the leading `@`.
    ::llvm::StringRef getSymName() const { return getName().getValue(); }
    /// How wide an entangling operation this site can host, falling back to the
    /// device's default when the site does not override it.
    uint64_t getEffectiveFacetRank(uint64_t deviceDefault) const {
      return getMaxFacetRank().value_or(deviceDefault);
    }
    /// How the qubits in a group held here are arranged, falling back to the
    /// device default. Overridable per site so that one machine can hold linear
    /// storage chains alongside an unordered gate zone.
    FacetOrdering getEffectiveFacetOrdering(FacetOrdering deviceDefault) const {
      return getFacetOrdering().value_or(deviceDefault);
    }
    /// Whether a chain arriving on segment `from` may depart on segment `to`.
    /// A site that declares no turn table permits every turn.
    bool permitsTurn(::llvm::StringRef from, ::llvm::StringRef to) const;
    /// What a rotation costs here, in microseconds. Zero when unstated.
    double getTurnLatencyUs() const;
    /// This site's position, when the description gives one.
    ::std::optional<::std::array<double, 3>> getPosition() const;
    /// Euclidean distance to `other`, when both sites carry a position. Read by
    /// the router to prefer a near zone over a far one.
    ::std::optional<double> distanceTo(SiteAttr other) const;
  }];
}

def QCC_HyperedgeAttr : QCCAttr<"Hyperedge", "edge"> {
  let summary = "A hyperedge of the substrate hypergraph";
  let description = [{
    The substrate is a hypergraph rather than a graph. `transport` is rank 2 on
    QCCD hardware, but the other three relations are not reducible to rank 2:
    expanding them into cliques loses information. If `{A, B, C, D}` share one
    AWG, the pairwise-conflict graph is a 4-clique; so is the case where `A-B`
    share one AWG, `C-D` share another, and the cross pairs conflict for an
    unrelated reason. The two cases have the same clique and different
    scheduling semantics.

    `props` carries the device-native metadata. Recognised keys:

    | key               | kind        | meaning                                   |
    |-------------------|-------------|-------------------------------------------|
    | `id`              | `transport` | name used by `qcc.ion.transport`'s `via`  |
    | `latency_us`      | any         | cost in microseconds, read by `getLatency`|
    | `bidirectional`   | `transport` | traversable in both directions            |
    | `max_chain_len` | `transport` | longest chain the segment can carry     |
    | `ends`            | `transport` | which end of each endpoint it meets       |
    | `persistent`      | `link`      | not consumed by use (a coupler, not a Bell pair) |
    | `axis`            | `rigid`     | which AOD axis the group travels along    |

    `ends` records the ends of the traps the segment reaches. A trap holds a
    line of ions and a segment joins it at one end, so only the chain at that
    end can leave; anything further in would have to pass through the ions in
    front of it. Give one entry per incident site, in the edge's own order:

    ```mlir
    #qcc.edge<transport, [@trap_1, @trap_2],
              {id = "seg", ends = ["head", "tail"], max_chain_len = 1 : i64}>
    ```

    This declares that the segment meets `@trap_1` at its head and `@trap_2` at
    its tail. An ion may therefore leave `@trap_1` only from the head and
    `@trap_2` only from the tail, and an ion arriving at `@trap_2` lands on its
    tail. Omitting the property treats the segment as reaching either end, which
    is the appropriate default for a gate zone holding a single chain.

    `id` is required on `transport`, `rigid` and `link` edges, since it is how a
    rule addresses an edge, and it is unique across the substrate. `persistent`
    is required on a `link` edge: it determines whether the edge contributes to
    `K` on its own or only once a link has been generated over it, and has no
    safe default.

    Example:
    ```mlir
    #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", latency_us = 8.0e+01}>
    ```
  }];

  let parameters = (ins EnumParameter<QCC_EdgeKind>:$kind,
                        ArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$sites,
                        "::mlir::DictionaryAttr":$props);

  let assemblyFormat = "`<` $kind `,` `[` $sites `]` `,` $props `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The `id` property, or the empty string if absent.
    ::llvm::StringRef getId() const;
    /// Which end of `site` this segment meets, when the edge declares its
    /// doors. Absent means the segment reaches either end.
    ::std::optional<ChainEnd> getEndAt(::llvm::StringRef site) const;
    /// Whether a `link` edge is a permanent coupler rather than a resource
    /// consumed by use. Meaningless on any other kind.
    bool isPersistent() const;
    /// The `latency_us` property, or 0.0 if absent.
    double getLatencyUs() const;
    /// Whether `key` is present and set to `true`.
    bool hasFlag(::llvm::StringRef key) const;
    /// The value of integer property `key`, if present.
    ::std::optional<int64_t> getIntProp(::llvm::StringRef key) const;
    /// Whether `site` is one of the incident sites.
    bool contains(::llvm::StringRef site) const;
  }];
}

def QCC_SubstrateAttr : QCCAttr<"Substrate", "substrate", [OpAsmAttrInterface]> {
  let summary = "The static machine description: a typed hypergraph";
  let description = [{
    Layer 0. The substrate is the type graph the configuration is typed over,
    and is not part of the rewritten object: rules are typed over it rather than
    carrying it as preserved context. Modelling it as preserved context would
    make every rewrite rule carry the whole machine description and would admit
    ill-formed rules that mutate the hardware.

    Example:
    ```mlir
    #qcc.substrate<
      sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 20>,
               #qcc.site<name = @j0, kind = junction, capacity = 10>],
      edges = [#qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj"}>]>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"::qcc::conn::SiteAttr">:$sites,
                        ArrayRefParameter<"::qcc::conn::HyperedgeAttr">:$edges);

  // Hand-written: a substrate may legitimately have no edges, as is the case
  // for a machine of isolated qubits, and the declarative list syntax cannot
  // parse an empty `[]`.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Index of the site named `name`, if it is declared.
    ::std::optional<unsigned> findSite(::llvm::StringRef name) const;
    /// The site named `name`, or a null attribute.
    SiteAttr lookupSite(::llvm::StringRef name) const;
    /// The edge of kind `kind` whose `id` property is `id`, or a null
    /// attribute. Ids are unique across the whole substrate, so the kind is a
    /// statement of what the caller expects to find rather than a disambiguator.
    HyperedgeAttr lookupEdge(EdgeKind kind, ::llvm::StringRef id) const;
    /// The transport edge whose `id` property is `id`, or a null attribute.
    HyperedgeAttr lookupTransportEdge(::llvm::StringRef id) const {
      return lookupEdge(EdgeKind::Transport, id);
    }
    /// All edges of kind `kind`.
    ::llvm::SmallVector<HyperedgeAttr> getEdgesOfKind(EdgeKind kind) const;

    //===------------------------------------------------------------------===//
    // OpAsmAttrInterface
    //===------------------------------------------------------------------===//

    /// A machine description is large and appears on the type of every value in
    /// a configuration chain. Aliasing it keeps the printed IR focused on what
    /// varies between operations, namely which rule ran and in what order.
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream& os) const {
      os << "substrate";
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}

def QCC_DeviceAttr : QCCAttr<"Device", "device", [OpAsmAttrInterface]> {
  let summary = "A substrate plus the algebraic properties that select a representation";
  let description = [{
    Two booleans and a maximum facet rank characterise the device families the
    dialect targets. Each affects how `K` must be stored and checked:

    | device                                | rank | downward closed | partitioning | representation     |
    |---------------------------------------|------|-----------------|--------------|--------------------|
    | superconducting                       | 2    | yes             | no           | static             |
    | QCCD gate zone                        | 2    | yes             | yes          | ordered-partition  |
    | long-chain MS, individual addressing  | k    | yes             | yes          | ordered-partition  |
    | global MS only                        | k    | **no**          | yes          | ordered-partition  |
    | neutral-atom Rydberg blockade         | 2..k | yes             | no           | injection          |
    | modular + photonic links              | 2    | yes             | **no**       | explicit-incidence |

    The representation is derived rather than declared, so the last column
    follows from the other three together with the substrate; a description
    states physics and the model follows from it. A machine that declares no
    transport, rigid or consumable-link edge has an empty rule set whatever else
    it says, so it is `static` even when its facets would otherwise partition.

    `downwardClosed = false` rules out representing `K` by its facets alone: a
    machine offering only a global MS gate can act on a whole chain but not on a
    subset of it, so `{0, 1}` is not executable even though `{0, 1, 2}` is.

    `partitioning = true` asserts the converse for the other axis: the facets of
    `K` are a partition of the qubits, so a configuration is an ordered
    partition with no loss of information and membership in `K` is a lookup
    rather than a search. Together these properties classify the device, which
    `getRepresentation` reports and `InteractionComplex` acts on.

    Example:
    ```mlir
    #qcc.device<substrate = #ion_substrate, maxFacetRank = 10,
                downwardClosed = true, partitioning = true,
                facetOrdering = linear>
    ```
  }];

  let parameters = (ins "::qcc::conn::SubstrateAttr":$substrate,
                        "uint64_t":$maxFacetRank,
                        QCC_BoolParam:$downwardClosed,
                        QCC_BoolParam:$partitioning,
                        EnumParameter<QCC_FacetOrdering>:$facetOrdering);

  // `qualified` throughout: the elided short form omits the dialect prefixes
  // that identify each nested attribute, which a machine description needs.
  let assemblyFormat = [{
    `<` `substrate` `=` qualified($substrate) `,`
        `maxFacetRank` `=` $maxFacetRank `,`
        `downwardClosed` `=` $downwardClosed `,`
        `partitioning` `=` $partitioning `,`
        `facetOrdering` `=` $facetOrdering `>`
  }];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Whether the device admits any rewrite rule. A device with neither a
    /// transport nor a rigid edge instantiates the empty rule set, so its
    /// connectivity is static and every `!qcc.config` use resolves to the value
    /// produced by `qcc.config.init`.
    bool isStatic() const;

    /// How a configuration over this device is represented, derived from the
    /// properties above. See `Representation` for what each case means and
    /// `InteractionComplex` for what it buys.
    Representation getRepresentation() const;

    //===------------------------------------------------------------------===//
    // OpAsmAttrInterface
    //===------------------------------------------------------------------===//

    /// A machine description is large and appears on the type of every value in
    /// a configuration chain. Aliasing it keeps the printed IR focused on what
    /// varies between operations, namely which rule ran and in what order.
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream& os) const {
      os << "device";
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Layer 1 — the configuration snapshot
//===----------------------------------------------------------------------===//

def QCC_PlacementAttr : QCCAttr<"Placement", "placement"> {
  let summary = "An ordered partition of the qubits over the sites";
  let description = [{
    A frozen Layer 1 configuration. A site holds an ordered list of groups, and
    a group is a word over the qubit alphabet rather than a set. On an ion
    machine a group is a chain; splitting one produces a second potential well
    in the same segment without moving anything, which is why a site holds a
    list of groups rather than a single word. The order within a group is
    physical state, since it determines which ions are detachable next, and a
    set-valued encoding would leave `qcc.ion.split` legality undecidable.

    A placement is a partial injection: a qubit appears at most once, and qubits
    absent from it are not yet placed.

    Example:
    ```mlir
    #qcc.placement<@trap_a = [[0, 1, 2]], @trap_b = [[3], [4, 5]], @j0 = []>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$sites,
                        ArrayRefParameter<"::mlir::ArrayAttr">:$groups);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Materialise the plain, mutable form the rewrite rules operate on.
    Configuration toConfiguration() const;
    /// Freeze a plain configuration back into a uniqued attribute.
    static PlacementAttr get(::mlir::MLIRContext *ctx, const Configuration &config);
  }];
  let skipDefaultBuilders = 0;
}

#endif // INCLUDE_QCC_DIALECT_QCC_IR_QCCATTRIBUTES_TD_
