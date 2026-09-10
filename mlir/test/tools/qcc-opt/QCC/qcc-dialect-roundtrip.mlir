// RUN: qcc-opt %s | qcc-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Layer 0: a QCCD substrate of two traps joined through a junction.
//
// `capacity` is the transient maximum rather than the nominal load: during a
// merge a trap briefly holds every ion it will hold afterwards plus the arriving
// chain.
//===----------------------------------------------------------------------===//

#ion_substrate = #qcc.substrate<
  sites = [
    #qcc.site<name = @trap_a, kind = trap, capacity = 20>,
    #qcc.site<name = @trap_b, kind = trap, capacity = 20>,
    #qcc.site<name = @j0, kind = junction, capacity = 10>
  ],
  edges = [
    // transport: rank 2. Displacement is binary, and the junction here is a site
    // of degree 2 rather than an edge of arity 2.
    #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 10 : i64}>,
    #qcc.edge<transport, [@j0, @trap_b], {id = "seg_jb", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 10 : i64}>,
    // control: rank 3. One AWG drives every shuttling electrode, so no two
    // transports may overlap in time even on disjoint segments. Expanding this
    // into a clique would lose which conflict comes from which resource.
    #qcc.edge<control, [@trap_a, @j0, @trap_b], {id = "awg_shuttle"}>,
    // control: rank 2. A single MS laser switched between the two traps.
    #qcc.edge<control, [@trap_a, @trap_b], {id = "laser_ms"}>
  ]>

#ion = #qcc.device<substrate = #ion_substrate,
                   maxFacetRank = 10,
                   downwardClosed = true,
                   partitioning = true,
                   facetOrdering = linear>

// The whole machine description survives the round trip, including hyperedge
// ranks and properties. It prints through an alias, since a substrate repeated on
// every value of a configuration chain is unreadable inline.
// CHECK-DAG: #[[$SUB:.+]] = #qcc.substrate<sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 20>, #qcc.site<name = @trap_b, kind = trap, capacity = 20>, #qcc.site<name = @j0, kind = junction, capacity = 10>]
// CHECK-DAG: #qcc.edge<transport, [@trap_a, @j0], {bidirectional = true, id = "seg_aj", latency_us = 8.000000e+01 : f64, max_chain_len = 10 : i64}>
// CHECK-DAG: #qcc.edge<control, [@trap_a, @j0, @trap_b], {id = "awg_shuttle"}>
// CHECK-DAG: #[[$ION:.+]] = #qcc.device<substrate = #[[$SUB]], maxFacetRank = 10, downwardClosed = true, partitioning = true, facetOrdering = linear>
// CHECK-DAG: !config = !qcc.config<#[[$ION]]>

// A site with no overrides prints only its required fields.
// CHECK-DAG: #qcc.site<name = @store, kind = trap, capacity = 8, coords = [0.000000e+00, 0.000000e+00]>
// Every override survives, in the order the struct declares them.
// CHECK-DAG: #qcc.site<name = @plane, kind = trap, capacity = 8, maxFacetRank = 4, facetOrdering = planar, coords = [3.000000e+00, 1.000000e+00, 2.000000e+00]>
// CHECK-DAG: #qcc.site<name = @j, kind = junction, capacity = 2, props = {turn_latency_us = 4.000000e+02 : f64, turns = ["seg_s>seg_p", "seg_p>seg_s"]}>

// CHECK-LABEL: func.func @shuttle
func.func @shuttle() {
  // CHECK: %[[C0:.+]] = qcc.config.init(#qcc.placement<@trap_a = {{\[\[}}0, 1, 2, 3, 4]], @trap_b = {{\[\[}}5, 6, 7, 8, 9]], @j0 = []>) : !config
  %c0 = qcc.config.init(#qcc.placement<
      @trap_a = [[0, 1, 2, 3, 4]],
      @trap_b = [[5, 6, 7, 8, 9]],
      @j0     = []>) : !qcc.config<#ion>

  // Detach ion 0 from the head of trap_a's chain: [0 1 2 3 4] -> [0] [1 2 3 4].
  // CHECK: %[[C1:.+]] = qcc.ion.split %[[C0]] {chain = 0 : i64, count = 1 : i64, end = #qcc.chain_end<head>, site = @trap_a} : !config
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64}
      : !qcc.config<#ion>

  // Two hops: a junction is traversed by two binary displacements.
  // CHECK: %[[C2:.+]] = qcc.ion.transport %[[C1]] {chain = 0 : i64, from = @trap_a, via = "seg_aj"} : !config
  %c2 = qcc.ion.transport %c1 {from = @trap_a, chain = 0 : i64, via = "seg_aj"}
      : !qcc.config<#ion>
  // CHECK: %[[C3:.+]] = qcc.ion.transport %[[C2]] {chain = 0 : i64, from = @j0, via = "seg_jb"} : !config
  %c3 = qcc.ion.transport %c2 {from = @j0, chain = 0 : i64, via = "seg_jb"}
      : !qcc.config<#ion>

  // Prepend the arrival onto trap_b's chain: [5 6 7 8 9] [0] -> [0 5 6 7 8 9].
  // CHECK: %[[C4:.+]] = qcc.ion.merge %[[C3]] {at = #qcc.chain_end<head>, dst = 0 : i64, site = @trap_b, src = 1 : i64} : !config
  %c4 = qcc.ion.merge %c3 {site = @trap_b, dst = 0 : i64, src = 1 : i64,
                           at = #qcc.chain_end<head>}
      : !qcc.config<#ion>

  // CHECK: %[[C5:.+]] = qcc.ion.reorder %[[C4]] {chain = 0 : i64, permutation = array<i64: 1, 0, 2, 3, 4, 5>, site = @trap_b} : !config
  %c5 = qcc.ion.reorder %c4 {site = @trap_b, chain = 0 : i64,
                             permutation = array<i64: 1, 0, 2, 3, 4, 5>}
      : !qcc.config<#ion>

  return
}

//===----------------------------------------------------------------------===//
// A superconducting target: the case where the rule set is empty. There are no
// transport edges, so nothing in the dialect can produce a new configuration.
// Couplers are rank-2 `link` edges that are never consumed.
//===----------------------------------------------------------------------===//

#sc = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @q0, kind = fixed, capacity = 1>,
      #qcc.site<name = @q1, kind = fixed, capacity = 1>,
      #qcc.site<name = @q2, kind = fixed, capacity = 1>
    ],
    edges = [
      #qcc.edge<link, [@q0, @q1], {id = "coupler_01", persistent = true, cz_err = 6.100000e-03}>,
      #qcc.edge<link, [@q1, @q2], {id = "coupler_12", persistent = true, cz_err = 5.800000e-03}>
    ]>,
  maxFacetRank = 2,
  downwardClosed = true,
  // couplers overlap: q1 is in both {q0,q1} and {q1,q2}
  partitioning = false,
  facetOrdering = none>

// CHECK-LABEL: func.func @static_chip
func.func @static_chip() {
  // CHECK: qcc.config.init(#qcc.placement<@q0 = {{\[\[}}0]], @q1 = {{\[\[}}1]], @q2 = {{\[\[}}2]]>)
  %c = qcc.config.init(#qcc.placement<@q0 = [[0]], @q1 = [[1]], @q2 = [[2]]>)
      : !qcc.config<#sc>
  return
}

// An empty placement is well-formed: nothing is placed yet.
// CHECK-LABEL: func.func @nothing_placed
func.func @nothing_placed() {
  // CHECK: qcc.config.init(#qcc.placement<>) : !config
  %c = qcc.config.init(#qcc.placement<>) : !qcc.config<#sc>
  return
}

//===----------------------------------------------------------------------===//
// The atom and link rule sets round-trip on the same footing as the ion ones.
// One dialect covers all three at no cost to the simpler targets: this target
// declares rigid and link edges, and an ion machine that declares neither never
// encounters these operations.
//===----------------------------------------------------------------------===//

#hybrid = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @t0, kind = trap, capacity = 2>,
      #qcc.site<name = @t1, kind = trap, capacity = 2>,
      #qcc.site<name = @t2, kind = trap, capacity = 2>,
      #qcc.site<name = @t3, kind = trap, capacity = 2>,
      #qcc.site<name = @m0, kind = module, capacity = 2>,
      #qcc.site<name = @m1, kind = module, capacity = 2>
    ],
    edges = [
      #qcc.edge<rigid, [@t0, @t1], {id = "aod_x", axis = "x", latency_us = 5.000000e+02}>,
      #qcc.edge<rigid, [@t2, @t3], {id = "aod_x2", axis = "x"}>,
      #qcc.edge<link, [@m0, @m1], {id = "fibre", persistent = false, latency_us = 5.500000e+03}>
    ]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

// CHECK-LABEL: func.func @atom_and_link_rules
func.func @atom_and_link_rules() {
  // CHECK: %[[C0:.+]] = qcc.config.init
  %c0 = qcc.config.init(#qcc.placement<
      @t0 = [[0]], @t1 = [[1]], @m0 = [[2]], @m1 = [[3]]>) : !qcc.config<#hybrid>

  // A whole AOD axis displaced in one step.
  // CHECK: %[[C1:.+]] = qcc.atom.aod_move %[[C0]] {to = [@t2, @t3], via = "aod_x"}
  %c1 = qcc.atom.aod_move %c0 {via = "aod_x", to = [@t2, @t3]} : !qcc.config<#hybrid>

  // A single tweezer handoff along an axis that reaches both traps.
  // CHECK: %[[C2:.+]] = qcc.atom.slm_transfer %[[C1]] {from = @t2, to = @t3}
  %c2 = qcc.atom.slm_transfer %c1 {from = @t2, to = @t3} : !qcc.config<#hybrid>

  // Herald a link, then consume it. The pair is a facet only in between.
  // CHECK: %[[C3:.+]] = qcc.link.generate %[[C2]] {qubits = array<i64: 2, 3>, via = "fibre"}
  %c3 = qcc.link.generate %c2 {via = "fibre", qubits = array<i64: 2, 3>} : !qcc.config<#hybrid>
  // CHECK: qcc.link.consume %[[C3]] {qubits = array<i64: 2, 3>}
  %c4 = qcc.link.consume %c3 {qubits = array<i64: 2, 3>} : !qcc.config<#hybrid>

  return
}

//===----------------------------------------------------------------------===//
// Per-site overrides, geometry and a junction's turn table all round-trip. Each
// is optional: a site that declares none inherits the device ordering and rank
// and permits every turn.
//===----------------------------------------------------------------------===//

#annotated = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @store, kind = trap, capacity = 8, coords = [0.000000e+00, 0.000000e+00]>,
      #qcc.site<name = @plane, kind = trap, capacity = 8, maxFacetRank = 4, facetOrdering = planar,
                coords = [3.000000e+00, 1.000000e+00, 2.000000e+00]>,
      #qcc.site<name = @j, kind = junction, capacity = 2,
                props = {turn_latency_us = 4.000000e+02, turns = ["seg_s>seg_p", "seg_p>seg_s"]}>
    ],
    edges = [
      #qcc.edge<transport, [@store, @j], {id = "seg_s", bidirectional = true}>,
      #qcc.edge<transport, [@j, @plane], {id = "seg_p", bidirectional = true}>
    ]>,
  maxFacetRank = 8, downwardClosed = true, partitioning = true, facetOrdering = linear>


// CHECK-LABEL: func.func @turn_at_a_junction
func.func @turn_at_a_junction() {
  // CHECK: %[[C0:.+]] = qcc.config.init
  %c0 = qcc.config.init(#qcc.placement<@store = [[0, 1]], @j = [], @plane = []>)
      : !qcc.config<#annotated>
  // CHECK: %[[C1:.+]] = qcc.ion.transport %[[C0]]
  %c1 = qcc.ion.transport %c0 {from = @store, chain = 0 : i64, via = "seg_s"} : !qcc.config<#annotated>
  // CHECK: qcc.ion.junction_turn %[[C1]] {chain = 0 : i64, from = "seg_s", site = @j, to = "seg_p"}
  %c2 = qcc.ion.junction_turn %c1 {site = @j, chain = 0 : i64, from = "seg_s", to = "seg_p"}
      : !qcc.config<#annotated>
  return
}
