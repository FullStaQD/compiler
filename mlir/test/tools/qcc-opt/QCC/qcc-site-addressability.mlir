// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Per-site addressability.
//
// Co-location is necessary but not sufficient for an entangling gate. On a QCCD
// machine a storage trap has no gate lasers, so ions there share a motional mode
// and nothing more; the gate is performed in a dedicated zone.
//
// `maxFacetRank` on a site gives the width of the widest operation that site can
// host; a rank below 2 means it can host none. Omitting it makes the site
// inherit the device rank, which is the case for a long-chain machine whose trap
// is also its gate zone.
//===----------------------------------------------------------------------===//

#h_series = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      // Storage: holds thirty ions and gates none of them.
      #qcc.site<name = @storage, kind = trap, capacity = 30, maxFacetRank = 0>,
      // A junction is a transit site, not a gate site.
      #qcc.site<name = @j0, kind = junction, capacity = 4, maxFacetRank = 0>,
      // The only site that can gate. Inherits the device rank of 2.
      #qcc.site<name = @gate, kind = gatezone, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@storage, @j0], {id = "s_j", bidirectional = true, max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @gate], {id = "j_g", bidirectional = true, max_chain_len = 2 : i64}>
    ]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Two ions travel from storage to the gate zone; {0, 1} becomes executable only
// once they arrive.

func.func @shuttle_to_the_gate_zone(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @storage = [[0, 1, 2, 3, 4]], @j0 = [], @gate = []>) : !qcc.config<#h_series>

  //   @storage: [0 1] [2 3 4]
  %c1 = qcc.ion.split %c0 {site = @storage, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 2 : i64} : !qcc.config<#h_series>
  %c2 = qcc.ion.transport %c1 {from = @storage, chain = 0 : i64, via = "s_j"} : !qcc.config<#h_series>
  %c3 = qcc.ion.transport %c2 {from = @j0, chain = 0 : i64, via = "j_g"} : !qcc.config<#h_series>
  //   @gate: [0 1]

  // The same pair now verifies, because of where the ions are.
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#h_series = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @storage, kind = trap, capacity = 30, maxFacetRank = 0>,
      #qcc.site<name = @j0, kind = junction, capacity = 4, maxFacetRank = 0>,
      #qcc.site<name = @gate, kind = gatezone, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@storage, @j0], {id = "s_j", bidirectional = true, max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @gate], {id = "j_g", bidirectional = true, max_chain_len = 2 : i64}>
    ]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Without the shuttle the ions are co-located but still cannot be gated, and the
// diagnostic distinguishes the two conditions.

func.func @co_located_is_not_enough(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit

  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c = qcc.config.init(#qcc.placement<
      @storage = [[0, 1, 2, 3, 4]], @j0 = [], @gate = []>) : !qcc.config<#h_series>

  // expected-error @+1 {{the qubits are co-located at @storage, but that site cannot host an entangling gate; they have to be moved to one that can}}
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#h_series = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @storage, kind = trap, capacity = 30, maxFacetRank = 0>,
      #qcc.site<name = @j0, kind = junction, capacity = 4, maxFacetRank = 0>,
      #qcc.site<name = @gate, kind = gatezone, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@storage, @j0], {id = "s_j", bidirectional = true, max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @gate], {id = "j_g", bidirectional = true, max_chain_len = 2 : i64}>
    ]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = true, facetOrdering = linear>

// A junction is a transit site: ions pass through it but are not acted on there.
func.func @a_junction_is_not_a_gate_zone(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit

  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c = qcc.config.init(#qcc.placement<@storage = [], @j0 = [[0, 1]], @gate = []>)
      : !qcc.config<#h_series>

  // expected-error @+1 {{the qubits are co-located at @j0, but that site cannot host an entangling gate}}
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

// On a long-chain machine the trap is also the gate zone, so it declares no
// override and inherits the wide device rank.

#long_chain = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap, kind = trap, capacity = 36>],
    edges = []>,
  maxFacetRank = 36, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @the_trap_is_the_gate_zone(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q9 = qc.static 9 : !qc.qubit
  %c = qcc.config.init(#qcc.placement<@trap = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]>)
      : !qcc.config<#long_chain>
  // Opposite ends of one chain. An MS gate couples through the collective
  // motional mode, so distance along the chain is not a constraint.
  qc.rzz(%theta) %q0, %q9 : !qc.qubit, !qc.qubit
  return
}

// -----

// A site can also be narrower than the device rather than unable to gate: a
// two-ion gate zone on a machine that elsewhere performs three-body gates. The
// diagnostic distinguishes a set that is too wide for the site from one that is
// not a facet at all.

#mixed = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @pair_zone, kind = gatezone, capacity = 4, maxFacetRank = 2>],
    edges = []>,
  maxFacetRank = 3, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @narrower_than_the_device(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q2 = qc.static 2 : !qc.qubit

  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c = qcc.config.init(#qcc.placement<@pair_zone = [[0, 1, 2]]>) : !qcc.config<#mixed>

  // A pair is within this zone's rank.
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit

  // Three is within the device rank but beyond this zone's.
  // expected-error @+1 {{facet {0, 1, 2} contains it, but that site hosts operations of at most 2 qubit(s)}}
  qc.ctrl(%q0, %q1) targets(%t = %q2) { qc.z %t : !qc.qubit
                                        qc.yield } : {!qc.qubit, !qc.qubit}, {!qc.qubit}
  return
}

// -----

// A site cannot host an operation on more qubits than it can hold.
#over_wide = #qcc.substrate<
  // expected-error @+1 {{site @gate declares maxFacetRank = 4 but holds at most 2 qubit(s)}}
  sites = [#qcc.site<name = @gate, kind = gatezone, capacity = 2, maxFacetRank = 4>],
  edges = []>
