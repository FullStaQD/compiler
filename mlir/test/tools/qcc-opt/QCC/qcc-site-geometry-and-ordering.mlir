// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Per-site ordering and geometry.
//
// `facetOrdering` and `maxFacetRank` are per-site overrides of the device
// default, so one machine can hold a linear storage chain alongside an unordered
// gate zone and a planar crystal. `coords` gives a site a position, which the
// router uses to prefer a near zone over a far one.
//===----------------------------------------------------------------------===//

#mixed = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      // A linear storage chain: splits work here.
      #qcc.site<name = @store, kind = trap, capacity = 8, coords = [0.000000e+00, 0.000000e+00]>,
      // A planar crystal in the same machine. Gates on it verify; the chain
      // rules do not apply, because a plane has no ends.
      #qcc.site<name = @plane, kind = trap, capacity = 8, facetOrdering = planar,
                coords = [3.000000e+00, 0.000000e+00]>
    ],
    edges = [#qcc.edge<transport, [@store, @plane], {id = "seg", bidirectional = true}>]>,
  maxFacetRank = 8, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Coordinates survive the round trip, and so does the per-site ordering.

func.func @linear_zone_beside_a_planar_one(%theta: f64) {
  %q4 = qc.static 4 : !qc.qubit
  %q6 = qc.static 6 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<@store = [[0, 1, 2]], @plane = [[4, 5, 6]]>)
      : !qcc.config<#mixed>

  // The storage chain is linear, so it can be cut.
  %c1 = qcc.ion.split %c0 {site = @store, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#mixed>

  // A gate on the planar crystal verifies exactly as anywhere else: `K` only
  // ever asks a group who is in it, never how they are arranged.
  qc.rzz(%theta) %q4, %q6 : !qc.qubit, !qc.qubit
  return
}

// -----

#mixed = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @plane, kind = trap, capacity = 8, facetOrdering = planar>],
    edges = []>,
  maxFacetRank = 8, downwardClosed = true, partitioning = true, facetOrdering = linear>

// The chain rules are defined in terms of chain ends, which a plane does not
// have, so they are rejected on a planar site rather than applied as though the
// site were linear. A planar rule set would sit alongside the ion rules; none
// exists, because no platform currently splits or restructures a planar
// crystal.
func.func @chain_rules_do_not_apply_to_a_plane() {
  %c0 = qcc.config.init(#qcc.placement<@plane = [[0, 1, 2]]>) : !qcc.config<#mixed>
  // expected-error @+1 {{ion.split requires facetOrdering = linear, but @plane is planar}}
  %c1 = qcc.ion.split %c0 {site = @plane, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#mixed>
  return
}

// -----

// The device default applies where a site declares nothing, so a machine that is
// linear throughout needs no per-site annotation.
#all_linear = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap, kind = trap, capacity = 8>],
    edges = []>,
  maxFacetRank = 8, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @inherited_ordering() {
  %c0 = qcc.config.init(#qcc.placement<@trap = [[0, 1, 2]]>) : !qcc.config<#all_linear>
  %c1 = qcc.ion.split %c0 {site = @trap, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#all_linear>
  return
}

// -----

// A position has two or three components. Reading the first two of a longer list
// would mask the error rather than report it.
#bad_coords = #qcc.substrate<
  // expected-error @+1 {{site @a declares 4 coordinate(s); a position is 2 or 3 numbers}}
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4, coords = [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]>],
  edges = []>
