// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Segment ends: which end of a trap a segment reaches.
//
// A trap holds a line of ions and a segment joins it at one end. Only the chain
// at that end can leave; anything further in would have to pass through the ions
// in front of it.
//
// `ends` declares the reached ends, one entry per incident site in the edge's
// own order. A site's chain list is maintained in spatial order, since a split
// leaves the detached part on the side it came from, so the reachable chain is
// either the first or the last index, and the same property positions arrivals.
//
// The machine below has two traps of ten ions each. The segment reaches the head
// of @trap_1 and the tail of @trap_2, so an ion may leave @trap_1 only from the
// head and @trap_2 only from the tail, one at a time.
//===----------------------------------------------------------------------===//

#two = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_1, kind = trap, capacity = 12>,
      #qcc.site<name = @trap_2, kind = trap, capacity = 12>
    ],
    edges = [
      #qcc.edge<transport, [@trap_1, @trap_2],
                {id = "seg", ends = ["head", "tail"], latency_us = 8.000000e+01,
                 bidirectional = true, max_chain_len = 1 : i64}>
    ]>,
  maxFacetRank = 10, downwardClosed = true, partitioning = true, facetOrdering = linear>

// The legal move: detach the head ion of @trap_1 and carry it across.

func.func @leftmost_ion_leaves(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q19 = qc.static 19 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @trap_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
      @trap_2 = [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]>) : !qcc.config<#two>

  // Detach one ion from the head, which is the end the segment reaches.
  //   @trap_1: [0] [1 2 3 4 5 6 7 8 9]
  %c1 = qcc.ion.split %c0 {site = @trap_1, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#two>

  // Chain 0 is at the head, so it can leave. It enters @trap_2 through the tail
  // and lands on the tail:
  //   @trap_2: [10 .. 19] [0]
  %c2 = qcc.ion.transport %c1 {from = @trap_1, chain = 0 : i64, via = "seg"} : !qcc.config<#two>

  %c3 = qcc.ion.merge %c2 {site = @trap_2, dst = 0 : i64, src = 1 : i64,
                           at = #qcc.chain_end<tail>} : !qcc.config<#two>
  //   @trap_2: [10 .. 19 0]

  qc.rzz(%theta) %q0, %q19 : !qc.qubit, !qc.qubit
  return
}

// -----

#two = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_1, kind = trap, capacity = 12>,
      #qcc.site<name = @trap_2, kind = trap, capacity = 12>
    ],
    edges = [
      #qcc.edge<transport, [@trap_1, @trap_2],
                {id = "seg", ends = ["head", "tail"], bidirectional = true, max_chain_len = 1 : i64}>
    ]>,
  maxFacetRank = 10, downwardClosed = true, partitioning = true, facetOrdering = linear>

// An arrival lands at the end it entered through, so it can leave again along
// the same segment. Ion 0 enters @trap_2 through the tail, and the return
// journey is legal only because it landed on the tail.

func.func @an_arrival_can_go_home(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @trap_1 = [[0, 1, 2]], @trap_2 = [[10, 11]]>) : !qcc.config<#two>

  %c1 = qcc.ion.split %c0 {site = @trap_1, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#two>
  //   @trap_1: [0] [1 2]   ->   @trap_2: [10 11] [0]
  %c2 = qcc.ion.transport %c1 {from = @trap_1, chain = 0 : i64, via = "seg"} : !qcc.config<#two>

  // Chain 1 is at @trap_2's tail, which is the end the segment reaches. Had the
  // arrival landed at the head, this would be rejected.
  %c3 = qcc.ion.transport %c2 {from = @trap_2, chain = 1 : i64, via = "seg"} : !qcc.config<#two>
  //   back to @trap_1 through its head, so it lands at the head again
  %c4 = qcc.ion.merge %c3 {site = @trap_1, dst = 1 : i64, src = 0 : i64,
                           at = #qcc.chain_end<head>} : !qcc.config<#two>

  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#two = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_1, kind = trap, capacity = 12>,
      #qcc.site<name = @trap_2, kind = trap, capacity = 12>
    ],
    edges = [
      #qcc.edge<transport, [@trap_1, @trap_2],
                {id = "seg", ends = ["head", "tail"], bidirectional = true, max_chain_len = 1 : i64}>
    ]>,
  maxFacetRank = 10, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Detaching the tail ion of @trap_1 is expressible but strands it: it ends up at
// the end opposite the segment and can no longer reach it.

func.func @the_rightmost_ion_is_stranded(%theta: f64) {
  %q9 = qc.static 9 : !qc.qubit
  %q10 = qc.static 10 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @trap_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
      @trap_2 = [[10, 11]]>) : !qcc.config<#two>

  //   @trap_1: [0 .. 8] [9]   -- ion 9 is chain 1, at the far end
  %c1 = qcc.ion.split %c0 {site = @trap_1, chain = 0 : i64,
                           end = #qcc.chain_end<tail>, count = 1 : i64} : !qcc.config<#two>
  // expected-error @+1 {{chain 1 is not at the door: segment 'seg' meets @trap_1 at its head end, so only chain 0 can leave; the 1 chain(s) in between would have to be moved first}}
  %c2 = qcc.ion.transport %c1 {from = @trap_1, chain = 1 : i64, via = "seg"} : !qcc.config<#two>

  qc.rzz(%theta) %q9, %q10 : !qc.qubit, !qc.qubit
  return
}

// -----

#two = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_1, kind = trap, capacity = 12>,
      #qcc.site<name = @trap_2, kind = trap, capacity = 12>
    ],
    edges = [
      #qcc.edge<transport, [@trap_1, @trap_2],
                {id = "seg", ends = ["head", "tail"], bidirectional = true, max_chain_len = 4 : i64}>
    ]>,
  maxFacetRank = 10, downwardClosed = true, partitioning = true, facetOrdering = linear>

// The same rule from the other side: a chain cannot be moved out past one
// sitting between it and the segment.

func.func @nothing_moves_past_the_ion_in_front(%theta: f64) {
  %q1 = qc.static 1 : !qc.qubit
  %q10 = qc.static 10 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<@trap_1 = [[0, 1, 2]], @trap_2 = [[10]]>)
      : !qcc.config<#two>

  //   @trap_1: [0] [1 2]
  %c1 = qcc.ion.split %c0 {site = @trap_1, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#two>
  // Ion 0 sits between chain 1 and the segment.
  // expected-error @+1 {{chain 1 is not at the door: segment 'seg' meets @trap_1 at its head end, so only chain 0 can leave}}
  %c2 = qcc.ion.transport %c1 {from = @trap_1, chain = 1 : i64, via = "seg"} : !qcc.config<#two>

  qc.rzz(%theta) %q1, %q10 : !qc.qubit, !qc.qubit
  return
}

// -----

// The property's shape is verified, since a mismatched list would silently
// associate the wrong end with the wrong trap.

#wrong_arity = #qcc.substrate<
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
  // expected-error @+1 {{`ends` has 1 entry but the edge is incident to 2 site(s); give one end per endpoint}}
  edges = [#qcc.edge<transport, [@a, @b], {id = "seg", ends = ["head"]}>]>

// -----

#wrong_value = #qcc.substrate<
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
  // expected-error @+1 {{`ends` entries must be "head" or "tail"}}
  edges = [#qcc.edge<transport, [@a, @b], {id = "seg", ends = ["left", "right"]}>]>

// -----

// Reached ends are a property of a segment. A control group has no ends, so
// declaring them there is an error rather than a no-op.
#wrong_kind = #qcc.substrate<
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
  // expected-error @+1 {{`ends` is only meaningful on a transport edge}}
  edges = [#qcc.edge<control, [@a, @b], {id = "awg", ends = ["head", "tail"]}>]>
