// RUN: qcc-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Junction turns.
//
// Carrying a crystal straight through a junction and turning it are different
// physical actions: the turn rotates the crystal, costs substantially more than
// a hop, and on much hardware is possible only between certain pairs of
// segments. `qcc.ion.junction_turn` exposes the cost through `getLatencyUs` and
// the legality through the verifier.
//
// The restriction is a `turns` table on the junction, in the same way that a
// road network attaches turn restrictions to the node rather than to a ternary
// edge. A junction that declares none permits every turn.
//===----------------------------------------------------------------------===//

#tee = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @north, kind = trap, capacity = 4>,
      #qcc.site<name = @east, kind = trap, capacity = 4>,
      #qcc.site<name = @south, kind = trap, capacity = 4>,
      // North may turn to east and back, but never to south.
      #qcc.site<name = @j, kind = junction, capacity = 2,
                props = {turns = ["sn>se", "se>sn"], turn_latency_us = 4.000000e+02}>
    ],
    edges = [
      #qcc.edge<transport, [@north, @j], {id = "sn", latency_us = 1.000000e+02, bidirectional = true}>,
      #qcc.edge<transport, [@east, @j], {id = "se", latency_us = 1.000000e+02, bidirectional = true}>,
      #qcc.edge<transport, [@south, @j], {id = "ss", latency_us = 1.000000e+02, bidirectional = true}>
    ]>,
  maxFacetRank = 4, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @north_to_east_is_permitted() {
  %c0 = qcc.config.init(#qcc.placement<@north = [[0]], @j = [], @east = []>) : !qcc.config<#tee>
  %c1 = qcc.ion.transport %c0 {from = @north, chain = 0 : i64, via = "sn"} : !qcc.config<#tee>
  %c2 = qcc.ion.junction_turn %c1 {site = @j, chain = 0 : i64, from = "sn", to = "se"} : !qcc.config<#tee>
  %c3 = qcc.ion.transport %c2 {from = @j, chain = 0 : i64, via = "se"} : !qcc.config<#tee>
  return
}

func.func @north_to_south_is_not() {
  %c0 = qcc.config.init(#qcc.placement<@north = [[0]], @j = []>) : !qcc.config<#tee>
  %c1 = qcc.ion.transport %c0 {from = @north, chain = 0 : i64, via = "sn"} : !qcc.config<#tee>
  // expected-error @+1 {{@j does not permit the turn 'sn' -> 'ss'; its turn table lists the rotations the hardware can actually perform}}
  %c2 = qcc.ion.junction_turn %c1 {site = @j, chain = 0 : i64, from = "sn", to = "ss"} : !qcc.config<#tee>
  return
}

func.func @a_turn_joins_two_segments() {
  %c0 = qcc.config.init(#qcc.placement<@north = [[0]], @j = []>) : !qcc.config<#tee>
  // expected-error @+1 {{a turn joins two different segments, but both are 'sn'; a crystal that leaves the way it came needs no rotation}}
  %c1 = qcc.ion.junction_turn %c0 {site = @j, chain = 0 : i64, from = "sn", to = "sn"} : !qcc.config<#tee>
  return
}

func.func @both_segments_must_reach_the_junction() {
  %c0 = qcc.config.init(#qcc.placement<@north = [[0]], @j = []>) : !qcc.config<#tee>
  // expected-error @+1 {{substrate declares no transport edge with id 'nowhere'}}
  %c1 = qcc.ion.junction_turn %c0 {site = @j, chain = 0 : i64, from = "sn", to = "nowhere"} : !qcc.config<#tee>
  return
}

// -----

// A junction that declares no table permits every turn, which is the case where
// the rotation is unrestricted.
#unrestricted = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @a, kind = trap, capacity = 4>,
             #qcc.site<name = @b, kind = trap, capacity = 4>,
             #qcc.site<name = @j, kind = junction, capacity = 2>],
    edges = [#qcc.edge<transport, [@a, @j], {id = "sa", bidirectional = true}>,
             #qcc.edge<transport, [@b, @j], {id = "sb", bidirectional = true}>]>,
  maxFacetRank = 4, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @no_table_permits_everything() {
  %c0 = qcc.config.init(#qcc.placement<@a = [[0]], @j = []>) : !qcc.config<#unrestricted>
  %c1 = qcc.ion.transport %c0 {from = @a, chain = 0 : i64, via = "sa"} : !qcc.config<#unrestricted>
  %c2 = qcc.ion.junction_turn %c1 {site = @j, chain = 0 : i64, from = "sa", to = "sb"} : !qcc.config<#unrestricted>
  return
}

// -----

// A malformed entry would silently forbid every turn, which is harder to
// diagnose than a rejected description.
#bad_table = #qcc.substrate<
  // expected-error @+1 {{site @j has a malformed `turns` entry; each is "<incoming-id>><outgoing-id>"}}
  sites = [#qcc.site<name = @j, kind = junction, capacity = 2, props = {turns = ["sn_to_se"]}>],
  edges = []>
