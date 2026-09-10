// RUN: qcc-opt %s --qcc-verify-connectivity --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Negative tests for the verifier. Each of these programs would be accepted by a
// dialect modelling connectivity as a static coupling graph, which cannot express
// why they are ill-formed.
//===----------------------------------------------------------------------===//

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 6>,
      #qcc.site<name = @j0, kind = junction, capacity = 4>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", latency_us = 8.000000e+01, max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @trap_b], {id = "seg_jb", latency_us = 8.000000e+01, bidirectional = false}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Ions 0 and 5 sit in different traps, so no facet of K contains both. A fixed
// coupling graph cannot make this check, since the answer depends on where the
// ions currently are.
func.func @not_colocated(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q5 = qc.static 5 : !qc.qubit
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]], @trap_b = [[5, 6]]>)
      : !qcc.config<#ion>
  // expected-error @+1 {{operand set {0, 5} is not executable in the configuration reaching it: no facet of K contains it; the facets here are {0, 1, 2}, {5, 6}}}
  qc.rxx(%theta) %q0, %q5 : !qc.qubit, !qc.qubit
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// The same two qubits are legal before the split and illegal after it. Nothing
// about the gate changes; the configuration dominating it does.
func.func @split_invalidates_a_later_gate(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]]>) : !qcc.config<#ion>
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  // @trap_a: [0 1 2] -> [0] [1 2]
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
  // expected-error @+1 {{operand set {0, 1} is not executable in the configuration reaching it}}
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// An unplaced qubit: the gate names a hardware qubit the configuration does not
// mention.
func.func @qubit_not_placed(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q9 = qc.static 9 : !qc.qubit
  %c = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]]>) : !qcc.config<#ion>
  // expected-error @+1 {{acts on qubit 9, which the configuration reaching it does not place}}
  qc.rxx(%theta) %q0, %q9 : !qc.qubit, !qc.qubit
  return
}

// -----

// A machine offering only a global MS gate acts on a chain as a whole or not at
// all. `downwardClosed = false` rules out representing K by its facets alone, and
// is why a subset check is insufficient.
#global_ms = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = false, partitioning = true, facetOrdering = linear>

func.func @not_downward_closed(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]]>) : !qcc.config<#global_ms>
  // expected-error @+1 {{the device is not downward closed, so facet {0, 1, 2} must be acted on whole rather than in part}}
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 6>,
      #qcc.site<name = @j0, kind = junction, capacity = 4>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @trap_b], {id = "seg_jb", bidirectional = false}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// There is no transport edge directly from @trap_a to @trap_b. The junction is a
// site of degree 2 rather than a ternary edge, so the route is two binary hops.
func.func @no_such_segment() {
  %q0 = qc.static 0 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1]]>) : !qcc.config<#ion>
  // expected-error @+1 {{substrate declares no transport edge with id 'seg_ab'}}
  %c1 = qcc.ion.transport %c0 {from = @trap_a, chain = 0 : i64, via = "seg_ab"} : !qcc.config<#ion>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 6>,
      #qcc.site<name = @j0, kind = junction, capacity = 4>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@j0, @trap_b], {id = "seg_jb", bidirectional = false}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// @seg_jb is declared one-way from @j0 to @trap_b.
func.func @against_the_declared_direction() {
  %c0 = qcc.config.init(#qcc.placement<@trap_b = [[0, 1]]>) : !qcc.config<#ion>
  // expected-error @+1 {{transport edge 'seg_jb' is unidirectional from @j0 to @trap_b, but this operation traverses it the other way}}
  %c1 = qcc.ion.transport %c0 {from = @trap_b, chain = 0 : i64, via = "seg_jb"} : !qcc.config<#ion>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @j0, kind = junction, capacity = 4>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", max_chain_len = 2 : i64}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// A configuration-dependent failure, reported when the rule is applied rather
// than by the operation's own verifier.
func.func @chain_too_long_for_the_segment(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]]>) : !qcc.config<#ion>
  // expected-error @+1 {{chain of length 3 exceeds the max_chain_len of 2 declared by segment 'seg_aj'}}
  %c1 = qcc.ion.transport %c0 {from = @trap_a, chain = 0 : i64, via = "seg_aj"} : !qcc.config<#ion>
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @j0, kind = junction, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj"}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Capacity is the transient maximum, and a junction has very little of it.
func.func @junction_overflows(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]], @j0 = []>) : !qcc.config<#ion>
  // expected-error @+1 {{would place 3 qubits at @j0, exceeding its capacity of 2}}
  %c1 = qcc.ion.transport %c0 {from = @trap_a, chain = 0 : i64, via = "seg_aj"} : !qcc.config<#ion>
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// A split must leave ions on both sides. Detaching the whole chain would produce
// an empty potential well.
func.func @split_leaves_nothing(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1]]>) : !qcc.config<#ion>
  // expected-error @+1 {{cannot detach 2 ion(s) from a chain of length 2}}
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 2 : i64} : !qcc.config<#ion>
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

// A target whose facets carry no order has no chain ends, so detaching at an end
// is undefined there rather than unrestricted.
#unordered = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @zone, kind = gatezone, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = none>

func.func @split_needs_an_order() {
  %c0 = qcc.config.init(#qcc.placement<@zone = [[0, 1, 2]]>) : !qcc.config<#unordered>
  // expected-error @+1 {{ion.split requires facetOrdering = linear}}
  %c1 = qcc.ion.split %c0 {site = @zone, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#unordered>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// The loop-carried case: the body's net rewrite is not the identity, so no single
// placement describes this program point. The pass reports it rather than
// deciding it.
func.func @loop_carried_configuration(%theta: f64, %n: index) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2, 3]]>) : !qcc.config<#ion>

  %cN = scf.for %i = %lb to %n step %step iter_args(%c = %c0) -> (!qcc.config<#ion>) {
    %ca = qcc.ion.split %c {site = @trap_a, chain = 0 : i64,
                            end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
    // expected-error @+2 {{the configuration reaching this operation is not statically known}}
    // expected-note @+1 {{statically known only if the body's net rewrite is the identity}}
    qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
    scf.yield %ca : !qcc.config<#ion>
  }
  return
}
