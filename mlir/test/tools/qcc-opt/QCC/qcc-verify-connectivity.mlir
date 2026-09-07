// RUN: qcc-opt %s --qcc-verify-connectivity | FileCheck %s

//===----------------------------------------------------------------------===//
// The same two-qubit gate on the same two qubits is legal at one program point
// and illegal at another. Nothing about the gate changes; only the configuration
// dominating it does.
//===----------------------------------------------------------------------===//

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 20>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 20>,
      #qcc.site<name = @j0, kind = junction, capacity = 10>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @j0], {id = "seg_aj", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 10 : i64}>,
      #qcc.edge<transport, [@j0, @trap_b], {id = "seg_jb", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 10 : i64}>,
      #qcc.edge<control, [@trap_a, @j0, @trap_b], {id = "awg_shuttle"}>
    ]>,
  maxFacetRank = 10,
  downwardClosed = true,   // individual addressing: any subset of a chain
  partitioning = true,     // every ion sits in exactly one chain
  facetOrdering = linear>  // chains are ordered; splits only at the ends

// CHECK-LABEL: func.func @shuttle_then_entangle
func.func @shuttle_then_entangle(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q5 = qc.static 5 : !qc.qubit
  %q6 = qc.static 6 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @trap_a = [[0, 1, 2, 3, 4]],
      @trap_b = [[5, 6, 7, 8, 9]],
      @j0     = []>) : !qcc.config<#ion>

  // {0, 1} is a face of the facet {0..4}. Contiguity is not required: with
  // individual addressing, {0, 4} verifies equally.
  // CHECK: qc.rxx
  qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  // {5, 6} is a face of the other facet.
  // CHECK: qc.rxx
  qc.rxx(%theta) %q5, %q6 : !qc.qubit, !qc.qubit

  // Now move ion 0 across the junction into trap_b.
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
  %c2 = qcc.ion.transport %c1 {from = @trap_a, chain = 0 : i64, via = "seg_aj"} : !qcc.config<#ion>
  %c3 = qcc.ion.transport %c2 {from = @j0, chain = 0 : i64, via = "seg_jb"} : !qcc.config<#ion>
  %c4 = qcc.ion.merge %c3 {site = @trap_b, dst = 0 : i64, src = 1 : i64,
                           at = #qcc.chain_end<head>} : !qcc.config<#ion>
  // @trap_a: [1 2 3 4]   @trap_b: [0 5 6 7 8 9]

  // {0, 5} is executable here and was not under %c0. The converse holds for
  // {0, 1}, which is why the gates above are ordered before the split.
  // CHECK: qc.rxx
  qc.rxx(%theta) %q0, %q5 : !qc.qubit, !qc.qubit

  // A single-qubit gate is a local rotation: it needs the qubit placed, not
  // co-located with anything.
  // CHECK: qc.h
  qc.h %q1 : !qc.qubit

  // A controlled gate is an interaction between every qubit it names, controls
  // included, so it is checked as the composite set {0, 5}.
  // CHECK: qc.ctrl
  qc.ctrl(%q0) targets(%a0 = %q5) {
    qc.x %a0 : !qc.qubit
    qc.yield
  } : {!qc.qubit}, {!qc.qubit}

  return
}

//===----------------------------------------------------------------------===//
// A function containing no configuration is not checked, so the pass is safe in
// a pipeline that also compiles programs with no target attached.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_target_declared
func.func @no_target_declared() {
  %q0 = qc.static 0 : !qc.qubit
  %q9 = qc.static 9 : !qc.qubit
  // Would be rejected under #ion; here there is no connectivity in force.
  // CHECK: qc.swap
  qc.swap %q0, %q9 : !qc.qubit, !qc.qubit
  return
}

//===----------------------------------------------------------------------===//
// A superconducting target has an empty rule set, so no operation can produce a
// new configuration and K is a fixed coupling graph. Couplers are link edges,
// whose facets do not partition the qubits: q1 belongs to both {q0, q1} and
// {q1, q2}.
//===----------------------------------------------------------------------===//

#sc = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @q0, kind = fixed, capacity = 1>,
      #qcc.site<name = @q1, kind = fixed, capacity = 1>,
      #qcc.site<name = @q2, kind = fixed, capacity = 1>
    ],
    edges = [
      #qcc.edge<link, [@q0, @q1], {id = "coupler_01", persistent = true}>,
      #qcc.edge<link, [@q1, @q2], {id = "coupler_12", persistent = true}>
    ]>,
  maxFacetRank = 2,
  downwardClosed = true,
  partitioning = false,
  facetOrdering = none>

// CHECK-LABEL: func.func @coupling_graph
func.func @coupling_graph() {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q2 = qc.static 2 : !qc.qubit

  %c = qcc.config.init(#qcc.placement<@q0 = [[0]], @q1 = [[1]], @q2 = [[2]]>)
      : !qcc.config<#sc>

  // Both couplers exist, so both CZs verify. {0, 2} does not; see the negative
  // tests.
  // CHECK-COUNT-2: qc.z
  qc.ctrl(%q0) targets(%a = %q1) { qc.z %a : !qc.qubit
                                   qc.yield } : {!qc.qubit}, {!qc.qubit}
  qc.ctrl(%q1) targets(%b = %q2) { qc.z %b : !qc.qubit
                                   qc.yield } : {!qc.qubit}, {!qc.qubit}

  return
}
