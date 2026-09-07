// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --qcc-optimize-shuttling 2>/dev/null | FileCheck %s
// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --qcc-optimize-shuttling 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK

//===----------------------------------------------------------------------===//
// Workflow 2: a hardware-aware program, checked and then optimised.
//
// The input already carries its own machine and its own shuttling schedule, as a
// hand-written or externally generated program does. The compiler does not decide
// where the qubits go; it confirms that the schedule is legal and then makes it
// cheaper without changing what the program computes.
//===----------------------------------------------------------------------===//

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 6>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @trap_b], {id = "seg", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 4 : i64}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// CHECK-LABEL: func.func @hand_written_schedule
func.func @hand_written_schedule(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q2 = qc.static 2 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]], @trap_b = []>)
      : !qcc.config<#ion>

  // Legal against the configuration in force, so verification leaves it alone.
  // CHECK: qc.rzz
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit

  // Ion 0 travels to @trap_b and returns. Nothing is executed in between or
  // afterwards, so the round trip costs 160 us of machine time for no observable
  // effect.
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
  %c2 = qcc.ion.transport %c1 {from = @trap_a, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  %c3 = qcc.ion.transport %c2 {from = @trap_b, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  %c4 = qcc.ion.merge %c3 {site = @trap_a, dst = 0 : i64, src = 1 : i64,
                           at = #qcc.chain_end<head>} : !qcc.config<#ion>

  // REMARK: remark: dropped 4 unobserved rewrite(s), saving 1.600000e+02 us
  // CHECK-NOT: qcc.ion.
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @trap_a, kind = trap, capacity = 6>,
      #qcc.site<name = @trap_b, kind = trap, capacity = 6>
    ],
    edges = [
      #qcc.edge<transport, [@trap_a, @trap_b], {id = "seg", latency_us = 8.000000e+01, bidirectional = true, max_chain_len = 4 : i64}>
    ]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

//===----------------------------------------------------------------------===//
// Motion a later gate depends on is retained. The pass removes only what nothing
// observes, so a schedule whose shuttles are all required passes through
// unchanged.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @every_shuttle_is_load_bearing
func.func @every_shuttle_is_load_bearing(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q3 = qc.static 3 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]], @trap_b = [[3, 4]]>)
      : !qcc.config<#ion>

  // CHECK: qcc.ion.split
  %c1 = qcc.ion.split %c0 {site = @trap_a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
  // CHECK: qcc.ion.transport
  %c2 = qcc.ion.transport %c1 {from = @trap_a, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  // CHECK: qcc.ion.merge
  %c3 = qcc.ion.merge %c2 {site = @trap_b, dst = 0 : i64, src = 1 : i64,
                           at = #qcc.chain_end<head>} : !qcc.config<#ion>

  // This gate requires all three rewrites above: without them ions 0 and 3 are in
  // different traps and {0, 3} is in no facet.
  // CHECK: qc.rzz
  qc.rzz(%theta) %q0, %q3 : !qc.qubit, !qc.qubit
  return
}
