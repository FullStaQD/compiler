// RUN: qcc-opt %s --canonicalize --cse | FileCheck %s
// RUN: qcc-opt %s --qcc-verify-connectivity=allow-dynamic=true | FileCheck %s --check-prefix=DYNAMIC

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @a, kind = trap, capacity = 6>,
      #qcc.site<name = @b, kind = trap, capacity = 6>
    ],
    edges = [#qcc.edge<transport, [@a, @b], {id = "seg", latency_us = 8.000000e+01}>]>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

//===----------------------------------------------------------------------===//
// A rewrite operation describes physical motion, and under the dominating-
// definition discipline no gate takes its result as an operand. Were these
// operations `Pure`, an unused configuration chain would be dead code and the
// shuttle it describes would be eliminated. They write a dedicated side-effect
// resource instead, which keeps the chain live and ordered.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @chain_with_no_consumer
func.func @chain_with_no_consumer() {
  // CHECK: %[[C0:.+]] = qcc.config.init
  %c0 = qcc.config.init(#qcc.placement<@a = [[0, 1]], @b = []>) : !qcc.config<#ion>
  // CHECK: %[[C1:.+]] = qcc.ion.split %[[C0]]
  %c1 = qcc.ion.split %c0 {site = @a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
  // CHECK: qcc.ion.transport %[[C1]]
  %c2 = qcc.ion.transport %c1 {from = @a, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  return
}

//===----------------------------------------------------------------------===//
// Two syntactically identical transports are two distinct physical moves. CSE
// must not merge them: the second acts on the configuration the first produced,
// not on the one the first consumed.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @identical_moves_are_not_common_subexpressions
func.func @identical_moves_are_not_common_subexpressions() {
  %c0 = qcc.config.init(#qcc.placement<@a = [[0], [1]]>) : !qcc.config<#ion>
  // CHECK-COUNT-2: qcc.ion.transport
  // CHECK-NOT: qcc.ion.transport
  %c1 = qcc.ion.transport %c0 {from = @a, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  %c2 = qcc.ion.transport %c1 {from = @a, chain = 0 : i64, via = "seg"} : !qcc.config<#ion>
  return
}

//===----------------------------------------------------------------------===//
// `allow-dynamic` downgrades the loop-carried case from an error to a warning,
// for pipelines that intend to emit a runtime co-location check rather than
// prove co-location statically.
//===----------------------------------------------------------------------===//

// DYNAMIC-LABEL: func.func @loop_carried
func.func @loop_carried(%theta: f64, %n: index) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %c0 = qcc.config.init(#qcc.placement<@a = [[0, 1, 2]], @b = []>) : !qcc.config<#ion>

  %cN = scf.for %i = %lb to %n step %step iter_args(%c = %c0) -> (!qcc.config<#ion>) {
    %ca = qcc.ion.split %c {site = @a, chain = 0 : i64,
                            end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#ion>
    // The pass warns rather than failing, and leaves the gate in place.
    // DYNAMIC: qc.rxx
    qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
    scf.yield %ca : !qcc.config<#ion>
  }
  return
}
