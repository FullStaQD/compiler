// RUN: qcc-opt %s --split-input-file --qcc-schedule-concurrency 2>&1 >/dev/null | FileCheck %s --check-prefix=REPORT
// RUN: qcc-opt %s --split-input-file --qcc-schedule-concurrency 2>/dev/null | FileCheck %s

//===----------------------------------------------------------------------===//
// Concurrency from control edges.
//
// Two independent constraints decide whether rewrites can share a time step.
// Their double-pushout footprints must be disjoint, since otherwise applying
// them in either order gives different configurations; and they must contend for
// no `control` hyperedge, since otherwise they compete for one waveform
// generator, which no reasoning about placement can discover.
//
// The two modules below differ only in how many waveform generators drive them,
// and their schedules differ by a factor of two.
//===----------------------------------------------------------------------===//

#shared_awg = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @a0, kind = trap, capacity = 6>, #qcc.site<name = @a1, kind = trap, capacity = 6>,
             #qcc.site<name = @b0, kind = trap, capacity = 6>, #qcc.site<name = @b1, kind = trap, capacity = 6>],
    edges = [#qcc.edge<transport, [@a0, @a1], {id = "sa", latency_us = 8.000000e+01, bidirectional = true}>,
             #qcc.edge<transport, [@b0, @b1], {id = "sb", latency_us = 8.000000e+01, bidirectional = true}>,
             // ONE generator drives every shuttling electrode on the machine.
             #qcc.edge<control, [@a0, @a1, @b0, @b1], {id = "awg_all"}>]>,
  maxFacetRank = 6, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @one_generator_for_the_whole_machine() {
  %c0 = qcc.config.init(#qcc.placement<@a0 = [[0, 1]], @a1 = [], @b0 = [[2, 3]], @b1 = []>)
      : !qcc.config<#shared_awg>
  %c1 = qcc.ion.split %c0 {site = @a0, chain = 0 : i64, end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#shared_awg>
  %c2 = qcc.ion.split %c1 {site = @b0, chain = 0 : i64, end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#shared_awg>
  %c3 = qcc.ion.transport %c2 {from = @a0, chain = 0 : i64, via = "sa"} : !qcc.config<#shared_awg>
  %c4 = qcc.ion.transport %c3 {from = @b0, chain = 0 : i64, via = "sb"} : !qcc.config<#shared_awg>
  return
}

// Nothing overlaps: four rewrites in four steps.
// REPORT: scheduled 4 rewrite(s) into 4 step(s); critical path 1.600000e+02 us
// The count of pairs an additional generator would allow to run concurrently.
// REPORT: pair(s) were serialised only by a shared control resource

// -----

#own_awg = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @a0, kind = trap, capacity = 6>, #qcc.site<name = @a1, kind = trap, capacity = 6>,
             #qcc.site<name = @b0, kind = trap, capacity = 6>, #qcc.site<name = @b1, kind = trap, capacity = 6>],
    edges = [#qcc.edge<transport, [@a0, @a1], {id = "sa", latency_us = 8.000000e+01, bidirectional = true}>,
             #qcc.edge<transport, [@b0, @b1], {id = "sb", latency_us = 8.000000e+01, bidirectional = true}>,
             // The same machine, with one generator per module.
             #qcc.edge<control, [@a0, @a1], {id = "awg_a"}>,
             #qcc.edge<control, [@b0, @b1], {id = "awg_b"}>]>,
  maxFacetRank = 6, downwardClosed = true, partitioning = true, facetOrdering = linear>

// CHECK-LABEL: func.func @one_generator_per_module
func.func @one_generator_per_module() {
  %c0 = qcc.config.init(#qcc.placement<@a0 = [[0, 1]], @a1 = [], @b0 = [[2, 3]], @b1 = []>)
      : !qcc.config<#own_awg>
  // The two splits touch different sites and different generators, so they go
  // in the same step -- even though the SSA chain totally orders them, which is
  // an artefact of threading one configuration value rather than a fact about
  // the machine.
  // CHECK: qcc.ion.split {{.*}}qcc.step = 0
  %c1 = qcc.ion.split %c0 {site = @a0, chain = 0 : i64, end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#own_awg>
  // CHECK: qcc.ion.split {{.*}}qcc.step = 0
  %c2 = qcc.ion.split %c1 {site = @b0, chain = 0 : i64, end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#own_awg>
  // CHECK: qcc.ion.transport {{.*}}qcc.step = 1
  %c3 = qcc.ion.transport %c2 {from = @a0, chain = 0 : i64, via = "sa"} : !qcc.config<#own_awg>
  // CHECK: qcc.ion.transport {{.*}}qcc.step = 1
  %c4 = qcc.ion.transport %c3 {from = @b0, chain = 0 : i64, via = "sb"} : !qcc.config<#own_awg>
  return
}

// The same program over the same segments, in half the steps.
// REPORT: scheduled 4 rewrite(s) into 2 step(s); critical path 8.000000e+01 us
