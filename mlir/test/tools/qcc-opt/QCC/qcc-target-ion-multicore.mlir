// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Trapped ions: a reconfigurable multicore machine with several long chains per
// module, a junction network for shuttling between them, and a photonic
// interconnect between modules.
//
// Every edge kind the dialect defines is exercised here:
//
//   transport  Binary displacement along a segment. Routing between two chains
//              is several hops through a junction, which is a site of degree 3
//              rather than an edge of arity 3.
//   control    Rank-4 AWG groups and rank-2 laser groups over overlapping site
//              sets. Expanding them into cliques yields one 4-clique and loses
//              which conflict comes from which resource.
//   link       A photonic interconnect between modules, forming a facet from
//              qubits that share no site and that no shuttle could bring
//              together, since there is no transport path between the modules.
//   rigid      Absent: on a QCCD machine ions are displaced one chain at a time,
//              which is why transport here is rank 2.
//
// This target is `facetOrdering = linear` and `partitioning = false` at the same
// time: ordered chains that a split can cut at the ends, with photonic facets
// laid across them. Tying chain order to partitioning would withdraw the whole
// ion rule set from a machine with an optical interconnect.
//===----------------------------------------------------------------------===//

#multicore = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      // Module A: two storage chains, a junction and a comm zone.
      #qcc.site<name = @a_c0, kind = trap, capacity = 36>,
      #qcc.site<name = @a_c1, kind = trap, capacity = 36>,
      #qcc.site<name = @a_j, kind = junction, capacity = 4>,
      #qcc.site<name = @a_comm, kind = gatezone, capacity = 2>,
      // Module B: identical.
      #qcc.site<name = @b_c0, kind = trap, capacity = 36>,
      #qcc.site<name = @b_c1, kind = trap, capacity = 36>,
      #qcc.site<name = @b_j, kind = junction, capacity = 4>,
      #qcc.site<name = @b_comm, kind = gatezone, capacity = 2>
    ],
    edges = [
      // transport: rank 2 throughout. @a_j has degree 3, that is, three segments
      // meet there, which is three binary displacements rather than one ternary
      // edge. Routing @a_c0 to @a_comm is two hops.
      #qcc.edge<transport, [@a_c0, @a_j], {id = "seg_a_c0j", latency_us = 1.500000e+02, bidirectional = true, max_chain_len = 4 : i64}>,
      #qcc.edge<transport, [@a_c1, @a_j], {id = "seg_a_c1j", latency_us = 1.500000e+02, bidirectional = true, max_chain_len = 4 : i64}>,
      #qcc.edge<transport, [@a_j, @a_comm], {id = "seg_a_jm", latency_us = 9.000000e+01, bidirectional = true, max_chain_len = 2 : i64}>,
      #qcc.edge<transport, [@b_c0, @b_j], {id = "seg_b_c0j", latency_us = 1.500000e+02, bidirectional = true, max_chain_len = 4 : i64}>,
      #qcc.edge<transport, [@b_c1, @b_j], {id = "seg_b_c1j", latency_us = 1.500000e+02, bidirectional = true, max_chain_len = 4 : i64}>,
      #qcc.edge<transport, [@b_j, @b_comm], {id = "seg_b_jm", latency_us = 9.000000e+01, bidirectional = true, max_chain_len = 2 : i64}>,

      // There is no transport edge between the modules: an ion cannot be carried
      // from A to B, and the only way across is the photonic link.

      // link: the photonic interconnect. Rank 2, and `persistent = false`, since
      // a heralded Bell pair is consumed by use.
      #qcc.edge<link, [@a_comm, @b_comm], {id = "photon_ab", persistent = false, bell_rate_hz = 1.820000e+02}>,

      // control: two hyperedges over overlapping site sets, from two different
      // resources. A graph cannot distinguish them.
      // rank 4: one AWG drives every shuttling electrode in module A, so no two
      // transports in A may overlap in time, including transports on disjoint
      // segments, which a placement-only analysis would call parallel.
      #qcc.edge<control, [@a_c0, @a_c1, @a_j, @a_comm], {id = "awg_a"}>,
      #qcc.edge<control, [@b_c0, @b_c1, @b_j, @b_comm], {id = "awg_b"}>,
      // rank 2 over a subset of the same sites: one Raman beam path is switched
      // between module A's two storage chains, so gates in @a_c0 and @a_c1
      // serialise even though they share no ion, no segment and no AWG conflict.
      #qcc.edge<control, [@a_c0, @a_c1], {id = "raman_a"}>,
      #qcc.edge<control, [@b_c0, @b_c1], {id = "raman_b"}>,
      // rank 2 spanning the modules: a single heralding detector serves both comm
      // zones, so the two modules cannot attempt entanglement concurrently.
      #qcc.edge<control, [@a_comm, @b_comm], {id = "bsa_detector"}>
    ]>,
  maxFacetRank = 36,       // full chain addressable in one MS, IonQ Forte scale
  downwardClosed = true,   // individual addressing: any subset of a chain
  partitioning = false,    // the photonic facet overlaps the comm-zone chains
  facetOrdering = linear>  // chains are ordered; splits only at the ends

// A cross-module entanglement, end to end.

func.func @entangle_across_modules(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q2 = qc.static 2 : !qc.qubit
  %q8 = qc.static 8 : !qc.qubit
  %q16 = qc.static 16 : !qc.qubit

  // a_c0: [0..7]   a_c1: [8..15]   b_c0: [16..23]   b_c1: [24..31]
  %c0 = qcc.config.init(#qcc.placement<
      @a_c0 = [[0, 1, 2, 3, 4, 5, 6, 7]],
      @a_c1 = [[8, 9, 10, 11, 12, 13, 14, 15]],
      @a_j = [], @a_comm = [],
      @b_c0 = [[16, 17, 18, 19, 20, 21, 22, 23]],
      @b_c1 = [[24, 25, 26, 27, 28, 29, 30, 31]],
      @b_j = [], @b_comm = []>) : !qcc.config<#multicore>

  // A three-body MS inside chain a_c0. {0, 1, 2} is a face of the facet {0..7};
  // contiguity is not required, so {0, 4, 7} verifies equally.
  qc.ctrl(%q0, %q1) targets(%m = %q2) { qc.z %m : !qc.qubit
                                        qc.yield } : {!qc.qubit, !qc.qubit}, {!qc.qubit}

  // Route ion 0 out of chain a_c0 and into module A's comm zone.
  %a1 = qcc.ion.split %c0 {site = @a_c0, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#multicore>
  // Two hops: the junction is traversed by two binary displacements.
  %a2 = qcc.ion.transport %a1 {from = @a_c0, chain = 0 : i64, via = "seg_a_c0j"} : !qcc.config<#multicore>
  %a3 = qcc.ion.transport %a2 {from = @a_j, chain = 0 : i64, via = "seg_a_jm"} : !qcc.config<#multicore>

  // The same for ion 16 in module B.
  %b1 = qcc.ion.split %a3 {site = @b_c0, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#multicore>
  %b2 = qcc.ion.transport %b1 {from = @b_c0, chain = 0 : i64, via = "seg_b_c0j"} : !qcc.config<#multicore>
  %b3 = qcc.ion.transport %b2 {from = @b_j, chain = 0 : i64, via = "seg_b_jm"} : !qcc.config<#multicore>
  // a_c0: [1..7]  a_comm: [0]   b_c0: [17..23]  b_comm: [16]

  // Herald a Bell pair between the two comm zones. No shuttle can achieve this,
  // since the modules are joined by no transport path. Both parties are routed
  // into their comm zones first, so the herald depends on the placement the
  // shuttles produced and the resulting facet depends on the herald.
  %l0 = qcc.link.generate %b3 {via = "photon_ab", qubits = array<i64: 0, 16>}
      : !qcc.config<#multicore>

  // Ions 0 and 16 sit in different modules, and {0, 16} is executable.
  qc.rzz(%theta) %q0, %q16 : !qc.qubit, !qc.qubit

  // Teleporting the gate consumes the pair. The two comm ions are then in
  // unrelated modules again, and the next cross-module gate heralds afresh.
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %l1 = qcc.link.consume %l0 {qubits = array<i64: 0, 16>} : !qcc.config<#multicore>

  // Ion 0 has left chain a_c0, so a gate legal at %c0 is not legal here. Ion 8 is
  // in the other chain and never was.
  // expected-error @+1 {{operand set {0, 8} is not executable in the configuration reaching it: no facet of K contains it}}
  qc.rzz(%theta) %q0, %q8 : !qc.qubit, !qc.qubit

  return
}

// -----

// Routing failures the substrate catches on its own.

#multicore = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @a_c0, kind = trap, capacity = 36>,
      #qcc.site<name = @a_j, kind = junction, capacity = 4>,
      #qcc.site<name = @a_comm, kind = gatezone, capacity = 2>,
      #qcc.site<name = @b_comm, kind = gatezone, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@a_c0, @a_j], {id = "seg_a_c0j", max_chain_len = 4 : i64}>,
      #qcc.edge<transport, [@a_j, @a_comm], {id = "seg_a_jm", max_chain_len = 2 : i64}>,
      #qcc.edge<link, [@a_comm, @b_comm], {id = "photon_ab", persistent = false}>
    ]>,
  maxFacetRank = 36, downwardClosed = true, partitioning = false, facetOrdering = linear>

// The interconnect joins @a_comm and @b_comm for the purposes of K only: no ion
// can travel along it.
func.func @cannot_shuttle_down_a_fibre() {
  %c0 = qcc.config.init(#qcc.placement<@a_comm = [[0]]>) : !qcc.config<#multicore>
  // expected-error @+1 {{substrate declares no transport edge with id 'photon_ab'}}
  %c1 = qcc.ion.transport %c0 {from = @a_comm, chain = 0 : i64, via = "photon_ab"} : !qcc.config<#multicore>
  return
}

// -----

#multicore = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @a_c0, kind = trap, capacity = 36>,
      #qcc.site<name = @a_j, kind = junction, capacity = 4>,
      #qcc.site<name = @a_comm, kind = gatezone, capacity = 2>
    ],
    edges = [
      #qcc.edge<transport, [@a_c0, @a_j], {id = "seg_a_c0j", max_chain_len = 4 : i64}>,
      // The segment can carry four ions; the destination cannot hold them.
      #qcc.edge<transport, [@a_j, @a_comm], {id = "seg_a_jm", max_chain_len = 4 : i64}>
    ]>,
  maxFacetRank = 36, downwardClosed = true, partitioning = false, facetOrdering = linear>

// The comm zone holds two ions in flight rather than a chain. Capacity is the
// transient maximum, and the router must respect it while planning.
func.func @comm_zone_overflows(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@a_j = [[0, 1, 2]], @a_comm = []>) : !qcc.config<#multicore>
  // expected-error @+1 {{would place 3 qubits at @a_comm, exceeding its capacity of 2}}
  %c1 = qcc.ion.transport %c0 {from = @a_j, chain = 0 : i64, via = "seg_a_jm"} : !qcc.config<#multicore>
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}
