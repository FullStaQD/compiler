// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Modular and photonic: three modules on an optical switch.
//
// This target exercises the component of a configuration that carries
// entanglement in its own right. A configuration is `C = (pi, <, Lambda)`, whose
// third component is the set of live entanglement links. A link exists because
// `qcc.link.generate` heralded it, so it is connectivity the program creates on
// demand rather than connectivity implied by where the qubits sit.
//
// Links are linear resources: `qcc.link.generate` produces one, a teleported gate
// consumes it, and `qcc.link.consume` records the consumption. K follows: the
// parties form a facet while the link is live and cease to once it is consumed,
// so a program that entangles across a fibre twice heralds twice.
//
// A `link` edge declares which of two kinds it is. `persistent = true` describes
// a coupler that exists unconditionally, such as a superconducting resonator or a
// Rydberg blockade disc. `persistent = false` describes a resource, which
// contributes to K exactly when one has been generated over it.
//===----------------------------------------------------------------------===//

#photonic = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @m0, kind = module, capacity = 4>,
      #qcc.site<name = @m1, kind = module, capacity = 4>,
      #qcc.site<name = @m2, kind = module, capacity = 4>
    ],
    edges = [
      // link, rank 2, consumed by use: a heralded Bell pair over a fibre.
      #qcc.edge<link, [@m0, @m1], {id = "fibre_01", persistent = false, latency_us = 5.500000e+03}>,
      #qcc.edge<link, [@m1, @m2], {id = "fibre_12", persistent = false, latency_us = 6.100000e+03}>,

      // The switch is a line: @m0 and @m2 have no fibre between them.

      // control, rank 3: one heralding detector array serves all three modules.
      // It constrains which entanglement attempts may overlap in time and leaves
      // K untouched, which is why scheduling structure has an edge kind of its
      // own rather than being folded into connectivity.
      #qcc.edge<control, [@m0, @m1, @m2], {id = "bsa_detector"}>
    ]>,
  maxFacetRank = 2,        // a fibre yields a Bell pair
  downwardClosed = true,
  partitioning = false,    // a live link's facet overlaps both module chains
  facetOrdering = none>

// A link is heralded, used and consumed.

func.func @herald_use_spend(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q4 = qc.static 4 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @m0 = [[0, 1, 2, 3]],
      @m1 = [[4, 5, 6, 7]],
      @m2 = [[8, 9, 10, 11]]>) : !qcc.config<#photonic>

  // Inside one module the qubits are co-located, so the chain is a facet and no
  // link is required.
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit

  // Herald a Bell pair between a qubit in @m0 and one in @m1. Party i occupies
  // site i of the edge, so the operands are given in the fibre's order.
  %c1 = qcc.link.generate %c0 {via = "fibre_01", qubits = array<i64: 0, 4>}
      : !qcc.config<#photonic>

  // {0, 4} is now a facet, and neither qubit moved.
  qc.rzz(%theta) %q0, %q4 : !qc.qubit, !qc.qubit

  // Teleporting the gate consumes the pair.
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c2 = qcc.link.consume %c1 {qubits = array<i64: 0, 4>} : !qcc.config<#photonic>

  // The same gate on the same two qubits, which have not moved. The resource is
  // gone and the facet with it: connectivity that was exhausted rather than
  // connectivity that moved.
  // expected-error @+1 {{operand set {0, 4} is not executable in the configuration reaching it: no facet of K contains it}}
  qc.rzz(%theta) %q0, %q4 : !qc.qubit, !qc.qubit

  return
}

// -----

#photonic = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @m0, kind = module, capacity = 4>,
      #qcc.site<name = @m1, kind = module, capacity = 4>,
      #qcc.site<name = @m2, kind = module, capacity = 4>
    ],
    edges = [
      #qcc.edge<link, [@m0, @m1], {id = "fibre_01", persistent = false}>,
      #qcc.edge<link, [@m1, @m2], {id = "fibre_12", persistent = false}>
    ]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

// Several links can be live at once, each forming its own facet.

func.func @two_hops_is_two_links(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q4 = qc.static 4 : !qc.qubit
  %q5 = qc.static 5 : !qc.qubit
  %q8 = qc.static 8 : !qc.qubit

  %c0 = qcc.config.init(#qcc.placement<
      @m0 = [[0, 1]], @m1 = [[4, 5]], @m2 = [[8, 9]]>) : !qcc.config<#photonic>

  // Both hops are heralded: two live links and two facets, with the middle module
  // holding one end of each.
  %c1 = qcc.link.generate %c0 {via = "fibre_01", qubits = array<i64: 0, 4>} : !qcc.config<#photonic>
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c2 = qcc.link.generate %c1 {via = "fibre_12", qubits = array<i64: 5, 8>} : !qcc.config<#photonic>

  qc.rzz(%theta) %q0, %q4 : !qc.qubit, !qc.qubit
  qc.rzz(%theta) %q5, %q8 : !qc.qubit, !qc.qubit

  // Two links end to end are not one link. Joining @m0 to @m2 requires an
  // entanglement swap, which the router emits from the operations above; the
  // verifier does not treat the path itself as a facet.
  // expected-error @+1 {{operand set {0, 8} is not executable in the configuration reaching it: no facet of K contains it}}
  qc.rzz(%theta) %q0, %q8 : !qc.qubit, !qc.qubit

  return
}

// -----

#photonic = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @m0, kind = module, capacity = 4>, #qcc.site<name = @m1, kind = module, capacity = 4>],
    edges = [#qcc.edge<link, [@m0, @m1], {id = "fibre_01", persistent = false}>]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

// A link is consumed by use, so consuming one that was never heralded is
// rejected.
func.func @a_link_is_heralded_before_it_is_spent(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q4 = qc.static 4 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@m0 = [[0]], @m1 = [[4]]>) : !qcc.config<#photonic>
  // expected-error @+1 {{no live link joins {0, 4} in the configuration reaching this operation}}
  %c1 = qcc.link.consume %c0 {qubits = array<i64: 0, 4>} : !qcc.config<#photonic>
  qc.rzz(%theta) %q0, %q4 : !qc.qubit, !qc.qubit
  return
}

// -----

#photonic = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @m0, kind = module, capacity = 4>, #qcc.site<name = @m1, kind = module, capacity = 4>],
    edges = [#qcc.edge<link, [@m0, @m1], {id = "fibre_01", persistent = false}>]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

// A communication qubit holds one pair at a time, so the second herald must wait
// for the first to be consumed.
func.func @one_link_per_qubit(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q4 = qc.static 4 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@m0 = [[0]], @m1 = [[4, 5]]>) : !qcc.config<#photonic>
  %c1 = qcc.link.generate %c0 {via = "fibre_01", qubits = array<i64: 0, 4>} : !qcc.config<#photonic>
  // expected-error @+1 {{qubit 0 already holds a live link}}
  %c2 = qcc.link.generate %c1 {via = "fibre_01", qubits = array<i64: 0, 5>} : !qcc.config<#photonic>
  qc.rzz(%theta) %q0, %q4 : !qc.qubit, !qc.qubit
  return
}

// -----

#photonic = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @m0, kind = module, capacity = 4>, #qcc.site<name = @m1, kind = module, capacity = 4>],
    edges = [#qcc.edge<link, [@m0, @m1], {id = "fibre_01", persistent = false}>]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

// The photon leaves the module its emitter is in, so party i must occupy site i
// of the fibre.
func.func @a_party_sits_at_its_end_of_the_fibre(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@m0 = [[0, 1]], @m1 = [[4]]>) : !qcc.config<#photonic>
  // expected-error @+1 {{qubit 1 sits at @m0, but link edge 'fibre_01' expects that party at @m1}}
  %c1 = qcc.link.generate %c0 {via = "fibre_01", qubits = array<i64: 0, 1>} : !qcc.config<#photonic>
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

// The same edge kind covers the permanent case. With `persistent = true` the
// coupler exists unconditionally: it is a facet from the start, with nothing to
// herald and nothing to exhaust.
#superconducting = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @q0, kind = fixed, capacity = 1>, #qcc.site<name = @q1, kind = fixed, capacity = 1>],
    edges = [#qcc.edge<link, [@q0, @q1], {id = "coupler_01", persistent = true}>]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

func.func @a_coupler_needs_no_heralding(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<@q0 = [[0]], @q1 = [[1]]>) : !qcc.config<#superconducting>
  // Legal immediately, with no rule applied.
  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  return
}

// -----

#superconducting = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @q0, kind = fixed, capacity = 1>, #qcc.site<name = @q1, kind = fixed, capacity = 1>],
    edges = [#qcc.edge<link, [@q0, @q1], {id = "coupler_01", persistent = true}>]>,
  maxFacetRank = 2, downwardClosed = true, partitioning = false, facetOrdering = none>

func.func @a_coupler_is_not_a_resource() {
  %c0 = qcc.config.init(#qcc.placement<@q0 = [[0]], @q1 = [[1]]>) : !qcc.config<#superconducting>
  // expected-error @+1 {{link edge 'coupler_01' is persistent, so it is available without being generated}}
  %c1 = qcc.link.generate %c0 {via = "coupler_01", qubits = array<i64: 0, 1>} : !qcc.config<#superconducting>
  return
}
