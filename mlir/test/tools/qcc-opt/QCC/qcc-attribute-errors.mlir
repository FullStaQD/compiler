// RUN: qcc-opt %s --split-input-file --verify-diagnostics

//===----------------------------------------------------------------------===//
// Layer 0 well-formedness: checks on the machine description itself, reported at
// parse time, before any program is written against it.
//===----------------------------------------------------------------------===//

// Displacement is a binary relation: there is no physical operation moving an
// ion out of {A, B, C}, and a junction is a site of high degree rather than an
// edge of high arity.
#bad_rank = #qcc.substrate<
  sites = [
    #qcc.site<name = @a, kind = trap, capacity = 4>,
    #qcc.site<name = @b, kind = trap, capacity = 4>,
    #qcc.site<name = @c, kind = trap, capacity = 4>
  ],
  // expected-error @+1 {{transport edge must have rank 2, but is incident to 3 sites}}
  edges = [#qcc.edge<transport, [@a, @b, @c], {id = "seg"}>]>

// -----

// A fixed site holds a single qubit rather than acting as a container, and
// nothing enters or leaves it. Permitting an incident transport edge would let a
// superconducting substrate express shuttling.
// expected-error @+1 {{transport edge 'seg' touches fixed site @q0; a fixed site cannot be entered or left}}
#shuttling_transmon = #qcc.substrate<
  sites = [
    #qcc.site<name = @q0, kind = fixed, capacity = 1>,
    #qcc.site<name = @t, kind = trap, capacity = 4>
  ],
  edges = [#qcc.edge<transport, [@q0, @t], {id = "seg"}>]>

// -----

// `via` addresses a transport edge by name, so an unnamed one is unreachable.
#no_id = #qcc.substrate<
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
  // expected-error @+1 {{transport edge must carry a non-empty string `id` property}}
  edges = [#qcc.edge<transport, [@a, @b], {}>]>

// -----

// expected-error @+1 {{edge refers to undeclared site @nowhere}}
#dangling = #qcc.substrate<
  sites = [#qcc.site<name = @a, kind = trap, capacity = 4>],
  edges = [#qcc.edge<control, [@a, @nowhere], {id = "awg"}>]>

// -----

// `partitioning` asserts that the facets of K partition the qubits, which is
// what allows the hypergraph to collapse to an ordered partition without loss. A
// link edge contradicts that by construction: it forms a facet from qubits at
// several sites, overlapping every group it drew from. Declaring the property
// regardless would admit a representation that discards the links.
// expected-error @+1 {{partitioning = true is inconsistent with 1 link edge(s)}}
#lying_about_partition = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
    edges = [#qcc.edge<link, [@a, @b], {id = "fibre", persistent = false}>]>,
  maxFacetRank = 4, downwardClosed = true, partitioning = true, facetOrdering = none>

// -----

// The converse is not an error: a modular ion machine has ordered chains and
// non-partitioning K simultaneously. Its chains are words a rule can split, and
// its photonic links lay facets across them. Rejecting this combination would
// withdraw the whole rule set from any machine with an optical interconnect.
#chains_plus_links = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @a, kind = trap, capacity = 4>, #qcc.site<name = @b, kind = trap, capacity = 4>],
    edges = [#qcc.edge<link, [@a, @b], {id = "fibre", persistent = false}>]>,
  maxFacetRank = 4, downwardClosed = true, partitioning = false, facetOrdering = linear>

func.func @split_is_available_on_a_modular_ion_machine() {
  %c0 = qcc.config.init(#qcc.placement<@a = [[0, 1]], @b = [[2, 3]]>) : !qcc.config<#chains_plus_links>
  %c1 = qcc.ion.split %c0 {site = @a, chain = 0 : i64,
                           end = #qcc.chain_end<head>, count = 1 : i64} : !qcc.config<#chains_plus_links>
  return
}

// -----

// expected-error @+1 {{fixed site @q0 must have capacity 1, but has 4}}
#fat_transmon = #qcc.site<name = @q0, kind = fixed, capacity = 4>

// -----

//===----------------------------------------------------------------------===//
// Layer 1 well-formedness: a placement is a partial injection.
//===----------------------------------------------------------------------===//

// expected-error @+1 {{qubit 1 is placed more than once}}
#bilocated = #qcc.placement<@a = [[0, 1]], @b = [[1, 2]]>

// -----

// An empty potential well is not a group.
// expected-error @+1 {{site @a holds an empty group}}
#empty_well = #qcc.placement<@a = [[0], []]>

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 3>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

// Capacity is the transient maximum a site can ever hold.
func.func @over_capacity() {
  // expected-error @+1 {{places 4 qubits at @trap_a, exceeding its capacity of 3}}
  %c = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2, 3]]>) : !qcc.config<#ion>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 3>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @undeclared_site() {
  // expected-error @+1 {{places qubits at @trap_z, which the substrate does not declare}}
  %c = qcc.config.init(#qcc.placement<@trap_z = [[0]]>) : !qcc.config<#ion>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @not_a_permutation() {
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0, 1, 2]]>) : !qcc.config<#ion>
  // expected-error @+1 {{permutation names position 1 twice}}
  %c1 = qcc.ion.reorder %c0 {site = @trap_a, chain = 0 : i64,
                             permutation = array<i64: 1, 1, 2>} : !qcc.config<#ion>
  return
}

// -----

#ion = #qcc.device<
  substrate = #qcc.substrate<
    sites = [#qcc.site<name = @trap_a, kind = trap, capacity = 6>],
    edges = []>,
  maxFacetRank = 5, downwardClosed = true, partitioning = true, facetOrdering = linear>

func.func @merge_into_itself() {
  %c0 = qcc.config.init(#qcc.placement<@trap_a = [[0], [1, 2]]>) : !qcc.config<#ion>
  // expected-error @+1 {{cannot merge chain 0 into itself}}
  %c1 = qcc.ion.merge %c0 {site = @trap_a, dst = 0 : i64, src = 0 : i64,
                           at = #qcc.chain_end<head>} : !qcc.config<#ion>
  return
}
