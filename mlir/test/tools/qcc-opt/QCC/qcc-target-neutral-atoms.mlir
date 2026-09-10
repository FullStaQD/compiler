// RUN: qcc-opt %s --split-input-file --qcc-verify-connectivity --verify-diagnostics

//===----------------------------------------------------------------------===//
// Neutral atoms: a 2x3 SLM trap array under an AOD.
//
// This target uses two structures that neither the ion nor the superconducting
// instantiation does:
//
//   * `rigid` edges carry a whole AOD axis. One waveform displaces every atom on
//     the axis simultaneously, so `qcc.atom.aod_move` is a single rule
//     application over k sites. The parallelism is stated once in the substrate
//     and is available to every pass downstream.
//
//   * Rydberg blockade discs are `link` edges, which may overlap. An atom in the
//     middle of the array lies inside several discs, so K records it as a member
//     of several facets, each of which is available to a gate.
//     `partitioning = false` records this.
//
// The rule set is `qcc.atom.aod_move` and `qcc.atom.slm_transfer`, both checked
// against the same K that checks every gate.
//===----------------------------------------------------------------------===//

#atoms = #qcc.device<
  substrate = #qcc.substrate<
    // A trap site holds one atom. It is a location rather than a qubit, so the
    // AOD may hand an atom to a different one.
    sites = [
      #qcc.site<name = @t00, kind = trap, capacity = 1>,
      #qcc.site<name = @t01, kind = trap, capacity = 1>,
      #qcc.site<name = @t02, kind = trap, capacity = 1>,
      #qcc.site<name = @t10, kind = trap, capacity = 1>,
      #qcc.site<name = @t11, kind = trap, capacity = 1>,
      #qcc.site<name = @t12, kind = trap, capacity = 1>
    ],
    edges = [
      // rigid, rank 3: an AOD row. One waveform moves three atoms in one step.
      #qcc.edge<rigid, [@t00, @t01, @t02], {id = "aod_row_0", axis = "x", latency_us = 5.000000e+02}>,
      #qcc.edge<rigid, [@t10, @t11, @t12], {id = "aod_row_1", axis = "x", latency_us = 5.000000e+02}>,
      // rigid, rank 2: an AOD column.
      #qcc.edge<rigid, [@t00, @t10], {id = "aod_col_0", axis = "y", latency_us = 5.000000e+02}>,
      #qcc.edge<rigid, [@t01, @t11], {id = "aod_col_1", axis = "y", latency_us = 5.000000e+02}>,
      #qcc.edge<rigid, [@t02, @t12], {id = "aod_col_2", axis = "y", latency_us = 5.000000e+02}>,

      // link, rank 4: a Rydberg blockade disc, present whenever the atoms are,
      // hence `persistent = true`. The two discs share @t01 and @t11, so an atom
      // at either belongs to both.
      #qcc.edge<link, [@t00, @t01, @t10, @t11], {id = "blockade_left", persistent = true}>,
      #qcc.edge<link, [@t01, @t02, @t11, @t12], {id = "blockade_right", persistent = true}>
    ]>,
  maxFacetRank = 3,        // multi-qubit Rydberg gates up to 3 atoms
  downwardClosed = true,   // any subset of a disc may be addressed
  partitioning = false,    // an atom in the overlap is in two facets at once
  facetOrdering = none>    // atoms in a disc carry no order

func.func @rigid_row_move(%theta: f64) {
  %a0 = qc.static 0 : !qc.qubit
  %a1 = qc.static 1 : !qc.qubit
  %a2 = qc.static 2 : !qc.qubit

  // Row 0 loaded, row 1 empty.
  %c0 = qcc.config.init(#qcc.placement<@t00 = [[0]], @t01 = [[1]], @t02 = [[2]]>)
      : !qcc.config<#atoms>

  // Both inside blockade_left.
  qc.rzz(%theta) %a0, %a1 : !qc.qubit, !qc.qubit
  // Atom 1 sits in the overlap, so it pairs to the right as readily as to the
  // left: one atom in two facets.
  qc.rzz(%theta) %a1, %a2 : !qc.qubit, !qc.qubit

  // One action moves three atoms one row down. This is why `rigid` is a
  // hyperedge: moving the row costs what moving one atom costs.
  // expected-note @+1 {{connectivity in force here is defined by this value}}
  %c1 = qcc.atom.aod_move %c0 {via = "aod_row_0", to = [@t10, @t11, @t12]}
      : !qcc.config<#atoms>

  // The discs span both rows, so the row retains its structure and the same pair
  // remains addressable after the move.
  qc.rzz(%theta) %a0, %a1 : !qc.qubit, !qc.qubit

  // The two ends of the row lie in different discs both before and after the
  // move: a rigid displacement moves the atoms, not the blockade geometry.
  // expected-error @+1 {{operand set {0, 2} is not executable in the configuration reaching it: no facet of K contains it}}
  qc.rzz(%theta) %a0, %a2 : !qc.qubit, !qc.qubit

  return
}

// -----

#atoms = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @t00, kind = trap, capacity = 1>,
      #qcc.site<name = @t01, kind = trap, capacity = 1>,
      #qcc.site<name = @t02, kind = trap, capacity = 1>,
      #qcc.site<name = @t10, kind = trap, capacity = 1>,
      #qcc.site<name = @t11, kind = trap, capacity = 1>,
      #qcc.site<name = @t12, kind = trap, capacity = 1>
    ],
    edges = [
      #qcc.edge<rigid, [@t00, @t01, @t02], {id = "aod_row_0", axis = "x"}>,
      #qcc.edge<rigid, [@t10, @t11, @t12], {id = "aod_row_1", axis = "x"}>,
      #qcc.edge<rigid, [@t00, @t10], {id = "aod_col_0", axis = "y"}>,
      #qcc.edge<rigid, [@t01, @t11], {id = "aod_col_1", axis = "y"}>,
      #qcc.edge<rigid, [@t02, @t12], {id = "aod_col_2", axis = "y"}>,
      #qcc.edge<link, [@t00, @t01, @t10, @t11], {id = "blockade_left", persistent = true}>,
      #qcc.edge<link, [@t01, @t02, @t11, @t12], {id = "blockade_right", persistent = true}>
    ]>,
  maxFacetRank = 3, downwardClosed = true, partitioning = false, facetOrdering = none>

// Defect-free assembly: stochastic loading leaves gaps, and the tweezer closes
// them. The gate the program requires is unavailable in the array as loaded, and
// one transfer makes it available.

func.func @close_the_gap(%theta: f64) {
  %a0 = qc.static 0 : !qc.qubit
  %a1 = qc.static 1 : !qc.qubit
  %a2 = qc.static 2 : !qc.qubit

  // Loading left a hole at @t01: atom 1 landed one trap too far right.
  //   blockade_left  = {0, 2}
  //   blockade_right = {1, 2}
  %c0 = qcc.config.init(#qcc.placement<@t00 = [[0]], @t02 = [[1]], @t11 = [[2]]>)
      : !qcc.config<#atoms>

  // Each of these is inside a disc as loaded.
  qc.rzz(%theta) %a0, %a2 : !qc.qubit, !qc.qubit
  qc.rzz(%theta) %a1, %a2 : !qc.qubit, !qc.qubit

  // Carry atom 1 one trap left, along the row the tweezer is already on.
  %c1 = qcc.atom.slm_transfer %c0 {from = @t02, to = @t01} : !qcc.config<#atoms>
  //   blockade_left  = {0, 1, 2}
  //   blockade_right = {1, 2}

  // {0, 1} was in no facet before the transfer and is in one now. The gate is
  // unchanged; the array beneath it is not.
  qc.rzz(%theta) %a0, %a1 : !qc.qubit, !qc.qubit

  // The three atoms now form a single facet, so one three-atom Rydberg gate
  // replaces three pairwise gates.
  qc.ctrl(%a0, %a1) targets(%y = %a2) { qc.z %y : !qc.qubit
                                        qc.yield } : {!qc.qubit, !qc.qubit}, {!qc.qubit}

  return
}

// -----

#atoms = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @t00, kind = trap, capacity = 1>,
      #qcc.site<name = @t01, kind = trap, capacity = 1>,
      #qcc.site<name = @t02, kind = trap, capacity = 1>,
      #qcc.site<name = @t10, kind = trap, capacity = 1>,
      #qcc.site<name = @t11, kind = trap, capacity = 1>,
      #qcc.site<name = @t12, kind = trap, capacity = 1>
    ],
    edges = [
      #qcc.edge<rigid, [@t00, @t01, @t02], {id = "aod_row_0", axis = "x"}>,
      #qcc.edge<rigid, [@t10, @t11, @t12], {id = "aod_row_1", axis = "x"}>,
      #qcc.edge<link, [@t00, @t01, @t10, @t11], {id = "blockade_left", persistent = true}>
    ]>,
  maxFacetRank = 3, downwardClosed = true, partitioning = false, facetOrdering = none>

// AOD traps are held by a pair of crossed acousto-optic deflectors, and two traps
// on one axis cannot swap places, since the frequencies addressing them would
// have to cross. The substrate's site order is the geometric order along the
// axis, so the check is order preservation.
func.func @traps_cannot_pass_through_each_other() {
  %c0 = qcc.config.init(#qcc.placement<@t00 = [[0]], @t01 = [[1]], @t02 = [[2]]>)
      : !qcc.config<#atoms>
  // expected-error @+1 {{move is crossing: @t00 and @t01 would swap order on the way to @t12 and @t11}}
  %c1 = qcc.atom.aod_move %c0 {via = "aod_row_0", to = [@t12, @t11, @t10]}
      : !qcc.config<#atoms>
  return
}

// -----

#atoms = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @t00, kind = trap, capacity = 1>,
      #qcc.site<name = @t01, kind = trap, capacity = 1>,
      #qcc.site<name = @t02, kind = trap, capacity = 1>,
      #qcc.site<name = @t10, kind = trap, capacity = 1>,
      #qcc.site<name = @t11, kind = trap, capacity = 1>,
      #qcc.site<name = @t12, kind = trap, capacity = 1>
    ],
    edges = [
      #qcc.edge<rigid, [@t00, @t01, @t02], {id = "aod_row_0", axis = "x"}>,
      #qcc.edge<rigid, [@t10, @t11, @t12], {id = "aod_row_1", axis = "x"}>,
      #qcc.edge<link, [@t00, @t01, @t10, @t11], {id = "blockade_left", persistent = true}>
    ]>,
  maxFacetRank = 3, downwardClosed = true, partitioning = false, facetOrdering = none>

// The row moves as a unit and arrives as a unit, so one occupied destination
// rejects the whole action. A per-atom model would have to re-derive this
// constraint for every atom on the axis.
func.func @the_whole_row_lands_or_none_of_it_does(%theta: f64) {
  %a0 = qc.static 0 : !qc.qubit
  %a1 = qc.static 1 : !qc.qubit
  %c0 = qcc.config.init(#qcc.placement<
      @t00 = [[0]], @t01 = [[1]], @t02 = [[2]], @t11 = [[3]]>) : !qcc.config<#atoms>
  // expected-error @+1 {{would place 2 atoms at @t11, exceeding its capacity of 1}}
  %c1 = qcc.atom.aod_move %c0 {via = "aod_row_0", to = [@t10, @t11, @t12]}
      : !qcc.config<#atoms>
  qc.rzz(%theta) %a0, %a1 : !qc.qubit, !qc.qubit
  return
}

// -----

#atoms = #qcc.device<
  substrate = #qcc.substrate<
    sites = [
      #qcc.site<name = @t00, kind = trap, capacity = 1>,
      #qcc.site<name = @t01, kind = trap, capacity = 1>,
      #qcc.site<name = @t02, kind = trap, capacity = 1>,
      #qcc.site<name = @t12, kind = trap, capacity = 1>
    ],
    edges = [
      #qcc.edge<rigid, [@t00, @t01, @t02], {id = "aod_row_0", axis = "x"}>,
      #qcc.edge<rigid, [@t02, @t12], {id = "aod_col_2", axis = "y"}>,
      #qcc.edge<link, [@t00, @t01], {id = "blockade_left", persistent = true}>
    ]>,
  maxFacetRank = 3, downwardClosed = true, partitioning = false, facetOrdering = none>

// A tweezer travels along one axis at a time. @t00 and @t12 share no axis, so the
// handoff requires two transfers through @t02.
func.func @a_tweezer_travels_along_an_axis() {
  %c0 = qcc.config.init(#qcc.placement<@t00 = [[0]]>) : !qcc.config<#atoms>
  // expected-error @+1 {{no rigid edge is incident to both @t00 and @t12}}
  %c1 = qcc.atom.slm_transfer %c0 {from = @t00, to = @t12} : !qcc.config<#atoms>
  return
}
