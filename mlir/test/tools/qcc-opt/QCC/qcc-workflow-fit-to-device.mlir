// RUN: qcc-opt %s --qcc-attach-device=device=sc-grid-4x4 --qcc-place-qubits | FileCheck %s --check-prefix=SC
// RUN: qcc-opt %s --qcc-attach-device=device=sc-grid-4x4 --qcc-place-qubits --qcc-route=emit-remarks=true 2>&1 | FileCheck %s --check-prefix=SC-ROUTE
// RUN: qcc-opt %s --qcc-attach-device=device=qccd-linear-2 --qcc-place-qubits | FileCheck %s --check-prefix=ION
// RUN: qcc-opt %s --qcc-attach-device=device=qccd-linear-2 --qcc-place-qubits --qcc-route --qcc-verify-connectivity | FileCheck %s --check-prefix=ION-OK
// RUN: not qcc-opt %s --qcc-attach-device=device=nonexistent 2>&1 | FileCheck %s --check-prefix=UNKNOWN

//===----------------------------------------------------------------------===//
// Workflow 1: a hardware-agnostic program fitted to a named machine.
//
// The input describes no device: no configuration, no substrate, only gates on
// hardware qubits. `--qcc-attach-device` records which machine to compile for,
// and `--qcc-place-qubits` decides where the qubits start out on it. Everything
// after that is the same dialect the hardware-aware workflow uses.
//
// The same input is fitted below to two machines with different outcomes, which
// is what keeping the machine out of the program allows.
//===----------------------------------------------------------------------===//

func.func @agnostic(%theta: f64) {
  %q0 = qc.static 0 : !qc.qubit
  %q1 = qc.static 1 : !qc.qubit
  %q5 = qc.static 5 : !qc.qubit

  qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
  qc.rzz(%theta) %q0, %q5 : !qc.qubit, !qc.qubit
  return
}

//===----------------------------------------------------------------------===//
// Superconducting: capacity-1 sites, so placement is the identity mapping and the
// coupling map is derived from the machine's reported couplings.
//===----------------------------------------------------------------------===//

// The reported coupling map becomes persistent link edges.
// SC-DAG: #qcc.edge<link, [@q0, @q1], {id = "coupler_q0_q1", persistent = true}>
// SC-DAG: #qcc.edge<link, [@q0, @q4], {id = "coupler_q0_q4", persistent = true}>
// SC-DAG: module attributes {qcc.device = #{{.+}}}
// SC-LABEL: func.func @agnostic
// Sites the program does not use are listed too, so the initial configuration
// describes the whole machine.
// SC: qcc.config.init(#qcc.placement<@q0 = {{\[\[}}0]], @q1 = {{\[\[}}1]], @q2 = {{\[\[}}5]], @q3 = [],

// The first gate lands on a coupled pair and the second does not. On a machine
// with no rewrite rules the only remedy is SWAP insertion, which the router
// reports.
// SC-ROUTE: remark: needs routing: {0, 5} is not executable here
// SC-ROUTE: note: currently 0 at @q0, 5 at @q2
// SC-ROUTE: note: this machine has no rewrite rules, so no shuttle can fix it

//===----------------------------------------------------------------------===//
// Trapped ions: a trap is a zone, so placement packs all three qubits into the
// first one and co-location makes both gates executable without routing. The
// program itself is unchanged.
//===----------------------------------------------------------------------===//

// The supplement supplies what a coupling map cannot express: the segments and
// the waveform generator that serialises them.
// ION-DAG: #qcc.edge<transport, [@trap_a, @j0], {bidirectional = true, id = "seg_aj"
// ION-DAG: #qcc.edge<control, [@trap_a, @j0, @trap_b], {id = "awg_shuttle"}>
// ION-DAG: module attributes {qcc.device = #{{.+}}}
// ION-LABEL: func.func @agnostic
// ION: qcc.config.init(#qcc.placement<@trap_a = {{\[\[}}0, 1, 5]], @j0 = [], @trap_b = []>)

// Routing and verification both pass, so the pipeline runs to completion.
// ION-OK-LABEL: func.func @agnostic
// ION-OK-COUNT-2: qc.rzz

//===----------------------------------------------------------------------===//
// An unknown machine name is rejected with the list of machines this build
// knows.
//===----------------------------------------------------------------------===//

// UNKNOWN: unknown machine 'nonexistent'; this build knows
// UNKNOWN-SAME: sc-grid-4x4
// UNKNOWN-SAME: qccd-linear-2
