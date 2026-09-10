// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//

#ifndef INCLUDE_QCC_DIALECT_QCC_TRANSFORMS_PASSES_TD_
#define INCLUDE_QCC_DIALECT_QCC_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def VerifyConnectivity : Pass<"qcc-verify-connectivity", "::mlir::func::FuncOp"> {
  let summary = "Check every gate against the connectivity in force where it stands.";
  let description = [{
    This pass is the interface between the `qcc` and `qc` dialects. Gates are
    left unchanged: they remain `qc` operations acting on `!qc.qubit`
    references and take no configuration operand. For each gate, the pass:

    1. resolves its qubits to program indices (`qc.static`);
    2. finds the `!qcc.config` definition that dominates it, which is the
       dialect's notion of connectivity at a program point;
    3. folds that definition's chain back to a `qcc.config.init` to recover the
       configuration, and derives `K` from it;
    4. checks the gate's qubit set against `K`.

    Multi-qubit gates must be a face of a facet (or, on a device that is not
    downward closed, a facet exactly). Single-qubit gates are checked only for
    being placed at all, since local rotations do not depend on co-location.

    A `qc.call` implements the same unitary interface and is checked as the
    joint set of the qubits it passes. This is conservative, since a callee that
    decomposes internally into pairwise interactions does not require all of its
    qubits in one facet. Checking the callee body instead is not possible: it is
    a separate `func.func` whose qubits arrive as block arguments carrying no
    program index, so skipping the call site would leave it unverified.

    A function containing no `!qcc.config` is skipped entirely, so adding this
    pass to a pipeline cannot break a program that does not opt in.

    Because gates carry no operand edge to the configuration, nothing in the IR
    prevents a later pass from sinking a gate across a `qcc.ion.split` that
    invalidates it. The intended discipline is to re-run this pass after any
    reordering; expressing the dependence explicitly would require adding a
    configuration operand to every gate in the `qc` dialect.

    Loop-carried configurations are reported rather than checked; see
    `allow-dynamic`.
  }];

  let options = [
    Option<"allowDynamic", "allow-dynamic", "bool", /*default=*/"false",
           "Emit a warning instead of an error for gates whose configuration is "
           "not statically known. A shuttle inside an `scf.for` makes the "
           "configuration a loop-carried block argument, and unless the body's "
           "net rewrite is the identity, no single placement describes that "
           "program point. Deciding this requires abstract interpretation over "
           "an ordered-partition lattice, which the pass does not implement, so "
           "such gates are rejected by default.">,
    Option<"requireResolvableQubits", "require-resolvable-qubits", "bool",
           /*default=*/"true",
           "Treat a gate whose qubits carry no program index as an error. Set "
           "to false to skip such gates instead, which is useful before "
           "dynamically allocated qubits have been rewritten to `qc.static`.">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::qc::QCDialect",
    "::qcc::conn::QCCDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Workflow 1 — fit a hardware-agnostic program to a machine
//===----------------------------------------------------------------------===//

def AttachDevice : Pass<"qcc-attach-device", "::mlir::ModuleOp"> {
  let summary = "Record which machine a program is being compiled for.";
  let description = [{
    Attaches a `qcc.device` attribute to the module, naming the machine that
    every later pass checks and rewrites against. It is the step that gives a
    hardware-agnostic program a target, and the only place a machine description
    enters the pipeline.

    A description has two possible sources, which compose. `device=<name>`
    selects one of the machines compiled into this build, as enumerated by
    `--list-devices`. A description read from QDMI supplies the coupling-graph
    part, namely sites, couplings, operation arities and blockade radius, and a
    supplement adds the transport, control and rigid edges that a device
    management interface does not describe.

    This option is orthogonal to `--target`: `--target` selects a backend, that
    is, how code is emitted, while `--device` selects a machine, that is, where
    the qubits are and how they move.
  }];

  let options = [
    Option<"device", "device", "std::string", /*default=*/"\"\"",
           "Name of a machine compiled into this build, as listed by "
           "`--list-devices`.">,
  ];

  let dependentDialects = ["::qcc::conn::QCCDialect"];
}

def PlaceQubits : Pass<"qcc-place-qubits", "::mlir::func::FuncOp"> {
  let summary = "Choose where each qubit starts out on the attached machine.";
  let description = [{
    Emits the `qcc.config.init` that the rest of the dialect reads, assigning
    every qubit the function mentions to a site of the machine on the module's
    `qcc.device` attribute. A function that already establishes a configuration
    is left alone, so a hardware-aware program passes through untouched.

    The placement strategy is the simplest correct one: qubits are laid into
    sites in declaration order, filling each to capacity before moving on. It
    respects capacity and never places a qubit twice, so its output always
    verifies. It does not consider which qubits the program interacts, so on a
    machine with a non-empty rule set it leaves the router more work than a
    placement heuristic would.

    A better strategy reads the gate graph and co-locates qubits that interact.
    Such a heuristic replaces the body of this pass and nothing else, since
    everything downstream consumes the emitted `qcc.config.init`.
  }];

  let dependentDialects = ["::qcc::conn::QCCDialect", "::mlir::qc::QCDialect"];
}

def Route : Pass<"qcc-route", "::mlir::func::FuncOp"> {
  let summary = "Report which gates the current configuration cannot execute.";
  let description = [{
    Walks the gates in program order against the configuration in force at each
    one and reports every gate whose operands are not a facet of `K`, along with
    where those qubits currently sit.

    The pass does not insert shuttles. Finding a sequence that brings the
    operands together is pebble motion on a graph with capacities and chain
    order; the pass instead reports precisely what is unreachable, in the order
    a router would have to address it.

    `emit-remarks` reports as remarks rather than errors, so that a pipeline
    runs to completion and the whole work list is visible at once.
  }];

  let options = [
    Option<"emitRemarks", "emit-remarks", "bool", /*default=*/"false",
           "Report unroutable gates as remarks rather than errors, so that a "
           "pipeline continues and the whole work list is visible at once.">,
  ];

  let dependentDialects = ["::qcc::conn::QCCDialect", "::mlir::qc::QCDialect"];
}

//===----------------------------------------------------------------------===//
// Workflow 2 — check and improve a hardware-aware program
//===----------------------------------------------------------------------===//

def OptimizeShuttling : Pass<"qcc-optimize-shuttling", "::mlir::func::FuncOp"> {
  let summary = "Remove shuttling work that the program does not need.";
  let description = [{
    Operates on a program that already carries its own configuration chain and
    shortens that chain without changing the configuration any gate observes.

    The transformation applied is removal of a maximal suffix of rewrite
    operations that no observer reads. A hand-written schedule routinely ends by
    returning ions to where they started; when nothing is executed afterwards,
    that motion costs machine time for no observable effect.

    The pass does not reorder. Two shuttles commute only when their
    double-pushout footprints are disjoint and no control edge serialises them,
    which is what `--qcc-schedule-concurrency` computes; reordering is left to
    that pass.

    Total latency saved, taken from `TopologyRewriteOpInterface::getLatencyUs`,
    is reported as a remark, so the effect is visible without diffing the IR.
  }];

  let dependentDialects = ["::qcc::conn::QCCDialect", "::mlir::qc::QCDialect"];
}

def ScheduleConcurrency : Pass<"qcc-schedule-concurrency", "::mlir::func::FuncOp"> {
  let summary = "Work out which rewrites can run at the same time.";
  let description = [{
    Packs the rewrite operations of a function into time steps and annotates
    each with the step it lands in, so a backend can emit them as a schedule
    rather than a sequence.

    Two independent constraints decide the packing:

    - Configuration dependence. Two rules may share a step only if their
      double-pushout footprints are disjoint. When the sites they touch do not
      overlap, applying them in either order yields the same configuration.
    - Control contention. Two rules may share a step only if they contend for no
      `control` hyperedge. A single arbitrary-waveform generator driving every
      shuttling electrode serialises transports on segments that are otherwise
      unrelated, which no reasoning about placement can discover.

    This is the only pass that derives anything from `control` edges. They never
    contribute to `K`, so connectivity checking ignores them; scheduling is the
    analysis they exist for.

    The SSA chain is not consulted. Threading a single configuration value
    through every rule totally orders them in the IR, but that order is a
    property of the representation rather than of the machine; the footprints
    are what bind.

    The remark reports how many pairs were serialised only by a shared control
    resource, which is the number of pairs an additional control resource would
    allow to run concurrently.
  }];

  let options = [
    Option<"annotate", "annotate", "bool", /*default=*/"true",
           "Record each rewrite's time step as a `qcc.step` attribute.">,
  ];

  let dependentDialects = ["::qcc::conn::QCCDialect"];
}

#endif // INCLUDE_QCC_DIALECT_QCC_TRANSFORMS_PASSES_TD_
