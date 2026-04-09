// RUN: qcc -o - %s | FileCheck %s

/// Prepare GHZ state (manually).
func.func @ghz_manual() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    %1 = qc.static 1 : !qc.qubit
    %2 = qc.static 2 : !qc.qubit

    qc.h %0 : !qc.qubit

    qc.ctrl(%0) { qc.x %1 : !qc.qubit } : !qc.qubit
    qc.ctrl(%1) { qc.x %2 : !qc.qubit } : !qc.qubit

    %m0 = qc.measure %0 : !qc.qubit -> i1
    %m1 = qc.measure %1 : !qc.qubit -> i1
    %m2 = qc.measure %2 : !qc.qubit -> i1

    return
}

// FIXME: better test and checks
// CHECK-LABEL: @ghz_manual
