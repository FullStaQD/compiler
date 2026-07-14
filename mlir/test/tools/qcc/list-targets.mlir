// RUN: qcc --list-targets | FileCheck %s

// Lists *all* targets, even the ones unavailable for the current build.
// CHECK: Available targets for --target:
// CHECK-DAG: qir - QIR (LLVM-based) target
// CHECK-DAG: hisep-q
