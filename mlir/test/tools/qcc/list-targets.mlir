// RUN: qcc --list-targets | FileCheck %s

// Lists the targets compiled into this build.
// CHECK: Available targets for --target:
// CHECK-DAG: qir - QIR (LLVM-based) target
