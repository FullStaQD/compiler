// UNSUPPORTED: hisep-q

// RUN: qcc --list-targets | FileCheck %s --check-prefix=CHECK-LIST
// RUN: not qcc --target=hisep-q %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

// CHECK-LIST: hisep-q (unavailable in this build)
// CHECK-ERR: error: target 'hisep-q' is not available in this build

func.func @main() attributes { qcc.entry_point } {
    return
}
