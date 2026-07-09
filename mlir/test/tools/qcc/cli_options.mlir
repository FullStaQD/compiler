// Exercises the qcc driver's --target / --compile-to option surface.

// Stage selection (extension form):
// RUN: qcc --compile-to=mlir %s | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: qcc --compile-to=ll %s | FileCheck %s --check-prefix=CHECK-LLVM
// Stage selection (word form):
// RUN: qcc --compile-to=mlir %s | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: qcc --compile-to=llvmir %s | FileCheck %s --check-prefix=CHECK-LLVM
// Default is LLVM-IR:
// RUN: qcc %s | FileCheck %s --check-prefix=CHECK-LLVM

// Assembly/object are not supported for --target=qir (they require --target=hisep-q):
// RUN: not qcc --target=qir --compile-to=s %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=assembly %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=o %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=object %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE

func.func @main() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    qc.h %0 : !qc.qubit
    %m0 = qc.measure %0 : !qc.qubit -> i1
    aux.record_int %m0 : i1
    return
}

// CHECK-MLIR: llvm.func @main()
// CHECK-LLVM: define void @main()
// CHECK-ERR-NATIVE: error: assembly/object output is not supported for --target=qir
