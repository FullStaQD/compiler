// Exercises the qcc driver's --target / --compile-to option surface.

// Stage selection (extension form):
// RUN: qcc --compile-to=mlir %s | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: qcc --compile-to=ll %s | FileCheck %s --check-prefix=CHECK-LLVM
// Stage selection (word form):
// RUN: qcc --compile-to=mlir %s | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: qcc --compile-to=llvmir %s | FileCheck %s --check-prefix=CHECK-LLVM
// Default is LLVM-IR:
// RUN: qcc %s | FileCheck %s --check-prefix=CHECK-LLVM

// The qir target stops at LLVM IR, so it rejects the stages below it:
// RUN: not qcc --target=qir --compile-to=s %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=assembly %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=o %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=object %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=elf %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE
// RUN: not qcc --target=qir --compile-to=mem %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-NATIVE

// An unknown target lists the ones qcc has:
// RUN: not qcc --target=does-not-exist %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-TARGET

// The hisep-q device implements h, x, cx and mz, so a T gate has no hardware to run on. The ideal
// device of the qir target implements the whole QIR gate set and takes it:
// RUN: not qcc --target=hisep-q --compile-to=ll %S/Inputs/t_gate.mlir 2>&1 | FileCheck %s --check-prefix=CHECK-ERR-GATE
// RUN: qcc --target=qir --compile-to=ll %S/Inputs/t_gate.mlir | FileCheck %s --check-prefix=CHECK-T-GATE

func.func @main() attributes { qcc.entry_point } {
    %0 = qc.static 0 : !qc.qubit
    qc.h %0 : !qc.qubit
    %m0 = qc.measure %0 : !qc.qubit -> i1
    aux.record_int %m0 : i1
    return
}

// CHECK-MLIR: llvm.func @main()
// CHECK-LLVM: define void @main()
// CHECK-ERR-NATIVE: error: target 'qir' does not support --compile-to=
// CHECK-ERR-TARGET: error: unknown target 'does-not-exist', expected one of: qir, hisep-q
// CHECK-ERR-GATE: error: the target device does not implement '__quantum__qis__t__body'
// CHECK-T-GATE: call void @__quantum__qis__t__body
