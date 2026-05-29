// RUN: qcc-opt %s --jasp-check-static-qubit-allocation --split-input-file --verify-diagnostics | FileCheck %s

// Test that all the amount of memories we need to allocate for qubits are compile-time constants
func.func public @main() {
    %alloc = memref.alloc() : memref<3x!qc.qubit>
    return
  }

// The output of the test will essential be the (unchanged) input since it is just a check pass
// CHECK-LABEL: func.func public @main()
// CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<3x!qc.qubit>
// CHECK:   return
// CHECK: }

// -----

func.func @dynamic_size(%arg0: i64) {
    %0 = arith.index_cast %arg0 : i64 to index
    // expected-error @+1 {{qubit array allocation size must be a compile-time constant}}
    %alloc = memref.alloc(%0) : memref<?x!qc.qubit>
    return
}

// -----

func.func @zero_size() {
    // expected-error @+1 {{qubit array size must be positive, got 0}}
    %alloc = memref.alloc() : memref<0x!qc.qubit>
    return
}
