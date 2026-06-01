// RUN: qcc-opt %s -integer-output-recording | FileCheck %s

//===----------------------------------------------------------------------===//
//
// Integer output recording pass tests
//
//===----------------------------------------------------------------------===//

// RUN: qcc-opt %s -integer-output-recording | FileCheck %s

// -----------------------------------------------------------------------------
// Test 1: Entry point function returning i64 should be transformed
// -----------------------------------------------------------------------------

func.func @test_entry_point_i64() -> i64 attributes { qcc.entry_point } {
  %q5 = qc.static 5 : !qc.qubit
  %q7 = qc.static 7 : !qc.qubit

  %c = arith.constant 1 : i64
  %c1 = arith.addi %c, %c : i64
  return %c1 : i64
}

// CHECK: module {
// CHECK:   func.func @test_entry_point_i64() attributes {qcc.entry_point} {
// CHECK:     %[[qubit0:.*]] = qc.static 5 : !qc.qubit
// CHECK:     %[[qubit1:.*]] = qc.static 7 : !qc.qubit
// CHECK:     %[[constant:.*]] = arith.constant 1 : i64
// CHECK:     %[[add:.*]] = arith.addi %[[constant]], %[[constant]] : i64
// CHECK:     aux.record_integer %[[add]]
// CHECK:     return
// CHECK:   }

// -----------------------------------------------------------------------------
// Test 2: No return value should not be transformed, even if the function is an entry point
// -----------------------------------------------------------------------------

func.func @test_void_entry_point() attributes { qcc.entry_point } {
  %c = arith.constant 1 : i64
  return
}
// CHECK:   func.func @test_void_entry_point() attributes {qcc.entry_point} {
// CHECK:     %[[constant:.*]] = arith.constant 1 : i64
// CHECK:     return
// CHECK:   }

// -----------------------------------------------------------------------------
// Test 3: Multiple i64 return values should be transformed
// -----------------------------------------------------------------------------

func.func @test_multiple_i64() -> (i64, i64) attributes { qcc.entry_point } {
  %a = arith.constant 1 : i64
  %b = arith.constant 2 : i64
  return %a, %b : i64, i64
}

// CHECK:   func.func @test_multiple_i64() attributes {qcc.entry_point} {
// CHECK:     %[[constant0:.*]] = arith.constant 1 : i64
// CHECK:     %[[constant1:.*]] = arith.constant 2 : i64
// CHECK:     aux.record_integer %[[constant0]]
// CHECK:     aux.record_integer %[[constant1]]
// CHECK:     return
// CHECK:   }

// -----------------------------------------------------------------------------
// Test 4: Function without entry point attribute should not be transformed
// -----------------------------------------------------------------------------

func.func @test_no_entry_point_attribute() -> (i64, i64) attributes {} {
  %a = arith.constant 42 : i64
  %b = arith.constant 2 : i64
  return %a, %b : i64, i64
}

// CHECK:   func.func @test_no_entry_point_attribute() -> (i64, i64) {
// CHECK:     %[[constant0:.*]] = arith.constant 42 : i64
// CHECK:     %[[constant1:.*]] = arith.constant 2 : i64
// CHECK:     return %[[constant0]], %[[constant1]] : i64, i64
// CHECK:   }

// -----------------------------------------------------------------------------
// Test 5: Multiple returns mixed between i64 and i1
// -----------------------------------------------------------------------------

func.func @test_multiple_returns() -> (i64, i1) attributes { qcc.entry_point } {
  %a = arith.constant 2 : i64
  %b = arith.constant 1 : i1
  return %a, %b : i64, i1
}

// CHECK:   func.func @test_multiple_returns() attributes {qcc.entry_point} {
// CHECK:     %[[constant0:.*]] = arith.constant 2 : i64
// CHECK:     %[[constant1:.*]] = arith.constant true
// CHECK:     aux.record_integer %[[constant0]]
// CHECK:     aux.record_bool %[[constant1]]
// CHECK:     return
// CHECK:   }

// CHECK: }
