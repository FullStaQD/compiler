// RUN: qcc-opt %s
module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %0 = qc.alloc("q", 1, 0) : !qc.qubit
    qc.x %0 : !qc.qubit
    qc.dealloc %0 : !qc.qubit
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
}
