include "mlir/Pass/PassBase.td"

def CheckStaticQubitAllocation : Pass<"check-static-qubit-allocation"> {
  let summary = "Validate that qubit array allocations use compile-time constant sizes.";
  let description = [{
      Verifies all qubit memref allocations have static sizes and reports diagnostic
      errors for non-constant or invalid sizes.

      An example of a correct IR is the following:
      `%0 = "arith.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
       %1, %2 = "jasp.create_qubits"(%0, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)`
      in which the size of `jasp.create_qubit` is known at runtime.

      which will result in the following snippet of compiled code:
      `%0 = qc.static 0 : !qc.qubit
       %1 = qc.static 1 : !qc.qubit
       %2 = qc.static 2 : !qc.qubit`
      in which the allocation of three qubits has been turned into 3 separate static allocations.

      On the other hand, in the following snippet of code:
      `func.func public @main(%arg14: !jasp.QuantumState, %s: tensor<i64>) attributes { qcc.entry_point } {
        %1, %2 = "jasp.create_qubits"(%s, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)`
      the compilation process will prompt an error since the number of element that `jasp.create_qubits` needs to allocate
      is unknown at compile time.

      The compiler will prompt an error message even if the size of the register we want to allocate is negative.

      We assume constant propagation to be performed via `mlir::Canonicalization` before calling this step.
   }];
}

def StaticizeQubitRefs : Pass<"staticize-qubit-refs"> {
  let summary = "Replace dynamic qubit array loads with static qubits.";
  let description = [{ First, substitutes `memref.alloc` with a list of `qc.static` allocation.Subsequently,
      transforms `memref.load` on qubit arrays into direct `qc.static` references.Requires compile -
          time constant indices(should be called after a `check - static - qubit - allocation` step)
              .It assumes dangling `memref.load` calls to be cleaned up via further `mlir::Canonicalization`.
}];
let dependentDialects = ["mlir::qc::QCDialect"];
}

def JaspToQC : Pass<"jasp-to-qc"> {
  let summary = "Convert jasp dialect to QC dialect.";

  let description = [{
      This pass converts all operations from qrisps jasp dialect
      (see https://qrisp.eu/reference/Jasp/MLIR%20Interface.html)
      to their equivalent operations in the QC dialect.
   }];

  let dependentDialects = [
    "mlir::qc::QCDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
}
