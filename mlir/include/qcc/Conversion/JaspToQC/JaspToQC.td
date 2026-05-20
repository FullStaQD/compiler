include "mlir/Pass/PassBase.td"

def JaspCheckStaticQubitAllocation : Pass<"jasp-check-static-qubit-allocation"> {
  let summary = "Validate that qubit array allocations use compile-time constant sizes.";
  let description = [{
      Verifies all qubit memref allocations have static sizes and reports diagnostic
      errors for non-constant or invalid sizes.

      An example of a correct IR is the following:

      ```mlir
      %0 = "arith.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
      %1, %2 = "jasp.create_qubits"(%0, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
      ```

      in which the size of `jasp.create_qubit` is known at runtime.

      On the other hand, in the following snippet of code:

      ```mlir
      func.func public @main(%arg14: !jasp.QuantumState, %s: tensor<i64>) attributes { qcc.entry_point } {
        %1, %2 = "jasp.create_qubits"(%s, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
        ...
      }
      ```

      the compilation process will prompt an error since the number of element that `jasp.create_qubits` needs to allocate
      is unknown at compile time.

      The compiler will prompt an error message even if the size of the register we want to allocate is negative (for example, wrongly initialized constants).

      We assume constant propagation to be performed via `mlir::Canonicalization` before calling this step.
   }];
}

def JaspToQC : Pass<"jasp-to-qc"> {
  let summary = "Convert jasp dialect to QC dialect.";

  let description = [{
      This pass converts all operations from qrisp jasp dialect
      (see https://qrisp.eu/reference/Jasp/MLIR%20Interface.html)
      to their equivalent operations in the QC dialect.
}];

let dependentDialects = ["mlir::qc::QCDialect", "mlir::tensor::TensorDialect", "mlir::memref::MemRefDialect"];
}

def ConvertMemrefToStaticQubits : Pass<"convert-memref-to-static-qubits"> {
  let summary = "Replace dynamic qubit array loads with static qubits.";
  let description = [{ First, substitutes `memref.alloc` with  a list of `qc.static` allocation. Subsequently,
      transforms `memref.load` on qubit arrays into direct `qc.static` references. Requires compile-time constant indices
      (should be called after a `check-static-qubit-allocation` step). At the end of the step the `qc` and the `memref` dialect live together in the same IR.
      The latter is actually dangling and it should be cleaned up via further `mlir::Canonicalization`.

      Providing as input the following IR:

      ```mlir
      %0 = "arith.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
      %1, %2 = "jasp.create_qubits"(%0, %arg14) : (tensor<i64>, !jasp.QuantumState) -> (!jasp.QubitArray, !jasp.QuantumState)
      ```

      the pass will translate it into:

      ```mlir
      %0 = qc.static 0 : !qc.qubit
      %1 = qc.static 1 : !qc.qubit
      %2 = qc.static 2 : !qc.qubit
      %alloc = memref.alloc() : memref<3x!qc.qubit>
      ```

      in which the allocation of three qubits has been turned into 3 separate static allocations, while the `memref.alloc` is dangling
      and will be cleaned up via `mlir::Canonicalization`.

}];
let dependentDialects = ["mlir::qc::QCDialect"];
}
