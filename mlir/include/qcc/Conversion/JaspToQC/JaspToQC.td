include "mlir/Pass/PassBase.td"

def CheckStaticQubitAllocation : Pass<"check-static-qubit-allocation"> {
  let summary = "Validate that qubit array allocations use compile-time constant sizes.";
  let description = [{
      Verifies all qubit memref allocations have static sizes and reports diagnostic
      errors for non-constant or invalid sizes.
   }];
  let dependentDialects = [
    "mlir::qc::QCDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def StaticizeQubitRefs : Pass<"staticize-qubit-refs"> {
  let summary = "Replace d ynamic qubit array loads with static qubit references.";
                    let description = [{
      Transforms memref.load on qubit arrays into direct qc.static references.
      Requires compile-time constant indices (typically from loop unrolling).
   }];
  let dependentDialects = [
    "mlir::qc::QCDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
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
