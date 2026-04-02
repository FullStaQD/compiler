include "mlir/Pass/PassBase.td"

def JaspToQC : Pass<"jasp-to-qc"> {
  let summary = "Convert jasp dialect to QC dialect.";

  let description = [{
      This pass converts all operations from qrisp's jasp dialect
      (see https://qrisp.eu/reference/Jasp/MLIR%20Interface.html)
      to their equivalent operations in the QC dialect.
   }];

  let dependentDialects = [
    "mlir::qc::QCDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
}
