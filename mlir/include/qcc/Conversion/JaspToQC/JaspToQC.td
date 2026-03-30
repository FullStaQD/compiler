include "mlir/Pass/PassBase.td"

def JaspToQC : Pass<"jasp-to-qc"> {
  let summary = "Convert jasp dialect to QC dialect.";

  let description = [{
      This pass converts all operations from the jasp dialect to their equivalent
      operations in the QC dialect.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "jasp::JaspDialect",
    "mlir::qc::QCDialect",
    "mlir::tensor::TensorDialect"
  ];
}
