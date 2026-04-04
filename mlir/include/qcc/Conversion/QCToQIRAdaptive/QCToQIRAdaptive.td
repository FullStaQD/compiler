include "mlir/Pass/PassBase.td"

def QCToQIRAdaptive : Pass<"qc-to-qir-adaptive", "mlir::func::FuncOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}

def QCToQIRAdaptiveCleanup : Pass<"qc-to-qir-adaptive-cleanup", "mlir::ModuleOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}
