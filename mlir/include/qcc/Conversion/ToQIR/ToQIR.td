include "mlir/Pass/PassBase.td"

def StdToQIR : Pass<"std-to-qir", "mlir::func::FuncOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}

def ToQIRFinalize : Pass<"to-qir-finalize", "mlir::ModuleOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}

def QCToQIR : Pass<"qc-to-qir", "mlir::func::FuncOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}
