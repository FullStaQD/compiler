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

def ToQIRModule : Pass<"to-qir-module", "mlir::ModuleOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}
