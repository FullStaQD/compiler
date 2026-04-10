include "mlir/Pass/PassBase.td"

def ToQIRPrep : Pass<"to-qir-prep", "mlir::ModuleOp"> {
  let summary = "Very first pass to apply if you want to convert to QIR (adaptive).";

  let description = [{
      Adds the QIR function declarations (runtime functions like `__quantum__rt__initialize` and QIS functions like
      `__quantum__qis__x__body`) as well as the llvm module flags (capabilities) to the module.

      Assumptions:
      - Input IR contains a single top-level module.

      TODO:
      - Currently we hardcode the capabilities, must be target specific in the future.
   }];

  let dependentDialects = [
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

def StdToLLVM : Pass<"std-to-llvm", "mlir::func::FuncOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}

def QCToQIR : Pass<"convert-qc-to-qir", "mlir::func::FuncOp"> {
  let summary = "FIXME: add summary";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}
