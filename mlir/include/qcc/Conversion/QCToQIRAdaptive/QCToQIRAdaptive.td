include "mlir/Pass/PassBase.td"

def QCToQIRAdaptive : Pass<"qc-to-qir-adaptive", "mlir::func::FuncOp"> {
  let summary = "Convert qc dialect QIR adaptive profile (LLVM-IR dialect).";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    // FIXME: which ones do I need?
    "mlir::LLVM::LLVMDialect"
  ];
}
