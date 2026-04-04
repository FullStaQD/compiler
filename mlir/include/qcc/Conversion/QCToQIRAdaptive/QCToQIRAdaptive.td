include "mlir/Pass/PassBase.td"

def QCToQIRAdaptive : Pass<"qc-to-qir-adaptive"> {
  let summary = "Convert qc dialect QIR adaptive profile (LLVM-IR dialect).";

  let description = [{
      FIXME: add description
   }];

  let dependentDialects = [
    //"mlir::LLVM::LLVMDialect"
  ];
}
