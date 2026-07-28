# The HiSEP-Q backend is the only component that may depend on the HiSEP-Q LLVM
# fork; its RISC-V/QV codegen components are pulled in via LLVM_LINK_COMPONENTS.
# The fork provides the experimental "xqv" RISC-V vector extension used to emit
# native QISA.
set(LLVM_LINK_COMPONENTS
    RISCVCodeGen
    RISCVAsmParser
    RISCVDesc
    RISCVInfo
    CodeGen
    MC
    Target
    SelectionDAG
    AsmPrinter)

add_mlir_library(
  QCCTargetHiSEPQ HiSEPQTarget.cpp LINK_LIBS MLIRPass
  # Shared QIR lowering + the QIR->intrinsics conversion:
  QCCTargetQIR QCCToIntrinsics)
