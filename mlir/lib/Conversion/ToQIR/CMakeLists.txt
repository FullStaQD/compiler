file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCToQIR
  ${CONVERSION_SOURCES}
  DEPENDS
  QCCToQIRIncGen
  QCCAuxDialectIncGen
  QCCAuxOpsIncGen
  AuxOutputRecordingIncGen
  LINK_LIBS
  # Dialects
  MLIRQCDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  QCCAuxDialect
  # Conversion infra
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  # The passes that `buildQIRPipeline` sequences alongside this library's own
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  QCCAuxOutputRecording
  # Core MLIR
  MLIRIR
  MLIRPass)
