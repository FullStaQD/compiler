add_mlir_library(
  QCCPipeline
  Pipeline.cpp
  LINK_LIBS
  MLIRPass
  MLIRTransforms
  # Dialects
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  QCCJaspDialect
  QCCAuxDialect
  MLIRQCDialect
  # Passes
  MLIRMemRefTransforms
  MLIRArithTransforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRSCFTransforms
  MLIRQCToQIR
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  QCCAuxOutputRecording
  QCCJaspToQC
  QCCToQIR
  # Extensions/Interfaces
  MLIRFuncInlinerExtension
  MLIRTransformUtils)
