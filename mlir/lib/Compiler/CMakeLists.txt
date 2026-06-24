add_mlir_library(
  QCCPipeline
  Pipeline.cpp
  DEPENDS
  AuxOutputRecordingIncGen
  JaspToQCIncGen
  QCCToIntrinsicsIncGen
  QCCToQIRIncGen
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
  QCCToIntrinsics
  QCCToQIR
  # Extensions/Interfaces
  MLIRFuncInlinerExtension
  MLIRTransformUtils)
