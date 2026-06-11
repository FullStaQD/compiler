add_mlir_library(
  QCCPipeline
  Pipeline.cpp
  DEPENDS # FIXME: What's the proper place to put these dependencies? Why do only I need them?
  JaspToQCIncGen
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
  QCCToQIR
  # Extensions/Interfaces
  MLIRFuncInlinerExtension
  MLIRTransformUtils)
