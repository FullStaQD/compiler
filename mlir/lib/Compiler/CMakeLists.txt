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
  MLIRQCDialect
  # Passes
  MLIRMemRefTransforms
  MLIRArithTransforms
  MLIRFuncTransforms
  MLIRLinalgTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  QCCJaspToQC
  # Extensions/Interfaces
  MLIRFuncInlinerExtension
  MLIRTransformUtils)
