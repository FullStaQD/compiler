add_mlir_library(
  QCCPipeline
  Pipeline.cpp
  DEPENDS
  JaspToQCIncGen
  LINK_LIBS
  MLIRPass
  MLIRTransforms
  # Dialects
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRAffineDialect
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
  MLIRAffineTransforms
  MLIRQCToQIR
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  QCCAffineRaise
  QCCJaspToQC
  # Targets
  QCCTarget
  # Extensions/Interfaces
  MLIRFuncInlinerExtension
  MLIRTransformUtils)
