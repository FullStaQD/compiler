file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCAuxOutputRecording
  ${CONVERSION_SOURCES}
  DEPENDS
  AuxOutputRecordingIncGen
  LINK_LIBS
  MLIRQCDialect
  QCCJaspDialect
  QCCAuxDialect
  MLIRArithDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms)

mqt_mlir_target_use_project_options(QCCAuxOutputRecording)
