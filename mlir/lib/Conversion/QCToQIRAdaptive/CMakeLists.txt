file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCQCToQIRAdaptive
  ${CONVERSION_SOURCES}
  DEPENDS
  # FIXME: which ones do I need in the end?
  QCCQCToQIRAdaptiveIncGen
  LINK_LIBS
  MLIRQCDialect
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRTransforms
  MLIRFuncTransforms)
