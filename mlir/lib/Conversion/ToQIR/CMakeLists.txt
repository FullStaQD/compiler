file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCToQIR
  ${CONVERSION_SOURCES}
  DEPENDS
  # FIXME: which ones do I need in the end?
  QCCToQIRIncGen
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
