file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCJaspToQC
  ${CONVERSION_SOURCES}
  DEPENDS
  JaspToQCIncGen
  LINK_LIBS
  MLIRQCDialect
  QCCJaspDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms)

mqt_mlir_target_use_project_options(QCCJaspoQC)
