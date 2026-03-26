file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCJaspToQC
  ${CONVERSION_SOURCES}
  DEPENDS
  JaspToQCIncGen
  LINK_LIBS
  MLIRQCDialect
  MLIRJaspDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms
  DISABLE_INSTALL)

mqt_mlir_target_use_project_options(QCCJaspoQC)
