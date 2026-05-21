file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
  QCCRaiseAffine
  ${CONVERSION_SOURCES}
  DEPENDS
  QCCRaiseAffineIncGen
  LINK_LIBS
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass)

mqt_mlir_target_use_project_options(QCCRaiseAffine)
