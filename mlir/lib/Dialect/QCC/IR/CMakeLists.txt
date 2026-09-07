add_mlir_dialect_library(
  QCCQCCDialect
  QCCDialect.cpp
  QCCOps.cpp
  DEPENDS
  QCCQCCEnumsIncGen
  QCCQCCDialectIncGen
  QCCQCCAttributesIncGen
  QCCQCCTypesIncGen
  QCCQCCInterfacesIncGen
  QCCQCCOpsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces)

mqt_mlir_target_use_project_options(QCCQCCDialect)
