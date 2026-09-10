add_mlir_library(
  QCCQCCAnalysis
  ConnectivityAnalysis.cpp
  DEPENDS
  QCCQCCEnumsIncGen
  QCCQCCDialectIncGen
  QCCQCCAttributesIncGen
  QCCQCCTypesIncGen
  QCCQCCInterfacesIncGen
  QCCQCCOpsIncGen
  LINK_LIBS
  PUBLIC
  QCCQCCDialect
  MLIRIR
  MLIRSupport
  MLIRQCDialect)

mqt_mlir_target_use_project_options(QCCQCCAnalysis)
