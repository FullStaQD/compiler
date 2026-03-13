add_mlir_dialect_library(QCCMLIRDummyDialect
  DummyDialect.cpp
  DummyOps.cpp

  DEPENDS
  QCCMLIRDummyDialectIncGen
  QCCMLIRDummyTypesIncGen
  QCCMLIRDummyOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)