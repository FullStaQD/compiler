add_mlir_dialect_library(QCCDummyDialect
  DummyDialect.cpp
  DummyOps.cpp

  DEPENDS
  QCCDummyDialectIncGen
  QCCDummyTypesIncGen
  QCCDummyOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)