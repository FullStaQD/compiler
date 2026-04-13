add_mlir_dialect_library(
  QCCAuxDialect
  AuxDialect.cpp
  DEPENDS
  QCCAuxDialectIncGen
  QCCAuxOpsIncGen
  LINK_LIBS
  PRIVATE # FIXME: remove those which are not needed.
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces)
