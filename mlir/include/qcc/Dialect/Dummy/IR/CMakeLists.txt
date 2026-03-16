set(LLVM_TARGET_DEFINITIONS DummyDialect.td)

mlir_tablegen(DummyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DummyDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(QCCMLIRDummyDialectIncGen)

set(LLVM_TARGET_DEFINITIONS DummyTypes.td)
mlir_tablegen(DummyTypes.h.inc -gen-typedef-decls --typedefs-dialect=dummy)
mlir_tablegen(DummyTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=dummy)
add_public_tablegen_target(QCCMLIRDummyTypesIncGen)

set(LLVM_TARGET_DEFINITIONS DummyOps.td)
mlir_tablegen(DummyOps.h.inc -gen-op-decls)
mlir_tablegen(DummyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QCCMLIRDummyOpsIncGen)