set(LLVM_TARGET_DEFINITIONS JaspDialect.td)
mlir_tablegen(JaspDialect.h.inc -gen-dialect-decls)
mlir_tablegen(JaspDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(QCCJaspDialectIncGen)

set(LLVM_TARGET_DEFINITIONS JaspDialect.td)
mlir_tablegen(JaspTypes.h.inc -gen-typedef-decls --typedefs-dialect=jasp)
mlir_tablegen(JaspTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=jasp)
add_public_tablegen_target(QCCJaspTypesIncGen)

set(LLVM_TARGET_DEFINITIONS JaspOps.td)
mlir_tablegen(JaspOps.h.inc -gen-op-decls)
mlir_tablegen(JaspOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QCCJaspOpsIncGen)
