set(LLVM_TARGET_DEFINITIONS QCCEnums.td)
mlir_tablegen(QCCEnums.h.inc -gen-enum-decls)
mlir_tablegen(QCCEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(QCCQCCEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS QCCDialect.td)
mlir_tablegen(QCCDialect.h.inc -gen-dialect-decls -dialect=qcc)
mlir_tablegen(QCCDialect.cpp.inc -gen-dialect-defs -dialect=qcc)
add_public_tablegen_target(QCCQCCDialectIncGen)

set(LLVM_TARGET_DEFINITIONS QCCAttributes.td)
mlir_tablegen(QCCAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=qcc)
mlir_tablegen(QCCAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=qcc)
add_public_tablegen_target(QCCQCCAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS QCCTypes.td)
mlir_tablegen(QCCTypes.h.inc -gen-typedef-decls -typedefs-dialect=qcc)
mlir_tablegen(QCCTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=qcc)
add_public_tablegen_target(QCCQCCTypesIncGen)

set(LLVM_TARGET_DEFINITIONS QCCInterfaces.td)
mlir_tablegen(QCCInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QCCInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(QCCQCCInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS QCCOps.td)
mlir_tablegen(QCCOps.h.inc -gen-op-decls)
mlir_tablegen(QCCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QCCQCCOpsIncGen)
