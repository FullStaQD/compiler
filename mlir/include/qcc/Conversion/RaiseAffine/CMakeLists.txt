set(LLVM_TARGET_DEFINITIONS RaiseAffine.td)
mlir_tablegen(RaiseAffine.h.inc -gen-pass-decls -name RaiseAffine)
add_public_tablegen_target(QCCRaiseAffineIncGen)

add_mlir_doc(RaiseAffine QCCRaiseAffine Conversions/ -gen-pass-doc)
