set(LLVM_TARGET_DEFINITIONS QCToQIRAdaptive.td)
mlir_tablegen(QCToQIRAdaptive.h.inc -gen-pass-decls -name QCToQIRAdaptive)
add_public_tablegen_target(QCCQCToQIRAdaptiveIncGen)

add_mlir_doc(QCToQIRAdaptive QCCQCToQIRAdaptive Conversions/ -gen-pass-doc)
