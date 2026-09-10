set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QCCTransforms)
add_public_tablegen_target(QCCQCCTransformsIncGen)

add_mlir_doc(Passes QCCTransforms Dialects/ -gen-pass-doc)
