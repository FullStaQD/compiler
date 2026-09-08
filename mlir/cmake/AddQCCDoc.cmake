# Umbrella target that collects all qcc-generated (dialect/pass) documentation.
add_custom_target(qcc-doc COMMENT "Generating qcc dialect/pass documentation")
set_target_properties(qcc-doc PROPERTIES FOLDER "QCC/Docs")

# Mirrors add_mlir_doc()'s signature: doc_filename, output_file, output_directory, command (e.g. -gen-pass-doc), plus
# any extra tablegen arguments. We cannot use `add_mlir_doc` directly as it hardcodes the output path to
# `${MLIR_BINARY_DIR}/docs/` which collapses to `/docs/` out-of-tree (e.g. for us).
function(add_qcc_doc doc_filename output_file output_directory command)
  # implementation almost identical to `add_mlir_doc`.
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  # The MLIR docs use Hugo, so we allow Hugo specific features here, matching add_mlir_doc.
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(GEN_DOC_FILE ${PROJECT_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md ${GEN_DOC_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  set_target_properties(${output_file}DocGen PROPERTIES FOLDER "QCC/Docs")
  add_dependencies(qcc-doc ${output_file}DocGen)
endfunction()
