# The linker script of the HiSEP-Q target is baked into the binary (rather than being a loose file qcc has to locate
# at runtime) so that `--compile-to=elf`/`mem` work as a single self-contained command.
set(EMBED_FILE_CMAKE "${PROJECT_SOURCE_DIR}/mlir/cmake/EmbedFile.cmake")
set(HISEPQ_LD "${CMAKE_CURRENT_SOURCE_DIR}/Targets/hisepq.ld")
set(HISEPQ_LD_INC "${CMAKE_CURRENT_BINARY_DIR}/HisepqLd.inc")
add_custom_command(
  OUTPUT ${HISEPQ_LD_INC}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${HISEPQ_LD} -DOUTPUT=${HISEPQ_LD_INC} -DVARNAME=kHisepqLd -P ${EMBED_FILE_CMAKE}
  DEPENDS ${HISEPQ_LD} ${EMBED_FILE_CMAKE}
  COMMENT "Embedding hisepq.ld into qcc")
add_custom_target(QCCEmbedHisepqLd DEPENDS ${HISEPQ_LD_INC})

add_mlir_library(
  QCCTarget
  Target.cpp
  Targets/QirPlatform.cpp
  Targets/HisepQPlatform.cpp
  DEPENDS
  QCCEmbedHisepqLd
  QCCToIntrinsicsIncGen
  QCCToQIRIncGen
  LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  CodeGen
  MC
  Target
  LINK_LIBS
  MLIRPass
  QCCToQIR
  QCCToIntrinsics
  QCCSupport)

target_include_directories(QCCTarget PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(QCCTarget QCCEmbedHisepqLd)
