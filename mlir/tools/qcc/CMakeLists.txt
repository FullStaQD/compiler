get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LLVM_LINK_COMPONENTS
    BitWriter
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    AllTargetsAsmParsers
    CodeGen
    MC
    Target
    SelectionDAG
    AsmPrinter)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRBytecodeWriter
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    QCCPipeline
    QCCSupport)

# hisepq.ld is baked into the qcc binary (rather than being a loose file qcc has to locate at runtime) so that
# `--compile-to=elf`/`mem` work as a single self-contained command. See EmbedFile.cmake.
set(HISEPQ_LD_INC "${CMAKE_CURRENT_BINARY_DIR}/HisepqLd.inc")
add_custom_command(
  OUTPUT ${HISEPQ_LD_INC}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/hisepq.ld -DOUTPUT=${HISEPQ_LD_INC} -DVARNAME=kHisepqLd
          -P ${CMAKE_CURRENT_SOURCE_DIR}/EmbedFile.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hisepq.ld ${CMAKE_CURRENT_SOURCE_DIR}/EmbedFile.cmake
  COMMENT "Embedding hisepq.ld into qcc")
add_custom_target(QCCEmbedHisepqLd DEPENDS ${HISEPQ_LD_INC})

add_llvm_executable(qcc qcc.cpp DEPENDS ${LIBS} QCCEmbedHisepqLd)

target_link_libraries(qcc PRIVATE ${LIBS})
target_include_directories(qcc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(qcc QCCEmbedHisepqLd)

llvm_update_compile_flags(qcc)
mlir_check_all_link_libraries(qcc)
