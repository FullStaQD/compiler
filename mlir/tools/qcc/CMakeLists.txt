get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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
    QCCPipeline)

add_llvm_executable(qcc qcc.cpp DEPENDS ${LIBS})

target_link_libraries(qcc PRIVATE ${LIBS})

llvm_update_compile_flags(qcc)
mlir_check_all_link_libraries(qcc)
