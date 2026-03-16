add_executable(qcc-opt main.cpp)
target_link_libraries(qcc-opt PRIVATE
    MLIRIR
    MLIROptLib
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    QCCMLIRDummyDialect
    MLIRQCDialect
)

# target_include_directories(qcc-opt PRIVATE
#     ${CMAKE_BINARY_DIR}/_deps/mqt-core-src/mlir/include
#     ${CMAKE_BINARY_DIR}/_deps/mqt-core-build/mlir/include
# )