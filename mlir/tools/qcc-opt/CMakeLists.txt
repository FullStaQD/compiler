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