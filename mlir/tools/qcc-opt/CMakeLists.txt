add_executable(qcc-opt qcc-opt.cpp)
target_link_libraries(
  qcc-opt
  PRIVATE MLIRIR
          MLIROptLib
          MLIRSupport
          MLIRFuncDialect
          MLIRArithDialect
          MLIRSCFDialect
          MLIRTensorDialect
          QCCJaspDialect
          QCCPrelimHLEPDialect
          QCCJaspToQC
          StablehloRegister
          MLIRQCDialect)
