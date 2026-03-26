add_executable(qcc qcc.cpp)
target_link_libraries(
  qcc
  PRIVATE MLIRIR
          MLIRPass
          MLIRTransforms
          MLIROptLib
          MLIRSupport
          MLIRFuncDialect
          MLIRArithDialect
          QCCPipeline
          QCCJaspDialect
          MLIRQCDialect
          MLIRQCODialect)
