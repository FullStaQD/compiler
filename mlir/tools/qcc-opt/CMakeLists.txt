add_executable(qcc-opt qcc-opt.cpp)
target_link_libraries(
  qcc-opt
  PRIVATE MLIRIR
          MLIROptLib
          MLIRSupport
          # Dialects
          MLIRFuncDialect
          MLIRArithDialect
          MLIRSCFDialect
          MLIRTensorDialect
          MLIRBufferizationDialect
          QCCJaspDialect
          StablehloRegister
          # Passes
          StablehloLinalgTransforms
          MLIRMemRefTransforms
          MLIRBufferizationTransforms
          QCCJaspToQC
          # Extensions
          MLIRFuncInlinerExtension
          MLIRQCDialect)
