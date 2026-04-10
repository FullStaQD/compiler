add_executable(qcc-opt qcc-opt.cpp)
target_link_libraries(
  qcc-opt
  PRIVATE MLIRIR
          MLIROptLib
          MLIRSupport
          MLIRFuncDialect
          MLIRArithDialect
          MLIRSCFDialect
          MLIRControlFlowDialect
          MLIRFuncInlinerExtension
          MLIRTensorDialect
          MLIRBufferizationDialect
          MLIRTosaDialect
          MLIRTosaToSCF
          MLIRSCFToControlFlow
          MLIRMemRefTransforms
          MLIRTosaToArith
          MLIRLinalgToStandard
          MLIRBufferizationTransforms
          QCCJaspDialect
          QCCJaspToQC
          StablehloRegister
          StablehloLinalgTransforms
          StablehloTOSATransforms
          MLIRQCDialect)
