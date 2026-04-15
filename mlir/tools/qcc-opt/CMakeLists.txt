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
          MLIRLinalgDialect
          MLIRMemRefDialect
          MLIRBufferizationDialect
          QCCJaspDialect
          # Passes
          MLIRMemRefTransforms
          MLIRArithTransforms
          MLIRFuncTransforms
          MLIRLinalgTransforms
          MLIRTensorTransforms
          MLIRBufferizationTransforms
          QCCPrelimHLEPDialect
          QCCJaspToQC
          # Extensions/Interfaces
          MLIRFuncInlinerExtension
          MLIRTransformUtils
          MLIRQCDialect)
