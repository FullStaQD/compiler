set(QCC_DEVICE_SOURCES DeviceLibrary.cpp BuiltinDevices.cpp)

# The QDMI importer is the only part that needs a live device-management session, so it is compiled in only when QDMI is
# present.
if(TARGET qdmi::qdmi)
  list(APPEND QCC_DEVICE_SOURCES QDMIImport.cpp)
endif()

add_mlir_library(
  QCCQCCDevices
  ${QCC_DEVICE_SOURCES}
  DEPENDS
  QCCQCCEnumsIncGen
  QCCQCCDialectIncGen
  QCCQCCAttributesIncGen
  QCCQCCTypesIncGen
  QCCQCCInterfacesIncGen
  QCCQCCOpsIncGen
  LINK_LIBS
  PUBLIC
  QCCQCCDialect
  MLIRIR
  MLIRSupport)

if(TARGET qdmi::qdmi)
  target_link_libraries(QCCQCCDevices PUBLIC qdmi::qdmi)
  target_compile_definitions(QCCQCCDevices PUBLIC QCC_HAVE_QDMI=1)
endif()

mqt_mlir_target_use_project_options(QCCQCCDevices)
