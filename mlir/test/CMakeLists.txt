configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MLIR_COMPILER_TEST_DEPENDS FileCheck count not qcc-opt qcc)

add_lit_testsuite(test-qcc-project "Run the QCC project tests" ${CMAKE_CURRENT_BINARY_DIR} DEPENDS
                  ${MLIR_COMPILER_TEST_DEPENDS})
