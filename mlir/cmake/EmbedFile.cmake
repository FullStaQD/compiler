# Embeds a file's raw bytes into a generated C++ header as a byte array, so it can be baked into
# the qcc binary instead of being a loose file qcc has to locate at runtime.
#
# Usage: cmake -DINPUT=<path> -DOUTPUT=<path> -DVARNAME=<identifier> -P EmbedFile.cmake

if(NOT DEFINED INPUT
   OR NOT DEFINED OUTPUT
   OR NOT DEFINED VARNAME)
  message(FATAL_ERROR "EmbedFile.cmake requires -DINPUT=, -DOUTPUT= and -DVARNAME= to be set")
endif()

file(READ "${INPUT}" hex_content HEX)
string(REGEX REPLACE "(..)" "0x\\1," hex_array "${hex_content}")

file(WRITE "${OUTPUT}" "// Auto-generated from ${INPUT} by EmbedFile.cmake. Do not edit.\n")
file(APPEND "${OUTPUT}" "static const unsigned char ${VARNAME}[] = {${hex_array}};\n")
file(APPEND "${OUTPUT}" "static const unsigned long ${VARNAME}Size = sizeof(${VARNAME});\n")
