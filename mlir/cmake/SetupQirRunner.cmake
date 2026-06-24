# Locate the `qir-runner` executable, which the integration tests use to
# simulate QIR. If it cannot be found, we try to install it on the user's behalf
# via `uv`; if that is not possible we fail with a descriptive error.

find_program(QCC_QIR_RUNNER_EXECUTABLE qir-runner)

if(NOT QCC_QIR_RUNNER_EXECUTABLE)
  find_program(QCC_UV_EXECUTABLE uv)
  if(QCC_UV_EXECUTABLE)
    message(STATUS "qir-runner not found; installing the 'qirrunner' package via 'uv tool install'...")
    execute_process(COMMAND ${QCC_UV_EXECUTABLE} tool install qirrunner RESULT_VARIABLE _exit_code)
    if(NOT _exit_code EQUAL 0)
      message(FATAL_ERROR "Failed to install 'qirrunner' via 'uv tool install qirrunner'.")
    endif()
    # Force a fresh search now that the tool should be installed.
    unset(QCC_QIR_RUNNER_EXECUTABLE CACHE)
    find_program(QCC_QIR_RUNNER_EXECUTABLE qir-runner)
  endif()
endif()

if(NOT QCC_QIR_RUNNER_EXECUTABLE)
  message(FATAL_ERROR "Could not find the 'qir-runner' executable, which is required to run the integration tests. "
                      "Install possible with `uv tool install qirrunner`.")
endif()

message(STATUS "Found qir-runner: ${QCC_QIR_RUNNER_EXECUTABLE}")
