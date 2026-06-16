#!/bin/bash
set -euo pipefail
exec 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Parse arguments: optional seed (-r <value>), optional shots (-n <value>), then the python test file
RNG_SEED=""
SHOTS="1"
while getopts "r:n:" opt; do
  case $opt in
    r) RNG_SEED="$OPTARG" ;;
    n) SHOTS="$OPTARG" ;;
    \?) echo "Usage: $0 [-r <seed>] [-n <shots>] <python_test_file>"; exit 1 ;;
  esac
done
shift $((OPTIND-1))

if [ $# -ne 1 ]; then
    echo "Usage: $0 [-r <seed>] [-n <shots>] <python_test_file>"
    exit 1
fi
if [ ! -f "$1" ]; then
    echo "Error: file '$1' not found"
    exit 1
fi

# Ensure qir-runner is installed. If not, attempt to install it as a uv tool.
which qir-runner >/dev/null 2>&1 || {
    echo "qir-runner not found, attempting to install via uv..."
    if which uv >/dev/null 2>&1; then
        uv tool install qirrunner
    else
        echo "uv not found, please install uv to run qrisp integration tests" >&2
        exit 1
    fi
}

# Create a unique temporary directory for this invocation.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Run our pipeline, lower to LLVM/QIR.
# Use a subshell to capture errors without triggering set -e, so
# that even failing test cases produce output for FileCheck.
set +e
(
  set -e
  qcc "$1" -o "$TMP_DIR/qcc_output.mlir"
  mlir-translate "$TMP_DIR/qcc_output.mlir" -mlir-to-llvmir -o "$TMP_DIR/qcc_output.ll"
  echo "---QIR---"
  cat "$TMP_DIR/qcc_output.ll"
)
PIPELINE_EXIT=$?
set -e

# Simulate via qir-runner (with optional seed for reproducibility)
# Only run if the pipeline succeeded; otherwise we just print the errors.
if [ "$PIPELINE_EXIT" -eq 0 ]; then
  echo "---OUTPUT-RECORDING---"
  if [ -n "$RNG_SEED" ]; then
    qir-runner -f "$TMP_DIR/qcc_output.ll" -s "$SHOTS" -r "$RNG_SEED"
  else
    qir-runner -f "$TMP_DIR/qcc_output.ll" -s "$SHOTS"
  fi
else
  # In case of pipeline failure, stderr messages from the subshell have
  # already been redirected to stdout above (2>&1).  All we need to do
  # is propagate the failure so test cases expecting the error pass.
  :
fi
exit "$PIPELINE_EXIT"
