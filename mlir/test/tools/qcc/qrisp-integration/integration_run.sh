#!/bin/bash
set -euo pipefail
exec 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Parse arguments: optional seed (-r <value>), then the python test file
RNG_SEED=""
while getopts "r:" opt; do
  case $opt in
    r) RNG_SEED="$OPTARG" ;;
    \?) echo "Usage: $0 [-r <seed>] <python_test_file>"; exit 1 ;;
  esac
done
shift $((OPTIND-1))

if [ $# -ne 1 ]; then
    echo "Usage: $0 [-r <seed>] <python_test_file>"
    exit 1
fi
if [ ! -f "$1" ]; then
    echo "Error: file '$1' not found"
    exit 1
fi

# Ensure uv is installed (stderr only so stdout stays clean for FileCheck)
which uv >/dev/null 2>&1 || { echo "uv not found, please install uv to run qrisp integration tests" >&2; exit 1; }

# Ensure uv dependencies are synced from the script directory
cd "$SCRIPT_DIR"
uv sync >/dev/null 2>&1

# Create a unique temporary directory for this invocation.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Generate MLIR from the Qrisp Python test file
"$VENV_DIR/bin/python3" "$1" > "$TMP_DIR/qcc_input.mlir"

# Run our pipeline, lower to LLVM/QIR.
# Use a subshell to capture errors without triggering set -e, so
# that even failing test cases produce output for FileCheck.
set +e
(
  set -e
  qcc "$TMP_DIR/qcc_input.mlir" -o "$TMP_DIR/qcc_output.mlir"
  mlir-translate "$TMP_DIR/qcc_output.mlir" -mlir-to-llvmir -o "$TMP_DIR/qcc_output.ll"
)
PIPELINE_EXIT=$?
set -e

# Simulate via qir-runner (with optional seed for reproducibility)
# Only run if the pipeline succeeded; otherwise we just print the errors.
if [ "$PIPELINE_EXIT" -eq 0 ]; then
  if [ -n "$RNG_SEED" ]; then
    "$VENV_DIR/bin/qir-runner" -f "$TMP_DIR/qcc_output.ll" -s 2 -r "$RNG_SEED"
  else
    "$VENV_DIR/bin/qir-runner" -f "$TMP_DIR/qcc_output.ll" -s 2
  fi
else
  # In case of pipeline failure, stderr messages from the subshell have
  # already been redirected to stdout above (2>&1).  All we need to do
  # is propagate the failure so test cases expecting the error pass.
  :
fi
exit "$PIPELINE_EXIT"
