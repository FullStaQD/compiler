#!/bin/bash

which python3 >/dev/null 2>&1 || { echo "python not found. Please activate a venv with qrisp and xDSL installed." >&2; exit 1; }
python3 -c "import qrisp" >/dev/null 2>&1 || { echo "qrisp not found. Please activate a venv with qrisp installed." >&2; exit 1; }
python3 -c "import xdsl" >/dev/null 2>&1 || { echo "xDSL not found. Please activate a venv with xDSL installed." >&2; exit 1; }


# Run all python scripts in the qrisp_sources directory, storing the results in the current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remove any previously generated .mlir files.
find "$SCRIPT_DIR" -maxdepth 1 -name "*.generated_test.mlir" -delete

cd "$SCRIPT_DIR/qrisp_sources"
for py_file in *.py; do
    echo "Processing $py_file..."
    python3 "$py_file" > "../${py_file%.py}.generated_test.mlir"
done

cd ../
