#!/usr/bin/env python3
"""Cross-platform integration runner for QIR simulation tests.

Usage:
    integration_run.py [-r <seed>] [-n <shots>] [-c <qcc_path>] [-t <mlir_translate_path>] <mlir_test_file>

Cross-platform replacement for integration_run.sh. Compiles an MLIR test file
using qcc, translates it to LLVM IR via mlir-translate, prints the QIR, and
optionally runs the QIR through qir-runner for simulation.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QIR integration tests: compile MLIR, lower to LLVM, simulate."
    )
    parser.add_argument(
        "-r", type=int, default=None,
        help="Random seed for reproducible simulations"
    )
    parser.add_argument(
        "-n", type=int, default=1,
        help="Number of shots for the simulation"
    )
    parser.add_argument(
        "-c", default="qcc",
        help="Path to the qcc compiler (default: qcc)"
    )
    parser.add_argument(
        "-t", default="mlir-translate",
        help="Path to mlir-translate (default: mlir-translate)"
    )
    parser.add_argument(
        "mlir_test_file",
        help="The MLIR test file to compile and simulate"
    )
    args = parser.parse_args()

    # Resolve full paths for qcc and mlir-translate
    qcc_path = args.c if os.path.isfile(args.c) else shutil.which(args.c) or args.c
    mlir_translate_path = (
        args.t if os.path.isfile(args.t) else shutil.which(args.t) or args.t
    )

    # Validate the input file
    mlir_test_file = args.mlir_test_file
    if not os.path.isfile(mlir_test_file):
        print(f"Error: file '{mlir_test_file}' not found")
        sys.exit(1)

    # Ensure qir-runner is available; try to install via uv if missing
    if shutil.which("qir-runner") is None:
        print("qir-runner not found, attempting to install via uv...", file=sys.stderr)
        uv_path = shutil.which("uv")
        if uv_path is not None:
            result = subprocess.run(
                [uv_path, "tool", "install", "qirrunner"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Error: failed to install qir-runner", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                sys.exit(1)
        else:
            print(
                "Error: uv not found, please install uv or qir-runner "
                "to run qrisp integration tests",
                file=sys.stderr
            )
            sys.exit(1)

    # Create a unique temporary directory for intermediate files
    tmp_dir = tempfile.mkdtemp(prefix="qcc_integration_")
    pipeline_exit = 0
    try:
        qcc_output = os.path.join(tmp_dir, "qcc_output.mlir")
        llvm_output = os.path.join(tmp_dir, "qcc_output.ll")

        # Step 1: Compile the MLIR test file using qcc
        result = subprocess.run(
            [qcc_path, mlir_test_file, "-o", qcc_output],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Print errors so FileCheck can see them (negative tests use "not")
            print(result.stdout, end="")
            print(result.stderr, end="")
            pipeline_exit = result.returncode
        else:
            # Step 2: Translate the compiled MLIR to LLVM IR
            result = subprocess.run(
                [mlir_translate_path, qcc_output, "-mlir-to-llvmir", "-o", llvm_output],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(result.stdout, end="")
                print(result.stderr, end="")
                pipeline_exit = result.returncode
            else:
                # Print the generated QIR
                print("---QIR---")
                with open(llvm_output) as f:
                    print(f.read(), end="")

        # Step 3: Run simulation only if the pipeline succeeded
        if pipeline_exit == 0:
            print("---OUTPUT-RECORDING---")
            qir_runner_args = [
                "qir-runner", "-f", llvm_output, "-s", str(args.n)
            ]
            if args.r is not None:
                qir_runner_args.extend(["-r", str(args.r)])
            result = subprocess.run(
                qir_runner_args, capture_output=True, text=True
            )
            # Print all output so FileCheck can verify it
            print(result.stdout, end="")
            print(result.stderr, end="")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    sys.exit(pipeline_exit)


if __name__ == "__main__":
    main()
