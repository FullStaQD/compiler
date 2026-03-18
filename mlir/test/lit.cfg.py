# -*- Python -*-

import os
from typing import TYPE_CHECKING, final

import lit.formats  # pyright: ignore[reportMissingTypeStubs]
import lit.util  # pyright: ignore[reportMissingTypeStubs]

from lit.llvm import llvm_config  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    @final
    class ConfigType:
        name: str = ""
        test_format = lit.formats.ShTest()
        suffixes: list[str] = []
        test_source_root: str = ""
        test_exec_root: str = ""
        project_binary_dir: str = ""
        project_source_dir: str = ""
        project_tools_dir: str = ""
        project_scripts_dir: str = ""
        llvm_tools_dir: str = ""
        llvm_shlib_ext: str = ""
        environment: dict[str, str] = {}
        substitutions: list[tuple[str, str]] = []
        excludes: list[str] = []

    config = ConfigType()

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "QCC_MLIR_COMPILER"

config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.project_binary_dir, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%project_source_dir", config.project_source_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories and filenames to exclude from the testsuite.
config.excludes = []

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.project_binary_dir, "test")
config.project_tools_dir = os.path.join(config.project_binary_dir, "bin")

# Tweak the PATH to include the tools and scripts dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
# llvm_config.with_environment("PATH", config.project_scripts_dir, append_path=True)

tool_dirs = [
    os.path.join(config.project_binary_dir, "mlir", "tools", "qcc-opt"),
    config.project_tools_dir,
    config.llvm_tools_dir
]
tools = [
    "qcc-opt"
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
