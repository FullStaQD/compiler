FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Installs the basic underlying framework
RUN apt-get -y update && \
  apt-get install -y --no-install-recommends clang clang-tools clangd curl ca-certificates zstd git gcc g++

# Download the latest uv installer
ADD https://astral.sh/uv/0.10.11/install.sh /uv-installer.sh

# Install uv into /usr/local/bin so all users can access it
RUN UV_INSTALL_DIR=/usr/local/bin sh /uv-installer.sh && rm /uv-installer.sh

# Store tool binaries and environments in system-wide locations
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_TOOL_DIR=/opt/uv/tools

# Install the latest versions of cmake, ninja, lit, and prek as tools
RUN uv tool install cmake && \
  uv tool install ninja && \
  uv tool install lit && \
  uv tool install prek

# Downloads and installs a prebuilt LLVM/MLIR toolchain for the current OS and architecture. Uses `zstd`.
RUN curl -LsSf https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.sh \
  | bash -s -- -v 22.1.0 -p /opt/llvm
