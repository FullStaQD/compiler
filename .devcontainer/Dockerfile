FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get -y update && \
  apt-get install -y cmake ninja-build clang clang-tools clangd curl zstd git gcc g++ zlib1g-dev

# Downloads and installs a prebuilt LLVM/MLIR toolchain for the current OS and architecture. Uses `zstd`.
RUN curl -LsSf https://github.com/munich-quantum-software/setup-mlir/releases/latest/download/setup-mlir.sh \
  | bash -s -- -v 22.1.0 -p /opt/llvm