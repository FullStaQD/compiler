// ===----------------------------------------------------------------------===//
//
// Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See <repo-root>/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// elf2mem converts a linked HiSEP-Q ELF image into the Verilog $readmemh
// memory file consumed by the opcode simulator (caps-tum/HiSEP-Q-2.0,
// demo/verilator). It does not interpret the program in any way: it just
// walks the ELF's PT_LOAD segments in address order and dumps their bytes as
// 32-bit hex words, emitting an `@<address>` line whenever a segment does
// not immediately follow the previous one. Getting the memory layout right
// (entry code at address 0, sensible section placement) is the linker
// script's job (see hisepq.ld); this tool assumes the ELF it is given
// already reflects the intended layout.
//
// ===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace cl = llvm::cl;

namespace {

struct Segment {
  uint32_t addr;
  llvm::ArrayRef<uint8_t> fileBytes;
  uint32_t memSize;
};

} // namespace

int main(int argc, char** argv) {
  static cl::OptionCategory elf2memCategory("elf2mem options");

  const cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<elf-file>"), cl::Required,
                                           cl::cat(elf2memCategory));
  const cl::opt<std::string> outputFilename("o", cl::desc("Output .mem file"), cl::value_desc("filename"),
                                            cl::init("-"), cl::cat(elf2memCategory));

  cl::ParseCommandLineOptions(argc, argv,
                              "elf2mem - convert a HiSEP-Q ELF image into a Verilog $readmemh memory file\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = bufferOrErr.getError()) {
    llvm::errs() << inputFilename << ": " << ec.message() << "\n";
    return 1;
  }

  llvm::Expected<std::unique_ptr<llvm::object::Binary>> binaryOrErr =
      llvm::object::createBinary((*bufferOrErr)->getMemBufferRef());
  if (!binaryOrErr) {
    llvm::errs() << inputFilename << ": " << llvm::toString(binaryOrErr.takeError()) << "\n";
    return 1;
  }

  auto* elfObj = llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(binaryOrErr->get());
  if (elfObj == nullptr) {
    llvm::errs() << inputFilename << ": expected a 32-bit little-endian ELF (e.g. riscv32)\n";
    return 1;
  }

  const llvm::object::ELFFile<llvm::object::ELF32LE>& elfFile = elfObj->getELFFile();

  auto phdrsOrErr = elfFile.program_headers();
  if (!phdrsOrErr) {
    llvm::errs() << inputFilename << ": " << llvm::toString(phdrsOrErr.takeError()) << "\n";
    return 1;
  }

  std::vector<Segment> segments;
  for (const auto& phdr : *phdrsOrErr) {
    if (phdr.p_type != llvm::ELF::PT_LOAD || phdr.p_memsz == 0) {
      continue;
    }

    auto bytesOrErr = elfFile.getSegmentContents(phdr);
    if (!bytesOrErr) {
      llvm::errs() << inputFilename << ": " << llvm::toString(bytesOrErr.takeError()) << "\n";
      return 1;
    }

    segments.push_back({static_cast<uint32_t>(phdr.p_vaddr), *bytesOrErr, static_cast<uint32_t>(phdr.p_memsz)});
  }

  if (segments.empty()) {
    llvm::errs() << inputFilename << ": no loadable (PT_LOAD) segments found\n";
    return 1;
  }

  std::sort(segments.begin(), segments.end(),
            [](const Segment& lhs, const Segment& rhs) { return lhs.addr < rhs.addr; });

  for (size_t i = 0; i < segments.size(); ++i) {
    if (segments[i].addr % 4 != 0) {
      llvm::errs() << inputFilename << ": segment at address " << llvm::format("0x%08X", segments[i].addr)
                   << " is not word (4-byte) aligned\n";
      return 1;
    }
    if (i > 0 && segments[i].addr < segments[i - 1].addr + segments[i - 1].memSize) {
      llvm::errs() << inputFilename << ": overlapping loadable segments\n";
      return 1;
    }
  }

  std::error_code ec;
  llvm::ToolOutputFile outFile(outputFilename, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << outputFilename << ": " << ec.message() << "\n";
    return 1;
  }

  llvm::raw_ostream& os = outFile.os();
  std::optional<uint32_t> nextExpectedAddr;
  for (const Segment& seg : segments) {
    if (nextExpectedAddr != seg.addr) {
      // $readmemh addresses a word array, not bytes: divide by 4 (segment alignment was already
      // checked above, so this is exact).
      os << llvm::format("@%08X\n", seg.addr / 4);
    }

    for (uint32_t off = 0; off < seg.memSize; off += 4) {
      uint32_t word = 0;
      for (uint32_t b = 0; b < 4; ++b) {
        uint32_t idx = off + b;
        uint8_t byte = idx < seg.fileBytes.size() ? seg.fileBytes[idx] : 0;
        word |= static_cast<uint32_t>(byte) << (8 * b);
      }
      os << llvm::format("%08X\n", word);
    }

    nextExpectedAddr = seg.addr + seg.memSize;
  }

  outFile.keep();
  return 0;
}
