/* ===----------------------------------------------------------------------===
 *
 * Part of the FullStaQD Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See <repo-root>/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===----------------------------------------------------------------------===
 *
 * Linker script for the --target=hisep-q output of qcc.
 *
 * The HiSEP-Q simulator (caps-tum/HiSEP-Q-2.0, demo/) loads the .mem image
 * into a single flat word memory with one $readmemh. elf2mem derives the
 * image's @word markers from the ELF's PT_LOAD vaddrs, so the loader imposes
 * no fixed addresses on this script; the only hard constraints are BOOT_ADDR
 * and the testbench memory size (MEM_SZ = 0x40000 bytes).
 *
 * BOOT_ADDR = 0x80 is fixed by hardware: `ibex_if_stage` hardwires the
 * post-reset fetch address to {boot_addr_i[31:8], 8'h80} (see PC_BOOT in
 * ibex_if_stage.sv) and vproc_top.sv instantiates the core with
 * boot_addr_i = 0. The 0x80 bytes below it are the trap-vector region of the
 * vectored mtvec derived from the same boot_addr_i -- unused, but reserved.
 *
 * The core jumps to BOOT_ADDR at reset with no caller, so `ra`/`sp` hold
 * garbage. qcc synthesizes a `_start` (see ConvertQIRToIntrinsics.cpp) that
 * loads `sp` from __stack_top, calls the designated entry point (`main` by
 * default, see qcc.entry_point in JaspToQC.td), and spins if it returns.
 * Functions are emitted in per-function sections, so pulling `.text._start`
 * to the front of .text pins it to BOOT_ADDR.
 *
 * All sections share one PT_LOAD so lld doesn't insert page-sized permission
 * gaps between them (the flat memory has no MMU). Sections pack at word
 * alignment, and elf2mem materializes the whole segment (p_memsz) into the
 * image, so layout size is image size. For that reason the stack is not a
 * section at all: __stack_top is just a symbol STACK_SIZE bytes past the end
 * of .bss, so the reservation adds nothing to the image. STACK_SIZE is the
 * headroom `sp` has before it grows down into .bss.
 */

BOOT_ADDR = 0x00000080;
STACK_SIZE = 0x1000;

OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS
{
  image PT_LOAD;
}

SECTIONS
{
  . = BOOT_ADDR;

  .text : {
    KEEP(*(.text._start))
    *(.text .text.*)
  } :image

  . = ALIGN(4);
  .rodata : {
    *(.rodata .rodata.*)
  } :image

  . = ALIGN(4);
  .data : {
    *(.data .data.*)
  } :image

  . = ALIGN(4);
  .bss : {
    *(.bss .bss.*)
  } :image

  /* Keep __stack_top 16-byte aligned per the RISC-V psABI. */
  __stack_top = ALIGN(16) + STACK_SIZE;

  /DISCARD/ : {
    *(.comment)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.note.*)
  }
}
