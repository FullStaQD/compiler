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
 * The HiSEP-Q opcode simulator (see caps-tum/HiSEP-Q-2.0, demo/verilator)
 * loads a single memory image and starts fetching at address 0, so .text
 * has to be the very first thing in the image, and the entry function has
 * to be the very first thing in .text.
 *
 * qcc always marks exactly one function as the entry point (`main` by
 * default, see qcc.entry_point / entry-point-name in JaspToQC.td) and the
 * target machine is configured with function sections, so that function is
 * emitted into its own `.text.main` input section. Pulling that section to
 * the front of the output .text guarantees it lands at address 0 regardless
 * of how the rest of the module is ordered.
 *
 * All sections share a single PT_LOAD segment (via the explicit PHDRS
 * below) so lld doesn't insert page-sized gaps between .text and
 * .rodata/.data for permission separation, which would waste memory in a
 * target this small. Sections are packed at 4-byte (word) alignment
 * instead.
 */

OUTPUT_ARCH(riscv)
ENTRY(main)

PHDRS
{
  image PT_LOAD;
}

SECTIONS
{
  . = 0x00000000;

  .text : {
    KEEP(*(.text.main))
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

  /DISCARD/ : {
    *(.comment)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.note.*)
  }
}
