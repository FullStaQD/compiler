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
 * The HiSEP-Q opcode simulator (see caps-tum/HiSEP-Q-2.0, demo/verilator and
 * demo/README.md) loads a "combined" memory image with two fixed regions:
 * instruction words starting at byte address 0, and data starting at the
 * fixed byte address 0x1000 (word address 0x400) -- regardless of how big
 * .text actually is. That reservation is a hardware/toolchain convention,
 * not something elf2mem can infer from the ELF, so it has to be baked into
 * this script.
 *
 * qcc always marks exactly one function as the entry point (`main` by
 * default, see qcc.entry_point / entry-point-name in JaspToQC.td) and the
 * target machine is configured with function sections, so that function is
 * emitted into its own `.text.main` input section. Pulling that section to
 * the front of the output .text guarantees it lands at address 0 regardless
 * of how the rest of the module is ordered.
 *
 * .rodata/.data/.bss are bundled together into the "data" region starting
 * at TEXT_REGION_SIZE; an explicit PHDRS split (rather than relying on
 * lld's default permission-based segment splitting) keeps the boundary
 * exactly at TEXT_REGION_SIZE regardless of linker defaults.
 */

TEXT_REGION_SIZE = 0x1000;

OUTPUT_ARCH(riscv)
ENTRY(main)

PHDRS
{
  text PT_LOAD;
  data PT_LOAD;
}

SECTIONS
{
  . = 0x00000000;

  .text : {
    KEEP(*(.text.main))
    *(.text .text.*)
  } :text

  ASSERT(. <= TEXT_REGION_SIZE, "hisepq.ld: .text overflows the reserved instruction region (TEXT_REGION_SIZE)")

  . = TEXT_REGION_SIZE;

  .rodata : {
    *(.rodata .rodata.*)
  } :data

  . = ALIGN(4);
  .data : {
    *(.data .data.*)
  } :data

  . = ALIGN(4);
  .bss : {
    *(.bss .bss.*)
  } :data

  /DISCARD/ : {
    *(.comment)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.note.*)
  }
}
