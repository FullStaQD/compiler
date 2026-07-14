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
 * The Ibex core that executes this image does not fetch its first
 * instruction from address 0: `ibex_if_stage` hardwires the post-reset
 * fetch address to `{boot_addr_i[31:8], 8'h80}` (see PC_BOOT in
 * ibex_if_stage.sv), and vproc_top.sv instantiates it with boot_addr_i =
 * 0, so the very first fetch is always from byte address 0x80. The first
 * 0x80 bytes are also where a vectored mtvec (also derived from the same
 * boot_addr_i) sends synchronous exceptions, i.e. they are reserved as the
 * (unused, but real) trap-vector region. Code must therefore start at
 * BOOT_ADDR, not address 0 -- placing it at 0 only appears to work because
 * an illegal-instruction trap on whatever garbage sits at 0x80 happens to
 * vector back to address 0.
 *
 * The hardware jumps to BOOT_ADDR directly at reset -- there is no caller,
 * so `ra`/`sp` hold whatever garbage the core reset with. qcc therefore
 * synthesizes a small `_start` function (see ConvertQIRToIntrinsics.cpp)
 * that initializes `sp` from __stack_top below, calls the user's designated
 * entry point (`main` by default, see qcc.entry_point / entry-point-name in
 * JaspToQC.td) with a real `jal` so it can return normally, and halts in an
 * infinite loop if that call ever returns. The target machine is configured
 * with function sections, so `_start` is emitted into its own `.text._start`
 * input section; pulling that section to the front of the output .text
 * guarantees it lands at BOOT_ADDR regardless of how the rest of the module
 * is ordered. `main` itself is an ordinary function with no placement
 * requirement -- `_start` finds it via a normal call relocation.
 *
 * .rodata/.data/.bss are bundled together into the "data" region starting
 * at TEXT_REGION_SIZE; an explicit PHDRS split (rather than relying on
 * lld's default permission-based segment splitting) keeps the boundary
 * exactly at TEXT_REGION_SIZE regardless of linker defaults. A small stack
 * region follows .bss in the same segment, growing down from __stack_top;
 * it needs no initialized bytes, so it costs nothing in the .mem image
 * beyond reserving address space (like .bss, it's zero-filled on load).
 */

BOOT_ADDR = 0x00000080;
TEXT_REGION_SIZE = 0x1000;
STACK_SIZE = 0x1000;

OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS
{
  text PT_LOAD;
  data PT_LOAD;
}

SECTIONS
{
  . = BOOT_ADDR;

  .text : {
    KEEP(*(.text._start))
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

  . = ALIGN(4);
  .stack (NOLOAD) : {
    . += STACK_SIZE;
    __stack_top = .;
  } :data

  /DISCARD/ : {
    *(.comment)
    *(.eh_frame*)
    *(.riscv.attributes)
    *(.note.*)
  }
}
