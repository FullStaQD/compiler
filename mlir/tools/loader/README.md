# HiSEP-Q loader

`qcc --target=hisep-q` only compiles: it lowers a circuit down to native RISC-V
QISA (`--compile-to=native`) and stops there. Turning that into something the
HiSEP-Q simulator or hardware can run is two more, independent steps:

1. **Link** the object file against `hisepq.ld`, the linker script in this
   directory. It places code at the hardware's boot address and lays out
   `.rodata`/`.data`/`.bss` the way the opcode simulator expects (see the
   comment block at the top of the script for the memory-layout contract).
2. **Convert** the linked ELF into the `$readmemh` memory image the simulator
   loads, with the `elf2mem` tool (`mlir/tools/elf2mem`).

```sh
qcc --target=hisep-q --compile-to=native --binary input.mlir -o out.o
ld.lld -T mlir/tools/loader/hisepq.ld out.o -o out.elf
elf2mem out.elf -o out.mem
```

Any RISC-V-capable linker works in step 2, not just `ld.lld` — pass `-T
hisepq.ld` to whichever one you use.
