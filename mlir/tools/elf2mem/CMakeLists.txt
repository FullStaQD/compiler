set(LLVM_LINK_COMPONENTS Object Support)

add_llvm_executable(elf2mem elf2mem.cpp)

llvm_update_compile_flags(elf2mem)
