/* One flat, position-independent image: code at 0, then read-only data. */
SECTIONS
{
  . = 0;
  .text : { *(.text .text.*) }
  . = ALIGN(8);
  .rodata : { *(.srodata .srodata.* .rodata .rodata.*) }
  .data : { *(.sdata .sdata.* .data .data.*) }
  .bss : { *(.sbss .sbss.* .bss .bss.* COMMON) }
  /DISCARD/ : { *(.eh_frame*) *(.comment) *(.riscv.attributes) }
}
