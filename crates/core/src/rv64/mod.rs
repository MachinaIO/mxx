//! RV64IM for proof images: the instruction set of the mathcal-A machine model ([`isa`]), an
//! assembler for glue code ([`asm`]), and an emulator of the machine for tests ([`machine`]).
//!
//! Leaf code comes from the Rust compiler (`no_std`, built for `riscv64imac-unknown-none-elf`
//! with `-C target-feature=-c,-zca,-a,-zaamo,-zalrsc` and `-Zbuild-std`), and is checked with
//! [`check_code`]: the model rejects compressed encodings, atomics, floating point, and RV64 word
//! operations other than `ADDIW`, and so does [`isa::Instr::decode`].

pub mod asm;
pub mod isa;
pub mod machine;

/// The index and word of the first instruction outside the machine's instruction set. The zero
/// word, which RISC-V defines as illegal and the machine does not decode, is allowed as padding:
/// it faults if it is ever executed.
pub fn check_code(words: &[u32]) -> Result<(), (usize, u32)> {
    match words
        .iter()
        .enumerate()
        .find(|(_, w)| **w != 0 && isa::Instr::decode(**w).is_none_or(|i| i == isa::Instr::Ebreak))
    {
        Some((i, w)) => Err((i, *w)),
        None => Ok(()),
    }
}
