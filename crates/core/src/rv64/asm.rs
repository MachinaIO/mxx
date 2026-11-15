//! A small assembler for the glue code of proof images.
//!
//! Labels resolve to instruction indices. Every conditional jump is emitted as an inverted short
//! branch over an unconditional `jal`, so its target may be anywhere within the `jal` range of one
//! mebibyte; [`Asm::steps_of_branch`] gives its cost. Calls ([`Asm::call`]) are `auipc` and
//! `jalr`, so a callee may be anywhere within two gibibytes.

use super::isa::{AluOp, BranchOp, EncodeError, ImmOp, Instr, Reg, ShiftOp, Width};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

#[derive(Clone, Copy, Debug)]
enum Fixup {
    Jal {
        rd: Reg,
        target: Label,
    },
    /// `auipc ra` here and `jalr ra` after it.
    Call {
        target: Label,
        words: u64,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum AsmError {
    #[error("label {0} is never placed")]
    Unplaced(usize),
    #[error(transparent)]
    Encode(#[from] EncodeError),
}

#[derive(Clone, Copy, Debug)]
enum Item {
    Instr(Instr),
    /// A word copied verbatim, such as compiled leaf code.
    Word(u32),
}

#[derive(Default)]
pub struct Asm {
    instrs: Vec<Item>,
    fixups: Vec<(usize, Fixup)>,
    labels: Vec<Option<usize>>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    /// The index of the next instruction.
    pub fn here(&self) -> usize {
        self.instrs.len()
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn place(&mut self, label: Label) {
        assert!(self.labels[label.0].is_none(), "label placed twice");
        self.labels[label.0] = Some(self.instrs.len());
    }

    pub fn emit(&mut self, instr: Instr) {
        self.instrs.push(Item::Instr(instr));
    }

    /// Appends a word verbatim.
    pub fn word(&mut self, word: u32) {
        self.instrs.push(Item::Word(word));
    }

    pub fn alu(&mut self, op: AluOp, rd: Reg, rs1: Reg, rs2: Reg) {
        self.emit(Instr::Alu { op, rd, rs1, rs2 });
    }

    pub fn addi(&mut self, rd: Reg, rs1: Reg, imm: i32) {
        self.emit(Instr::Imm { op: ImmOp::Addi, rd, rs1, imm });
    }

    pub fn mv(&mut self, rd: Reg, rs: Reg) {
        self.addi(rd, rs, 0);
    }

    pub fn shift(&mut self, op: ShiftOp, rd: Reg, rs1: Reg, shamt: u32) {
        self.emit(Instr::Shift { op, rd, rs1, shamt });
    }

    pub fn ld(&mut self, rd: Reg, rs1: Reg, imm: i32) {
        self.emit(Instr::Load { width: Width::D, unsigned: false, rd, rs1, imm });
    }

    pub fn sd(&mut self, rs2: Reg, rs1: Reg, imm: i32) {
        self.emit(Instr::Store { width: Width::D, rs1, rs2, imm });
    }

    pub fn lbu(&mut self, rd: Reg, rs1: Reg, imm: i32) {
        self.emit(Instr::Load { width: Width::B, unsigned: true, rd, rs1, imm });
    }

    pub fn sb(&mut self, rs2: Reg, rs1: Reg, imm: i32) {
        self.emit(Instr::Store { width: Width::B, rs1, rs2, imm });
    }

    /// The number of instructions [`Asm::li`] emits for `value`.
    pub fn li_len(value: i64) -> u64 {
        let mut probe = Asm::new();
        probe.li(Reg::T0, value);
        probe.instrs.len() as u64
    }

    /// Loads a 64-bit constant.
    pub fn li(&mut self, rd: Reg, value: i64) {
        if (-2048..2048).contains(&value) {
            self.addi(rd, Reg::ZERO, value as i32);
        } else if i32::try_from(value).is_ok() {
            let low = ((value << 52) >> 52) as i32;
            let high = (((value - i64::from(low)) >> 12) as u32) & 0xf_ffff;
            self.emit(Instr::Lui { rd, imm: high });
            if low != 0 {
                self.emit(Instr::Addiw { rd, rs1: rd, imm: low });
            }
        } else {
            let low = (value << 52) >> 52;
            // `value = rest * 4096 + low` modulo 2^64, which is all the machine computes.
            let rest = value.wrapping_sub(low) >> 12;
            self.li(rd, rest);
            self.shift(ShiftOp::Slli, rd, rd, 12);
            if low != 0 {
                self.addi(rd, rd, low as i32);
            }
        }
    }

    /// `jal x0, target`: one step.
    pub fn jump(&mut self, target: Label) {
        self.jal(Reg::ZERO, target);
    }

    /// `jal rd, target`: one step.
    pub fn jal(&mut self, rd: Reg, target: Label) {
        self.fixups.push((self.instrs.len(), Fixup::Jal { rd, target }));
        self.instrs.push(Item::Instr(Instr::Ecall));
    }

    /// `auipc ra; jalr ra`: calls `words` instructions past `target`, anywhere within two
    /// gibibytes, in two steps.
    pub fn call(&mut self, target: Label, words: u64) {
        self.fixups.push((self.instrs.len(), Fixup::Call { target, words }));
        self.instrs.push(Item::Instr(Instr::Ecall));
        self.instrs.push(Item::Instr(Instr::Ecall));
    }

    /// Jumps to `target` when `op` holds: two steps when it does, one when it does not.
    pub fn branch(&mut self, op: BranchOp, rs1: Reg, rs2: Reg, target: Label) {
        self.emit(Instr::Branch { op: op.invert(), rs1, rs2, offset: 8 });
        self.jump(target);
    }

    /// The most steps [`Asm::branch`] takes.
    pub const fn steps_of_branch() -> u64 {
        2
    }

    pub fn ecall(&mut self) {
        self.emit(Instr::Ecall);
    }

    /// The encoded instructions.
    pub fn finish(mut self) -> Result<Vec<u32>, AsmError> {
        for (at, fixup) in std::mem::take(&mut self.fixups) {
            let (target, words) = match fixup {
                Fixup::Jal { target, .. } => (target, 0),
                Fixup::Call { target, words } => (target, words),
            };
            let to = self.labels[target.0].ok_or(AsmError::Unplaced(target.0))? + words as usize;
            let offset = (to as i64 - at as i64) * 4;
            match fixup {
                Fixup::Jal { rd, .. } => {
                    let offset =
                        i32::try_from(offset).map_err(|_| EncodeError::Immediate(offset))?;
                    self.instrs[at] = Item::Instr(Instr::Jal { rd, offset });
                }
                Fixup::Call { .. } => {
                    // offset = high * 4096 + low with a sign-extended 12-bit low part.
                    let high = (offset + 0x800) >> 12;
                    let low = offset - (high << 12);
                    if !(-(1 << 19)..1 << 19).contains(&high) {
                        return Err(EncodeError::Immediate(offset).into());
                    }
                    let imm = (high as u32) & 0xf_ffff;
                    self.instrs[at] = Item::Instr(Instr::Auipc { rd: Reg::RA, imm });
                    self.instrs[at + 1] =
                        Item::Instr(Instr::Jalr { rd: Reg::RA, rs1: Reg::RA, imm: low as i32 });
                }
            }
        }
        Ok(self
            .instrs
            .into_iter()
            .map(|item| match item {
                Item::Instr(instr) => instr.encode(),
                Item::Word(word) => Ok(word),
            })
            .collect::<Result<_, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rv64::machine::{Machine, Outcome};
    use rand::Rng;

    #[test]
    fn test_li_loads_every_value() {
        let mut rng = rand::rng();
        let mut values: Vec<i64> = (0..300).map(|_| rng.random()).collect();
        values.extend((0..300).map(|_| rng.random_range(-70000..70000)));
        values.extend([
            0,
            -1,
            2047,
            -2048,
            2048,
            i64::MAX,
            i64::MIN,
            0x7fff_f800,
            0x7fff_ffff,
            -0x8000_0000,
        ]);
        for value in values {
            let mut asm = Asm::new();
            asm.li(Reg::A1, value);
            // HALT with the eight bytes of a1 stored just past the image.
            let words = {
                let mut tail = Asm::new();
                tail.li(Reg::A0, 0x20000);
                tail.sd(Reg::A1, Reg::A0, 0);
                tail.li(Reg::A1, 8);
                tail.li(Reg::A7, 0);
                tail.ecall();
                let mut all = asm.finish().unwrap();
                all.extend(tail.finish().unwrap());
                all
            };
            let image: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut machine = Machine::load(&image, 0x20000).unwrap();
            let (outcome, _) = machine.run(10_000, &mut crate::rv64::machine::NoEnvironment);
            assert_eq!(outcome, Outcome::Halted(value.to_le_bytes().to_vec()), "{value:#x}");
        }
    }
}
