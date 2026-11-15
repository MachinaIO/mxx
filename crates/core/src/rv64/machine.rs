//! An emulator of the mathcal-A machine, for tests.
//!
//! It mirrors `MathcalA/Trusted/Machine.lean`: a flat image is loaded at [`BASE`] with `pc =
//! BASE`, `sp = BASE + mem_size`, other registers zero, and memory zero past the image. Code is
//! the decoded image and is never writable. Loads must be naturally aligned and inside the address
//! space; stores must also lie past the image. `EBREAK` and undecodable words fault. Environment
//! calls take their number from `a7`:
//!
//! | `a7` | Call | Arguments | Extra steps |
//! | --- | --- | --- | --- |
//! | 0 | `HALT` | `a0` pointer, `a1` length of the output | the length |
//! | 1 | `QUERY` | `a0`, `a1` request; `a2`, `a3` response buffer and capacity; returns the length in `a0` | request plus response length |
//! | 2 | `COINS` | `a0` pointer, `a1` count of uniform bytes | the count |
//!
//! Every instruction costs one step and an environment call `1 + extra`; a run that would exceed
//! its budget times out. The emulator is a test tool: claims are about the Lean model, not about
//! this code.

use super::isa::{AluOp, BranchOp, ImmOp, Instr, Reg, ShiftOp, Width};
use rand::RngCore;

pub const BASE: u64 = 0x10000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Halted(Vec<u8>),
    Fault,
    Timeout,
}

/// The external side of a run: the game answering `QUERY`, and the source of `COINS`.
pub trait Environment {
    fn query(&mut self, request: &[u8]) -> Vec<u8>;
    fn coins(&mut self, out: &mut [u8]) {
        rand::rng().fill_bytes(out);
    }
}

/// Answers every query with nothing.
pub struct NoEnvironment;

impl Environment for NoEnvironment {
    fn query(&mut self, _: &[u8]) -> Vec<u8> {
        Vec::new()
    }
}

pub struct Machine {
    regs: [u64; 32],
    pc: u64,
    mem: Vec<u8>,
    image_len: u64,
    code: Vec<Option<Instr>>,
}

enum Step {
    Next,
    Ecall,
    Fault,
}

impl Machine {
    /// Loads `image`, or `None` when it does not fit.
    pub fn load(image: &[u8], mem_size: u64) -> Option<Self> {
        if image.len() as u64 > mem_size || BASE.checked_add(mem_size).is_none() {
            return None;
        }
        let mut mem = vec![0u8; usize::try_from(mem_size).ok()?];
        mem[..image.len()].copy_from_slice(image);
        let code = image
            .chunks(4)
            .map(|w| {
                if w.len() == 4 {
                    Instr::decode(u32::from_le_bytes(w.try_into().unwrap()))
                } else {
                    None
                }
            })
            .collect();
        let mut regs = [0u64; 32];
        regs[2] = BASE + mem_size;
        Some(Self { regs, pc: BASE, mem, image_len: image.len() as u64, code })
    }

    pub fn pc(&self) -> u64 {
        self.pc
    }

    pub fn reg(&self, r: Reg) -> u64 {
        self.regs[usize::from(r.0)]
    }

    fn set(&mut self, r: Reg, v: u64) {
        if r.0 != 0 {
            self.regs[usize::from(r.0)] = v;
        }
    }

    fn readable(&self, a: u64, w: u64) -> bool {
        a >= BASE && a.checked_add(w).is_some_and(|end| end <= BASE + self.mem.len() as u64)
    }

    fn writable(&self, a: u64, w: u64) -> bool {
        a >= BASE + self.image_len &&
            a.checked_add(w).is_some_and(|end| end <= BASE + self.mem.len() as u64)
    }

    pub fn read(&self, a: u64, n: u64) -> &[u8] {
        let start = (a - BASE) as usize;
        &self.mem[start..start + n as usize]
    }

    fn write(&mut self, a: u64, bytes: &[u8]) {
        let start = (a - BASE) as usize;
        self.mem[start..start + bytes.len()].copy_from_slice(bytes);
    }

    fn fetch(&self) -> Option<Instr> {
        let off = self.pc.checked_sub(BASE)?;
        if off % 4 != 0 {
            return None;
        }
        self.code.get((off / 4) as usize).copied().flatten()
    }

    fn step(&mut self) -> Step {
        let Some(instr) = self.fetch() else { return Step::Fault };
        let next = self.pc.wrapping_add(4);
        let x = |r: Reg| self.regs[usize::from(r.0)];
        match instr {
            Instr::Alu { op, rd, rs1, rs2 } => {
                let (a, b) = (x(rs1), x(rs2));
                let v = match op {
                    AluOp::Add => a.wrapping_add(b),
                    AluOp::Sub => a.wrapping_sub(b),
                    AluOp::Sll => a << (b & 63),
                    AluOp::Srl => a >> (b & 63),
                    AluOp::Sra => ((a as i64) >> (b & 63)) as u64,
                    AluOp::And => a & b,
                    AluOp::Or => a | b,
                    AluOp::Xor => a ^ b,
                    AluOp::Slt => u64::from((a as i64) < (b as i64)),
                    AluOp::Sltu => u64::from(a < b),
                    AluOp::Mul => a.wrapping_mul(b),
                    AluOp::Mulh => ((i128::from(a as i64) * i128::from(b as i64)) >> 64) as u64,
                    AluOp::Mulhsu => ((i128::from(a as i64) * i128::from(b)) >> 64) as u64,
                    AluOp::Mulhu => ((u128::from(a) * u128::from(b)) >> 64) as u64,
                    AluOp::Div => match (a as i64, b as i64) {
                        (_, 0) => u64::MAX,
                        (i64::MIN, -1) => a,
                        (a, b) => (a / b) as u64,
                    },
                    AluOp::Divu => a.checked_div(b).unwrap_or(u64::MAX),
                    AluOp::Rem => match (a as i64, b as i64) {
                        (_, 0) => a,
                        (i64::MIN, -1) => 0,
                        (a, b) => (a % b) as u64,
                    },
                    AluOp::Remu => a.checked_rem(b).unwrap_or(a),
                };
                self.set(rd, v);
            }
            Instr::Imm { op, rd, rs1, imm } => {
                let (a, i) = (x(rs1), i64::from(imm) as u64);
                let v = match op {
                    ImmOp::Addi => a.wrapping_add(i),
                    ImmOp::Slti => u64::from((a as i64) < (i as i64)),
                    ImmOp::Sltiu => u64::from(a < i),
                    ImmOp::Xori => a ^ i,
                    ImmOp::Ori => a | i,
                    ImmOp::Andi => a & i,
                };
                self.set(rd, v);
            }
            Instr::Shift { op, rd, rs1, shamt } => {
                let a = x(rs1);
                let v = match op {
                    ShiftOp::Slli => a << shamt,
                    ShiftOp::Srli => a >> shamt,
                    ShiftOp::Srai => ((a as i64) >> shamt) as u64,
                };
                self.set(rd, v);
            }
            Instr::Addiw { rd, rs1, imm } => {
                let v = (x(rs1) as i32).wrapping_add(imm);
                self.set(rd, i64::from(v) as u64);
            }
            Instr::Lui { rd, imm } => self.set(rd, i64::from((imm << 12) as i32) as u64),
            Instr::Auipc { rd, imm } => {
                self.set(rd, self.pc.wrapping_add(i64::from((imm << 12) as i32) as u64))
            }
            Instr::Load { width, unsigned, rd, rs1, imm } => {
                let a = x(rs1).wrapping_add(i64::from(imm) as u64);
                let w = width.bytes();
                if a % w != 0 || !self.readable(a, w) {
                    return Step::Fault;
                }
                let mut buf = [0u8; 8];
                buf[..w as usize].copy_from_slice(self.read(a, w));
                let raw = u64::from_le_bytes(buf);
                let v = if unsigned || width == Width::D {
                    raw
                } else {
                    let shift = 64 - 8 * w;
                    (((raw << shift) as i64) >> shift) as u64
                };
                self.set(rd, v);
            }
            Instr::Store { width, rs1, rs2, imm } => {
                let a = x(rs1).wrapping_add(i64::from(imm) as u64);
                let w = width.bytes();
                if a % w != 0 || !self.writable(a, w) {
                    return Step::Fault;
                }
                let bytes = x(rs2).to_le_bytes();
                self.write(a, &bytes[..w as usize]);
            }
            Instr::Branch { op, rs1, rs2, offset } => {
                let (a, b) = (x(rs1), x(rs2));
                let taken = match op {
                    BranchOp::Beq => a == b,
                    BranchOp::Bne => a != b,
                    BranchOp::Blt => (a as i64) < (b as i64),
                    BranchOp::Bge => (a as i64) >= (b as i64),
                    BranchOp::Bltu => a < b,
                    BranchOp::Bgeu => a >= b,
                };
                if taken {
                    self.pc = self.pc.wrapping_add(i64::from(offset) as u64);
                    return Step::Next;
                }
            }
            Instr::Jal { rd, offset } => {
                self.set(rd, next);
                self.pc = self.pc.wrapping_add(i64::from(offset) as u64);
                return Step::Next;
            }
            Instr::Jalr { rd, rs1, imm } => {
                let target = x(rs1).wrapping_add(i64::from(imm) as u64) & !1;
                self.set(rd, next);
                self.pc = target;
                return Step::Next;
            }
            Instr::Ecall => {
                self.pc = next;
                return Step::Ecall;
            }
            Instr::Ebreak => return Step::Fault,
            Instr::Fence => {}
        }
        self.pc = next;
        Step::Next
    }

    /// Runs with a budget of `fuel` steps. Returns the outcome and the steps charged.
    pub fn run(&mut self, fuel: u64, env: &mut dyn Environment) -> (Outcome, u64) {
        let mut used = 0u64;
        loop {
            if used >= fuel {
                return (Outcome::Timeout, used);
            }
            used += 1;
            match self.step() {
                Step::Next => {}
                Step::Fault => return (Outcome::Fault, used),
                Step::Ecall => {
                    let (p, n) = (self.reg(Reg::A0), self.reg(Reg::A1));
                    let (extra, halt) = match self.reg(Reg::A7) {
                        0 => {
                            if !self.readable(p, n) {
                                return (Outcome::Fault, used);
                            }
                            (n, Some(self.read(p, n).to_vec()))
                        }
                        1 => {
                            let (buf, cap) = (self.reg(Reg::A2), self.reg(Reg::A3));
                            if !self.readable(p, n) || !self.writable(buf, cap) {
                                return (Outcome::Fault, used);
                            }
                            let response = env.query(self.read(p, n));
                            if response.len() as u64 > cap {
                                return (Outcome::Fault, used);
                            }
                            self.write(buf, &response);
                            self.set(Reg::A0, response.len() as u64);
                            (n + response.len() as u64, None)
                        }
                        2 => {
                            if !self.writable(p, n) {
                                return (Outcome::Fault, used);
                            }
                            let mut bytes = vec![0u8; n as usize];
                            env.coins(&mut bytes);
                            self.write(p, &bytes);
                            (n, None)
                        }
                        _ => return (Outcome::Fault, used),
                    };
                    if extra > fuel - used {
                        return (Outcome::Timeout, fuel);
                    }
                    used += extra;
                    if let Some(out) = halt {
                        return (Outcome::Halted(out), used);
                    }
                }
            }
        }
    }
}
