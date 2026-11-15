//! The instruction set of the mathcal-A machine model: RV64IM without compressed encodings,
//! atomics, floating point, CSR access, or RV64 word operations other than `ADDIW`.
//!
//! [`Instr::decode`] accepts exactly the encodings of this set (and `FENCE` and `EBREAK`, which
//! the model decodes; the machine rejects `EBREAK`), and [`Instr::encode`] is its inverse.

use std::fmt;

/// A register number `0..32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(pub u8);

impl Reg {
    pub const ZERO: Reg = Reg(0);
    pub const RA: Reg = Reg(1);
    pub const SP: Reg = Reg(2);
    pub const T0: Reg = Reg(5);
    pub const T1: Reg = Reg(6);
    pub const T2: Reg = Reg(7);
    pub const A0: Reg = Reg(10);
    pub const A1: Reg = Reg(11);
    pub const A2: Reg = Reg(12);
    pub const A3: Reg = Reg(13);
    pub const A4: Reg = Reg(14);
    pub const A5: Reg = Reg(15);
    pub const A6: Reg = Reg(16);
    pub const A7: Reg = Reg(17);
    pub const T3: Reg = Reg(28);
    pub const T4: Reg = Reg(29);
    pub const T5: Reg = Reg(30);
    pub const T6: Reg = Reg(31);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AluOp {
    Add,
    Sub,
    Sll,
    Srl,
    Sra,
    And,
    Or,
    Xor,
    Slt,
    Sltu,
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImmOp {
    Addi,
    Slti,
    Sltiu,
    Xori,
    Ori,
    Andi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShiftOp {
    Slli,
    Srli,
    Srai,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Width {
    B,
    H,
    W,
    D,
}

impl Width {
    pub fn bytes(self) -> u64 {
        match self {
            Self::B => 1,
            Self::H => 2,
            Self::W => 4,
            Self::D => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BranchOp {
    Beq,
    Bne,
    Blt,
    Bge,
    Bltu,
    Bgeu,
}

impl BranchOp {
    pub fn invert(self) -> Self {
        match self {
            Self::Beq => Self::Bne,
            Self::Bne => Self::Beq,
            Self::Blt => Self::Bge,
            Self::Bge => Self::Blt,
            Self::Bltu => Self::Bgeu,
            Self::Bgeu => Self::Bltu,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    Alu {
        op: AluOp,
        rd: Reg,
        rs1: Reg,
        rs2: Reg,
    },
    Imm {
        op: ImmOp,
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    Shift {
        op: ShiftOp,
        rd: Reg,
        rs1: Reg,
        shamt: u32,
    },
    Addiw {
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    /// `rd := sext(imm << 12)`, `imm` a 20-bit value.
    Lui {
        rd: Reg,
        imm: u32,
    },
    Auipc {
        rd: Reg,
        imm: u32,
    },
    Load {
        width: Width,
        unsigned: bool,
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    Store {
        width: Width,
        rs1: Reg,
        rs2: Reg,
        imm: i32,
    },
    Branch {
        op: BranchOp,
        rs1: Reg,
        rs2: Reg,
        offset: i32,
    },
    Jal {
        rd: Reg,
        offset: i32,
    },
    Jalr {
        rd: Reg,
        rs1: Reg,
        imm: i32,
    },
    Ecall,
    Ebreak,
    Fence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    #[error("immediate {0} out of range")]
    Immediate(i64),
}

fn check(imm: i64, bits: u32, multiple: i64) -> Result<(), EncodeError> {
    let half = 1i64 << (bits - 1);
    if imm < -half || imm >= half || imm % multiple != 0 {
        return Err(EncodeError::Immediate(imm));
    }
    Ok(())
}

fn r(op: u32, rd: Reg, f3: u32, rs1: Reg, rs2: Reg, f7: u32) -> u32 {
    op | (u32::from(rd.0) << 7) |
        (f3 << 12) |
        (u32::from(rs1.0) << 15) |
        (u32::from(rs2.0) << 20) |
        (f7 << 25)
}

fn i(op: u32, rd: Reg, f3: u32, rs1: Reg, imm: i32) -> u32 {
    op | (u32::from(rd.0) << 7) |
        (f3 << 12) |
        (u32::from(rs1.0) << 15) |
        (((imm as u32) & 0xfff) << 20)
}

impl Instr {
    pub fn encode(self) -> Result<u32, EncodeError> {
        Ok(match self {
            Self::Alu { op, rd, rs1, rs2 } => {
                let (f3, f7) = match op {
                    AluOp::Add => (0, 0),
                    AluOp::Sub => (0, 0x20),
                    AluOp::Sll => (1, 0),
                    AluOp::Slt => (2, 0),
                    AluOp::Sltu => (3, 0),
                    AluOp::Xor => (4, 0),
                    AluOp::Srl => (5, 0),
                    AluOp::Sra => (5, 0x20),
                    AluOp::Or => (6, 0),
                    AluOp::And => (7, 0),
                    AluOp::Mul => (0, 1),
                    AluOp::Mulh => (1, 1),
                    AluOp::Mulhsu => (2, 1),
                    AluOp::Mulhu => (3, 1),
                    AluOp::Div => (4, 1),
                    AluOp::Divu => (5, 1),
                    AluOp::Rem => (6, 1),
                    AluOp::Remu => (7, 1),
                };
                r(0x33, rd, f3, rs1, rs2, f7)
            }
            Self::Imm { op, rd, rs1, imm } => {
                check(imm.into(), 12, 1)?;
                let f3 = match op {
                    ImmOp::Addi => 0,
                    ImmOp::Slti => 2,
                    ImmOp::Sltiu => 3,
                    ImmOp::Xori => 4,
                    ImmOp::Ori => 6,
                    ImmOp::Andi => 7,
                };
                i(0x13, rd, f3, rs1, imm)
            }
            Self::Shift { op, rd, rs1, shamt } => {
                if shamt >= 64 {
                    return Err(EncodeError::Immediate(shamt.into()));
                }
                let (f3, high) = match op {
                    ShiftOp::Slli => (1, 0),
                    ShiftOp::Srli => (5, 0),
                    ShiftOp::Srai => (5, 0x10),
                };
                i(0x13, rd, f3, rs1, ((high << 6) | shamt) as i32)
            }
            Self::Addiw { rd, rs1, imm } => {
                check(imm.into(), 12, 1)?;
                i(0x1b, rd, 0, rs1, imm)
            }
            Self::Lui { rd, imm } | Self::Auipc { rd, imm } => {
                if imm >= 1 << 20 {
                    return Err(EncodeError::Immediate(imm.into()));
                }
                let op = if matches!(self, Self::Lui { .. }) { 0x37 } else { 0x17 };
                op | (u32::from(rd.0) << 7) | (imm << 12)
            }
            Self::Load { width, unsigned, rd, rs1, imm } => {
                check(imm.into(), 12, 1)?;
                let f3 = match (width, unsigned) {
                    (Width::B, false) => 0,
                    (Width::H, false) => 1,
                    (Width::W, false) => 2,
                    (Width::D, false) => 3,
                    (Width::B, true) => 4,
                    (Width::H, true) => 5,
                    (Width::W, true) => 6,
                    (Width::D, true) => return Err(EncodeError::Immediate(0)),
                };
                i(0x03, rd, f3, rs1, imm)
            }
            Self::Store { width, rs1, rs2, imm } => {
                check(imm.into(), 12, 1)?;
                let f3 = match width {
                    Width::B => 0,
                    Width::H => 1,
                    Width::W => 2,
                    Width::D => 3,
                };
                let imm = imm as u32;
                0x23 | ((imm & 0x1f) << 7) |
                    (f3 << 12) |
                    (u32::from(rs1.0) << 15) |
                    (u32::from(rs2.0) << 20) |
                    (((imm >> 5) & 0x7f) << 25)
            }
            Self::Branch { op, rs1, rs2, offset } => {
                check(offset.into(), 13, 2)?;
                let f3 = match op {
                    BranchOp::Beq => 0,
                    BranchOp::Bne => 1,
                    BranchOp::Blt => 4,
                    BranchOp::Bge => 5,
                    BranchOp::Bltu => 6,
                    BranchOp::Bgeu => 7,
                };
                let o = offset as u32;
                0x63 | (((o >> 11) & 1) << 7) |
                    (((o >> 1) & 0xf) << 8) |
                    (f3 << 12) |
                    (u32::from(rs1.0) << 15) |
                    (u32::from(rs2.0) << 20) |
                    (((o >> 5) & 0x3f) << 25) |
                    (((o >> 12) & 1) << 31)
            }
            Self::Jal { rd, offset } => {
                check(offset.into(), 21, 2)?;
                let o = offset as u32;
                0x6f | (u32::from(rd.0) << 7) |
                    (((o >> 12) & 0xff) << 12) |
                    (((o >> 11) & 1) << 20) |
                    (((o >> 1) & 0x3ff) << 21) |
                    (((o >> 20) & 1) << 31)
            }
            Self::Jalr { rd, rs1, imm } => {
                check(imm.into(), 12, 1)?;
                i(0x67, rd, 0, rs1, imm)
            }
            Self::Ecall => 0x0000_0073,
            Self::Ebreak => 0x0010_0073,
            Self::Fence => 0x0ff0_000f,
        })
    }

    pub fn decode(word: u32) -> Option<Self> {
        let rd = Reg(((word >> 7) & 0x1f) as u8);
        let rs1 = Reg(((word >> 15) & 0x1f) as u8);
        let rs2 = Reg(((word >> 20) & 0x1f) as u8);
        let f3 = (word >> 12) & 7;
        let f7 = word >> 25;
        let imm_i = (word as i32) >> 20;
        Some(match word & 0x7f {
            0x33 => {
                let op = match (f7, f3) {
                    (0, 0) => AluOp::Add,
                    (0x20, 0) => AluOp::Sub,
                    (0, 1) => AluOp::Sll,
                    (0, 2) => AluOp::Slt,
                    (0, 3) => AluOp::Sltu,
                    (0, 4) => AluOp::Xor,
                    (0, 5) => AluOp::Srl,
                    (0x20, 5) => AluOp::Sra,
                    (0, 6) => AluOp::Or,
                    (0, 7) => AluOp::And,
                    (1, 0) => AluOp::Mul,
                    (1, 1) => AluOp::Mulh,
                    (1, 2) => AluOp::Mulhsu,
                    (1, 3) => AluOp::Mulhu,
                    (1, 4) => AluOp::Div,
                    (1, 5) => AluOp::Divu,
                    (1, 6) => AluOp::Rem,
                    (1, 7) => AluOp::Remu,
                    _ => return None,
                };
                Self::Alu { op, rd, rs1, rs2 }
            }
            0x13 => match f3 {
                1 | 5 => {
                    let shamt = (word >> 20) & 0x3f;
                    let op = match (f3, word >> 26) {
                        (1, 0) => ShiftOp::Slli,
                        (5, 0) => ShiftOp::Srli,
                        (5, 0x10) => ShiftOp::Srai,
                        _ => return None,
                    };
                    Self::Shift { op, rd, rs1, shamt }
                }
                _ => {
                    let op = match f3 {
                        0 => ImmOp::Addi,
                        2 => ImmOp::Slti,
                        3 => ImmOp::Sltiu,
                        4 => ImmOp::Xori,
                        6 => ImmOp::Ori,
                        _ => ImmOp::Andi,
                    };
                    Self::Imm { op, rd, rs1, imm: imm_i }
                }
            },
            0x1b if f3 == 0 => Self::Addiw { rd, rs1, imm: imm_i },
            0x37 => Self::Lui { rd, imm: word >> 12 },
            0x17 => Self::Auipc { rd, imm: word >> 12 },
            0x03 => {
                let (width, unsigned) = match f3 {
                    0 => (Width::B, false),
                    1 => (Width::H, false),
                    2 => (Width::W, false),
                    3 => (Width::D, false),
                    4 => (Width::B, true),
                    5 => (Width::H, true),
                    6 => (Width::W, true),
                    _ => return None,
                };
                Self::Load { width, unsigned, rd, rs1, imm: imm_i }
            }
            0x23 => {
                let width = match f3 {
                    0 => Width::B,
                    1 => Width::H,
                    2 => Width::W,
                    3 => Width::D,
                    _ => return None,
                };
                let imm = ((word as i32 >> 25) << 5) | ((word >> 7) & 0x1f) as i32;
                Self::Store { width, rs1, rs2, imm }
            }
            0x63 => {
                let op = match f3 {
                    0 => BranchOp::Beq,
                    1 => BranchOp::Bne,
                    4 => BranchOp::Blt,
                    5 => BranchOp::Bge,
                    6 => BranchOp::Bltu,
                    7 => BranchOp::Bgeu,
                    _ => return None,
                };
                let offset = (((word as i32) >> 31) << 12) |
                    ((((word >> 7) & 1) << 11) as i32) |
                    ((((word >> 25) & 0x3f) << 5) as i32) |
                    ((((word >> 8) & 0xf) << 1) as i32);
                Self::Branch { op, rs1, rs2, offset }
            }
            0x6f => {
                let offset = (((word as i32) >> 31) << 20) |
                    ((((word >> 12) & 0xff) << 12) as i32) |
                    ((((word >> 20) & 1) << 11) as i32) |
                    ((((word >> 21) & 0x3ff) << 1) as i32);
                Self::Jal { rd, offset }
            }
            0x67 if f3 == 0 => Self::Jalr { rd, rs1, imm: imm_i },
            0x73 if word == 0x0000_0073 => Self::Ecall,
            0x73 if word == 0x0010_0073 => Self::Ebreak,
            0x0f if f3 == 0 => Self::Fence,
            _ => return None,
        })
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_instr(rng: &mut impl Rng) -> Instr {
        let reg = |rng: &mut dyn rand::RngCore| Reg(rng.random_range(0..32));
        match rng.random_range(0..12) {
            0 => Instr::Alu {
                op: [AluOp::Add, AluOp::Sub, AluOp::Sra, AluOp::Mulhu, AluOp::Remu, AluOp::Sltu]
                    [rng.random_range(0..6)],
                rd: reg(rng),
                rs1: reg(rng),
                rs2: reg(rng),
            },
            1 => Instr::Imm {
                op: ImmOp::Addi,
                rd: reg(rng),
                rs1: reg(rng),
                imm: rng.random_range(-2048..2048),
            },
            2 => Instr::Shift {
                op: [ShiftOp::Slli, ShiftOp::Srli, ShiftOp::Srai][rng.random_range(0..3)],
                rd: reg(rng),
                rs1: reg(rng),
                shamt: rng.random_range(0..64),
            },
            3 => Instr::Addiw { rd: reg(rng), rs1: reg(rng), imm: rng.random_range(-2048..2048) },
            4 => Instr::Lui { rd: reg(rng), imm: rng.random_range(0..1 << 20) },
            5 => Instr::Load {
                width: Width::D,
                unsigned: false,
                rd: reg(rng),
                rs1: reg(rng),
                imm: rng.random_range(-2048..2048),
            },
            6 => Instr::Store {
                width: Width::W,
                rs1: reg(rng),
                rs2: reg(rng),
                imm: rng.random_range(-2048..2048),
            },
            7 => Instr::Branch {
                op: BranchOp::Bgeu,
                rs1: reg(rng),
                rs2: reg(rng),
                offset: rng.random_range(-2048..2048) * 2,
            },
            8 => Instr::Jal { rd: reg(rng), offset: rng.random_range(-(1 << 19)..(1 << 19)) * 2 },
            9 => Instr::Jalr { rd: reg(rng), rs1: reg(rng), imm: rng.random_range(-2048..2048) },
            10 => Instr::Auipc { rd: reg(rng), imm: rng.random_range(0..1 << 20) },
            _ => Instr::Ecall,
        }
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let mut rng = rand::rng();
        for _ in 0..10_000 {
            let instr = random_instr(&mut rng);
            let word = instr.encode().unwrap();
            assert_eq!(Instr::decode(word), Some(instr), "{word:#010x}");
        }
    }

    #[test]
    fn test_word_operations_other_than_addiw_do_not_decode() {
        // addw a0, a1, a0 (as rustc emits for u32 addition) and slliw.
        assert_eq!(Instr::decode(0x00a5_853b), None);
        assert_eq!(Instr::decode(0x0015_951b), None);
        // A compressed encoding.
        assert_eq!(Instr::decode(0x0000_8082), None);
    }
}
