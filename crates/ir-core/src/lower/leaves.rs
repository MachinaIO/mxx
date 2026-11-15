//! Leaves of the lattice DSL: elementwise arithmetic on one limb.
//!
//! Each leaf takes the arguments `n` (the number of residues) and `q` (the limb's modulus,
//! `q < 2^63`), reads canonical residues in `0..q`, and writes canonical residues. Its RV64IM code
//! follows the core's calling convention: the descriptor at `a0` holds `n`, `q`, the input
//! addresses, the output address, and the scratch address.

use super::rv64_leaves;
use mxx_core::{
    expr::IntExpr,
    leaf::{Builtin, HostCall, Leaf, LeafSignature, Rv64Code, Rv64Library},
    rv64::{
        asm::Asm,
        isa::{AluOp, BranchOp, Instr, Reg, ShiftOp},
    },
    types::CoreType,
};
use std::sync::{Arc, OnceLock};

/// The registers holding the current input addresses, in operand order.
const INPUTS: [Reg; 2] = [Reg::T2, Reg::A1];

/// A leaf applying `body` to each residue. `body` reads the operands from `a3` (and `a4`) and the
/// modulus from `t1`, and leaves the result in `a5`.
fn elementwise(
    name: &str,
    arity: usize,
    body: impl Fn(&mut Asm),
    host: impl Fn(&[u64], u64) -> u64 + Send + Sync + 'static,
) -> Arc<Leaf> {
    let mut asm = Asm::new();
    let (top, done) = (asm.label(), asm.label());
    // Prologue: n, q, input addresses, output address.
    asm.ld(Reg::T0, Reg::A0, 0);
    asm.ld(Reg::T1, Reg::A0, 8);
    for (k, reg) in INPUTS[..arity].iter().enumerate() {
        asm.ld(*reg, Reg::A0, 16 + 8 * k as i32);
    }
    asm.ld(Reg::A2, Reg::A0, 16 + 8 * arity as i32);
    let prologue = asm.here() as i64;
    asm.place(top);
    asm.branch(BranchOp::Beq, Reg::T0, Reg::ZERO, done);
    for (reg, operand) in INPUTS[..arity].iter().zip([Reg::A3, Reg::A4]) {
        asm.ld(operand, *reg, 0);
    }
    body(&mut asm);
    asm.sd(Reg::A5, Reg::A2, 0);
    for reg in &INPUTS[..arity] {
        asm.addi(*reg, *reg, 8);
    }
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::T0, Reg::T0, -1);
    asm.jump(top);
    let per_residue = asm.here() as i64 - prologue;
    asm.place(done);
    asm.emit(Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
    let words = asm.finish().expect("leaf code assembles");
    let limb = CoreType::buffer(8, "n");
    Arc::new(Leaf {
        signature: LeafSignature {
            name: name.into(),
            arguments: vec!["n".into(), "q".into()],
            inputs: vec![limb.clone(); arity],
            outputs: vec![limb],
            scratch_bytes: 0.into(),
            in_place: Vec::new(),
            associative: false,
        },
        // Every residue runs the loop body once; the final check takes two steps and `ret` one.
        rv64: Some(Rv64Code::standalone(
            name,
            words,
            IntExpr::from(prologue + 3) + IntExpr::var("n") * per_residue,
        )),
        host: Some(Arc::new(move |call: HostCall<'_>| {
            let q = call.arguments[1] as u64;
            let word = |bytes: &[u8], j: usize| {
                u64::from_le_bytes(bytes[8 * j..8 * j + 8].try_into().unwrap())
            };
            for (j, out) in call.outputs[0].chunks_mut(8).enumerate() {
                let operands: Vec<u64> = call.inputs.iter().map(|input| word(input, j)).collect();
                out.copy_from_slice(&host(&operands, q).to_le_bytes());
            }
        })),
    })
}

/// `(a + b) mod q`.
pub fn add_mod() -> Arc<Leaf> {
    elementwise(
        "lattice.add_mod",
        2,
        |asm| {
            let skip = asm.label();
            asm.alu(AluOp::Add, Reg::A5, Reg::A3, Reg::A4);
            asm.branch(BranchOp::Bltu, Reg::A5, Reg::T1, skip);
            asm.alu(AluOp::Sub, Reg::A5, Reg::A5, Reg::T1);
            asm.place(skip);
        },
        |x, q| {
            let s = x[0] + x[1];
            if s >= q { s - q } else { s }
        },
    )
}

/// `(a - b) mod q`.
pub fn sub_mod() -> Arc<Leaf> {
    elementwise(
        "lattice.sub_mod",
        2,
        |asm| {
            let skip = asm.label();
            asm.alu(AluOp::Sub, Reg::A5, Reg::A3, Reg::A4);
            asm.branch(BranchOp::Bgeu, Reg::A3, Reg::A4, skip);
            asm.alu(AluOp::Add, Reg::A5, Reg::A5, Reg::T1);
            asm.place(skip);
        },
        |x, q| if x[0] >= x[1] { x[0] - x[1] } else { x[0] + q - x[1] },
    )
}

/// `(-a) mod q`.
pub fn neg_mod() -> Arc<Leaf> {
    elementwise(
        "lattice.neg_mod",
        1,
        |asm| {
            let skip = asm.label();
            asm.alu(AluOp::Sub, Reg::A5, Reg::T1, Reg::A3);
            asm.branch(BranchOp::Bne, Reg::A3, Reg::ZERO, skip);
            asm.mv(Reg::A5, Reg::ZERO);
            asm.place(skip);
        },
        |x, q| if x[0] == 0 { 0 } else { q - x[0] },
    )
}

/// A leaf writing one `W`-word integer (`W >= 1`, an argument): word 0 is `first`, the others
/// `rest`, each loaded or computed by `load` into `a5` (word 0) and then turned into the others'
/// word by `extend`. The descriptor's words after `W` are read by `load`; the output address is
/// at `out_slot`.
fn word_fill(
    name: &str,
    arguments: &[&str],
    inputs: Vec<CoreType>,
    output: CoreType,
    out_slot: i32,
    load: impl Fn(&mut Asm),
    extend: impl Fn(&mut Asm),
    host: impl Fn(&HostCall<'_>) -> (u64, u64) + Send + Sync + 'static,
) -> Arc<Leaf> {
    let mut asm = Asm::new();
    let (top, done) = (asm.label(), asm.label());
    asm.ld(Reg::T0, Reg::A0, 0);
    load(&mut asm);
    asm.ld(Reg::A2, Reg::A0, out_slot);
    asm.sd(Reg::A5, Reg::A2, 0);
    extend(&mut asm);
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::T0, Reg::T0, -1);
    let prologue = asm.here() as i64;
    asm.place(top);
    asm.branch(BranchOp::Beq, Reg::T0, Reg::ZERO, done);
    asm.sd(Reg::A5, Reg::A2, 0);
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::T0, Reg::T0, -1);
    asm.jump(top);
    let per_word = asm.here() as i64 - prologue;
    asm.place(done);
    asm.emit(Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
    let words = asm.finish().expect("leaf code assembles");
    Arc::new(Leaf {
        signature: LeafSignature {
            name: name.into(),
            arguments: arguments.iter().map(|a| (*a).to_owned()).collect(),
            inputs,
            outputs: vec![output],
            scratch_bytes: 0.into(),
            in_place: Vec::new(),
            associative: false,
        },
        // Word 0, then `W - 1` iterations (the head's branch skips its jump to the exit), and
        // the exit: the branch, the jump, and `ret`.
        rv64: Some(Rv64Code::standalone(
            name,
            words,
            IntExpr::from(prologue + 3 - (per_word - 1)) + IntExpr::var("W") * (per_word - 1),
        )),
        host: Some(Arc::new(move |call: HostCall<'_>| {
            let (first, rest) = host(&call);
            for (k, out) in call.outputs[0].chunks_mut(8).enumerate() {
                out.copy_from_slice(&if k == 0 { first } else { rest }.to_le_bytes());
            }
        })),
    })
}

/// The word of a Bool as a `W`-word Int: the word, then zeros. Descriptor: `W, a, out, scratch`.
pub fn bool_to_int(words: u32) -> Arc<Leaf> {
    word_fill(
        "lattice.bool_to_int",
        &["W"],
        vec![CoreType::Bool],
        int(words),
        16,
        |asm| {
            asm.ld(Reg::T1, Reg::A0, 8);
            asm.ld(Reg::A5, Reg::T1, 0);
        },
        |asm| asm.mv(Reg::A5, Reg::ZERO),
        |call| (u64::from_le_bytes(call.inputs[0][..8].try_into().unwrap()), 0),
    )
}

/// A `W`-word integer constant: its one-word argument `value`, sign-extended. Descriptor: `W,
/// value, out, scratch`.
pub fn int_constant(words: u32) -> Arc<Leaf> {
    word_fill(
        "lattice.int_constant",
        &["W", "value"],
        Vec::new(),
        int(words),
        16,
        |asm| asm.ld(Reg::A5, Reg::A0, 8),
        |asm| asm.shift(ShiftOp::Srai, Reg::A5, Reg::A5, 63),
        |call| {
            let value = call.arguments[1] as u64;
            (value, ((value as i64) >> 63) as u64)
        },
    )
}

/// Emits `a5 = a3 a4 mod q` inline (`q` in `t1`, `a3, a4 < q < 2^63`; 15 instructions, falling
/// through): the high word of the product (below `q`) starts 64 steps of binary long division,
/// each doubling the remainder, adding the next bit of the low word, and subtracting `q` when the
/// remainder reaches it. It clobbers `t3`, `t4`, and `t5`; `MxxLatticeLeaves.MulMod` specifies it
/// wherever a leaf places it.
fn emit_mulmod(asm: &mut Asm) {
    let (top, skip, end) = (asm.label(), asm.label(), asm.label());
    asm.alu(AluOp::Mulhu, Reg(28), Reg::A3, Reg::A4);
    asm.alu(AluOp::Mul, Reg(29), Reg::A3, Reg::A4);
    asm.addi(Reg(30), Reg::ZERO, 64);
    asm.place(top);
    asm.branch(BranchOp::Beq, Reg(30), Reg::ZERO, end);
    asm.shift(ShiftOp::Srli, Reg::A5, Reg(29), 63);
    asm.shift(ShiftOp::Slli, Reg(29), Reg(29), 1);
    asm.alu(AluOp::Add, Reg(28), Reg(28), Reg(28));
    asm.alu(AluOp::Add, Reg(28), Reg(28), Reg::A5);
    asm.branch(BranchOp::Bltu, Reg(28), Reg::T1, skip);
    asm.alu(AluOp::Sub, Reg(28), Reg(28), Reg::T1);
    asm.place(skip);
    asm.addi(Reg(30), Reg(30), -1);
    asm.jump(top);
    asm.place(end);
    asm.mv(Reg::A5, Reg(28));
}

/// The steps of one `emit_mulmod`: the products and the counter (3), 64 division steps of at most
/// 9, the exit (2), and the move (1).
const MULMOD_STEPS: i128 = 3 + 64 * 9 + 2 + 1;

/// `acc + a b` for polynomials of `Z_q[X]/(X^N + 1)` (`q < 2^63`, canonical residues), one output
/// coefficient at a time: coefficient `k` adds `a[i] b[k - i]` for `i <= k` and subtracts
/// `a[i] b[N + k - i]` for `i > k`, each product by the inline code of [`emit_mulmod`].
/// Descriptor: `N, q, acc, a, b, out, scratch`.
pub fn poly_mul_acc() -> Arc<Leaf> {
    let mut asm = Asm::new();
    let (otop, done, itop, store, wrap, product, add_done) =
        (asm.label(), asm.label(), asm.label(), asm.label(), asm.label(), asm.label(), asm.label());
    let (i, wrapped) = (Reg(3), Reg(4));
    asm.ld(Reg::T0, Reg::A0, 0);
    asm.ld(Reg::T1, Reg::A0, 8);
    asm.ld(Reg::A1, Reg::A0, 16);
    asm.ld(Reg::A6, Reg::A0, 24);
    asm.ld(Reg::A7, Reg::A0, 32);
    asm.ld(Reg::A2, Reg::A0, 40);
    asm.mv(Reg::T2, Reg::ZERO);
    // Coefficient k = t2: the sum in a0, from acc[k].
    asm.place(otop);
    asm.branch(BranchOp::Beq, Reg::T2, Reg::T0, done);
    asm.ld(Reg::A0, Reg::A1, 0);
    asm.mv(i, Reg::ZERO);
    asm.place(itop);
    asm.branch(BranchOp::Beq, i, Reg::T0, store);
    // a3 = a[i].
    asm.shift(ShiftOp::Slli, Reg::A3, i, 3);
    asm.alu(AluOp::Add, Reg::A3, Reg::A6, Reg::A3);
    asm.ld(Reg::A3, Reg::A3, 0);
    // j = k - i, or N + k - i when i > k (then the product is subtracted).
    asm.alu(AluOp::Sltu, wrapped, Reg::T2, i);
    asm.alu(AluOp::Sub, Reg::A4, Reg::T2, i);
    asm.branch(BranchOp::Beq, wrapped, Reg::ZERO, wrap);
    asm.alu(AluOp::Add, Reg::A4, Reg::A4, Reg::T0);
    asm.place(wrap);
    asm.shift(ShiftOp::Slli, Reg::A4, Reg::A4, 3);
    asm.alu(AluOp::Add, Reg::A4, Reg::A7, Reg::A4);
    asm.ld(Reg::A4, Reg::A4, 0);
    emit_mulmod(&mut asm);
    asm.branch(BranchOp::Bne, wrapped, Reg::ZERO, product);
    // sum += product.
    asm.alu(AluOp::Add, Reg::A0, Reg::A0, Reg::A5);
    asm.branch(BranchOp::Bltu, Reg::A0, Reg::T1, add_done);
    asm.alu(AluOp::Sub, Reg::A0, Reg::A0, Reg::T1);
    asm.jump(add_done);
    // sum -= product, adding q first when the product is larger.
    let no_borrow = asm.label();
    asm.place(product);
    asm.branch(BranchOp::Bgeu, Reg::A0, Reg::A5, no_borrow);
    asm.alu(AluOp::Add, Reg::A0, Reg::A0, Reg::T1);
    asm.place(no_borrow);
    asm.alu(AluOp::Sub, Reg::A0, Reg::A0, Reg::A5);
    asm.place(add_done);
    asm.addi(i, i, 1);
    asm.jump(itop);
    asm.place(store);
    asm.sd(Reg::A0, Reg::A2, 0);
    asm.addi(Reg::A1, Reg::A1, 8);
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::T2, Reg::T2, 1);
    asm.jump(otop);
    asm.place(done);
    asm.emit(Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
    let code = asm.finish().expect("leaf code assembles");
    let limb = CoreType::buffer(8, "N");
    Arc::new(Leaf {
        signature: LeafSignature {
            name: "lattice.poly_mul_acc".into(),
            arguments: vec!["N".into(), "q".into()],
            inputs: vec![limb.clone(), limb.clone(), limb.clone()],
            outputs: vec![limb],
            scratch_bytes: 0.into(),
            in_place: Vec::new(),
            associative: false,
        },
        // The setup (7), per coefficient its head and load (4) and exit with the store (7), per
        // term at most 22 and a product, and the exit with `ret` (3).
        rv64: Some(Rv64Code::standalone(
            "lattice.poly_mul_acc",
            code,
            IntExpr::from(10) +
                IntExpr::var("N") * (IntExpr::from(11) + IntExpr::var("N") * (22 + MULMOD_STEPS)),
        )),
        host: Some(Arc::new(|call: HostCall<'_>| {
            let (n, q) = (call.arguments[0] as usize, call.arguments[1] as u64);
            let word = |bytes: &[u8], j: usize| {
                u64::from_le_bytes(bytes[8 * j..8 * j + 8].try_into().unwrap())
            };
            for (k, out) in call.outputs[0].chunks_mut(8).enumerate() {
                let mut sum = u128::from(word(call.inputs[0], k));
                let q128 = u128::from(q);
                for i in 0..n {
                    let a = u128::from(word(call.inputs[1], i));
                    let (j, wrap) = if i <= k { (k - i, false) } else { (n + k - i, true) };
                    let p = a * u128::from(word(call.inputs[2], j)) % q128;
                    sum = if wrap { (sum + q128 - p) % q128 } else { (sum + p) % q128 };
                }
                out.copy_from_slice(&(sum as u64).to_le_bytes());
            }
        })),
    })
}

/// The code of `mxx-lattice-leaves` for the mathcal-A machine, built by
/// `scripts/build_rv64_leaves.sh`.
fn library() -> Arc<Rv64Library> {
    static LIBRARY: OnceLock<Arc<Rv64Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            Arc::new(Rv64Library {
                name: "mxx-lattice-leaves".into(),
                words: rv64_leaves::WORDS.to_vec(),
                rodata: rv64_leaves::RODATA.to_vec(),
            })
        })
        .clone()
}

/// A leaf of `mxx-lattice-leaves`: its host kernel calls `function` with a descriptor of host
/// addresses, on word-aligned copies of the inputs.
fn compiled(
    signature: LeafSignature,
    symbol: &str,
    steps: IntExpr,
    function: unsafe extern "C" fn(*const u64),
) -> Arc<Leaf> {
    let entry = rv64_leaves::ENTRIES
        .iter()
        .find(|(name, _)| *name == symbol)
        .unwrap_or_else(|| {
            panic!("`{symbol}` is not in the RV64IM build; run scripts/build_rv64_leaves.sh")
        })
        .1;
    let words = |bytes: &[u8]| -> Vec<u64> {
        bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("whole words"))).collect()
    };
    Arc::new(Leaf {
        signature,
        rv64: Some(Rv64Code { library: library(), entry, steps }),
        host: Some(Arc::new(move |call: HostCall<'_>| {
            let inputs: Vec<Vec<u64>> = call.inputs.iter().map(|input| words(input)).collect();
            let mut outputs: Vec<Vec<u64>> =
                call.outputs.iter().map(|o| vec![0u64; o.len() / 8]).collect();
            let mut scratch = vec![0u64; call.scratch.len().div_ceil(8)];
            let mut descriptor: Vec<u64> = call.arguments.iter().map(|a| *a as u64).collect();
            descriptor.extend(inputs.iter().map(|i| i.as_ptr() as u64));
            descriptor.extend(outputs.iter_mut().map(|o| o.as_mut_ptr() as u64));
            descriptor.push(scratch.as_mut_ptr() as u64);
            // SAFETY: the descriptor follows the leaf's layout over buffers of the declared sizes,
            // which validation checked against the signature.
            unsafe { function(descriptor.as_ptr()) };
            for (out, words) in call.outputs.iter_mut().zip(&outputs) {
                for (chunk, word) in out.chunks_mut(8).zip(words) {
                    chunk.copy_from_slice(&word.to_le_bytes());
                }
            }
        })),
    })
}

/// `c * a mod q` for the integer `c < q` on `n` residues, by Shoup's multiplication: the leaf first
/// computes `c' = floor(c 2^64 / q)` by 64 steps of binary long division, then per residue
/// `t = floor(a c' / 2^64)` (`mulhu`) and `c a - t q`, which lies in `[0, 2 q)`, less `q` when it
/// is at least `q`. Descriptor: `n, q, c, a, out, scratch`, with `q < 2^63` and `c, a[i] < q`.
pub fn scale_mod() -> Arc<Leaf> {
    let mut asm = Asm::new();
    let (dtop, ddone, skip, etop, ok, done) =
        (asm.label(), asm.label(), asm.label(), asm.label(), asm.label(), asm.label());
    asm.ld(Reg::T0, Reg::A0, 0);
    asm.ld(Reg::T1, Reg::A0, 8);
    asm.ld(Reg::A3, Reg::A0, 16);
    asm.ld(Reg::T2, Reg::A0, 24);
    asm.ld(Reg::A2, Reg::A0, 32);
    asm.mv(Reg::A4, Reg::ZERO);
    asm.mv(Reg::A5, Reg::A3);
    asm.addi(Reg::A6, Reg::ZERO, 64);
    // c' = floor(c 2^64 / q): the remainder `a5` doubles, and the quotient `a4` takes a bit.
    asm.place(dtop);
    asm.branch(BranchOp::Beq, Reg::A6, Reg::ZERO, ddone);
    asm.alu(AluOp::Add, Reg::A5, Reg::A5, Reg::A5);
    asm.alu(AluOp::Add, Reg::A4, Reg::A4, Reg::A4);
    asm.branch(BranchOp::Bltu, Reg::A5, Reg::T1, skip);
    asm.alu(AluOp::Sub, Reg::A5, Reg::A5, Reg::T1);
    asm.addi(Reg::A4, Reg::A4, 1);
    asm.place(skip);
    asm.addi(Reg::A6, Reg::A6, -1);
    asm.jump(dtop);
    asm.place(ddone);
    // Per residue.
    asm.place(etop);
    asm.branch(BranchOp::Beq, Reg::T0, Reg::ZERO, done);
    asm.ld(Reg::A7, Reg::T2, 0);
    asm.alu(AluOp::Mulhu, Reg::A5, Reg::A7, Reg::A4);
    asm.alu(AluOp::Mul, Reg::A5, Reg::A5, Reg::T1);
    asm.alu(AluOp::Mul, Reg::A6, Reg::A7, Reg::A3);
    asm.alu(AluOp::Sub, Reg::A5, Reg::A6, Reg::A5);
    asm.branch(BranchOp::Bltu, Reg::A5, Reg::T1, ok);
    asm.alu(AluOp::Sub, Reg::A5, Reg::A5, Reg::T1);
    asm.place(ok);
    asm.sd(Reg::A5, Reg::A2, 0);
    asm.addi(Reg::T2, Reg::T2, 8);
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::T0, Reg::T0, -1);
    asm.jump(etop);
    asm.place(done);
    asm.emit(Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
    let words = asm.finish().expect("leaf code assembles");
    let limb = CoreType::buffer(8, "n");
    Arc::new(Leaf {
        signature: LeafSignature {
            name: "lattice.scale_mod".into(),
            arguments: vec!["n".into(), "q".into(), "c".into()],
            inputs: vec![limb.clone()],
            outputs: vec![limb],
            scratch_bytes: 0.into(),
            in_place: Vec::new(),
            associative: false,
        },
        // The setup (8), 64 division steps of at most 8, the division's exit (2), at most 14 per
        // residue, and the exit with `ret` (3).
        rv64: Some(Rv64Code::standalone(
            "lattice.scale_mod",
            words,
            IntExpr::from(8 + 64 * 8 + 2 + 3) + IntExpr::var("n") * 14,
        )),
        host: Some(Arc::new(|call: HostCall<'_>| {
            let (q, c) = (call.arguments[1] as u64, call.arguments[2] as u64);
            let word = |bytes: &[u8], j: usize| {
                u64::from_le_bytes(bytes[8 * j..8 * j + 8].try_into().unwrap())
            };
            for (j, out) in call.outputs[0].chunks_mut(8).enumerate() {
                let a = word(call.inputs[0], j);
                let product = (u128::from(c) * u128::from(a) % u128::from(q)) as u64;
                out.copy_from_slice(&product.to_le_bytes());
            }
        })),
    })
}

fn signature(
    name: &str,
    arguments: &[&str],
    inputs: Vec<CoreType>,
    outputs: Vec<CoreType>,
) -> LeafSignature {
    LeafSignature {
        name: name.into(),
        arguments: arguments.iter().map(|a| (*a).to_owned()).collect(),
        inputs,
        outputs,
        scratch_bytes: 0.into(),
        in_place: Vec::new(),
        associative: false,
    }
}

/// The lowering's integers: two's complement of `words` words.
fn int(words: u32) -> CoreType {
    CoreType::Int { words }
}

/// The step bound of one integer operation on `W`-word integers: a division is the longest.
fn int_steps() -> IntExpr {
    IntExpr::from(512) + IntExpr::var("W") * IntExpr::var("W") * 64 * 400
}

/// `W`-word integer `+`, `-`, `*` (wrapping), and the quotient and remainder by `|b|` rounding
/// down (argument `op` 0 to 4).
pub fn int_binary(words: u32) -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.int_binary",
            &["W", "op"],
            vec![int(words), int(words)],
            vec![int(words)],
        ),
        "mxx_int_binary",
        int_steps(),
        mxx_lattice_leaves::mxx_int_binary,
    )
}

/// `W`-word integer `=`, `<`, `<=` (argument `op` 0, 1, 2).
pub fn int_compare(words: u32) -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.int_compare",
            &["W", "op"],
            vec![int(words), int(words)],
            vec![CoreType::Bool],
        ),
        "mxx_int_compare",
        int_steps(),
        mxx_lattice_leaves::mxx_int_compare,
    )
}

/// Bit `bit` of a `W`-word integer.
pub fn bit_extract(words: u32) -> Arc<Leaf> {
    compiled(
        signature("lattice.bit_extract", &["W", "bit"], vec![int(words)], vec![CoreType::Bool]),
        "mxx_bit_extract",
        int_steps(),
        mxx_lattice_leaves::mxx_bit_extract,
    )
}

/// The transpose of one limb of a `rows x columns` matrix. Its step bound is measured.
pub fn transpose() -> Arc<Leaf> {
    let limb =
        CoreType::buffer(8, IntExpr::var("rows") * IntExpr::var("columns") * IntExpr::var("N"));
    compiled(
        signature("lattice.transpose", &["N", "rows", "columns"], vec![limb.clone()], vec![limb]),
        "mxx_transpose",
        IntExpr::from(64) +
            IntExpr::var("rows") * IntExpr::var("columns") * (IntExpr::var("N") * 8 + 32),
        mxx_lattice_leaves::mxx_transpose,
    )
}

/// `X -> X^k` on each entry of one limb. Its step bound is measured.
pub fn automorphism() -> Arc<Leaf> {
    let limb = CoreType::buffer(8, IntExpr::var("entries") * IntExpr::var("N"));
    compiled(
        signature(
            "lattice.automorphism",
            &["N", "q", "k", "entries"],
            vec![limb.clone()],
            vec![limb],
        ),
        "mxx_automorphism",
        IntExpr::from(64) + IntExpr::var("entries") * (IntExpr::var("N") * 64 + 32),
        mxx_lattice_leaves::mxx_automorphism,
    )
}

/// Multiplication of each of `entries` entries of one limb by `X^k` for a runtime `W`-word `k`
/// (`N` a power of two), gathered: coefficient `p` of entry `e` is `A[e][t]` for
/// `t = (p - k) mod 2 N` below `N`, else `-A[e][t - N] mod q` (the low word of `k` suffices, as
/// `2 N` divides `2^64`). Descriptor: `N, q, entries, A, k, out, scratch`.
pub fn multiply_monomial(words: u32) -> Arc<Leaf> {
    let mut asm = Asm::new();
    let (etop, done, itop, inext, low, store, stored) =
        (asm.label(), asm.label(), asm.label(), asm.label(), asm.label(), asm.label(), asm.label());
    asm.ld(Reg::T0, Reg::A0, 0);
    asm.ld(Reg::T1, Reg::A0, 8);
    asm.ld(Reg::T2, Reg::A0, 16);
    asm.ld(Reg::A1, Reg::A0, 24);
    asm.ld(Reg::A3, Reg::A0, 32);
    asm.ld(Reg::A3, Reg::A3, 0);
    asm.ld(Reg::A2, Reg::A0, 40);
    // mask = 2 N - 1, a3 = -k mod 2 N, t3 = 8 N.
    asm.alu(AluOp::Add, Reg::A4, Reg::T0, Reg::T0);
    asm.addi(Reg::A4, Reg::A4, -1);
    asm.alu(AluOp::Sub, Reg::A3, Reg::ZERO, Reg::A3);
    asm.alu(AluOp::And, Reg::A3, Reg::A3, Reg::A4);
    asm.shift(ShiftOp::Slli, Reg(28), Reg::T0, 3);
    asm.place(etop);
    asm.branch(BranchOp::Beq, Reg::T2, Reg::ZERO, done);
    asm.mv(Reg::A5, Reg::ZERO);
    asm.place(itop);
    asm.branch(BranchOp::Beq, Reg::A5, Reg::T0, inext);
    asm.alu(AluOp::Add, Reg::A7, Reg::A5, Reg::A3);
    asm.alu(AluOp::And, Reg::A7, Reg::A7, Reg::A4);
    asm.branch(BranchOp::Bltu, Reg::A7, Reg::T0, low);
    asm.alu(AluOp::Sub, Reg::A7, Reg::A7, Reg::T0);
    asm.shift(ShiftOp::Slli, Reg::A7, Reg::A7, 3);
    asm.alu(AluOp::Add, Reg::A7, Reg::A1, Reg::A7);
    asm.ld(Reg::A6, Reg::A7, 0);
    asm.branch(BranchOp::Beq, Reg::A6, Reg::ZERO, stored);
    asm.alu(AluOp::Sub, Reg::A6, Reg::T1, Reg::A6);
    asm.jump(stored);
    asm.place(low);
    asm.shift(ShiftOp::Slli, Reg::A7, Reg::A7, 3);
    asm.alu(AluOp::Add, Reg::A7, Reg::A1, Reg::A7);
    asm.ld(Reg::A6, Reg::A7, 0);
    asm.place(stored);
    asm.place(store);
    asm.sd(Reg::A6, Reg::A2, 0);
    asm.addi(Reg::A2, Reg::A2, 8);
    asm.addi(Reg::A5, Reg::A5, 1);
    asm.jump(itop);
    asm.place(inext);
    asm.alu(AluOp::Add, Reg::A1, Reg::A1, Reg(28));
    asm.addi(Reg::T2, Reg::T2, -1);
    asm.jump(etop);
    asm.place(done);
    asm.emit(Instr::Jalr { rd: Reg::ZERO, rs1: Reg::RA, imm: 0 });
    let code = asm.finish().expect("leaf code assembles");
    let limb = CoreType::buffer(8, IntExpr::var("entries") * IntExpr::var("N"));
    Arc::new(Leaf {
        signature: LeafSignature {
            name: "lattice.multiply_monomial".into(),
            arguments: vec!["N".into(), "q".into(), "entries".into()],
            inputs: vec![limb.clone(), int(words)],
            outputs: vec![limb],
            scratch_bytes: 0.into(),
            in_place: Vec::new(),
            associative: false,
        },
        // The setup (12), per entry its head (3), per coefficient at most 16, and the inner exit
        // with the pointer updates (5); the exit with `ret` (3).
        rv64: Some(Rv64Code::standalone(
            "lattice.multiply_monomial",
            code,
            IntExpr::from(15) + IntExpr::var("entries") * (IntExpr::var("N") * 16 + 8),
        )),
        host: Some(Arc::new(|call: HostCall<'_>| {
            let (n, q) = (call.arguments[0] as u64, call.arguments[1] as u64);
            let word = |bytes: &[u8], j: usize| {
                u64::from_le_bytes(bytes[8 * j..8 * j + 8].try_into().unwrap())
            };
            let negk = 0u64.wrapping_sub(word(call.inputs[1], 0)) & (2 * n - 1);
            for (j, out) in call.outputs[0].chunks_mut(8).enumerate() {
                let (e, p) = (j as u64 / n, j as u64 % n);
                let t = (p + negk) & (2 * n - 1);
                let v = if t < n {
                    word(call.inputs[0], (e * n + t) as usize)
                } else {
                    let v = word(call.inputs[0], (e * n + t - n) as usize);
                    if v == 0 { 0 } else { q - v }
                };
                out.copy_from_slice(&v.to_le_bytes());
            }
        })),
    })
}

/// `n` residues uniform modulo `q` from `16 n` bytes of coins.
pub fn uniform_mod() -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.uniform_mod",
            &["n", "q"],
            vec![CoreType::bytes(IntExpr::var("n") * 16)],
            vec![CoreType::buffer(8, "n")],
        ),
        "mxx_uniform_mod",
        IntExpr::from(64) + IntExpr::var("n") * 3200,
        mxx_lattice_leaves::mxx_uniform_mod,
    )
}

/// `n` integers uniform on `[minimum, minimum + width)` modulo `q`, from `16 n` bytes of coins
/// that every limb shares.
pub fn interval_mod() -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.interval_mod",
            &["n", "q", "minimum_mod_q", "width"],
            vec![CoreType::bytes(IntExpr::var("n") * 16)],
            vec![CoreType::buffer(8, "n")],
        ),
        "mxx_interval_mod",
        IntExpr::from(64) + IntExpr::var("n") * 6400,
        mxx_lattice_leaves::mxx_interval_mod,
    )
}

/// The constant of type `ty` whose bytes are `bytes`, as the core's built-in.
pub fn constant(ty: CoreType, bytes: Vec<u8>) -> Builtin {
    Builtin::Constant { ty, bytes }
}

/// `n` integers drawn by a cumulative distribution table of `entries` words, from `8 n` bytes of
/// coins every limb shares. Its step bound is measured.
pub fn cdt_mod() -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.cdt_mod",
            &["n", "q", "entries", "offset_mod_q"],
            vec![CoreType::bytes(IntExpr::var("n") * 8), CoreType::buffer(8, "entries")],
            vec![CoreType::buffer(8, "n")],
        ),
        "mxx_cdt_mod",
        IntExpr::from(64) + IntExpr::var("n") * (IntExpr::var("entries") * 8 + 64),
        mxx_lattice_leaves::mxx_cdt_mod,
    )
}

/// A matrix of residues uniform modulo `Q` from a 32-byte key and a tag, as the CPU backend's hash
/// sampler derives it. Its step bound is measured, assuming at most two candidates per
/// coefficient on average.
pub fn hash_uniform() -> Arc<Leaf> {
    let depth = IntExpr::var("L");
    compiled(
        signature(
            "lattice.hash_uniform",
            &[
                "N",
                "L",
                "rows",
                "columns",
                "bytes_per_coefficient",
                "high_mask",
                "limb_words",
                "tag_words",
                "modulus_words",
            ],
            vec![
                CoreType::bytes(32),
                CoreType::buffer(8, "tag_words"),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, "modulus_words"),
            ],
            vec![CoreType::family(CoreType::buffer(8, "limb_words"), depth.clone())],
        ),
        "mxx_hash_uniform",
        IntExpr::from(200_000) +
            IntExpr::var("rows") *
                IntExpr::var("columns") *
                IntExpr::var("N") *
                (depth * 4000 + 150_000),
        mxx_lattice_leaves::mxx_hash_uniform,
    )
}

/// `count` integers uniform on `[0, 2^b)` from a key and a tag, as `W`-word integers. Its step
/// bound is measured.
pub fn hash_ints(words: u32) -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.hash_ints",
            &["count", "bytes_per_coefficient", "high_mask", "tag_words", "W"],
            vec![CoreType::bytes(32), CoreType::buffer(8, "tag_words")],
            vec![CoreType::family(int(words), "count")],
        ),
        "mxx_hash_ints",
        IntExpr::from(200_000) +
            IntExpr::var("count") * (IntExpr::var("bytes_per_coefficient") * 6_000 + 150_000),
        mxx_lattice_leaves::mxx_hash_ints,
    )
}

/// A hash tag built at run time from a template and `count` `W`-word integer operands. Its step
/// bound is measured.
pub fn hash_tag(words: u32) -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.hash_tag",
            &["W", "template_words", "out_words", "count"],
            vec![CoreType::buffer(8, "template_words"), CoreType::family(int(words), "count")],
            vec![CoreType::buffer(8, "out_words")],
        ),
        "mxx_hash_tag",
        IntExpr::from(1024) +
            IntExpr::var("template_words") * 300 +
            IntExpr::var("count") * IntExpr::var("W") * 400_000,
        mxx_lattice_leaves::mxx_hash_tag,
    )
}

/// The balanced CRT gadget decompositions: with dropped moduli, or small per limb. Its step bound
/// is measured.
pub fn gadget_decompose() -> Arc<Leaf> {
    let depth = IntExpr::var("L");
    compiled(
        signature(
            "lattice.gadget_decompose",
            &[
                "N",
                "L",
                "rows",
                "columns",
                "base",
                "digits_per_tower",
                "limb_words_in",
                "limb_words_out",
                "retained",
                "small",
            ],
            vec![
                CoreType::family(CoreType::buffer(8, "limb_words_in"), depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, depth.clone() * depth.clone()),
            ],
            vec![CoreType::family(CoreType::buffer(8, "limb_words_out"), depth.clone())],
        ),
        "mxx_gadget_decompose",
        IntExpr::from(256) +
            IntExpr::var("limb_words_in") *
                depth.clone() *
                (depth.clone() * 3_000 + IntExpr::var("digits_per_tower") * (depth * 64 + 128)),
        mxx_lattice_leaves::mxx_gadget_decompose,
    )
}

/// One candidate preimage under a sampled trapdoor (computational or statistical) and whether it
/// is within the bound (`mxx_lattice_leaves::mxx_trapdoor_preimage`). Its step bound and scratch
/// are the arguments `steps` ([`trapdoor_preimage_steps`]) and `scratch_words`.
pub fn trapdoor_preimage() -> Arc<Leaf> {
    let depth = IntExpr::var("L");
    let (n, rows, digits, columns, secret) = (
        IntExpr::var("N"),
        IntExpr::var("rows"),
        IntExpr::var("digits"),
        IntExpr::var("columns"),
        IntExpr::var("secret_rows"),
    );
    let secret_columns = rows.clone() * digits;
    let public = secret.clone() + secret_columns.clone();
    let trapdoor_words = n.clone() * (rows.clone() * public.clone() + secret * secret_columns);
    let mut signature = signature(
        "lattice.trapdoor_preimage",
        &[
            "N",
            "L",
            "rows",
            "digits",
            "tower_digits",
            "columns",
            "secret_rows",
            "statistical",
            "base",
            "sigma",
            "width",
            "bound",
            "budget",
            "steps",
            "scratch_words",
        ],
        vec![
            CoreType::family(CoreType::buffer(8, trapdoor_words), depth.clone()),
            CoreType::family(
                CoreType::buffer(8, n.clone() * rows * columns.clone()),
                depth.clone(),
            ),
            CoreType::buffer(8, depth.clone()),
            CoreType::bytes(32),
        ],
        vec![CoreType::family(CoreType::buffer(8, n * public * columns), depth), CoreType::Bool],
    );
    signature.scratch_bytes = IntExpr::var("scratch_words") * 8;
    compiled(
        signature,
        "mxx_trapdoor_preimage",
        IntExpr::var("steps"),
        mxx_lattice_leaves::mxx_trapdoor_preimage,
    )
}

/// The step bound of one `trapdoor_preimage` call of `shape` with a stream of `budget` words:
/// a constant per stream word (Keccak and the rejection loops around each word), per binary64
/// operation of the transforms, the Schur complements, and the factorizations, and per term of
/// the integer polynomial products, measured on the emulator with a margin.
pub fn trapdoor_preimage_steps(shape: &mxx_lattice_leaves::preimage::Shape, budget: u64) -> u64 {
    let (n, towers, d, columns) =
        (shape.n as u64, shape.towers as u64, shape.rows as u64, shape.columns as u64);
    let (m, public, k) =
        (shape.trapdoor_columns() as u64, shape.public_columns() as u64, shape.tower_digits as u64);
    let dim = shape.secret_rows() as u64;
    let log = 64 - n.leading_zeros() as u64;
    let transform = n * (log + 2);
    let floats = (m * (dim + columns) + columns * dim) * transform +
        columns * dim * transform * (log + 2) * 4 +
        columns * dim * dim * dim * n * 8 +
        m * (dim * dim + dim * columns) * n +
        n * dim * dim * dim;
    let products = columns * (towers * d * n * n * public + dim * n * n * m);
    let gadget = columns * towers * d * n * (k + 2);
    TRAPDOOR_PREIMAGE_FIXED +
        TRAPDOOR_PREIMAGE_PER_WORD * budget +
        TRAPDOOR_PREIMAGE_PER_FLOAT * floats +
        TRAPDOOR_PREIMAGE_PER_PRODUCT * products +
        TRAPDOOR_PREIMAGE_PER_GADGET * gadget
}

/// Statistical TrapGen's draw of `T` and whether its preimages' perturbation covariance is
/// positive definite (`mxx_lattice_leaves::mxx_statistical_trapgen`). Its step bound and scratch
/// are the arguments `steps` ([`statistical_trapgen_steps`]) and `scratch_words`.
pub fn statistical_trapgen() -> Arc<Leaf> {
    let depth = IntExpr::var("L");
    let words = IntExpr::var("N") * IntExpr::var("m_bar") * IntExpr::var("digits");
    let mut signature = signature(
        "lattice.statistical_trapgen",
        &[
            "N",
            "L",
            "m_bar",
            "digits",
            "base",
            "sigma",
            "width",
            "budget",
            "steps",
            "scratch_words",
        ],
        vec![CoreType::buffer(8, depth.clone()), CoreType::bytes(32)],
        vec![CoreType::family(CoreType::buffer(8, words), depth), CoreType::Bool],
    );
    signature.scratch_bytes = IntExpr::var("scratch_words") * 8;
    compiled(
        signature,
        "mxx_statistical_trapgen",
        IntExpr::var("steps"),
        mxx_lattice_leaves::mxx_statistical_trapgen,
    )
}

/// The step bound of one `statistical_trapgen` call of `shape` (its `m̄` rows and digits) with a
/// stream of `budget` words, measured like [`trapdoor_preimage_steps`]'s.
pub fn statistical_trapgen_steps(shape: &mxx_lattice_leaves::preimage::Shape, budget: u64) -> u64 {
    let (n, towers, m_bar, digits) =
        (shape.n as u64, shape.towers as u64, shape.statistical as u64, shape.digits as u64);
    let log = 64 - n.leading_zeros() as u64;
    let floats =
        digits * m_bar * n * (log + 2) + digits * m_bar * m_bar * n + n * m_bar * m_bar * m_bar;
    let residues = m_bar * digits * n * towers;
    TRAPDOOR_PREIMAGE_FIXED +
        TRAPDOOR_PREIMAGE_PER_WORD * budget +
        TRAPDOOR_PREIMAGE_PER_FLOAT * floats +
        TRAPDOOR_PREIMAGE_PER_GADGET * residues
}

const TRAPDOOR_PREIMAGE_FIXED: u64 = 200_000;
const TRAPDOOR_PREIMAGE_PER_WORD: u64 = 1_000;
const TRAPDOOR_PREIMAGE_PER_FLOAT: u64 = 400;
const TRAPDOOR_PREIMAGE_PER_PRODUCT: u64 = 60;
const TRAPDOOR_PREIMAGE_PER_GADGET: u64 = 4_000;

/// A family of `count` buffers of `words` zero words, the first value of a loop that redraws.
/// Its step bound is measured.
pub fn zero() -> Arc<Leaf> {
    compiled(
        signature(
            "lattice.zero",
            &["words", "count"],
            Vec::new(),
            vec![CoreType::family(CoreType::buffer(8, "words"), "count")],
        ),
        "mxx_zero",
        IntExpr::from(64) + IntExpr::var("words") * IntExpr::var("count") * 8,
        mxx_lattice_leaves::mxx_zero,
    )
}

/// `I_count ⊗ A` over one limb. Its step bound is measured.
pub fn block_diagonal() -> Arc<Leaf> {
    let (n, rows, columns, count) =
        (IntExpr::var("N"), IntExpr::var("rows"), IntExpr::var("columns"), IntExpr::var("count"));
    compiled(
        signature(
            "lattice.block_diagonal",
            &["N", "rows", "columns", "count"],
            vec![CoreType::buffer(8, rows.clone() * columns.clone() * n.clone())],
            vec![CoreType::buffer(
                8,
                rows.clone() * columns.clone() * count.clone() * count.clone() * n.clone(),
            )],
        ),
        "mxx_block_diagonal",
        IntExpr::from(256) + rows * columns * count.clone() * count * n * 16,
        mxx_lattice_leaves::mxx_block_diagonal,
    )
}

/// The `W`-word integer `m` on the diagonal of a `size x size` matrix, modulo `q`.
pub fn lift_int(words: u32) -> Arc<Leaf> {
    let limb = CoreType::buffer(8, IntExpr::var("size") * IntExpr::var("size") * IntExpr::var("N"));
    compiled(
        signature("lattice.lift_int", &["N", "q", "size", "W"], vec![int(words)], vec![limb]),
        "mxx_lift_int",
        IntExpr::from(20_000) + IntExpr::var("size") * IntExpr::var("size") * IntExpr::var("N") * 8,
        mxx_lattice_leaves::mxx_lift_int,
    )
}

/// `round(t x / Q) mod t` of coefficients `0..length` of entry `(0, 0)`, exactly. Its step bound
/// is measured.
pub fn threshold_decode(words: u32) -> Arc<Leaf> {
    let (depth, length) = (IntExpr::var("L"), IntExpr::var("length"));
    compiled(
        signature(
            "lattice.threshold_decode",
            &["N", "L", "t", "length", "limb_words", "W"],
            vec![
                CoreType::family(CoreType::buffer(8, "limb_words"), depth.clone()),
                CoreType::buffer(8, depth.clone()),
            ],
            vec![CoreType::family(int(words), length.clone())],
        ),
        "mxx_threshold_decode",
        IntExpr::from(4096) + length * depth.clone() * depth * 200_000,
        mxx_lattice_leaves::mxx_threshold_decode,
    )
}

/// The number of words of the big integers of `mxx_crt_map`.
pub const CRT_MAP_WORDS: usize = 20;

/// The exact coefficientwise CRT map `mxx_crt_map`: modulus switch (mode 0), centered rebase
/// (mode 1), or centered round division (mode 2) from `L` source limbs to `M` destination limbs.
/// Its step bound is measured.
pub fn crt_map() -> Arc<Leaf> {
    let (depth, targets, words) =
        (IntExpr::var("L"), IntExpr::var("M"), IntExpr::var("limb_words"));
    compiled(
        signature(
            "lattice.crt_map",
            &["limb_words", "L", "M", "mode"],
            vec![
                CoreType::family(CoreType::buffer(8, words.clone()), depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, targets.clone()),
                CoreType::buffer(8, CRT_MAP_WORDS),
            ],
            vec![CoreType::family(CoreType::buffer(8, words.clone()), targets.clone())],
        ),
        "mxx_crt_map",
        IntExpr::from(400_000) +
            depth.clone() * depth.clone() * 40_000 +
            words *
                (depth.clone() * depth.clone() * 8_000 +
                    (depth + 2) * 200_000 +
                    targets * 20_000),
        mxx_lattice_leaves::mxx_crt_map,
    )
}

/// OpenFHE's fast RNS base conversion `mxx_fast_base_conversion` from `L` source limbs to `M`
/// destination limbs of `groups` blocks each. Its step bound is measured.
pub fn fast_base_conversion() -> Arc<Leaf> {
    let (depth, targets, groups, words) =
        (IntExpr::var("L"), IntExpr::var("M"), IntExpr::var("groups"), IntExpr::var("limb_words"));
    compiled(
        signature(
            "lattice.fast_base_conversion",
            &["limb_words", "L", "M", "groups", "down"],
            vec![
                CoreType::family(CoreType::buffer(8, words.clone()), depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, targets.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, depth.clone() * targets.clone()),
                CoreType::buffer(8, targets.clone() * 3),
            ],
            vec![CoreType::family(
                CoreType::buffer(8, groups.clone() * words.clone()),
                targets.clone(),
            )],
        ),
        "mxx_fast_base_conversion",
        IntExpr::from(256) + targets * groups * words * (depth * 3_000 + 3_000),
        mxx_lattice_leaves::mxx_fast_base_conversion,
    )
}

/// A real: one word holding the bits of an IEEE 754 binary64.
pub const REAL: CoreType =
    CoreType::Buffer { element_bytes: IntExpr::Const(8), count: IntExpr::Const(1) };

/// Binary64 `+`, `-`, `*`, `/`, and square root (argument `op` 0 to 4; the square root ignores
/// its second input). Its step bound is measured.
pub fn real_op() -> Arc<Leaf> {
    compiled(
        signature("lattice.real_op", &["op"], vec![REAL, REAL], vec![REAL]),
        "mxx_real_op",
        IntExpr::from(60_000),
        mxx_lattice_leaves::mxx_real_op,
    )
}

/// A `W`-word integer as the nearest binary64. Its step bound is measured.
pub fn int_to_real(words: u32) -> Arc<Leaf> {
    compiled(
        signature("lattice.int_to_real", &["W"], vec![int(words)], vec![REAL]),
        "mxx_int_to_real",
        IntExpr::from(20_000),
        mxx_lattice_leaves::mxx_int_to_real,
    )
}

/// One level of a CRT recomposition, `round(t x / Q) mod t` times a constant, from `L` source
/// limbs to `M` destination limbs. Its step bound is measured.
pub fn round_lift() -> Arc<Leaf> {
    let (depth, targets, words) =
        (IntExpr::var("L"), IntExpr::var("M"), IntExpr::var("limb_words"));
    compiled(
        signature(
            "lattice.round_lift",
            &["limb_words", "L", "M", "t"],
            vec![
                CoreType::family(CoreType::buffer(8, words.clone()), depth.clone()),
                CoreType::buffer(8, depth.clone()),
                CoreType::buffer(8, targets.clone()),
                CoreType::buffer(8, targets.clone()),
            ],
            vec![CoreType::family(CoreType::buffer(8, words.clone()), targets.clone())],
        ),
        "mxx_round_lift",
        IntExpr::from(4096) +
            depth.clone() * depth.clone() * 80_000 +
            words * (depth.clone() * depth * 4_000 + 200_000 + targets * 2_000),
        mxx_lattice_leaves::mxx_round_lift,
    )
}

/// A scalar polynomial from canonical coefficient bits. Its step bound is measured.
pub fn pack_bits() -> Arc<Leaf> {
    let (n, bits, depth) = (IntExpr::var("N"), IntExpr::var("B"), IntExpr::var("L"));
    compiled(
        signature(
            "lattice.pack_bits",
            &["N", "B", "L"],
            vec![
                CoreType::family(CoreType::Bool, n.clone() * bits.clone()),
                CoreType::buffer(8, depth.clone()),
            ],
            vec![CoreType::family(CoreType::buffer(8, n.clone()), depth.clone())],
        ),
        "mxx_pack_bits",
        IntExpr::from(256) + n * (bits * 16 + depth * 12_000),
        mxx_lattice_leaves::mxx_pack_bits,
    )
}

/// OpenFHE's negacyclic NTT, forward or inverse, of `count` polynomials over one limb. Its step
/// bound is measured.
pub fn ntt() -> Arc<Leaf> {
    let (n, count) = (IntExpr::var("N"), IntExpr::var("count"));
    let values = CoreType::buffer(8, count.clone() * n.clone());
    compiled(
        signature(
            "lattice.ntt",
            &["N", "q", "count", "inverse", "n_inverse"],
            vec![values.clone(), CoreType::buffer(8, n.clone())],
            vec![values],
        ),
        "mxx_ntt",
        IntExpr::from(256) + count * n.clone() * (IntExpr::Log2Ceil(Box::new(n)) + 2) * 1_200,
        mxx_lattice_leaves::mxx_ntt,
    )
}

/// The values in `[0, Q)` of `count` CRT coefficients as one-word integers. Its step bound is
/// measured.
pub fn crt_values(words: u32) -> Arc<Leaf> {
    let (count, depth) = (IntExpr::var("count"), IntExpr::var("L"));
    compiled(
        signature(
            "lattice.crt_values",
            &["count", "L", "W"],
            vec![
                CoreType::family(CoreType::buffer(8, count.clone()), depth.clone()),
                CoreType::buffer(8, depth.clone()),
            ],
            vec![CoreType::family(int(words), count.clone())],
        ),
        "mxx_crt_values",
        IntExpr::from(4096) +
            depth.clone() * depth.clone() * 80_000 +
            count * depth.clone() * depth * 4_000,
        mxx_lattice_leaves::mxx_crt_values,
    )
}

/// The residues of `count` one-word integers modulo each of `L` moduli. Its step bound is
/// measured.
pub fn int_residues(words: u32) -> Arc<Leaf> {
    let (count, depth) = (IntExpr::var("count"), IntExpr::var("L"));
    compiled(
        signature(
            "lattice.int_residues",
            &["count", "L", "W"],
            vec![CoreType::family(int(words), count.clone()), CoreType::buffer(8, depth.clone())],
            vec![CoreType::family(CoreType::buffer(8, count.clone()), depth.clone())],
        ),
        "mxx_int_residues",
        IntExpr::from(256) + count * depth * (IntExpr::var("W") * 600 + 600),
        mxx_lattice_leaves::mxx_int_residues,
    )
}

/// The wrapping one-word integer matrix-vector product. Its step bound is measured.
pub fn int_mat_vec(words: u32) -> Arc<Leaf> {
    let (outer, inner) = (IntExpr::var("outer"), IntExpr::var("inner"));
    compiled(
        signature(
            "lattice.int_mat_vec",
            &["outer", "inner", "transpose", "W"],
            vec![
                CoreType::family(int(words), outer.clone() * inner.clone()),
                CoreType::family(int(words), inner.clone()),
            ],
            vec![CoreType::family(int(words), outer.clone())],
        ),
        "mxx_int_mat_vec",
        IntExpr::from(256) + outer * inner * (IntExpr::var("W") * IntExpr::var("W") * 40 + 400),
        mxx_lattice_leaves::mxx_int_mat_vec,
    )
}

/// The hand-written leaves, by the Lean module name of their code.
pub fn all() -> Vec<(&'static str, Arc<Leaf>)> {
    vec![
        ("AddMod", add_mod()),
        ("SubMod", sub_mod()),
        ("NegMod", neg_mod()),
        ("BoolToInt", bool_to_int(1)),
        ("IntConstant", int_constant(1)),
        ("ScaleMod", scale_mod()),
        ("MultiplyMonomial", multiply_monomial(1)),
        ("PolyMulAcc", poly_mul_acc()),
    ]
}

/// The Lean modules of the leaves' code: `(file name, source)`, written to
/// `crates/ir-core/lean/leaves/MxxLatticeLeaves/Code` by the `export_leaf_code` example.
pub fn lean_code_modules() -> Vec<(String, String)> {
    all()
        .into_iter()
        .map(|(module, leaf)| {
            let code = leaf.rv64.as_ref().expect("lattice leaves have RV64IM code");
            let doc = format!(
                "The RV64IM code of the lattice leaf `{}`, entry at instruction 0.",
                leaf.signature.name
            );
            let namespace = format!("MxxLatticeLeaves.Code.{module}");
            (
                format!("{module}.lean"),
                mxx_core::lean::code_module(&namespace, &doc, &code.library.words, &[]),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real leaves give the host's binary64 results, and `num_bigint`'s conversion of
    /// integers, both on the host and in a proof image on the emulator (soft float and the
    /// leaves' integer square root and conversion).
    #[test]
    fn test_real_leaves_match_host_binary64_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use num_bigint::BigInt;
        use num_traits::ToPrimitive;
        use rand::Rng;
        const WORDS: u32 = 3;
        let mut root = DefinitionBuilder::new("root", vec![REAL, REAL, int(WORDS)]);
        let (a, b, i) = (root.param(0), root.param(1), root.param(2));
        let op = real_op();
        let mut outputs: Vec<_> =
            (0..4).map(|code| root.leaf(&op, vec![code.into()], vec![a, b])[0]).collect();
        outputs.push(root.leaf(&op, vec![4.into()], vec![a, a])[0]);
        outputs.push(root.leaf(&int_to_real(WORDS), vec![WORDS.into()], vec![i])[0]);
        let program = ProgramBuilder::new("reals").finish(
            vec!["a".into(), "b".into(), "i".into()],
            root.finish(outputs),
            (0..6).map(|k| format!("out{k}")).collect(),
        );
        let program = validate(Arc::new(program), Bindings::new()).unwrap();
        let image = flatten(&program, &FlattenOptions::default()).unwrap();
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        let mut rng = rand::rng();
        for case in 0..200 {
            // Positive operands of wide range, including subnormals, so every operation is
            // defined; integers of up to 190 bits of either sign, and small ones exactly.
            let real = |rng: &mut rand::rngs::ThreadRng| {
                f64::from_bits(rng.random_range(1..0x7fe0_0000_0000_0000u64))
            };
            let (x, y) = (real(&mut rng), real(&mut rng));
            let bits = if case % 4 == 0 { 50 } else { rng.random_range(1..190) };
            let mut n: BigInt = BigInt::from(rng.random::<u64>()) << 128 |
                BigInt::from(rng.random::<u64>()) << 64 |
                BigInt::from(rng.random::<u64>());
            n %= BigInt::from(1) << bits;
            if rng.random::<bool>() {
                n = -n;
            }
            let mut n_bytes = n.to_signed_bytes_le();
            n_bytes.resize(8 * WORDS as usize, if n < BigInt::from(0) { 0xff } else { 0 });
            let inputs = vec![
                x.to_bits().to_le_bytes().to_vec(),
                y.to_bits().to_le_bytes().to_vec(),
                n_bytes,
            ];
            let expected: Vec<Vec<u8>> =
                [x + y, x - y, x * y, x / y, x.sqrt(), n.to_f64().unwrap()]
                    .iter()
                    .map(|v| v.to_bits().to_le_bytes().to_vec())
                    .collect();
            assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), expected, "{x} {y} {n}");
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, steps) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
            let Outcome::Halted(payload) = outcome else { panic!("{outcome:?}") };
            let decoded: Vec<Vec<u8>> = payload.chunks(16).map(|c| c[8..].to_vec()).collect();
            assert_eq!(decoded, expected, "{x} {y} {n}");
            assert!(u128::from(steps) <= image.step_bound);
        }
    }

    /// The claim of `MxxLatticeLeaves.Examples.AddModSum` on the image of `add_mod` over two
    /// queried vectors, written for the Lean check of `scripts/check_core_lean.sh` (whose proof is
    /// `crates/ir-core/lean/leaves/MxxLatticeLeaves/Proofs/AddModSum.lean`).
    #[test]
    fn test_generated_claim_of_the_add_mod_sum_example() {
        use mxx_core::{
            claims::{AlgorithmImage, ClaimRequest, SpecManifest, generate},
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const Q: i128 = (1 << 61) - 1;
        let manifest = SpecManifest::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/lean/leaves/MxxLatticeLeaves/Examples/AddModSum.toml"
            ))
            .unwrap(),
        )
        .unwrap();
        let limb = CoreType::buffer(8, N);
        let mut root = DefinitionBuilder::new("root", vec![limb.clone(), limb]);
        let (a, b) = (root.param(0), root.param(1));
        let sum = root.leaf(&add_mod(), vec![N.into(), Q.into()], vec![a, b])[0];
        let program = ProgramBuilder::new("add-mod-sum").finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![sum]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: format!("⟨{N}, {Q}⟩"),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let modules = generate(
            "Generated.AddModSum",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap();
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
        std::fs::write(directory.join("../add_mod_sum.bin"), &image.bytes).unwrap();
        let leaves = std::collections::BTreeMap::from([(
            "lattice.add_mod".to_owned(),
            mxx_core::compose::LeanLeaf {
                module: "MxxLatticeLeaves.AddMod".into(),
                code_at: "MxxLatticeLeaves.Code.AddMod.At".into(),
                spec: "MxxLatticeLeaves.AddMod.byteSpec".into(),
            },
        )]);
        let (module, source) =
            mxx_core::compose::run_module("Generated.AddModSum", "run", &image, &leaves).unwrap();
        let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
        let path = directory.join(format!("{file}.lean"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
    }

    /// The same claim as `AddModSum` on `n k` residues, computed by a loop of `add_mod` calls
    /// over `k` columns of `n` residues each (`Generated.Columns`).
    #[test]
    fn test_generated_claim_of_the_columns_example() {
        use mxx_core::{
            claims::{AlgorithmImage, ClaimRequest, SpecManifest, generate},
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 1;
        let manifest = SpecManifest::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/lean/leaves/MxxLatticeLeaves/Examples/AddModSum.toml"
            ))
            .unwrap(),
        )
        .unwrap();
        let matrix = CoreType::buffer(8, N * K);
        let mut builder = ProgramBuilder::new("columns");
        let mut body = DefinitionBuilder::new("column", vec![matrix.clone(), matrix.clone()]);
        let column = |b: &mut DefinitionBuilder, input: usize| {
            let input = b.param(input);
            b.builtin(
                Builtin::Window {
                    offset: IntExpr::LoopIndex(0) * (N * 8),
                    runs: 1.into(),
                    length: (N * 8).into(),
                    stride: 0.into(),
                },
                vec![input],
            )
        };
        let (a, b) = (column(&mut body, 0), column(&mut body, 1));
        let sum = body.leaf(&add_mod(), vec![N.into(), Q.into()], vec![a, b])[0];
        let body = builder.add_definition(body.finish(vec![sum]));
        let mut root = DefinitionBuilder::new("root", vec![matrix.clone(), matrix]);
        let (pa, pb) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pa, pb],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: format!("⟨{}, {Q}⟩", N * K),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let modules = generate(
            "Generated.Columns",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap();
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
        let leaves = std::collections::BTreeMap::from([(
            "lattice.add_mod".to_owned(),
            mxx_core::compose::LeanLeaf {
                module: "MxxLatticeLeaves.AddMod".into(),
                code_at: "MxxLatticeLeaves.Code.AddMod.At".into(),
                spec: "MxxLatticeLeaves.AddMod.byteSpec".into(),
            },
        )]);
        let (module, source) =
            mxx_core::compose::run_module("Generated.Columns", "run", &image, &leaves).unwrap();
        let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
        let path = directory.join(format!("{file}.lean"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, source).unwrap();
    }

    /// The same claim on `rows · columns · n` residues by two nested loops, rows outside and
    /// columns inside, whose windows read both indices (`Generated.Grid`).
    #[test]
    fn test_generated_claim_of_the_grid_example() {
        use mxx_core::{
            claims::{AlgorithmImage, ClaimRequest, SpecManifest, generate},
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const COLUMNS: i128 = 3;
        const ROWS: i128 = 2;
        const Q: i128 = (1 << 61) - 1;
        let manifest = SpecManifest::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/lean/leaves/MxxLatticeLeaves/Examples/AddModSum.toml"
            ))
            .unwrap(),
        )
        .unwrap();
        let matrix = CoreType::buffer(8, N * COLUMNS * ROWS);
        let mut builder = ProgramBuilder::new("grid");
        let mut cell = DefinitionBuilder::new("cell", vec![matrix.clone(), matrix.clone()]);
        let window = |b: &mut DefinitionBuilder, input: usize| {
            let input = b.param(input);
            b.builtin(
                Builtin::Window {
                    offset: (IntExpr::LoopIndex(0) * COLUMNS + IntExpr::LoopIndex(1)) * (N * 8),
                    runs: 1.into(),
                    length: (N * 8).into(),
                    stride: 0.into(),
                },
                vec![input],
            )
        };
        let (a, b) = (window(&mut cell, 0), window(&mut cell, 1));
        let sum = cell.leaf(&add_mod(), vec![N.into(), Q.into()], vec![a, b])[0];
        let cell = builder.add_definition(cell.finish(vec![sum]));
        let mut row = DefinitionBuilder::new("row", vec![matrix.clone(), matrix.clone()]);
        let (ra, rb) = (row.param(0), row.param(1));
        let cells = row.add_loop(
            Loop {
                count: COLUMNS.into(),
                index_slot: 1,
                body: cell,
                bindings: vec![],
                carried: vec![],
                inputs: vec![ra, rb],
                until: false,
            },
            1,
        )[0];
        let row = builder.add_definition(row.finish(vec![cells]));
        let mut root = DefinitionBuilder::new("root", vec![matrix.clone(), matrix]);
        let (pa, pb) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: ROWS.into(),
                index_slot: 0,
                body: row,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pa, pb],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: format!("⟨{}, {Q}⟩", N * COLUMNS * ROWS),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let mut modules = generate(
            "Generated.Grid",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap();
        let leaves = std::collections::BTreeMap::from([(
            "lattice.add_mod".to_owned(),
            mxx_core::compose::LeanLeaf {
                module: "MxxLatticeLeaves.AddMod".into(),
                code_at: "MxxLatticeLeaves.Code.AddMod.At".into(),
                spec: "MxxLatticeLeaves.AddMod.byteSpec".into(),
            },
        )]);
        modules
            .push(mxx_core::compose::run_module("Generated.Grid", "run", &image, &leaves).unwrap());
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
    }

    /// A loop with a carried value: `c ← add_mod(c, column i)` over the columns of a matrix
    /// (`Generated.Fold`). Its run is generated and checked; it has no claim of its own, so only
    /// the image, program, and run modules are written.
    #[test]
    fn test_generated_run_of_the_fold_example() {
        use mxx_core::{
            claims::{AlgorithmImage, ClaimRequest, SpecManifest, generate},
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 1;
        let manifest = SpecManifest::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/lean/leaves/MxxLatticeLeaves/Examples/AddModSum.toml"
            ))
            .unwrap(),
        )
        .unwrap();
        let (vector, matrix) = (CoreType::buffer(8, N), CoreType::buffer(8, N * K));
        let mut builder = ProgramBuilder::new("fold");
        let mut body = DefinitionBuilder::new("step", vec![vector.clone(), matrix.clone()]);
        let (acc, m) = (body.param(0), body.param(1));
        let column = body.builtin(
            Builtin::Window {
                offset: IntExpr::LoopIndex(0) * (N * 8),
                runs: 1.into(),
                length: (N * 8).into(),
                stride: 0.into(),
            },
            vec![m],
        );
        let next = body.leaf(&add_mod(), vec![N.into(), Q.into()], vec![acc, column])[0];
        let body = builder.add_definition(body.finish(vec![next]));
        let mut root = DefinitionBuilder::new("root", vec![vector, matrix]);
        let (init, pm) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![init],
                inputs: vec![pm],
                until: false,
            },
            0,
        )[0];
        let program = builder.finish(
            vec!["init".into(), "m".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: format!("⟨{N}, {Q}⟩"),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let mut modules: Vec<(String, String)> = generate(
            "Generated.Fold",
            &manifest,
            &[AlgorithmImage { name: "run".into(), image: &image }],
            &request,
        )
        .unwrap()
        .into_iter()
        .filter(|(m, _)| m != "Generated.Fold")
        .collect();
        let leaves = std::collections::BTreeMap::from([(
            "lattice.add_mod".to_owned(),
            mxx_core::compose::LeanLeaf {
                module: "MxxLatticeLeaves.AddMod".into(),
                code_at: "MxxLatticeLeaves.Code.AddMod.At".into(),
                spec: "MxxLatticeLeaves.AddMod.byteSpec".into(),
            },
        )]);
        modules
            .push(mxx_core::compose::run_module("Generated.Fold", "run", &image, &leaves).unwrap());
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
    }

    /// Writes the image, program, and generated run modules of `image` (algorithm `run`) under
    /// `Generated.<name>`, with the AddModSum manifest and no claim, for the Lean check.
    fn write_run_modules(name: &str, image: &mxx_core::flatten::ProofImage) {
        use mxx_core::claims::{AlgorithmImage, ClaimRequest, SpecManifest, generate};
        let manifest = SpecManifest::parse(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/lean/leaves/MxxLatticeLeaves/Examples/AddModSum.toml"
            ))
            .unwrap(),
        )
        .unwrap();
        let request = ClaimRequest {
            params: "⟨0, 0⟩".into(),
            failure_probability: "0".into(),
            security: Vec::new(),
        };
        let namespace = format!("Generated.{name}");
        let mut modules: Vec<(String, String)> = generate(
            &namespace,
            &manifest,
            &[AlgorithmImage { name: "run".into(), image }],
            &request,
        )
        .unwrap()
        .into_iter()
        .filter(|(m, _)| *m != namespace)
        .collect();
        let leaves = std::collections::BTreeMap::from([(
            "lattice.add_mod".to_owned(),
            mxx_core::compose::LeanLeaf {
                module: "MxxLatticeLeaves.AddMod".into(),
                code_at: "MxxLatticeLeaves.Code.AddMod.At".into(),
                spec: "MxxLatticeLeaves.AddMod.byteSpec".into(),
            },
        )]);
        modules.push(mxx_core::compose::run_module(&namespace, "run", image, &leaves).unwrap());
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/core_claims/Generated");
        for (module, source) in &modules {
            let file = module.strip_prefix("Generated.").unwrap().replace('.', "/");
            let path = directory.join(format!("{file}.lean"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, source).unwrap();
        }
    }

    /// A loop whose leaf argument reads the index: column `i` is added modulo `q₀ + 2 i`, so
    /// each call's descriptor is written at run time (`Generated.Moduli`).
    #[test]
    fn test_generated_run_of_the_moduli_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 101;
        let matrix = CoreType::buffer(8, N * K);
        let mut builder = ProgramBuilder::new("moduli");
        let mut body = DefinitionBuilder::new("column", vec![matrix.clone(), matrix.clone()]);
        let column = |b: &mut DefinitionBuilder, input: usize| {
            let input = b.param(input);
            b.builtin(
                Builtin::Window {
                    offset: IntExpr::LoopIndex(0) * (N * 8),
                    runs: 1.into(),
                    length: (N * 8).into(),
                    stride: 0.into(),
                },
                vec![input],
            )
        };
        let (a, b) = (column(&mut body, 0), column(&mut body, 1));
        let q = IntExpr::from(Q) + IntExpr::LoopIndex(0) * 2;
        let sum = body.leaf(&add_mod(), vec![N.into(), q], vec![a, b])[0];
        let body = builder.add_definition(body.finish(vec![sum]));
        let mut root = DefinitionBuilder::new("root", vec![matrix.clone(), matrix]);
        let (pa, pb) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pa, pb],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Moduli", &image);
    }

    /// A loop whose leaf argument is a table over the index: column `i` is added modulo `q_i`
    /// of three moduli, read from read-only data (`Generated.Tables`).
    #[test]
    fn test_generated_run_of_the_tables_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 101;
        let matrix = CoreType::buffer(8, N * K);
        let mut builder = ProgramBuilder::new("tables");
        let mut body = DefinitionBuilder::new("column", vec![matrix.clone(), matrix.clone()]);
        let column = |b: &mut DefinitionBuilder, input: usize| {
            let input = b.param(input);
            b.builtin(
                Builtin::Window {
                    offset: IntExpr::LoopIndex(0) * (N * 8),
                    runs: 1.into(),
                    length: (N * 8).into(),
                    stride: 0.into(),
                },
                vec![input],
            )
        };
        let (a, b) = (column(&mut body, 0), column(&mut body, 1));
        let q = IntExpr::Select {
            selector: Box::new(IntExpr::LoopIndex(0)),
            branches: vec![Q.into(), (Q - 4).into(), (Q - 10).into()],
        };
        let sum = body.leaf(&add_mod(), vec![N.into(), q], vec![a, b])[0];
        let body = builder.add_definition(body.finish(vec![sum]));
        let mut root = DefinitionBuilder::new("root", vec![matrix.clone(), matrix]);
        let (pa, pb) = (root.param(0), root.param(1));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pa, pb],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(
            vec!["a".into(), "b".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Tables", &image);
    }

    /// A member chosen at run time: member `v` of a family of three vectors, `v` a 2-word
    /// integer input, added to itself (`Generated.Gather`).
    #[test]
    fn test_generated_run_of_the_gather_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const Q: i128 = (1 << 61) - 1;
        let vector = CoreType::buffer(8, N);
        let family = CoreType::family(vector, 3);
        let mut root = DefinitionBuilder::new("root", vec![family, CoreType::Int { words: 2 }]);
        let (members, index) = (root.param(0), root.param(1));
        let member = root.builtin(Builtin::GetDynamic, vec![members, index]);
        let sum = root.leaf(&add_mod(), vec![N.into(), Q.into()], vec![member, member])[0];
        let program = ProgramBuilder::new("gather").finish(
            vec!["members".into(), "index".into()],
            root.finish(vec![sum]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Gather", &image);
    }

    /// Runtime selection: one of three vectors, chosen by a 2-word integer input, added to
    /// itself (`Generated.Choose`).
    #[test]
    fn test_generated_run_of_the_choose_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const Q: i128 = (1 << 61) - 1;
        let vector = CoreType::buffer(8, N);
        let mut root = DefinitionBuilder::new(
            "root",
            vec![CoreType::Int { words: 2 }, vector.clone(), vector.clone(), vector],
        );
        let params: Vec<_> = (0..4).map(|i| root.param(i)).collect();
        let chosen = root.builtin(Builtin::Select, params);
        let sum = root.leaf(&add_mod(), vec![N.into(), Q.into()], vec![chosen, chosen])[0];
        let program = ProgramBuilder::new("choose").finish(
            vec!["selector".into(), "a".into(), "b".into(), "c".into()],
            root.finish(vec![sum]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Choose", &image);
    }

    /// Coins: 4 residues' worth of uniform bytes, drawn once, added to the input
    /// (`Generated.Noise`); and drawn once per column of a loop (`Generated.Samples`). Their runs
    /// are distributions over the draws.
    #[test]
    fn test_generated_runs_of_the_coins_examples() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 1;
        let image = |program| {
            flatten(
                &validate(Arc::new(program), Bindings::new()).unwrap(),
                &FlattenOptions::default(),
            )
            .unwrap()
        };
        // One draw.
        let mut root = DefinitionBuilder::new("root", vec![CoreType::buffer(8, N)]);
        let input = root.param(0);
        let coins = root.builtin(Builtin::Coins { bytes: (N * 8).into() }, vec![]);
        let sum = root.leaf(&add_mod(), vec![N.into(), Q.into()], vec![input, coins])[0];
        let program = ProgramBuilder::new("noise").finish(
            vec!["x".into()],
            root.finish(vec![sum]),
            vec!["sum".into()],
        );
        write_run_modules("Noise", &image(program));
        // One draw per column.
        let matrix = CoreType::buffer(8, N * K);
        let mut builder = ProgramBuilder::new("samples");
        let mut body = DefinitionBuilder::new("column", vec![matrix.clone()]);
        let m = body.param(0);
        let column = body.builtin(
            Builtin::Window {
                offset: IntExpr::LoopIndex(0) * (N * 8),
                runs: 1.into(),
                length: (N * 8).into(),
                stride: 0.into(),
            },
            vec![m],
        );
        let coins = body.builtin(Builtin::Coins { bytes: (N * 8).into() }, vec![]);
        let sum = body.leaf(&add_mod(), vec![N.into(), Q.into()], vec![column, coins])[0];
        let body = builder.add_definition(body.finish(vec![sum]));
        let mut root = DefinitionBuilder::new("root", vec![matrix]);
        let pm = root.param(0);
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![],
                inputs: vec![pm],
                until: false,
            },
            1,
        )[0];
        let program = builder.finish(vec!["m".into()], root.finish(vec![out]), vec!["sum".into()]);
        write_run_modules("Samples", &image(program));
    }

    /// A loop that draws until it stops: `c ← add_mod(x, coins)` until the flag (an input) is
    /// true, at most `K` times (`Generated.Retry`).
    #[test]
    fn test_generated_run_of_the_retry_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 1;
        let vector = CoreType::buffer(8, N);
        let flags = CoreType::family(CoreType::Bool, 1);
        let mut builder = ProgramBuilder::new("retry");
        let mut body =
            DefinitionBuilder::new("attempt", vec![vector.clone(), vector.clone(), flags.clone()]);
        let (x, fs) = (body.param(1), body.param(2));
        let coins = body.builtin(Builtin::Coins { bytes: (N * 8).into() }, vec![]);
        let next = body.leaf(&add_mod(), vec![N.into(), Q.into()], vec![x, coins])[0];
        let stop = body.builtin(Builtin::Get { index: 0.into() }, vec![fs]);
        let body = builder.add_definition(body.finish(vec![next, stop]));
        let mut root = DefinitionBuilder::new("root", vec![vector.clone(), vector, flags]);
        let (init, px, pf) = (root.param(0), root.param(1), root.param(2));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![init],
                inputs: vec![px, pf],
                until: true,
            },
            0,
        )[0];
        let program = builder.finish(
            vec!["init".into(), "x".into(), "flags".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Retry", &image);
    }

    /// A loop that stops early: `c ← add_mod(c, column i)` until the `i`-th of a family of
    /// flags is true (`Generated.Until`).
    #[test]
    fn test_generated_run_of_the_until_example() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            program::{DefinitionBuilder, Loop, ProgramBuilder},
            validate::validate,
        };
        const N: i128 = 4;
        const K: i128 = 3;
        const Q: i128 = (1 << 61) - 1;
        let (vector, matrix) = (CoreType::buffer(8, N), CoreType::buffer(8, N * K));
        let flags = CoreType::family(CoreType::Bool, K);
        let mut builder = ProgramBuilder::new("until");
        let mut body =
            DefinitionBuilder::new("step", vec![vector.clone(), matrix.clone(), flags.clone()]);
        let (acc, m, fs) = (body.param(0), body.param(1), body.param(2));
        let column = body.builtin(
            Builtin::Window {
                offset: IntExpr::LoopIndex(0) * (N * 8),
                runs: 1.into(),
                length: (N * 8).into(),
                stride: 0.into(),
            },
            vec![m],
        );
        let next = body.leaf(&add_mod(), vec![N.into(), Q.into()], vec![acc, column])[0];
        let stop = body.builtin(Builtin::Get { index: IntExpr::LoopIndex(0) }, vec![fs]);
        let body = builder.add_definition(body.finish(vec![next, stop]));
        let mut root = DefinitionBuilder::new("root", vec![vector, matrix, flags]);
        let (init, pm, pf) = (root.param(0), root.param(1), root.param(2));
        let out = root.add_loop(
            Loop {
                count: K.into(),
                index_slot: 0,
                body,
                bindings: vec![],
                carried: vec![init],
                inputs: vec![pm, pf],
                until: true,
            },
            0,
        )[0];
        let program = builder.finish(
            vec!["init".into(), "m".into(), "flags".into()],
            root.finish(vec![out]),
            vec!["sum".into()],
        );
        let image = flatten(
            &validate(Arc::new(program), Bindings::new()).unwrap(),
            &FlattenOptions::default(),
        )
        .unwrap();
        write_run_modules("Until", &image);
    }

    /// The preimage leaf gives the same candidate and flag on the host and in a proof image on the
    /// emulator from the same coins, within its step bound, for random small shapes (override the
    /// count with `MXX_TEST_PREIMAGE_LEAF_CASES`).
    #[test]
    fn test_trapdoor_preimage_leaf_matches_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use mxx_lattice_leaves::preimage::{Shape, scratch_words};
        use rand::Rng;
        let cases: usize = std::env::var("MXX_TEST_PREIMAGE_LEAF_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        let mut rng = rand::rng();
        let moduli_pool = [(1u64 << 30) - 35, (1 << 31) - 1];
        for _ in 0..cases {
            let towers = rng.random_range(1..3);
            let base_bits = rng.random_range(2..4) * 5;
            let tower_digits = 31usize.div_ceil(base_bits);
            let log_n = std::env::var("MXX_TEST_PREIMAGE_LEAF_LOG_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4);
            let mut shape = Shape {
                n: 1 << std::env::var("MXX_TEST_PREIMAGE_LEAF_MIN_LOG_N")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .map_or_else(|| rng.random_range(1..log_n), |min: u32| min.max(1)),
                towers,
                rows: 1,
                digits: towers * tower_digits,
                tower_digits,
                columns: rng.random_range(1..3),
                statistical: if rng.random::<bool>() { rng.random_range(1..4) } else { 0 },
            };
            if shape.statistical == 0 {
                shape.rows = rng.random_range(1..3);
            }
            let moduli = &moduli_pool[..towers];
            let (base, sigma) = (1u64 << base_bits, 4.578f64);
            let c = (base as f64 + 1.0) * sigma;
            let n = shape.n as f64;
            let width = if shape.statistical > 0 {
                let s1 = sigma *
                    n.sqrt() *
                    ((shape.statistical as f64).sqrt() +
                        (shape.digits as f64).sqrt() +
                        (n * 512.0).ln().sqrt());
                (c * c * (1.0 + s1 * s1) + sigma * sigma).sqrt()
            } else {
                1.8 * c *
                    sigma *
                    (((shape.n * shape.rows * shape.digits) as f64).sqrt() +
                        (2.0 * n).sqrt() +
                        4.7)
            };
            let bound = (6.5 * width) as u64;
            let budget = 32 * shape.gaussian_samples() as u64 + 65_536;
            let steps = trapdoor_preimage_steps(&shape, budget);
            let scratch = scratch_words(&shape);
            // Any public part; the secret block small (within the statistical covariance's range).
            let small_from = shape.rows * shape.n * shape.public_columns();
            let spread = if shape.statistical > 0 { 2i64 } else { 9 };
            let words = |count: usize, small: bool, rng: &mut rand::rngs::ThreadRng| {
                (0..towers)
                    .flat_map(|l| {
                        let q = moduli[l];
                        (0..count)
                            .map(|i| {
                                if small && i >= small_from {
                                    (rng.random_range(-spread..=spread)).rem_euclid(q as i64) as u64
                                } else {
                                    rng.random_range(0..q)
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<u8>>()
            };
            let trapdoor = words(shape.trapdoor_limb_words(), true, &mut rng);
            let target = words(shape.target_limb_words(), false, &mut rng);
            let moduli_bytes: Vec<u8> = moduli.iter().flat_map(|q| q.to_le_bytes()).collect();
            let coins: Vec<u8> = (0..32).map(|_| rng.random()).collect();
            let leaf = trapdoor_preimage();
            let family = |words: usize| CoreType::family(CoreType::buffer(8, words), towers);
            let mut root = DefinitionBuilder::new(
                "root",
                vec![
                    family(shape.trapdoor_limb_words()),
                    family(shape.target_limb_words()),
                    CoreType::buffer(8, towers),
                    CoreType::bytes(32),
                ],
            );
            let params: Vec<_> = (0..4).map(|i| root.param(i)).collect();
            let arguments: Vec<IntExpr> = [
                shape.n as u64,
                towers as u64,
                shape.rows as u64,
                shape.digits as u64,
                tower_digits as u64,
                shape.columns as u64,
                shape.secret_rows() as u64,
                u64::from(shape.statistical > 0),
                base,
                sigma.to_bits(),
                width.to_bits(),
                bound,
                budget,
                steps,
                scratch as u64,
            ]
            .iter()
            .map(|w| IntExpr::from(*w as i128))
            .collect();
            let outputs = root.leaf(&leaf, arguments, params);
            let program = ProgramBuilder::new("preimage").finish(
                (0..4).map(|k| format!("in{k}")).collect(),
                root.finish(outputs),
                vec!["x".into(), "ok".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let inputs = vec![trapdoor, target, moduli_bytes, coins];
            let expected = host::run(&program, inputs.clone(), None).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
            let Outcome::Halted(payload) = outcome else { panic!("{shape:?}: {outcome:?}") };
            let x_bytes = expected[0].len();
            assert_eq!(&payload[8..8 + x_bytes], &expected[0][..], "{shape:?}");
            assert_eq!(&payload[16 + x_bytes..], &expected[1][..], "{shape:?}");
            println!("{shape:?}: {used} steps of {steps}");
            assert!(u128::from(used) <= image.step_bound);
        }
    }

    /// The statistical TrapGen leaf gives the same `T` and flag on the host and on the emulator
    /// from the same coins, within its step bound, and accepts at the minimum width.
    #[test]
    fn test_statistical_trapgen_leaf_matches_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use mxx_lattice_leaves::preimage::{Shape, trapgen_scratch_words};
        use rand::Rng;
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        let mut rng = rand::rng();
        let moduli_pool = [(1u64 << 30) - 35, (1 << 31) - 1];
        let mut accepted = 0;
        for _ in 0..4 {
            let towers = rng.random_range(1..3);
            let base_bits = rng.random_range(2..4) * 5;
            let tower_digits = 31usize.div_ceil(base_bits);
            let shape = Shape {
                n: 1 << rng.random_range(1..5),
                towers,
                rows: 1,
                digits: towers * tower_digits,
                tower_digits,
                columns: 1,
                statistical: rng.random_range(1..5),
            };
            let (n, m_bar, digits) = (shape.n, shape.statistical, shape.digits);
            let (base, sigma) = (1u64 << base_bits, 4.578f64);
            let c = (base as f64 + 1.0) * sigma;
            let s1 = sigma *
                (n as f64).sqrt() *
                ((m_bar as f64).sqrt() + (digits as f64).sqrt() + (n as f64 * 512.0).ln().sqrt());
            let width = (c * c * (1.0 + s1 * s1) + sigma * sigma).sqrt();
            let budget = 32 * (m_bar * digits * n) as u64 + 65_536;
            let steps = statistical_trapgen_steps(&shape, budget);
            let leaf = statistical_trapgen();
            let mut root = DefinitionBuilder::new(
                "root",
                vec![CoreType::buffer(8, towers), CoreType::bytes(32)],
            );
            let params = vec![root.param(0), root.param(1)];
            let arguments: Vec<IntExpr> = [
                n as u64,
                towers as u64,
                m_bar as u64,
                digits as u64,
                base,
                sigma.to_bits(),
                width.to_bits(),
                budget,
                steps,
                trapgen_scratch_words(n, m_bar, digits) as u64,
            ]
            .iter()
            .map(|w| IntExpr::from(*w as i128))
            .collect();
            let outputs = root.leaf(&leaf, arguments, params);
            let program = ProgramBuilder::new("trapgen").finish(
                vec!["moduli".into(), "coins".into()],
                root.finish(outputs),
                vec!["t".into(), "ok".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let moduli: Vec<u8> =
                moduli_pool[..towers].iter().flat_map(|q| q.to_le_bytes()).collect();
            let coins: Vec<u8> = (0..32).map(|_| rng.random()).collect();
            let inputs = vec![moduli, coins];
            let expected = host::run(&program, inputs.clone(), None).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
            let Outcome::Halted(payload) = outcome else { panic!("{shape:?}: {outcome:?}") };
            let t_bytes = expected[0].len();
            assert_eq!(&payload[8..8 + t_bytes], &expected[0][..], "{shape:?}");
            assert_eq!(&payload[16 + t_bytes..], &expected[1][..], "{shape:?}");
            accepted += usize::from(expected[1][0] == 1);
            println!("{shape:?}: {used} steps of {steps}");
            assert!(u128::from(used) <= image.step_bound);
        }
        assert!(accepted > 0, "TrapGen rejected every T");
    }

    /// The hand-written word fills (`bool_to_int`, `int_constant`) give their host results on the
    /// emulator within their step bounds, for random widths and values.
    #[test]
    fn test_word_fill_leaves_match_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Machine, NoEnvironment, Outcome},
            validate::validate,
        };
        use rand::Rng;
        let mut rng = rand::rng();
        for _ in 0..20 {
            let words = rng.random_range(1..8u32);
            let value: i64 = rng.random();
            let flag = u64::from(rng.random::<bool>());
            let mut root = DefinitionBuilder::new("root", Vec::new());
            let constant =
                root.leaf(&int_constant(words), vec![words.into(), value.into()], vec![])[0];
            let bool_bytes = flag.to_le_bytes().to_vec();
            let b =
                root.builtin(Builtin::Constant { ty: CoreType::Bool, bytes: bool_bytes }, vec![]);
            let converted = root.leaf(&bool_to_int(words), vec![words.into()], vec![b])[0];
            let program = ProgramBuilder::new("fill").finish(
                Vec::new(),
                root.finish(vec![constant, converted]),
                vec!["constant".into(), "converted".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let outputs = host::run(&program, Vec::new(), None).unwrap();
            let sign = if value < 0 { u64::MAX } else { 0 };
            let expect = |first: u64, rest: u64| -> Vec<u8> {
                (0..words).flat_map(|k| if k == 0 { first } else { rest }.to_le_bytes()).collect()
            };
            assert_eq!(outputs, vec![expect(value as u64, sign), expect(flag, 0)]);
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut NoEnvironment);
            let Outcome::Halted(payload) = outcome else { panic!("{outcome:?}") };
            let n = 8 * words as usize;
            assert_eq!(&payload[8..8 + n], &outputs[0][..]);
            assert_eq!(&payload[16 + n..], &outputs[1][..]);
            assert!(u128::from(used) <= image.step_bound);
        }
    }

    /// The hand-written `scale_mod` gives `c a mod q` on the emulator within its step bound, for
    /// random moduli below `2^63`, scalars, and residues (the extremes included).
    #[test]
    fn test_scale_mod_matches_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use rand::Rng;
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        let mut rng = rand::rng();
        for case in 0..40 {
            let q: u64 = match case {
                0 => (1 << 63) - 25,
                1 => 2,
                _ => {
                    let bits = rng.random_range(2..64);
                    rng.random_range(2..1u64 << bits)
                }
            };
            let n = rng.random_range(1..9usize);
            let c = if case % 3 == 0 { q - 1 } else { rng.random_range(0..q) };
            let a: Vec<u64> =
                (0..n).map(|i| if i == 0 { q - 1 } else { rng.random_range(0..q) }).collect();
            let mut root = DefinitionBuilder::new("root", vec![CoreType::buffer(8, n as i128)]);
            let x = root.param(0);
            let out = root.leaf(
                &scale_mod(),
                vec![(n as i128).into(), (q as i128).into(), (c as i128).into()],
                vec![x],
            )[0];
            let program = ProgramBuilder::new("scale").finish(
                vec!["a".into()],
                root.finish(vec![out]),
                vec!["out".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
            let expected: Vec<u8> = a
                .iter()
                .flat_map(|v| {
                    ((u128::from(c) * u128::from(*v) % u128::from(q)) as u64).to_le_bytes()
                })
                .collect();
            assert_eq!(
                host::run(&program, vec![bytes.clone()], None).unwrap(),
                vec![expected.clone()]
            );
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(vec![bytes]));
            let Outcome::Halted(payload) = outcome else { panic!("{outcome:?}") };
            assert_eq!(&payload[8..], &expected[..], "q {q} c {c} a {a:?}");
            assert!(u128::from(used) <= image.step_bound);
        }
    }

    /// The hand-written `multiply_monomial` multiplies each entry by `X^k` (checked against
    /// repeated multiplication by `X`) on the host and the emulator within its step bound, for
    /// random sizes, moduli, and exponents of either sign.
    #[test]
    fn test_multiply_monomial_matches_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use rand::Rng;
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        const WORDS: u32 = 2;
        let mut rng = rand::rng();
        for _ in 0..40 {
            let n = 1usize << rng.random_range(0..5);
            let entries = rng.random_range(1..4usize);
            let q: u64 = rng.random_range(2..1u64 << 62);
            let k: i64 = rng.random_range(-100..100);
            let a: Vec<u64> = (0..entries * n).map(|_| rng.random_range(0..q)).collect();
            // X^k by |k| steps of X (or X^-1 = -X^(N-1)).
            let mut expected = a.clone();
            for e in 0..entries {
                let mut poly: Vec<u64> = a[e * n..(e + 1) * n].to_vec();
                let neg = |v: u64| if v == 0 { 0 } else { q - v };
                for _ in 0..k.unsigned_abs() {
                    poly = if k > 0 {
                        (0..n)
                            .map(|i| if i == 0 { neg(poly[n - 1]) } else { poly[i - 1] })
                            .collect()
                    } else {
                        (0..n)
                            .map(|i| if i == n - 1 { neg(poly[0]) } else { poly[i + 1] })
                            .collect()
                    };
                }
                expected[e * n..(e + 1) * n].copy_from_slice(&poly);
            }
            let leaf = multiply_monomial(WORDS);
            let mut root = DefinitionBuilder::new(
                "root",
                vec![CoreType::buffer(8, (entries * n) as i128), int(WORDS)],
            );
            let (x, kk) = (root.param(0), root.param(1));
            let args = vec![(n as i128).into(), (q as i128).into(), (entries as i128).into()];
            let out = root.leaf(&leaf, args, vec![x, kk])[0];
            let program = ProgramBuilder::new("monomial").finish(
                vec!["a".into(), "k".into()],
                root.finish(vec![out]),
                vec!["out".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
            let k_bytes: Vec<u8> = [k as u64, if k < 0 { u64::MAX } else { 0 }]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let expected: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
            let inputs = vec![a_bytes, k_bytes];
            assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), vec![expected.clone()]);
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
            let Outcome::Halted(payload) = outcome else { panic!("{outcome:?}") };
            assert_eq!(&payload[8..], &expected[..], "n {n} k {k}");
            assert!(u128::from(used) <= image.step_bound);
        }
    }

    /// The hand-written `poly_mul_acc` gives `acc + a b` in `Z_q[X]/(X^N + 1)` (checked against a
    /// direct negacyclic product) on the host and the emulator within its step bound, for random
    /// sizes and moduli below `2^63`.
    #[test]
    fn test_poly_mul_acc_matches_on_host_and_emulator() {
        use mxx_core::{
            expr::Bindings,
            flatten::{FlattenOptions, flatten},
            host,
            program::{DefinitionBuilder, ProgramBuilder},
            rv64::machine::{Environment, Machine, Outcome},
            validate::validate,
        };
        use rand::Rng;
        struct Inputs(Vec<Vec<u8>>);
        impl Environment for Inputs {
            fn query(&mut self, request: &[u8]) -> Vec<u8> {
                self.0[u64::from_le_bytes(request[1..9].try_into().unwrap()) as usize].clone()
            }
        }
        let mut rng = rand::rng();
        for case in 0..30 {
            let n = rng.random_range(1..9usize);
            let q: u64 = if case == 0 { (1 << 63) - 25 } else { rng.random_range(2..1u64 << 63) };
            let poly = |rng: &mut rand::rngs::ThreadRng| -> Vec<u64> {
                (0..n).map(|_| rng.random_range(0..q)).collect()
            };
            let (acc, a, b) = (poly(&mut rng), poly(&mut rng), poly(&mut rng));
            let mut expected: Vec<i128> = acc.iter().map(|v| i128::from(*v)).collect();
            for i in 0..n {
                for j in 0..n {
                    let p = (u128::from(a[i]) * u128::from(b[j]) % u128::from(q)) as i128;
                    if i + j < n {
                        expected[i + j] += p;
                    } else {
                        expected[i + j - n] -= p;
                    }
                }
            }
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|v| (v.rem_euclid(i128::from(q)) as u64).to_le_bytes())
                .collect();
            let limb = CoreType::buffer(8, n as i128);
            let mut root = DefinitionBuilder::new("root", vec![limb.clone(), limb.clone(), limb]);
            let params: Vec<_> = (0..3).map(|k| root.param(k)).collect();
            let out =
                root.leaf(&poly_mul_acc(), vec![(n as i128).into(), (q as i128).into()], params)[0];
            let program = ProgramBuilder::new("polymul").finish(
                vec!["acc".into(), "a".into(), "b".into()],
                root.finish(vec![out]),
                vec!["out".into()],
            );
            let program = validate(Arc::new(program), Bindings::new()).unwrap();
            let bytes = |v: &[u64]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
            let inputs = vec![bytes(&acc), bytes(&a), bytes(&b)];
            assert_eq!(host::run(&program, inputs.clone(), None).unwrap(), vec![expected.clone()]);
            let image = flatten(&program, &FlattenOptions::default()).unwrap();
            let mut machine = Machine::load(&image.bytes, image.mem_size).unwrap();
            let (outcome, used) =
                machine.run(u64::try_from(image.step_bound).unwrap(), &mut Inputs(inputs));
            let Outcome::Halted(payload) = outcome else { panic!("{outcome:?}") };
            assert_eq!(&payload[8..], &expected[..], "n {n} q {q}");
            assert!(u128::from(used) <= image.step_bound);
        }
    }

    #[test]
    fn test_committed_leaf_code_modules_are_current() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("lean/leaves/MxxLatticeLeaves/Code");
        for (file, source) in lean_code_modules() {
            let committed = std::fs::read_to_string(directory.join(&file)).unwrap_or_else(|e| {
                panic!("{file}: {e}; run `cargo run -p mxx-ir-core --example export_leaf_code`")
            });
            assert_eq!(
                committed, source,
                "{file} is stale; run `cargo run -p mxx-ir-core --example export_leaf_code`"
            );
        }
    }
}
