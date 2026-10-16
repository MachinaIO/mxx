//! The public streaming function an obfuscation evaluates (see `crates/io/DIO2-SPEC.md`, §3).
//!
//! A [`StreamingFunction`] has a state of `state_bits` bits, a chunk of `eval_chunk_bits` bits
//! the evaluator supplies and a chunk of `obf_chunk_bits` bits the obfuscator supplies at every
//! step, and at most `stream_length` steps. Step `i` is an Exponent-LUT Program body `Pi_i`
//! ([`StepProgram`]) that maps the state monomial `mono(z_i)`, the evaluator's bits and the
//! obfuscator's bits to `mono(z_(i+1))`:
//!
//! - the state is a private exponent wire of [`StreamingFunction::state_layout`];
//! - evaluator bit `b` enters as the public monomial `X^(x_monomial(b, x_b))`, the one-encoding
//!   multiplied by that monomial (so the step's public keys depend on the chunk `x_i`, which
//!   preprocessing knows);
//! - obfuscator bit `b` enters as RHS member `b`, the private monomial `X^(y_monomial(b, y_b))`;
//! - the body returns `mono(z_(i+1))` as the output of an exponent LUT of the state layout (a
//!   refresh target is a signed monomial times the payload secret, see the spec's §9).
//!
//! The function also gives every step's clear evaluation, against which an obfuscated evaluation
//! is checked, and the same step as a DSL graph on integers, the ideal functionality of the
//! correctness claim ([`super::claim`]).

use mxx_dsl::{DslError, Int};
use mxx_exponent_lut::{
    ExpLayout,
    program::{
        ExponentLutProgramBuilder, LutOutputRepresentation, LutTable, Operand,
        ProgramValidationError, RhsId, WireId,
    },
};
use mxx_ir_core::IntExpr;
use std::sync::Arc;

/// The wires a step body reads.
pub struct StepWires {
    /// The private state monomial `mono(z_i)`.
    pub state: WireId,
    /// The evaluator's bits: the public monomial of each bit.
    pub x: Vec<Operand<WireId>>,
    /// The obfuscator's bits: an RHS family of `obf_chunk_bits` private monomials.
    pub y: RhsId,
}

type BuildStep = dyn Fn(&mut ExponentLutProgramBuilder, &StepWires) -> Result<WireId, ProgramValidationError>
    + Send
    + Sync;
type ClearStep = dyn Fn(usize, &[bool], &[bool]) -> usize + Send + Sync;
type IdealStep = dyn Fn(Int, &[bool], &[Int]) -> Result<Int, DslError> + Send + Sync;

/// One step `Pi_i` of a streaming function: its Program body, its clear evaluation, and its
/// ideal evaluation.
#[derive(Clone)]
pub struct StepProgram {
    build: Arc<BuildStep>,
    clear: Arc<ClearStep>,
    ideal: Arc<IdealStep>,
}

impl StepProgram {
    /// The step whose body `build` appends to a builder (returning the exponent LUT output
    /// `mono(z_(i+1))`), whose clear evaluation is `clear(z_i, x_i, y_i) = z_(i+1)`, and whose
    /// ideal evaluation `ideal(z_i, x_i, y_i)` computes `z_(i+1)` in a DSL graph from the state and
    /// the obfuscator's bits as integers (the evaluator's bits are fixed).
    pub fn new(
        build: impl Fn(
            &mut ExponentLutProgramBuilder,
            &StepWires,
        ) -> Result<WireId, ProgramValidationError>
        + Send
        + Sync
        + 'static,
        clear: impl Fn(usize, &[bool], &[bool]) -> usize + Send + Sync + 'static,
        ideal: impl Fn(Int, &[bool], &[Int]) -> Result<Int, DslError> + Send + Sync + 'static,
    ) -> Self {
        Self { build: Arc::new(build), clear: Arc::new(clear), ideal: Arc::new(ideal) }
    }

    /// The table step `z_(i+1) = table[z_i + val(x_i) + val(y_i)]` of `function`, where
    /// `val(bits) = sum_b bits_b 2^b`: the state monomial is multiplied by the evaluator's public
    /// monomials and the obfuscator's private ones (the exponents add), and one LUT of the state
    /// layout reads the sum. The table has one entry per exponent of the state layout's window.
    pub fn table(function: &StreamingFunction, table: Vec<usize>) -> Self {
        let layout = function.state_layout;
        let state_bits = function.state_bits;
        let y_bits = function.obf_chunk_bits;
        let width = layout.width();
        let values = table.clone();
        let entries = table.iter().map(|entry| Int::constant(*entry as i64)).collect::<Vec<_>>();
        let value = |bits: &[bool]| {
            bits.iter().enumerate().map(|(bit, set)| usize::from(*set) << bit).sum::<usize>()
        };
        Self::new(
            move |builder, wires| {
                let mut state = wires.state;
                for monomial in &wires.x {
                    state = builder.mul_public(monomial.clone(), Operand::Scalar(state))?;
                }
                for bit in 0..y_bits {
                    state = builder.mul_private(
                        Operand::Scalar(state),
                        Operand::Member { family: wires.y, index: IntExpr::constant(bit) },
                    )?;
                }
                let lut = builder.register_lut(LutTable::unary(
                    width,
                    layout,
                    1 << state_bits,
                    values.clone(),
                    LutOutputRepresentation::Exponent(layout),
                )?)?;
                builder.lut(Operand::Scalar(state), lut)
            },
            move |z, x, y| table[z + value(x) + value(y)],
            move |z, x, y| {
                let index = y.iter().enumerate().fold(z.add(value(x) as i64), |index, (bit, y)| {
                    index.add(y.clone().mul(1i64 << bit))
                });
                mxx_dsl::select(index, entries.clone())
            },
        )
    }

    pub(crate) fn build(
        &self,
        builder: &mut ExponentLutProgramBuilder,
        wires: &StepWires,
    ) -> Result<WireId, ProgramValidationError> {
        (self.build)(builder, wires)
    }

    /// `z_(i+1)` of the ideal evaluation on the state `z`, the evaluator's bits `x` and the
    /// obfuscator's bits `y` (integers in `{0, 1}`).
    pub fn ideal(&self, z: Int, x: &[bool], y: &[Int]) -> Result<Int, DslError> {
        (self.ideal)(z, x, y)
    }

    /// `z_(i+1)` of the clear evaluation.
    pub fn clear(&self, z: usize, x: &[bool], y: &[bool]) -> usize {
        (self.clear)(z, x, y)
    }
}

/// A public streaming function (see the module documentation).
#[derive(Clone)]
pub struct StreamingFunction {
    pub state_bits: usize,
    pub eval_chunk_bits: usize,
    pub obf_chunk_bits: usize,
    /// The exponent layout of the state monomial `mono(z) = X^(z 2^stride_log2)`; its window has
    /// at least `2^state_bits` exponents.
    pub state_layout: ExpLayout,
    /// The exponent (in powers of `X`) of evaluator bit `b` set to `bit`.
    pub x_monomial: Arc<dyn Fn(usize, bool) -> usize + Send + Sync>,
    /// The exponent (in powers of `X`) of obfuscator bit `b` set to `bit`.
    pub y_monomial: Arc<dyn Fn(usize, bool) -> usize + Send + Sync>,
    steps: Vec<StepProgram>,
}

impl StreamingFunction {
    /// The function of `state_bits` state bits on `state_layout` with chunks of
    /// `eval_chunk_bits` and `obf_chunk_bits` bits whose bit `b` set to `bit` adds
    /// `bit 2^b` to the state exponent (the bits of [`StepProgram::table`]), without steps.
    pub fn additive(
        state_bits: usize,
        eval_chunk_bits: usize,
        obf_chunk_bits: usize,
        state_layout: ExpLayout,
    ) -> Self {
        let stride = state_layout.stride();
        let monomial =
            Arc::new(move |bit: usize, set: bool| usize::from(set) * (1usize << bit) * stride);
        Self {
            state_bits,
            eval_chunk_bits,
            obf_chunk_bits,
            state_layout,
            x_monomial: monomial.clone(),
            y_monomial: monomial,
            steps: Vec::new(),
        }
    }

    /// This function with `step` appended.
    pub fn with_step(mut self, step: StepProgram) -> Self {
        self.steps.push(step);
        self
    }

    /// The number of steps `stream_length`.
    pub fn stream_length(&self) -> usize {
        self.steps.len()
    }

    pub fn step(&self, index: usize) -> &StepProgram {
        &self.steps[index]
    }

    /// The final state of the clear evaluation from `z_0` on the chunks `x` and `y`.
    pub fn evaluate(&self, z_0: usize, x: &[Vec<bool>], y: &[Vec<bool>]) -> usize {
        self.steps.iter().zip(x).zip(y).fold(z_0, |z, ((step, x), y)| step.clear(z, x, y))
    }
}
