//! One streaming step of an AIR (algebraic intermediate representation)
//! constraint check, over one public attribute per field.
//!
//! An AIR declaration fixes `T >= 2` rows, `W >= 1` columns, a witness bound
//! `B >= 2` (every cell lies in `[0, B)`), `M >= 0` instance values, and a
//! list of constraints. A constraint is a predicate over a fixed list of
//! cells, each an instance value, a current-row value, or a next-row value,
//! applied by its gate: [`AirGate::FirstRow`] at step `i == 0` (the initial
//! condition on the current row), [`AirGate::LastRow`] at step `i == T - 2`
//! (the terminal condition on the next row), and [`AirGate::EveryRow`] at
//! every step (the transition polynomial `P(current, next) = 0`).
//!
//! [`AirConstraintProgram`] evaluates one step `i in [0, T - 1)`: from the
//! index `i`, the flag `f_i in {0, 1}`, the current row `r_i in Z_B^W`, the
//! next row `r_{i+1} in Z_B^W`, and the instance `x in Z_B^M`, it outputs the
//! next state `(i + 1, f_{i+1}, r_{i+1}, x)`, with `f_{i+1} = f_i` if every
//! constraint gated at `i` holds, else `0`. The state `(i, f_i, r_i, x)` has a
//! fixed size, so the AIR is checked by streaming the rows: starting from
//! `(0, 1, r_0, x)`, step `i` reads only `r_{i+1}`, and `f_{T-1} = 1` iff
//! every gated constraint holds at every step. `f_i` is the acceptance of the
//! completed steps `0..i-1`; the initial condition is first checked by step
//! `0`.
//!
//! One step proves only the one-step relation. The streaming driver must
//! feed each output state unchanged into the next step and start from index
//! `0` and `f_0 = 1`; a step cannot detect a substituted state or a reset
//! flag.
//!
//! # Packing
//!
//! The Program has no RHS input, and every computation after the product of
//! its inputs is a LUT. The bit fields of the step exponent `E`, least
//! significant first, are the index (`d_0` bits), the flag (one bit),
//! `current[0..W]`, `next[0..W]`, and `instance[0..M]` (`b` bits each, the
//! bits of `B - 1`); `P` is their total width, and any ring with `n >= 2^P`
//! works. Each of the `F = 2 W + M + 2` fields is its own public attribute
//! `X^(v 2^(p_f))`, the field value `v` at its position `p_f`. The Program
//! multiplies them by public products, which add the disjoint exponents, in
//! a balanced binary tree of depth `ceil(log2 F)`, the least depth of a
//! product of `F` factors by two-operand products: `prod_f X^(v_f 2^(p_f)) =
//! X^E`. Every LUT then reads a public payload (see
//! [`crate::program::LutOutputRepresentation`]), so both outputs are public.
//! The input contract is that each field value fits its field, that the
//! index lies in `[0, T - 1)`, the flag in `{0, 1}`, and every digit in
//! `[0, B)`; the Program does not check it.
//!
//! A LUT with input layout `(s, width_log2)` reads the exponent bit window
//! `[s, s + width_log2)` of a monomial with no bits below `s`, replaces it
//! with its output, and keeps the bits above (see
//! [`LutOutputRepresentation`]).
//!
//! # Flag chain
//!
//! The next flag is a streaming automaton over the fields, one LUT per
//! field. After field `k` its state is `(ok, gate bits, live values)`: the
//! gate bits `[i == 0]` and `[i == T - 2]` while a constraint of that gate is
//! unevaluated, and the values of the cells read so far that an unevaluated
//! constraint still references. Before LUT `k` the state code sits just below
//! field `k`, in its `sigma_k` bits `[p_k - sigma_k, p_k)`, with nothing
//! below, so LUT `k` reads the window `(state, field k)` and writes the next
//! state just below field `k + 1`. The index field is widened to
//! `d_0 >= max sigma_k`, which keeps every state inside the consumed bits.
//! LUT `0` initialises the gate bits; the flag LUT sets `ok = [f == 1]`; a
//! cell LUT clears `ok` for every active constraint whose last cell it reads
//! and whose predicate fails. The last LUT outputs `X^ok`. A table entry
//! outside the input contract (an index `>= T - 1`, a digit `>= B`, or an
//! unused state code) is never read; the tables fill it with `0`.
//!
//! # State outputs
//!
//! The next state is output in the form of the field inputs, one public
//! encoding per field but the next row ([`AirConstraintProgram::state_outputs`]),
//! so that it re-enters the next step as its state inputs
//! ([`AirConstraintProgram::state_inputs`]) with only the next row fresh:
//!
//! - the index `X^((i + 1) 2^(p_0))`, one LUT on the index input (the index field holds `T - 1`);
//! - the flag `X^(f_{i+1} 2^(p_1))`, the flag chain, whose last LUT writes `ok` at the flag field;
//! - the current row `X^(r_{i+1}[c] 2^(p_c))`, one LUT on each next-row input that moves its value
//!   to the current-row field `c`;
//! - the instance, its input unchanged.
//!
//! Each of these LUTs reads one field input, so its window has nothing below
//! or above it.

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        LutOutputRepresentation, LutTable, Operand, ProgramValidationError, WireId,
    },
};
use mxx_backends::poly::Poly;
use std::collections::{BTreeMap, BTreeSet};

/// Errors in an AIR declaration.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum AirError {
    #[error("invalid AIR dimensions or constraint cells, or an unrepresentable packing")]
    InvalidDeclaration,
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
}

/// A cell read by a constraint.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum AirCell {
    /// Instance value `x[m]`.
    Instance(usize),
    /// Current-row value `r_i[c]`.
    Current(usize),
    /// Next-row value `r_{i+1}[c]`.
    Next(usize),
}

/// The steps at which a constraint applies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirGate {
    /// Step `i == 0`: an initial condition.
    FirstRow,
    /// Every step: a transition constraint.
    EveryRow,
    /// Step `i == T - 2`: a terminal condition.
    LastRow,
}

/// A constraint predicate: it receives the values of the constraint's cells
/// in order and returns whether the constraint is satisfied.
pub type AirPredicate = Box<dyn Fn(&[usize]) -> bool>;

/// A constraint: `holds` over the values of `cells`, applied by `gate`.
/// `cells` must be nonempty.
pub struct AirConstraint {
    pub gate: AirGate,
    pub cells: Vec<AirCell>,
    pub holds: AirPredicate,
}

/// The Program of one streaming AIR step.
///
/// Its inputs are the public scalars `X^(v 2^(p_f))` of the fields (see the
/// module documentation for the packing); it multiplies them in a balanced
/// tree of public products and then uses LUTs only. Its outputs, in order,
/// are the public state `X^(2^p R)` and the public next flag `X^(f_{i+1})`.
#[derive(Clone, Debug)]
pub struct AirConstraintProgram {
    program: ExponentLutProgram,
    rows: usize,
    width: usize,
    bound: usize,
    instance_count: usize,
    /// The field positions `p_0, ..., p_K`; `p_K = P`.
    positions: Vec<u32>,
    /// The public input of every field, in field order.
    field_inputs: Vec<WireId>,
    /// The next state, one output per field but the next row, in field order.
    state_outputs: Vec<WireId>,
}

impl AirConstraintProgram {
    /// Builds the step Program of the AIR with `rows = T`, `width = W`,
    /// `bound = B`, `instance_count = M`, and `constraints`, whose
    /// predicates are consumed into the LUT tables.
    pub fn new(
        rows: usize,
        width: usize,
        bound: usize,
        instance_count: usize,
        constraints: Vec<AirConstraint>,
    ) -> Result<Self, AirError> {
        let invalid = || AirError::InvalidDeclaration;
        if rows < 2 || width == 0 || bound < 2 {
            return Err(invalid());
        }
        let fields = width
            .checked_mul(2)
            .and_then(|count| count.checked_add(2))
            .and_then(|count| count.checked_add(instance_count))
            .ok_or_else(invalid)?;
        let field_of = |cell: AirCell| match cell {
            AirCell::Current(c) => 2 + c,
            AirCell::Next(c) => 2 + width + c,
            AirCell::Instance(m) => 2 + 2 * width + m,
        };
        let cell_at = |field: usize| match field - 2 {
            c if c < width => AirCell::Current(c),
            c if c < 2 * width => AirCell::Next(c - width),
            m => AirCell::Instance(m - 2 * width),
        };
        // The field of each constraint's last cell, which also checks its cells.
        let lasts = constraints
            .iter()
            .map(|constraint| {
                let in_range = |cell: &AirCell| match *cell {
                    AirCell::Instance(m) => m < instance_count,
                    AirCell::Current(c) | AirCell::Next(c) => c < width,
                };
                if !constraint.cells.iter().all(in_range) {
                    return Err(invalid());
                }
                constraint.cells.iter().copied().map(field_of).max().ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The automaton state after each field but the last, which outputs
        // the flag.
        let shapes = (0..fields - 1)
            .map(|field| {
                let pending = || constraints.iter().zip(&lasts).filter(|(_, last)| **last > field);
                let gated = |gate| pending().any(|(constraint, _)| constraint.gate == gate);
                let cells = pending()
                    .flat_map(|(constraint, _)| constraint.cells.iter().copied())
                    .filter(|cell| field_of(*cell) <= field)
                    .collect::<BTreeSet<_>>();
                let (first, last) = (gated(AirGate::FirstRow), gated(AirGate::LastRow));
                let digits = u32::try_from(cells.len()).map_err(|_| invalid())?;
                let count = bound
                    .checked_pow(digits)
                    .and_then(|values| values.checked_mul(2 << (u8::from(first) + u8::from(last))))
                    .ok_or_else(invalid)?;
                Ok(StateShape { first, last, cells: cells.into_iter().collect(), count })
            })
            .collect::<Result<Vec<_>, AirError>>()?;
        // `sigma[k]` is the state width before field `k`.
        let sigma = std::iter::once(0)
            .chain(shapes.iter().map(|shape| bit_width(shape.count - 1)))
            .collect::<Vec<_>>();
        let digit_width = bit_width(bound - 1);
        // The index field also holds the next index `T - 1` of the last step.
        let index_width =
            sigma.iter().copied().chain([bit_width(rows - 1), 1]).max().expect("nonempty");
        let field_widths = [index_width, 1]
            .into_iter()
            .chain(std::iter::repeat_n(digit_width, fields - 2))
            .collect::<Vec<_>>();
        let mut positions = vec![0u32];
        for field_width in &field_widths {
            let end = positions.last().unwrap().checked_add(*field_width).ok_or_else(invalid)?;
            positions.push(end);
        }
        if *positions.last().unwrap() >= usize::BITS {
            return Err(invalid());
        }

        let mut builder = ExponentLutProgramBuilder::new();
        let field_inputs = (0..fields)
            .map(|_| builder.input(AttributeKnowledge::PublicPayload))
            .collect::<Result<Vec<_>, _>>()?;
        // `prod_f X^(v_f 2^(p_f)) = X^E` by a balanced tree: each level
        // multiplies adjacent pairs and carries an unpaired last factor, so
        // the depth is `ceil(log2 F)`.
        let mut layer = field_inputs.clone();
        while layer.len() > 1 {
            layer = layer
                .chunks(2)
                .map(|pair| match pair {
                    [lhs, rhs] => builder.mul_public(Operand::Scalar(*lhs), Operand::Scalar(*rhs)),
                    [factor] => Ok(*factor),
                    _ => unreachable!("chunks of two"),
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        let input = layer[0];
        let mut flag = input;
        for field in 0..fields {
            let (state_width, field_width) = (sigma[field], field_widths[field]);
            let input_layout = ExpLayout {
                stride_log2: positions[field] - state_width,
                width_log2: state_width + field_width,
            };
            let values = (0..1usize << input_layout.width_log2)
                .map(|window| {
                    let code = window & ((1 << state_width) - 1);
                    let digit = window >> state_width;
                    let outside = match field {
                        0 => digit >= rows - 1,
                        1 => code >= shapes[0].count,
                        _ => digit >= bound || code >= shapes[field - 1].count,
                    };
                    if outside {
                        return 0;
                    }
                    let mut state = match field {
                        0 => AutomatonState {
                            ok: true,
                            first: digit == 0,
                            last: digit == rows - 2,
                            values: BTreeMap::new(),
                        },
                        _ => shapes[field - 1].decode(code, bound),
                    };
                    if field == 1 {
                        state.ok &= digit == 1;
                    }
                    if field >= 2 {
                        state.values.insert(cell_at(field), digit);
                        for (constraint, _) in
                            constraints.iter().zip(&lasts).filter(|(_, last)| **last == field)
                        {
                            let active = match constraint.gate {
                                AirGate::FirstRow => state.first,
                                AirGate::EveryRow => true,
                                AirGate::LastRow => state.last,
                            };
                            let cells = constraint
                                .cells
                                .iter()
                                .map(|cell| state.values[cell])
                                .collect::<Vec<_>>();
                            state.ok &= !active || (constraint.holds)(&cells);
                        }
                    }
                    match shapes.get(field) {
                        Some(shape) => shape.encode(&state, bound),
                        None => usize::from(state.ok),
                    }
                })
                .collect();
            let (output_layout, output_width) = match shapes.get(field) {
                Some(shape) => (
                    ExpLayout {
                        stride_log2: positions[field + 1] - sigma[field + 1],
                        width_log2: sigma[field + 1],
                    },
                    shape.count,
                ),
                // The flag at the flag field, as the flag input.
                None => (ExpLayout { stride_log2: positions[1], width_log2: 1 }, 2),
            };
            let table = builder.register_lut(LutTable::unary(
                1 << input_layout.width_log2,
                input_layout,
                output_width,
                values,
                LutOutputRepresentation::Exponent(output_layout),
            )?)?;
            flag = builder.lut(Operand::Scalar(flag), table)?;
        }
        // The state outputs in the form of the field inputs: one LUT on the
        // index input, the flag chain, one LUT moving each next-row input to
        // its current-row field, and the instance inputs.
        let mut relocate = |input: WireId,
                            from: usize,
                            to: usize,
                            values: Vec<usize>|
         -> Result<WireId, AirError> {
            let width_log2 = field_widths[from];
            let table = builder.register_lut(LutTable::unary(
                1 << width_log2,
                ExpLayout { stride_log2: positions[from], width_log2 },
                1 << field_widths[to],
                values,
                LutOutputRepresentation::Exponent(ExpLayout {
                    stride_log2: positions[to],
                    width_log2: field_widths[to],
                }),
            )?)?;
            Ok(builder.lut(Operand::Scalar(input), table)?)
        };
        let index_values =
            (0..1usize << index_width).map(|i| if i < rows - 1 { i + 1 } else { 0 }).collect();
        let mut state_outputs = vec![relocate(field_inputs[0], 0, 0, index_values)?, flag];
        for c in 0..width {
            let values =
                (0..1usize << digit_width).map(|v| if v < bound { v } else { 0 }).collect();
            state_outputs.push(relocate(
                field_inputs[2 + width + c],
                2 + width + c,
                2 + c,
                values,
            )?);
        }
        state_outputs.extend_from_slice(&field_inputs[2 + 2 * width..]);
        let program = builder
            .finish(state_outputs.iter().map(|output| Operand::Scalar(*output)).collect())?;
        Ok(Self {
            program,
            rows,
            width,
            bound,
            instance_count,
            positions,
            field_inputs,
            state_outputs,
        })
    }

    /// The immutable Program.
    pub const fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    /// The row count `T`.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The column count `W`.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The witness bound `B`.
    pub const fn bound(&self) -> usize {
        self.bound
    }

    /// The instance count `M`.
    pub const fn instance_count(&self) -> usize {
        self.instance_count
    }

    /// The packed width `P`; the ring dimension must be at least `2^P`.
    pub fn packed_bits(&self) -> u32 {
        *self.positions.last().expect("positions are nonempty")
    }

    /// The public input `X^(v 2^(p_f))` of every field, in field order: the
    /// index, the flag, `current[0..W]`, `next[0..W]`, and `instance[0..M]`.
    pub fn field_inputs(&self) -> &[WireId] {
        &self.field_inputs
    }

    /// The inputs of the state fields, every field but the next row, in
    /// field order: the index, the flag, `current[0..W]`, and
    /// `instance[0..M]`. Output `j` of [`Self::state_outputs`] re-enters the
    /// next step as input `j` here.
    pub fn state_inputs(&self) -> Vec<WireId> {
        self.field_inputs[..2 + self.width]
            .iter()
            .chain(&self.field_inputs[2 + 2 * self.width..])
            .copied()
            .collect()
    }

    /// The inputs of the next-row fields `next[0..W]`, fresh at every step.
    pub fn next_inputs(&self) -> &[WireId] {
        &self.field_inputs[2 + self.width..2 + 2 * self.width]
    }

    /// The Program outputs: the next state in the form of
    /// [`Self::state_inputs`], `X^((i + 1) 2^(p_0))`, `X^(f_{i+1} 2^(p_1))`,
    /// the next row at the current-row fields, and the instance.
    pub fn state_outputs(&self) -> &[WireId] {
        &self.state_outputs
    }

    /// The next-flag output `X^(f_{i+1} 2^(p_1))`.
    pub fn next_flag(&self) -> WireId {
        self.state_outputs[1]
    }

    /// The exponents `v 2^(p_f)` of a step's field inputs, in the order of
    /// [`Self::field_inputs`], whose values follow the input contract, with
    /// `current.len() == next.len() == W` and `instance.len() == M`. Their
    /// sum is the step exponent `E`.
    pub fn pack(
        &self,
        index: usize,
        flag: usize,
        current: &[usize],
        next: &[usize],
        instance: &[usize],
    ) -> Vec<usize> {
        [index, flag]
            .into_iter()
            .chain(current.iter().chain(next).chain(instance).copied())
            .zip(&self.positions)
            .map(|(value, position)| value << position)
            .collect()
    }

    /// The field exponents of the next step's inputs, in the order of
    /// [`Self::field_inputs`], from the exponents of a step's
    /// [`Self::state_outputs`] and the next step's row `next`.
    pub fn next_step(&self, state: &[usize], next: &[usize]) -> Vec<usize> {
        let fields = self.pack(0, 0, &vec![0; self.width], next, &vec![0; self.instance_count]);
        state[..2 + self.width]
            .iter()
            .chain(&fields[2 + self.width..2 + 2 * self.width])
            .chain(&state[2 + self.width..])
            .copied()
            .collect()
    }

    /// The plaintext bindings `X^(v 2^(p_f))` of the field inputs for the
    /// exponents of [`Self::pack`], over a ring of dimension at least `2^P`.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        exponents: &[usize],
    ) -> PlaintextInputs<WireId, P> {
        self.field_inputs
            .iter()
            .zip(exponents)
            .map(|(input, exponent)| {
                (*input, InputBinding::Scalar(P::const_rotate_poly(parameters, *exponent)))
            })
            .collect()
    }
}

/// The automaton state: `ok`, the gate bits, and the live cell values.
struct AutomatonState {
    ok: bool,
    first: bool,
    last: bool,
    values: BTreeMap<AirCell, usize>,
}

/// The live gates and cells of the state after one field, and its code
/// count `2^(1 + gates) B^cells`.
struct StateShape {
    first: bool,
    last: bool,
    cells: Vec<AirCell>,
    count: usize,
}

impl StateShape {
    /// The mixed-radix code `ok + 2 (gate bits) + 2^(1 + gates) (cells in base B)`.
    fn encode(&self, state: &AutomatonState, bound: usize) -> usize {
        let cells = self.cells.iter().rev().fold(0, |code, cell| code * bound + state.values[cell]);
        let gates = [(self.last, state.last), (self.first, state.first)]
            .into_iter()
            .filter(|(live, _)| *live)
            .fold(cells, |code, (_, bit)| 2 * code + usize::from(bit));
        2 * gates + usize::from(state.ok)
    }

    fn decode(&self, code: usize, bound: usize) -> AutomatonState {
        let mut rest = code;
        let mut bit = |live: bool| {
            let value = live && rest % 2 == 1;
            if live {
                rest /= 2;
            }
            value
        };
        let (ok, first, last) = (bit(true), bit(self.first), bit(self.last));
        let values = self
            .cells
            .iter()
            .map(|cell| {
                let value = rest % bound;
                rest /= bound;
                (*cell, value)
            })
            .collect();
        AutomatonState { ok, first, last, values }
    }
}

/// The number of bits that write every value in `0..=max`.
const fn bit_width(max: usize) -> u32 {
    usize::BITS - max.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        program::{
            Environment, ExponentLUTProgramLowering, FamilyExecution, FamilyRange, InputShape,
            LoweringSite, ProgramLowerError, ProgramLoweringContext, lower_program,
        },
        rns_bgg::{RnsBggDerivedWorkload, RnsBggParameters, RnsOutputUse, verify_output},
    };
    use mxx_backends::{
        MemoryArtifactStore,
        backend::poly::cpu_backend,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_ir_core::IntExpr;
    use num_bigint::{BigInt, BigUint};

    #[derive(Debug, thiserror::Error)]
    #[error("clear AIR evaluator error: {0}")]
    struct ClearError(&'static str);

    /// Evaluates the Program on monomials in the clear: a wire is its
    /// exponent and its depth of public products, a public multiplication
    /// adds exponents (here of disjoint fields), and a LUT applies the window
    /// semantics of [`LutOutputRepresentation`], asserting its preconditions.
    struct Clear {
        site: Option<LoweringSite>,
        packed_bits: u32,
        luts: usize,
        multiplications: usize,
    }

    /// An exponent and its depth of public products.
    type Monomial = (usize, usize);

    impl ExponentLUTProgramLowering for Clear {
        type Wire = Monomial;
        type Rhs = ();
        type Error = ClearError;

        fn add(&mut self, _: &Monomial, _: &Monomial) -> Result<Monomial, ClearError> {
            Err(ClearError("the AIR step has no additions"))
        }
        fn mul_public(&mut self, lhs: &Monomial, rhs: &Monomial) -> Result<Monomial, ClearError> {
            assert_eq!(lhs.0 & rhs.0, 0, "the multiplied exponents share a field");
            assert!(lhs.0 + rhs.0 < 1 << self.packed_bits, "the product leaves the packing");
            self.multiplications += 1;
            Ok((lhs.0 + rhs.0, lhs.1.max(rhs.1) + 1))
        }
        fn mul_private(&mut self, _: &Monomial, _: &()) -> Result<Monomial, ClearError> {
            Err(ClearError("the AIR step has no private multiplications"))
        }
        fn lut(&mut self, input: &Monomial, table: &LutTable) -> Result<Monomial, ClearError> {
            let (input, depth) = *input;
            let ExpLayout { stride_log2, width_log2 } = table.input_layout();
            let window = stride_log2 + width_log2;
            assert_eq!(input & ((1 << stride_log2) - 1), 0, "exponent bits below the window");
            let entry = (input >> stride_log2) & ((1 << width_log2) - 1);
            let LutOutputRepresentation::Exponent(output) = table.output_representation() else {
                return Err(ClearError("coefficient output"));
            };
            let replacement =
                table.values().ok_or(ClearError("withheld values"))?[entry] << output.stride_log2;
            assert!(replacement < 1 << window, "the output carries into the kept bits");
            let result = replacement + ((input >> window) << window);
            assert!(result < 1 << self.packed_bits, "exponent {result} leaves the packing");
            self.luts += 1;
            Ok((result, depth))
        }
        fn mul_small_constant(
            &mut self,
            _: &Monomial,
            _: &[i64],
            _: &IntExpr,
        ) -> Result<Monomial, ClearError> {
            Err(ClearError("the AIR step has no small constants"))
        }
    }

    impl ProgramLoweringContext for Clear {
        fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
            std::mem::replace(&mut self.site, site)
        }
    }

    struct NoFamilies;

    impl FamilyExecution<Clear> for NoFamilies {
        type WireFamily = ();
        type RhsFamily = ();

        fn wire_at(
            &mut self,
            _: &mut Clear,
            _: &(),
            _: &IntExpr,
        ) -> Result<Monomial, ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no families")))
        }
        fn rhs_at(&mut self, _: &(), _: &IntExpr) -> Result<(), ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no RHS")))
        }
        fn rhs_selected(
            &mut self,
            _: &(),
            _: crate::program::DataId,
            _: &IntExpr,
        ) -> Result<(), ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no RHS")))
        }
        fn parallel(
            &mut self,
            _: &mut Clear,
            _: &FamilyRange,
            _: impl FnOnce(
                &mut Self,
                &mut Clear,
                IntExpr,
            ) -> Result<Monomial, ProgramLowerError<ClearError>>,
        ) -> Result<(), ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no families")))
        }
        fn sum(
            &mut self,
            _: &mut Clear,
            _: &(),
        ) -> Result<Monomial, ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no sums")))
        }
    }

    /// Evaluates one step on the field inputs `X^(v 2^(p_f))` of
    /// `exponents`: the exponents of the state outputs, and the numbers of
    /// LUTs and public multiplications evaluated with the product depth of
    /// the flag.
    fn step(air: &AirConstraintProgram, exponents: Vec<usize>) -> (Vec<usize>, [usize; 3]) {
        let mut environment = Environment::new();
        assert_eq!(exponents.len(), air.field_inputs().len());
        for (input, exponent) in air.field_inputs().iter().zip(exponents) {
            environment.bind_wire(*input, InputBinding::Scalar((exponent, 0))).unwrap();
        }
        let mut clear =
            Clear { site: None, packed_bits: air.packed_bits(), luts: 0, multiplications: 0 };
        let outputs =
            lower_program(air.program(), &mut clear, &mut NoFamilies, &mut environment).unwrap();
        let state = outputs.iter().map(|(exponent, _)| *exponent).collect();
        (state, [clear.luts, clear.multiplications, outputs[1].1])
    }

    /// The state fields of `fields` (every field but the next row).
    fn state_fields(air: &AirConstraintProgram, fields: &[usize]) -> Vec<usize> {
        fields[..2 + air.width].iter().chain(&fields[2 + 2 * air.width..]).copied().collect()
    }

    /// The flag held by a flag field exponent.
    fn flag_of(air: &AirConstraintProgram, exponent: usize) -> usize {
        exponent >> air.positions[1]
    }

    /// The next flag of one step, after checking that the state outputs are
    /// the field inputs of `(i + 1, f_{i+1}, next, instance)`.
    fn next_flag(
        air: &AirConstraintProgram,
        index: usize,
        flag: usize,
        current: &[usize],
        next: &[usize],
        instance: &[usize],
    ) -> usize {
        let (state, _) = step(air, air.pack(index, flag, current, next, instance));
        let flag = flag_of(air, state[1]);
        assert!(flag <= 1, "flag {flag}");
        let zeros = vec![0; air.width];
        assert_eq!(state, state_fields(air, &air.pack(index + 1, flag, next, &zeros, instance)));
        flag
    }

    /// Streams `rows` through the Program from `(0, flag, rows[0], instance)`,
    /// feeding each step's state outputs into the next step with the next
    /// row, and returns the flag after every step.
    fn stream(
        air: &AirConstraintProgram,
        instance: &[usize],
        rows: &[Vec<usize>],
        flag: usize,
    ) -> Vec<usize> {
        assert_eq!(rows.len(), air.rows());
        let zeros = vec![0; air.width];
        let mut state = state_fields(air, &air.pack(0, flag, &rows[0], &zeros, instance));
        let mut flags = Vec::new();
        for (index, next) in rows[1..].iter().enumerate() {
            (state, _) = step(air, air.next_step(&state, next));
            assert_eq!(
                state,
                state_fields(
                    air,
                    &air.pack(index + 1, flag_of(air, state[1]), next, &zeros, instance)
                )
            );
            flags.push(flag_of(air, state[1]));
        }
        // Once a step rejects, every later step rejects.
        assert!(flags.windows(2).all(|pair| pair[1] <= pair[0]), "flags {flags:?}");
        flags
    }

    fn final_flag(
        air: &AirConstraintProgram,
        instance: &[usize],
        rows: &[Vec<usize>],
        flag: usize,
    ) -> usize {
        *stream(air, instance, rows, flag).last().unwrap()
    }

    fn constraint(
        gate: AirGate,
        cells: Vec<AirCell>,
        holds: impl Fn(&[usize]) -> bool + 'static,
    ) -> AirConstraint {
        AirConstraint { gate, cells, holds: Box::new(holds) }
    }

    /// `B = 3`, a bound that is not a power of two.
    const B: usize = 3;
    const ROW_COUNTS: [usize; 3] = [2, 3, 5];

    /// Row `i` is `(F_{2i}, F_{2i+1}) mod B` and the instance is
    /// `y = F_{2T-1} mod B`.
    fn fibonacci(rows: usize) -> AirConstraintProgram {
        use AirCell::{Current, Instance, Next};
        let constraints = vec![
            constraint(AirGate::FirstRow, vec![Current(0)], |v| v[0] == 0),
            constraint(AirGate::FirstRow, vec![Current(1)], |v| v[0] == 1),
            constraint(AirGate::EveryRow, vec![Next(0), Current(0), Current(1)], |v| {
                v[0] == (v[1] + v[2]) % B
            }),
            constraint(AirGate::EveryRow, vec![Next(1), Next(0), Current(1)], |v| {
                v[0] == (v[1] + v[2]) % B
            }),
            constraint(AirGate::LastRow, vec![Next(1), Instance(0)], |v| v[0] == v[1]),
        ];
        AirConstraintProgram::new(rows, 2, B, 1, constraints).unwrap()
    }

    /// The rows from `start` under the Fibonacci recurrence.
    fn fibonacci_trace(rows: usize, start: [usize; 2]) -> Vec<Vec<usize>> {
        std::iter::successors(Some(start), |[a, c]| {
            let next = (a + c) % B;
            Some([next, (next + c) % B])
        })
        .take(rows)
        .map(Vec::from)
        .collect()
    }

    #[test]
    fn fibonacci_packing_and_structure() {
        let air = fibonacci(5);
        assert_eq!(air.packed_bits(), 17);
        // One public input per field: index, flag, two current, two next,
        // and one instance value.
        let inputs = air.program().inputs();
        assert_eq!(inputs.len(), 7);
        assert_eq!(air.field_inputs().len(), 7);
        for input in air.field_inputs() {
            assert_eq!(inputs[input].shape(), &InputShape::Scalar(()));
            assert_eq!(inputs[input].knowledge(), AttributeKnowledge::PublicPayload);
        }
        assert_eq!(air.state_outputs().len(), 5);
        assert_eq!(air.state_inputs().len(), 5);
        for output in air.state_outputs() {
            assert_eq!(
                air.program().wire_knowledge(*output).unwrap(),
                AttributeKnowledge::PublicPayload
            );
        }
        assert!(air.program().rhs_inputs().is_empty());
        // Flag chain: windows `2^(sigma_k + d_k)` for sigma = (0, 3, 3, 5, 6,
        // 6, 4) and d = (6, 1, 2, ...); state outputs: the index and the two
        // next-row fields moved to the current row.
        let widths = air
            .program()
            .encryption_inputs()
            .luts()
            .iter()
            .map(LutTable::input_width)
            .collect::<Vec<_>>();
        assert_eq!(widths, [64, 16, 32, 128, 256, 256, 64, 64, 4, 4]);
        // Six public multiplications of depth `ceil(log2 7) = 3`, then LUTs
        // only: the clear evaluator rejects any other step.
        let inputs = air.pack(0, 1, &[0, 1], &[1, 2], &[0]);
        assert_eq!(step(&air, inputs).1, [widths.len(), 6, 3]);
    }

    #[test]
    fn fibonacci_accepts_the_honest_trace() {
        assert_eq!(
            fibonacci_trace(4, [0, 1]),
            vec![vec![0, 1], vec![1, 2], vec![0, 2], vec![2, 1]]
        );
        for rows in ROW_COUNTS {
            let air = fibonacci(rows);
            let trace = fibonacci_trace(rows, [0, 1]);
            let y = trace[rows - 1][1];
            assert_eq!(stream(&air, &[y], &trace, 1), vec![1; rows - 1], "T = {rows}");
            assert_eq!(final_flag(&air, &[y], &trace, 0), 0);
        }
    }

    #[test]
    fn fibonacci_rejects_every_single_cell_tamper_and_wrong_instance() {
        for rows in ROW_COUNTS {
            let air = fibonacci(rows);
            let trace = fibonacci_trace(rows, [0, 1]);
            let y = trace[rows - 1][1];
            for row in 0..rows {
                for column in 0..2 {
                    for value in (0..B).filter(|v| *v != trace[row][column]) {
                        let mut tampered = trace.clone();
                        tampered[row][column] = value;
                        assert_eq!(
                            final_flag(&air, &[y], &tampered, 1),
                            0,
                            "T = {rows}, row {row}, column {column}, value {value}"
                        );
                    }
                }
            }
            for wrong in (0..B).filter(|v| *v != y) {
                assert_eq!(final_flag(&air, &[wrong], &trace, 1), 0, "T = {rows}, y = {wrong}");
            }
        }
    }

    #[test]
    fn fibonacci_enforces_the_initial_row_independently() {
        // Every transition holds from the wrong start (1, 1), and the
        // instance is its own terminal value.
        for rows in ROW_COUNTS {
            let air = fibonacci(rows);
            let trace = fibonacci_trace(rows, [1, 1]);
            let y = trace[rows - 1][1];
            assert_eq!(stream(&air, &[y], &trace, 1), vec![0; rows - 1], "T = {rows}");
        }
    }

    #[test]
    fn gates_apply_exactly_at_their_index_without_an_instance() {
        for rows in [2, 4] {
            let declare = |gate, cell, expected| {
                let holds = move |v: &[usize]| v[0] == expected;
                AirConstraintProgram::new(rows, 1, B, 0, vec![constraint(gate, vec![cell], holds)])
                    .unwrap()
            };
            let first = declare(AirGate::FirstRow, AirCell::Current(0), 0);
            let last = declare(AirGate::LastRow, AirCell::Next(0), 2);
            for index in 0..rows - 1 {
                // Row values violating both gated constraints.
                let violating = |air| next_flag(air, index, 1, &[1], &[0], &[]);
                assert_eq!(violating(&first), usize::from(index != 0), "T = {rows}, i = {index}");
                assert_eq!(
                    violating(&last),
                    usize::from(index != rows - 2),
                    "T = {rows}, i = {index}"
                );
                assert_eq!(next_flag(&first, index, 1, &[0], &[0], &[]), 1);
                assert_eq!(next_flag(&last, index, 1, &[1], &[2], &[]), 1);
            }
        }
    }

    #[test]
    fn one_step_matches_the_relation_on_every_contract_input() {
        // T = 3, W = 1, M = 1: fields index (7 bits), flag, current, next,
        // instance (2 bits each), so P = 14.
        let rows = 3;
        let constraints = vec![
            constraint(AirGate::FirstRow, vec![AirCell::Current(0), AirCell::Instance(0)], |v| {
                v[0] == v[1]
            }),
            constraint(AirGate::EveryRow, vec![AirCell::Next(0), AirCell::Current(0)], |v| {
                v[0] == (v[1] + 1) % B
            }),
            constraint(AirGate::LastRow, vec![AirCell::Next(0), AirCell::Instance(0)], |v| {
                v[0] == (v[1] + 2) % B
            }),
        ];
        let air = AirConstraintProgram::new(rows, 1, B, 1, constraints).unwrap();
        assert_eq!(air.packed_bits(), 14);
        let mut accepted = 0;
        let contract = (0..rows - 1).flat_map(|index| {
            (0..2).flat_map(move |flag| {
                (0..B).flat_map(move |current| {
                    (0..B)
                        .flat_map(move |next| (0..B).map(move |x| (index, flag, current, next, x)))
                })
            })
        });
        for (index, flag, current, next, x) in contract {
            let fields = vec![index, flag << 7, current << 8, next << 10, x << 12];
            let exponent = fields.iter().sum::<usize>();
            let expected = flag == 1 &&
                (index != 0 || current == x) &&
                next == (current + 1) % B &&
                (index != rows - 2 || next == (x + 2) % B);
            assert_eq!(air.pack(index, flag, &[current], &[next], &[x]), fields);
            let (state, _) = step(&air, fields);
            let next_flag = flag_of(&air, state[1]);
            assert_eq!(state, [index + 1, next_flag << 7, next << 8, x << 12], "E = {exponent}");
            assert_eq!(next_flag, usize::from(expected), "E = {exponent}");
            accepted += next_flag;
        }
        // Step 0 accepts (x, x, x + 1); step 1 accepts (x, c, c + 1 = x + 2).
        assert_eq!(accepted, 2 * B);
    }

    #[test]
    fn packing_places_every_field_and_the_next_step_reuses_the_state() {
        let air = fibonacci(3);
        let fields = air.pack(1, 1, &[1, 2], &[0, 2], &[1]);
        assert_eq!(fields, [1, 1 << 6, 1 << 7, 2 << 9, 0, 2 << 13, 1 << 15]);
        let state = [2, 1 << 6, 0, 2 << 9, 1 << 15];
        assert_eq!(air.next_step(&state, &[1, 0]), air.pack(2, 1, &[0, 2], &[1, 0], &[1]));
        assert_eq!(
            air.state_inputs(),
            [0, 1, 2, 3, 6].map(|field| air.field_inputs()[field]).to_vec()
        );
        assert_eq!(air.next_inputs(), &air.field_inputs()[4..6]);
    }

    /// The product of `F = 2 W + M + 2` field inputs uses `F - 1` public
    /// multiplications of depth `ceil(log2 F)`, the least for two-operand
    /// products, and the step still evaluates its relation.
    #[test]
    fn field_inputs_multiply_in_a_tree_of_least_depth() {
        for (width, instances) in [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 2)] {
            let constraints = vec![constraint(
                AirGate::EveryRow,
                vec![AirCell::Next(0), AirCell::Current(0)],
                |v| v[0] == v[1],
            )];
            let air = AirConstraintProgram::new(2, width, 2, instances, constraints).unwrap();
            let fields = 2 * width + instances + 2;
            let depth = fields.next_power_of_two().trailing_zeros() as usize;
            let row = vec![1; width];
            let instance = vec![0; instances];
            let (state, [_, multiplications, product_depth]) =
                step(&air, air.pack(0, 1, &row, &row, &instance));
            let flag = flag_of(&air, state[1]);
            assert_eq!(
                (flag, multiplications, product_depth),
                (1, fields - 1, depth),
                "F = {fields}"
            );
        }
    }

    #[test]
    fn plaintext_inputs_are_the_field_monomials() {
        // P = 5: index (2 bits), flag, current, next (bit 4).
        let constraints =
            vec![constraint(AirGate::EveryRow, vec![AirCell::Next(0), AirCell::Current(0)], |v| {
                v[0] == v[1]
            })];
        let air = AirConstraintProgram::new(2, 1, 2, 0, constraints).unwrap();
        assert_eq!(air.packed_bits(), 5);
        let ring = DCRTPolyParams::new(32, 1, 17, 1, None, None);
        let fields = air.pack(0, 1, &[1], &[1], &[]);
        assert_eq!(fields, [0, 0b100, 0b1000, 0b10000]);
        let monomial =
            |exponent| InputBinding::Scalar(DCRTPoly::const_rotate_poly(&ring, exponent));
        assert_eq!(
            air.plaintext_inputs::<DCRTPoly>(&ring, &fields),
            air.field_inputs()
                .iter()
                .zip(&fields)
                .map(|(input, exponent)| (*input, monomial(*exponent)))
                .collect::<BTreeMap<_, _>>()
        );
        assert_eq!(next_flag(&air, 0, 1, &[1], &[1], &[]), 1);
        assert_eq!(next_flag(&air, 0, 1, &[1], &[0], &[]), 0);
    }

    /// One step of an AIR whose fields are all public attributes, evaluated
    /// encrypted in Static mode: the field inputs multiply in a tree of
    /// public products and every LUT reads a public payload, one level each.
    /// The public next flag verifies as `X^(2^(p_1))` (the flag field holding
    /// one) on a satisfying step and as `X^0` on a violating one, and the
    /// simulated bound of the flag fits its level.
    #[test]
    #[ignore = "the encrypted CPU step takes about fifteen minutes; run with --ignored"]
    fn public_air_step_verifies_its_next_flag() {
        let air = AirConstraintProgram::new(
            2,
            1,
            2,
            0,
            vec![AirConstraint {
                gate: AirGate::EveryRow,
                cells: vec![AirCell::Next(0), AirCell::Current(0)],
                holds: Box::new(|v: &[usize]| v[0] == v[1]),
            }],
        )
        .unwrap();
        let program = air.program();
        // The flag chain has one LUT per field, each dropping one prime; the
        // terminal level keeps two. The gadget base bits are overridable by
        // `MXX_EXPONENT_LUT_PUBLIC_AIR_TEST_BASE_BITS`.
        let base_bits = std::env::var("MXX_EXPONENT_LUT_PUBLIC_AIR_TEST_BASE_BITS")
            .map_or(4, |bits| bits.parse().unwrap());
        let levels = air.field_inputs().len() + 1;
        let pool =
            DCRTPolyParams::new(1 << air.packed_bits(), levels + 1, 17, base_bits, None, None);
        let primes = pool.to_crt().0;
        let parameters = RnsBggParameters::new(
            (0..levels)
                .map(|level| {
                    let modulus = primes[..levels + 1 - level]
                        .iter()
                        .fold(BigUint::from(1u8), |product, prime| product * BigUint::from(*prime));
                    pool.select_modulus(&modulus).unwrap()
                })
                .collect(),
            1,
            1,
        )
        .unwrap();
        let workload = RnsBggDerivedWorkload::from_program(program).unwrap();
        assert_eq!(workload.level(air.next_flag()).unwrap(), levels - 1);
        let bindings = crate::rns_bgg::RnsBggNoiseBindings::from_inputs(
            program,
            &air.field_inputs()
                .iter()
                .map(|_| {
                    crate::rns_bgg::RnsBggNoiseValue::sampled_encoding(
                        &parameters,
                        b"public-air-field",
                        &[BigInt::from(1u8)],
                    )
                })
                .collect::<Vec<_>>(),
            &[],
        )
        .unwrap();
        let report = crate::api::noise_simulate(
            program,
            crate::rns_bgg::RnsNoiseBuild::new(&parameters, program, &workload, &bindings).unwrap(),
            |_, outputs| Ok(outputs),
        )
        .unwrap();
        let bound = &report.outputs[1].bound.error;
        let modulus = parameters.level(levels - 1).unwrap().modulus();
        assert!(bound * 4u8 < *modulus.as_ref(), "the flag bound {bound} exceeds q / 4");
        let terminal = &parameters.levels()[levels - 1];
        for (next, accepted) in [(1, true), (0, false)] {
            let fields = air.pack(0, 1, &[1], &[next], &[]);
            let lhs = air.plaintext_inputs::<DCRTPoly>(&parameters.levels()[0], &fields);
            let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
                parameters.levels().iter().cloned(),
            ));
            let mut store = MemoryArtifactStore::default();
            let (result, payload, mask) = crate::test_support::run_without_rhs(
                &parameters,
                program,
                &crate::program::ProgramData::new(),
                false,
                &lhs,
                &mut execution,
                &mut store,
                |_| (),
            );
            let verify = |flag: usize, execution: &mut _, store: &mut _| {
                verify_output(
                    program,
                    &result,
                    1,
                    &DCRTPoly::const_rotate_poly(terminal, air.pack(0, flag, &[0], &[0], &[])[1]),
                    &parameters,
                    &payload,
                    Some(&mask),
                    RnsOutputUse::DirectDecode,
                    execution,
                    store,
                )
            };
            let flag = usize::from(accepted);
            verify(flag, &mut execution, &mut store)
                .unwrap_or_else(|error| panic!("next row {next}: {error}"));
            assert!(verify(1 - flag, &mut execution, &mut store).is_err(), "next row {next}");
        }
    }

    #[test]
    fn rejects_invalid_declarations() {
        let declare = |rows, width, bound, instances, cells: Vec<AirCell>| {
            let constraints = vec![constraint(AirGate::EveryRow, cells, |_| true)];
            AirConstraintProgram::new(rows, width, bound, instances, constraints).map(|_| ())
        };
        let next = || vec![AirCell::Next(0)];
        assert_eq!(declare(2, 1, 2, 0, next()), Ok(()));
        assert!(AirConstraintProgram::new(2, 1, 2, 0, vec![]).is_ok());
        let invalid = Err(AirError::InvalidDeclaration);
        assert_eq!(declare(1, 1, 2, 0, next()), invalid);
        assert_eq!(declare(2, 0, 2, 0, vec![]), invalid);
        assert_eq!(declare(2, 1, 1, 0, next()), invalid);
        assert_eq!(declare(2, 1, 2, 0, vec![]), invalid);
        assert_eq!(declare(2, 1, 2, 0, vec![AirCell::Current(1)]), invalid);
        assert_eq!(declare(2, 1, 2, 0, vec![AirCell::Instance(0)]), invalid);
        // The packing exceeds the exponent width, or a state count overflows.
        assert_eq!(declare(2, 1, 1 << 40, 0, next()), invalid);
        let wide = (0..40).map(AirCell::Current).chain([AirCell::Next(0)]).collect();
        assert_eq!(declare(2, 40, 3, 0, wide), invalid);
    }
}
