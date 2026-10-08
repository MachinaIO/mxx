//! One streaming step of an AIR (algebraic intermediate representation)
//! constraint check, over a packed private state and a public next row.
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
//! next row `r_{i+1} in Z_B^W`, and the instance `x in Z_B^M`, it outputs
//! `(r_{i+1}, x)` and `f_{i+1} = f_i` if every constraint gated at `i` holds,
//! else `0`. The state `(x, r_i, f_i)` has a fixed size, so the AIR is checked
//! by streaming the rows: starting from `(x, r_0, 1)`, step `i` reads only
//! `r_{i+1}`, and `f_{T-1} = 1` iff every gated constraint holds at every
//! step. `f_i` is the acceptance of the completed steps `0..i-1`; the initial
//! condition is first checked by step `0`.
//!
//! One step proves only the one-step relation. The streaming driver must
//! supply the indices `0` through `T - 2` in order, carry each output state
//! unchanged into the next step, and start from `f_0 = 1`; a step cannot
//! detect a skipped step, a substituted state, or a reset flag.
//!
//! # Packing
//!
//! The Program has no RHS input, so the private state is one packed
//! attribute, and every computation on it is a LUT. The bit fields of the
//! step exponent `E`, least significant first, are the index (`d_0` bits),
//! the flag (one bit), `current[0..W]`, `next[0..W]`, and `instance[0..M]`
//! (`b` bits each, the bits of `B - 1`); `P` is their total width, and any
//! ring with `n >= 2^P` works. The private attribute `X^(E_state)` holds
//! every field but the next row, whose fields are zero; the public attribute
//! `X^(E_next)` holds only the next-row fields. The Program multiplies them,
//! which adds the disjoint exponents: `X^(E_next) X^(E_state) = X^E`. The
//! input contract is that both exponents only use their own fields below
//! `2^P`; within it every bit pattern is handled: a digit `>= B` or an index
//! `>= T - 1` is rejected.
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
//! `d_0 >= max sigma_k`, which keeps every state inside the consumed bits;
//! its extra bits only encode rejected indices. LUT `0` initialises
//! `ok = [i < T - 1]` and the gate bits; the flag LUT clears `ok` unless
//! `f == 1`; a cell LUT clears `ok` for a digit `>= B` (keeping `0` as its
//! value) and for every active constraint whose last cell it reads and whose
//! predicate fails. The last LUT outputs `X^ok`.
//!
//! # State chain
//!
//! A LUT cannot leave bits below its window, so the flag chain cannot
//! forward fields. A second chain on the same product deletes the index, flag,
//! and current-row fields with all-zero tables (a zero exponent is
//! `X^0 = 1`), leaving `X^(2^p R)`, where `p` is the position of `next[0]`
//! and `R` packs `(r_{i+1}, x)`.
//!
//! The Program outputs are `[state, next_flag]`; the driver unpacks them
//! ([`AirConstraintProgram::unpack_state`]) and packs the next step's inputs
//! ([`AirConstraintProgram::pack`]).

use crate::{
    api::PlaintextInputs,
    program::{
        AttributeKnowledge, ExpLayout, ExponentLutProgram, ExponentLutProgramBuilder, InputBinding,
        LutOutputRepresentation, LutTable, Operand, ProgramValidationError, WireId,
    },
};
use mxx_backends::poly::{Poly, PolyParams};
use std::collections::{BTreeMap, BTreeSet};

/// Errors in an AIR declaration or in a packed value.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum AirError {
    #[error("invalid AIR dimensions or constraint cells, or an unrepresentable packing")]
    InvalidDeclaration,
    #[error("a value does not fit its packed field, or a packed exponent or ring is malformed")]
    InvalidValue,
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
pub struct AirConstraint {
    pub gate: AirGate,
    pub cells: Vec<AirCell>,
    pub holds: AirPredicate,
}

/// The Program of one streaming AIR step.
///
/// Its inputs are the private scalar `X^(E_state)` and the public scalar
/// `X^(E_next)` (see the module documentation for the packing); it
/// multiplies them once and then uses LUTs only. Its outputs, in order, are
/// the state `X^(2^p R)` and the next flag `X^(f_{i+1})`.
#[derive(Clone, Debug)]
pub struct AirConstraintProgram {
    program: ExponentLutProgram,
    rows: usize,
    width: usize,
    bound: usize,
    instance_count: usize,
    /// The field positions `p_0, ..., p_K`; `p_K = P`.
    positions: Vec<u32>,
    private_state: WireId,
    public_next: WireId,
    state: WireId,
    next_flag: WireId,
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
        let index_width =
            sigma.iter().copied().chain([bit_width(rows - 2), 1]).max().expect("nonempty");
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
        let private_state = builder.input(AttributeKnowledge::PrivatePayload)?;
        let public_next = builder.input(AttributeKnowledge::PublicPayload)?;
        // `X^(E_next) X^(E_state) = X^E`, a private wire.
        let input =
            builder.mul_public(Operand::Scalar(public_next), Operand::Scalar(private_state))?;
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
                    let mut state = match field {
                        0 => AutomatonState {
                            ok: digit < rows - 1,
                            first: digit == 0,
                            last: digit == rows - 2,
                            values: BTreeMap::new(),
                        },
                        _ if code < shapes[field - 1].count => {
                            shapes[field - 1].decode(code, bound)
                        }
                        _ => AutomatonState::default(),
                    };
                    if field == 1 {
                        state.ok &= digit == 1;
                    }
                    if field >= 2 {
                        let in_range = digit < bound;
                        state.ok &= in_range;
                        state.values.insert(cell_at(field), if in_range { digit } else { 0 });
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
                                .map(|cell| state.value(*cell))
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
                None => (ExpLayout { stride_log2: 0, width_log2: 1 }, 2),
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
        let mut state = input;
        for field in 0..2 + width {
            let field_width = field_widths[field];
            let table = builder.register_lut(LutTable::unary(
                1 << field_width,
                ExpLayout { stride_log2: positions[field], width_log2: field_width },
                1,
                vec![0; 1 << field_width],
                LutOutputRepresentation::Exponent(ExpLayout { stride_log2: 0, width_log2: 1 }),
            )?)?;
            state = builder.lut(Operand::Scalar(state), table)?;
        }
        let program = builder.finish(vec![Operand::Scalar(state), Operand::Scalar(flag)])?;
        Ok(Self {
            program,
            rows,
            width,
            bound,
            instance_count,
            positions,
            private_state,
            public_next,
            state,
            next_flag: flag,
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

    /// The private input `X^(E_state)`.
    pub const fn private_state(&self) -> WireId {
        self.private_state
    }

    /// The public input `X^(E_next)`.
    pub const fn public_next(&self) -> WireId {
        self.public_next
    }

    /// The state output `X^(2^p R)`.
    pub const fn state(&self) -> WireId {
        self.state
    }

    /// The next-flag output `X^(f_{i+1})`.
    pub const fn next_flag(&self) -> WireId {
        self.next_flag
    }

    /// The packed exponents `(E_state, E_next)` of a step's private and
    /// public inputs. Every value must fit its field; a digit `>= B` or an
    /// index `>= T - 1` that fits is packed, and the Program rejects it.
    pub fn pack(
        &self,
        index: usize,
        flag: usize,
        current: &[usize],
        next: &[usize],
        instance: &[usize],
    ) -> Result<(usize, usize), AirError> {
        if current.len() != self.width ||
            next.len() != self.width ||
            instance.len() != self.instance_count
        {
            return Err(AirError::InvalidValue);
        }
        let values =
            [index, flag].into_iter().chain(current.iter().chain(next).chain(instance).copied());
        let exponent =
            values.zip(self.positions.windows(2)).try_fold(0, |exponent, (value, field)| {
                if value >> (field[1] - field[0]) != 0 {
                    return Err(AirError::InvalidValue);
                }
                Ok(exponent | value << field[0])
            })?;
        let next = self.next_mask();
        Ok((exponent & !next, exponent & next))
    }

    /// The bits of the next-row fields.
    fn next_mask(&self) -> usize {
        let (low, high) = (self.positions[2 + self.width], self.positions[2 + 2 * self.width]);
        (1 << high) - (1 << low)
    }

    /// The next row and the instance packed in a state output exponent,
    /// which must be a multiple of `2^p` below `2^P`.
    pub fn unpack_state(&self, exponent: usize) -> Result<(Vec<usize>, Vec<usize>), AirError> {
        let low = self.positions[2 + self.width];
        if exponent >> self.packed_bits() != 0 || exponent & ((1 << low) - 1) != 0 {
            return Err(AirError::InvalidValue);
        }
        let mut values = self.positions[2 + self.width..]
            .windows(2)
            .map(|field| (exponent >> field[0]) & ((1 << (field[1] - field[0])) - 1));
        let next = values.by_ref().take(self.width).collect();
        Ok((next, values.collect()))
    }

    /// The plaintext bindings `X^(E_state)` and `X^(E_next)` of the inputs,
    /// over a ring of dimension at least `2^P`; each exponent may only use
    /// its own fields.
    pub fn plaintext_inputs<P: Poly>(
        &self,
        parameters: &P::Params,
        state: usize,
        next: usize,
    ) -> Result<PlaintextInputs<WireId, P>, AirError> {
        let packed = 1usize << self.packed_bits();
        let mask = self.next_mask();
        if state >= packed ||
            state & mask != 0 ||
            next & !mask != 0 ||
            (parameters.ring_dimension() as usize) < packed
        {
            return Err(AirError::InvalidValue);
        }
        let monomial = |exponent| InputBinding::Scalar(P::const_rotate_poly(parameters, exponent));
        Ok(BTreeMap::from([
            (self.private_state, monomial(state)),
            (self.public_next, monomial(next)),
        ]))
    }
}

/// The automaton state: `ok`, the gate bits, and the live cell values.
#[derive(Default)]
struct AutomatonState {
    ok: bool,
    first: bool,
    last: bool,
    values: BTreeMap<AirCell, usize>,
}

impl AutomatonState {
    /// A cell value; a rejected state, decoded from an unused code, has none
    /// and reads `0`.
    fn value(&self, cell: AirCell) -> usize {
        self.values.get(&cell).copied().unwrap_or(0)
    }
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
        let cells = self.cells.iter().rev().fold(0, |code, cell| code * bound + state.value(*cell));
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
    use crate::program::{
        Environment, ExponentLUTProgramLowering, FamilyExecution, FamilyRange, InputShape,
        LoweringSite, ProgramLowerError, ProgramLoweringContext, lower_program,
    };
    use mxx_backends::poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly};
    use mxx_ir_core::IntExpr;

    #[derive(Debug, thiserror::Error)]
    #[error("clear AIR evaluator error: {0}")]
    struct ClearError(&'static str);

    /// Evaluates the Program on monomials in the clear: a wire is its
    /// exponent, a public multiplication adds exponents (here of disjoint
    /// fields), and a LUT applies the window semantics of
    /// [`LutOutputRepresentation`], asserting its preconditions.
    struct Clear {
        site: Option<LoweringSite>,
        packed_bits: u32,
        luts: usize,
        multiplications: usize,
    }

    impl ExponentLUTProgramLowering for Clear {
        type Wire = usize;
        type Rhs = ();
        type Error = ClearError;

        fn add(&mut self, _: &usize, _: &usize) -> Result<usize, ClearError> {
            Err(ClearError("the AIR step has no additions"))
        }
        fn mul_public(&mut self, lhs: &usize, rhs: &usize) -> Result<usize, ClearError> {
            assert_eq!(lhs & rhs, 0, "the multiplied exponents share a field");
            assert!(lhs + rhs < 1 << self.packed_bits, "the product leaves the packing");
            self.multiplications += 1;
            Ok(lhs + rhs)
        }
        fn mul_private(&mut self, _: &usize, _: &()) -> Result<usize, ClearError> {
            Err(ClearError("the AIR step has no private multiplications"))
        }
        fn lut(&mut self, input: &usize, table: &LutTable) -> Result<usize, ClearError> {
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
            Ok(result)
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

        fn wire_at(&mut self, _: &(), _: &IntExpr) -> Result<usize, ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no families")))
        }
        fn rhs_at(&mut self, _: &(), _: &IntExpr) -> Result<(), ProgramLowerError<ClearError>> {
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
            ) -> Result<usize, ProgramLowerError<ClearError>>,
        ) -> Result<(), ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no families")))
        }
        fn sum(&mut self, _: &mut Clear, _: &()) -> Result<usize, ProgramLowerError<ClearError>> {
            Err(ProgramLowerError::Backend(ClearError("the AIR step has no sums")))
        }
    }

    /// Evaluates one step on `X^(E_state)` and `X^(E_next)`: the state
    /// exponent, the flag exponent, and the numbers of LUTs and public
    /// multiplications evaluated.
    fn step(
        air: &AirConstraintProgram,
        (state, next): (usize, usize),
    ) -> (usize, usize, [usize; 2]) {
        let mut environment = Environment::new();
        environment.bind_wire(air.private_state(), InputBinding::Scalar(state)).unwrap();
        environment.bind_wire(air.public_next(), InputBinding::Scalar(next)).unwrap();
        let mut clear =
            Clear { site: None, packed_bits: air.packed_bits(), luts: 0, multiplications: 0 };
        let outputs =
            lower_program(air.program(), &mut clear, &mut NoFamilies, &mut environment).unwrap();
        (outputs[0], outputs[1], [clear.luts, clear.multiplications])
    }

    /// The next flag of one step, after checking that the state output
    /// forwards the next row and the instance.
    fn next_flag(
        air: &AirConstraintProgram,
        index: usize,
        flag: usize,
        current: &[usize],
        next: &[usize],
        instance: &[usize],
    ) -> usize {
        let (state, flag, _) = step(air, air.pack(index, flag, current, next, instance).unwrap());
        assert_eq!(air.unpack_state(state).unwrap(), (next.to_vec(), instance.to_vec()));
        assert!(flag <= 1, "flag exponent {flag}");
        flag
    }

    /// Streams `rows` through the Program from `(instance, rows[0], flag)`,
    /// carrying the unpacked state, and returns the flag after every step.
    fn stream(
        air: &AirConstraintProgram,
        instance: &[usize],
        rows: &[Vec<usize>],
        flag: usize,
    ) -> Vec<usize> {
        assert_eq!(rows.len(), air.rows());
        let (mut instance, mut current, mut flag) = (instance.to_vec(), rows[0].clone(), flag);
        let mut flags = Vec::new();
        for (index, next) in rows[1..].iter().enumerate() {
            let inputs = air.pack(index, flag, &current, next, &instance).unwrap();
            let (state, next_flag, _) = step(air, inputs);
            (current, instance) = air.unpack_state(state).unwrap();
            assert_eq!(&current, next);
            flag = next_flag;
            flags.push(flag);
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

    /// `B = 3`, so the two-bit digit `3` is an out-of-range witness value.
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
        let inputs = air.program().inputs();
        assert_eq!(inputs.len(), 2);
        for (input, knowledge) in [
            (air.private_state(), AttributeKnowledge::PrivatePayload),
            (air.public_next(), AttributeKnowledge::PublicPayload),
        ] {
            assert_eq!(inputs[&input].shape(), &InputShape::Scalar(()));
            assert_eq!(inputs[&input].knowledge(), knowledge);
        }
        assert!(air.program().rhs_inputs().is_empty());
        // Flag chain: windows `2^(sigma_k + d_k)` for sigma = (0, 3, 3, 5, 6,
        // 6, 4) and d = (6, 1, 2, ...); state chain: the index, flag, and
        // current-row fields.
        let widths = air
            .program()
            .encryption_inputs()
            .luts()
            .iter()
            .map(LutTable::input_width)
            .collect::<Vec<_>>();
        assert_eq!(widths, [64, 16, 32, 128, 256, 256, 64, 64, 2, 4, 4]);
        // One public multiplication, then LUTs only: the clear evaluator
        // rejects any other step.
        let inputs = air.pack(0, 1, &[0, 1], &[1, 2], &[0]).unwrap();
        assert_eq!(step(&air, inputs).2, [widths.len(), 1]);
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
                    for value in (0..4).filter(|v| *v != trace[row][column]) {
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
            for wrong in (0..4).filter(|v| *v != y) {
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
    fn fibonacci_rejects_out_of_range_indices_and_digits() {
        let rows = 3;
        let air = fibonacci(rows);
        let trace = fibonacci_trace(rows, [0, 1]);
        let y = [trace[rows - 1][1]];
        let (current, next) = (&trace[1][..], &trace[2][..]);
        assert_eq!(next_flag(&air, 1, 1, current, next, &y), 1);
        for index in [rows - 1, 63] {
            assert_eq!(next_flag(&air, index, 1, current, next, &y), 0, "index {index}");
        }
        assert_eq!(next_flag(&air, 1, 1, current, next, &[B]), 0);
        assert_eq!(next_flag(&air, 1, 1, &[B, current[1]], next, &y), 0);
        assert_eq!(next_flag(&air, 1, 1, current, &[next[0], B], &y), 0);
        // Every check fails at once.
        assert_eq!(next_flag(&air, 63, 0, &[B, B], &[B, B], &[B]), 0);
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
    fn one_step_matches_the_relation_on_every_packed_input() {
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
        for exponent in 0..1 << 14 {
            let field = |position: u32, width: u32| (exponent >> position) & ((1 << width) - 1);
            let (index, flag) = (field(0, 7), field(7, 1));
            let (current, next, x) = (field(8, 2), field(10, 2), field(12, 2));
            let expected = index < rows - 1 &&
                flag == 1 &&
                current < B &&
                next < B &&
                x < B &&
                (index != 0 || current == x) &&
                next == (current + 1) % B &&
                (index != rows - 2 || next == (x + 2) % B);
            // The next-row field, bits [10, 12), is the public input.
            let next_bits = 0b11 << 10;
            let (state, next_flag, _) = step(&air, (exponent & !next_bits, exponent & next_bits));
            assert_eq!(state, exponent & !((1 << 10) - 1), "E = {exponent}");
            assert_eq!(next_flag, usize::from(expected), "E = {exponent}");
            accepted += next_flag;
        }
        // Step 0 accepts (x, x, x + 1); step 1 accepts (x, c, c + 1 = x + 2).
        assert_eq!(accepted, 2 * B);
    }

    #[test]
    fn packing_round_trips_and_rejects_malformed_values() {
        let air = fibonacci(3);
        let (state, next) = air.pack(1, 1, &[1, 2], &[0, 2], &[1]).unwrap();
        assert_eq!((state, next), (1 | 1 << 6 | 1 << 7 | 2 << 9 | 1 << 15, 2 << 13));
        let exponent = state | next;
        assert_eq!(air.unpack_state(exponent & !((1 << 11) - 1)).unwrap(), (vec![0, 2], vec![1]));
        for malformed in [
            air.pack(64, 1, &[0, 0], &[0, 0], &[0]),
            air.pack(0, 2, &[0, 0], &[0, 0], &[0]),
            air.pack(0, 1, &[4, 0], &[0, 0], &[0]),
            air.pack(0, 1, &[0], &[0, 0], &[0]),
            air.pack(0, 1, &[0, 0], &[0, 0], &[]),
        ] {
            assert_eq!(malformed, Err(AirError::InvalidValue));
        }
        assert_eq!(air.unpack_state(exponent), Err(AirError::InvalidValue));
        assert_eq!(air.unpack_state(1 << 17), Err(AirError::InvalidValue));
    }

    #[test]
    fn plaintext_inputs_are_the_packed_monomials() {
        // P = 5: index (2 bits), flag, current, next (bit 4).
        let constraints =
            vec![constraint(AirGate::EveryRow, vec![AirCell::Next(0), AirCell::Current(0)], |v| {
                v[0] == v[1]
            })];
        let air = AirConstraintProgram::new(2, 1, 2, 0, constraints).unwrap();
        assert_eq!(air.packed_bits(), 5);
        let ring = DCRTPolyParams::new(32, 1, 17, 1, None, None);
        let (state, next) = air.pack(0, 1, &[1], &[1], &[]).unwrap();
        assert_eq!((state, next), (0b1100, 0b10000));
        let bindings = air.plaintext_inputs::<DCRTPoly>(&ring, state, next).unwrap();
        let monomial =
            |exponent| InputBinding::Scalar(DCRTPoly::const_rotate_poly(&ring, exponent));
        assert_eq!(
            bindings,
            BTreeMap::from([
                (air.private_state(), monomial(state)),
                (air.public_next(), monomial(next))
            ])
        );
        assert_eq!(next_flag(&air, 0, 1, &[1], &[1], &[]), 1);
        assert_eq!(next_flag(&air, 0, 1, &[1], &[0], &[]), 0);
        let small = DCRTPolyParams::new(16, 1, 17, 1, None, None);
        for (parameters, state, next) in [
            (&small, state, next),
            (&ring, 32, next),
            (&ring, state | next, next),
            (&ring, state, next | 1),
        ] {
            assert_eq!(
                air.plaintext_inputs::<DCRTPoly>(parameters, state, next),
                Err(AirError::InvalidValue)
            );
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
