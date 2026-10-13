//! Branching programs and their compilation into Exponent-LUT Programs.
//!
//! A branching program of length `T` over attributes `x in [0, Bx)^l`
//! starts at the state `z_0 = 0` and applies `z_(i+1) = f_i(z_i, x_(iota(i)))`;
//! the last table outputs a bit, and a key for the program decrypts exactly
//! when that bit is zero (manuscript Definition 2).
//!
//! The compiled Program keeps the state as the exponent of a monomial: the
//! initial state `X^0 = 1` is a private encoding under the setup's input key,
//! step `i` multiplies it by the RHS package of attribute `iota(i)`, whose
//! payload is `X^(Bz x)`, giving `X^(z + Bz x)`, and then applies the table
//! `L_i(z + Bz x) = f_i(z, x)` over the class lookup width `W`. Intermediate
//! tables return exponent payloads; the last returns the bit as a
//! coefficient. Since `z + Bz x < W <= n`, no negacyclic sign enters a
//! reachable state; unreachable table entries are zero.
//!
//! One Program serves the whole class ([`PeClass::compile`]): it has `T_max`
//! steps, and the program is its integer data ([`BranchingProgram::data`]),
//! which key generation and decryption supply at run time. Step `i` selects
//! its attribute by entry `i` of the read data, and its table is window `i`
//! of the table data. A program shorter than `T_max` starts with identity
//! steps, which read attribute `0` and keep the state.
//!
//! A class may fix the kind of every step ([`PeClass::with_steps`]). Besides the
//! lookup step above, a *map* step only applies a table to the state, reading no
//! attribute. Each kind is fixed by the class because the Program is one graph
//! for every program. A program of such a class has exactly `T_max` steps.
use crate::PeError;
use mxx_exponent_lut::{
    ExpLayout, ExponentLutProgram,
    program::{
        AttributeKnowledge, DataId, ExponentLutProgramBuilder, LutOutputRepresentation, LutTable,
        Operand, ProgramData,
    },
    rns_bgg::RnsBggDynamicBinding,
};
use mxx_ir_core::IntExpr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Setup coordinate of the initial state encoding `c_0`.
pub const INITIAL_STATE_COORDINATE: u32 = 0;
/// Setup coordinate of the attribute packages: a family of `l` members.
pub const ATTRIBUTE_COORDINATE: u32 = 0;
/// The kind of one step of a class ([`PeClass::with_steps`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum StepKind {
    /// `z <- f(z + Bz x)`: the product by the package of `X^(Bz x)`, then a table.
    Lookup,
    /// `z <- f(z)`: a table only.
    Map,
}

/// One step of a program of a class ([`BranchingProgram::from_steps`]).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramStep {
    /// Read attribute `read` and apply `table` at `z + Bz x`.
    Lookup { read: usize, table: Vec<usize> },
    /// Apply `table` at `z`.
    Map { table: Vec<usize> },
}

/// The function class of a setup: attribute length `l`, alphabet size `Bx`,
/// state width `Bz`, the largest program length `T_max`, and the kind of every
/// step (all lookups unless [`Self::with_steps`] fixes them, as a bit mask of
/// the map steps). One lookup width `W` and automorphism group serve
/// every program of the class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeClass {
    attribute_length: usize,
    alphabet_size: usize,
    state_width: usize,
    max_steps: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    map_steps: u64,
}

fn is_zero(mask: &u64) -> bool {
    *mask == 0
}

impl PeClass {
    /// A class with every dimension positive and `Bz Bx` representable.
    pub fn new(
        attribute_length: usize,
        alphabet_size: usize,
        state_width: usize,
        max_steps: usize,
    ) -> Result<Self, PeError> {
        if attribute_length == 0 || alphabet_size == 0 || state_width == 0 || max_steps == 0 {
            return Err(PeError::Class("every class dimension must be positive"));
        }
        state_width
            .checked_mul(alphabet_size)
            .and_then(usize::checked_next_power_of_two)
            .ok_or(PeError::Class("the lookup width Bz Bx overflows"))?;
        Ok(Self { attribute_length, alphabet_size, state_width, max_steps, map_steps: 0 })
    }

    /// The class of the decimal range programs over `digits` decimal digits
    /// ([`BranchingProgram::decimal_range`]): one attribute symbol of 4 bits per digit
    /// (`Bx = 16`, BCD, with `10..=15` not digits), one lookup step per digit, and the 5 states
    /// of the range comparison padded to `Bz = 8`, so `W = 128`.
    pub fn decimal_range(digits: usize) -> Result<Self, PeError> {
        Self::new(digits, DECIMAL_ALPHABET, DECIMAL_STATES, digits)
    }

    /// A class whose step `i` has kind `steps[i]`, `T_max = steps.len() <= 64`.
    /// Every step applies a table; the last one's coefficient is the output.
    pub fn with_steps(
        attribute_length: usize,
        alphabet_size: usize,
        state_width: usize,
        steps: &[StepKind],
    ) -> Result<Self, PeError> {
        if steps.len() > 64 {
            return Err(PeError::Class("a class of step kinds has at most 64 steps"));
        }
        let mask = |kind| {
            steps
                .iter()
                .enumerate()
                .filter(|(_, step)| **step == kind)
                .map(|(i, _)| 1u64 << i)
                .sum()
        };
        Ok(Self {
            map_steps: mask(StepKind::Map),
            ..Self::new(attribute_length, alphabet_size, state_width, steps.len())?
        })
    }

    /// The kind of step `step`.
    pub const fn step_kind(&self, step: usize) -> StepKind {
        if step < 64 && self.map_steps >> step & 1 == 1 { StepKind::Map } else { StepKind::Lookup }
    }

    /// Whether every step is a lookup.
    pub const fn lookups_only(&self) -> bool {
        self.map_steps == 0
    }

    fn has(&self, kind: StepKind) -> bool {
        (0..self.max_steps).any(|step| self.step_kind(step) == kind)
    }

    /// The RHS packages of a ciphertext: `l` lookup packages if a lookup step
    /// reads them.
    pub fn rhs_packages(&self) -> usize {
        self.attribute_length * usize::from(self.has(StepKind::Lookup))
    }

    /// Whether encryption publishes the lookup packages `X^(Bz x_t)`.
    pub fn has_lookup_packages(&self) -> bool {
        self.has(StepKind::Lookup)
    }

    pub const fn attribute_length(&self) -> usize {
        self.attribute_length
    }

    pub const fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    pub const fn state_width(&self) -> usize {
        self.state_width
    }

    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// The lookup width `W = next_power_of_two(Bz Bx)`.
    pub fn lookup_width(&self) -> usize {
        (self.state_width * self.alphabet_size).next_power_of_two()
    }

    /// The entries of a table of step `step`: `Bz Bx` for a lookup, `Bz` for a map.
    pub fn table_entries(&self, step: usize) -> usize {
        match self.step_kind(step) {
            StepKind::Lookup => self.state_width * self.alphabet_size,
            StepKind::Map => self.state_width,
        }
    }

    /// The default radix `B` of the LUT automorphism decomposition
    /// ([`crate::PeParameters::with_lut_radix`]): `B = W^(1/r)` for the
    /// largest `r` in `{3, 2}` that divides `log2 W`, so that a branch is
    /// switched at most 2 or 3 times; `None` (the direct switching, one
    /// switch per branch) when neither divides it. Each switch multiplies
    /// the noise by a decomposition gain, while the direct switching
    /// publishes `W - 1` switching packages.
    pub fn default_lut_radix(&self) -> Option<usize> {
        let bits = self.lookup_width().trailing_zeros();
        [3, 2]
            .into_iter()
            .find(|switches| bits % switches == 0)
            .map(|switches| 1 << (bits / switches))
    }

    /// The exponent layout of every table: stride zero, width `W`.
    pub fn layout(&self) -> ExpLayout {
        ExpLayout { stride_log2: 0, width_log2: self.lookup_width().trailing_zeros() }
    }

    /// Reject a ring of dimension `n` with `W > n`.
    pub fn validate_ring_dimension(&self, ring_dimension: usize) -> Result<(), PeError> {
        if self.lookup_width() > ring_dimension {
            return Err(PeError::Class("the lookup width exceeds the ring dimension"));
        }
        Ok(())
    }

    /// The number of logical levels of the class chain, `2 T_max + 2`: a
    /// step's product and table each take a level, the final coefficient
    /// output is normalized once to level `2 T`, and one gadget successor
    /// follows the longest program's output.
    pub fn levels(&self) -> usize {
        2 * self.max_steps + 2
    }

    /// Reject an attribute of another length or with a symbol outside the
    /// alphabet.
    pub fn validate_attribute(&self, attribute: &[usize]) -> Result<(), PeError> {
        if attribute.len() != self.attribute_length ||
            attribute.iter().any(|symbol| *symbol >= self.alphabet_size)
        {
            return Err(PeError::Attribute);
        }
        Ok(())
    }

    /// Compile the class into the Exponent-LUT Program the setup evaluates,
    /// with its mapping to the setup coordinates: the Program input is the
    /// initial state, its RHS family the attribute packages, and its integer
    /// data a program of the class ([`BranchingProgram::data`]).
    pub fn compile(&self) -> Result<CompiledPredicate, PeError> {
        let invalid = |error: mxx_exponent_lut::program::ProgramValidationError| {
            PeError::Compilation(error.to_string())
        };
        let width = self.lookup_width();
        let layout = self.layout();
        let mut builder = ExponentLutProgramBuilder::new();
        let initial = builder.input(AttributeKnowledge::PrivatePayload).map_err(invalid)?;
        let mut rhs = BTreeMap::new();
        let attributes = self
            .has_lookup_packages()
            .then(|| builder.rhs_family(IntExpr::constant(self.attribute_length)))
            .transpose()
            .map_err(invalid)?;
        rhs.extend(attributes.map(|id| (id, ATTRIBUTE_COORDINATE)));
        let reads = builder.data_family(self.max_steps).map_err(invalid)?;
        let tables = builder.data_family(self.max_steps * width).map_err(invalid)?;
        let mut state = initial;
        for step in 0..self.max_steps {
            let selected = |family| Operand::Selected {
                family,
                data: reads,
                position: IntExpr::constant(step),
            };
            let input = match (self.step_kind(step), attributes) {
                (StepKind::Lookup, Some(attributes)) => builder
                    .mul_private(Operand::Scalar(state), selected(attributes))
                    .map_err(invalid)?,
                (StepKind::Map, _) => state,
                _ => return Err(PeError::Class("a step reads a package the class lacks")),
            };
            let (output_width, representation) = if step + 1 == self.max_steps {
                (2, LutOutputRepresentation::Coefficient)
            } else {
                (width, LutOutputRepresentation::Exponent(layout))
            };
            let table = builder
                .register_lut(
                    LutTable::from_data(
                        width,
                        layout,
                        output_width,
                        tables,
                        step * width,
                        representation,
                    )
                    .map_err(invalid)?,
                )
                .map_err(invalid)?;
            state = builder.lut(Operand::Scalar(input), table).map_err(invalid)?;
        }
        let program = builder.finish(vec![Operand::Scalar(state)]).map_err(invalid)?;
        let binding =
            RnsBggDynamicBinding::new(BTreeMap::from([(initial, INITIAL_STATE_COORDINATE)]), rhs);
        Ok(CompiledPredicate { program, binding, reads, tables })
    }
}

/// A checked branching program of a class: its read map and transition
/// tables, table `i` at index `z + Bz x` holding `f_i(z, x)` (at `z` for a map
/// step).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BranchingProgram {
    class: PeClass,
    reads: Vec<usize>,
    tables: Vec<Vec<usize>>,
}

/// The alphabet of the decimal range programs: 4-bit symbols, of which `0..=9` are digits.
const DECIMAL_ALPHABET: usize = 16;
/// The state width of the decimal range programs: the states `EQ_EQ`, `GT_EQ`, `EQ_LT`, `GT_LT`,
/// and `REJ` padded to a power of two.
const DECIMAL_STATES: usize = 8;
/// The rejecting state of the decimal range programs, which absorbs the padding states.
const DECIMAL_REJECT: usize = 4;

impl BranchingProgram {
    /// The program of [`PeClass::decimal_range`] that accepts (`F(x) = 0`) exactly the
    /// attributes `x = (x_1, ..., x_D)` of `digits = D` decimal digits, most significant first,
    /// that match `[1-9][0-9]{D-1}` and whose value `N(x)` satisfies `low <= N(x) <= high`, for
    /// `10^(D-1) <= low <= high <= 10^D - 1`.
    ///
    /// Step `i` reads digit `x_(i+1)` and tracks whether the prefix equals the prefix of `low`
    /// (`EQ`) or exceeds it (`GT`), and whether it equals the prefix of `high` (`EQ`) or is below
    /// it (`LT`): states `0 = (EQ, EQ)`, `1 = (GT, EQ)`, `2 = (EQ, LT)`, `3 = (GT, LT)`, and the
    /// absorbing `4 = REJ`, which a prefix below `low`, above `high`, a symbol `>= 10`, or a
    /// leading zero reaches; the padding states `5..=7` go to `REJ`. The table at `z + 8 x` is
    /// the next state, and the last step's is `0` unless it is `REJ`.
    pub fn decimal_range(digits: usize, low: u64, high: u64) -> Result<Self, PeError> {
        let class = PeClass::decimal_range(digits)?;
        let bound = u32::try_from(digits)
            .ok()
            .and_then(|digits| 10u64.checked_pow(digits))
            .ok_or(PeError::Predicate("the number of digits overflows"))?;
        if low < bound / 10 || low > high || high >= bound {
            return Err(PeError::Predicate("the range is not of D-digit numbers"));
        }
        let digit =
            |value: u64, i: usize| (value / 10u64.pow((digits - 1 - i) as u32) % 10) as usize;
        let step = |i: usize, z: usize, x: usize| -> usize {
            if z >= DECIMAL_REJECT || x >= 10 || (i == 0 && x == 0) {
                return DECIMAL_REJECT;
            }
            // `z = lo + 2 hi` with `lo, hi` one when the prefix exceeds `low`'s or is below
            // `high`'s.
            let (mut above, mut below) = (z & 1 == 1, z & 2 == 2);
            if !above {
                if x < digit(low, i) {
                    return DECIMAL_REJECT;
                }
                above = x > digit(low, i);
            }
            if !below {
                if x > digit(high, i) {
                    return DECIMAL_REJECT;
                }
                below = x < digit(high, i);
            }
            usize::from(above) + 2 * usize::from(below)
        };
        let tables = (0..digits)
            .map(|i| {
                (0..DECIMAL_STATES * DECIMAL_ALPHABET)
                    .map(|u| {
                        let next = step(i, u % DECIMAL_STATES, u / DECIMAL_STATES);
                        if i + 1 == digits { usize::from(next == DECIMAL_REJECT) } else { next }
                    })
                    .collect()
            })
            .collect();
        Self::new(class, (0..digits).collect(), tables)
    }

    /// Check a program of lookup steps of `class`: `1 <= T <= T_max` steps,
    /// every read an attribute index, every table of `Bz Bx` entries,
    /// intermediate outputs below `Bz`, and final outputs in `{0, 1}`.
    pub fn new(
        class: PeClass,
        reads: Vec<usize>,
        tables: Vec<Vec<usize>>,
    ) -> Result<Self, PeError> {
        if reads.len() != tables.len() {
            return Err(PeError::Predicate("the program length is outside 1..=T_max"));
        }
        Self::from_steps(
            class,
            reads
                .into_iter()
                .zip(tables)
                .map(|(read, table)| ProgramStep::Lookup { read, table })
                .collect(),
        )
    }

    /// Check a program of `class` given step by step: `1 <= T <= T_max` steps
    /// (exactly `T_max`, of the class's kinds, unless every step is a lookup),
    /// every read an attribute index, every table of [`PeClass::table_entries`]
    /// entries with intermediate outputs below `Bz` and final outputs in
    /// `{0, 1}`.
    pub fn from_steps(class: PeClass, steps: Vec<ProgramStep>) -> Result<Self, PeError> {
        let length = steps.len();
        let full = class.lookups_only() || length == class.max_steps;
        if length == 0 || length > class.max_steps || !full {
            return Err(PeError::Predicate("the program length is outside 1..=T_max"));
        }
        let offset = class.max_steps - length;
        let (mut reads, mut tables) = (vec![], vec![]);
        for (index, step) in steps.into_iter().enumerate() {
            let position = offset + index;
            let last = index + 1 == length;
            let (kind, read, table) = match step {
                ProgramStep::Lookup { read, table } => (StepKind::Lookup, Some(read), table),
                ProgramStep::Map { table } => (StepKind::Map, None, table),
            };
            if kind != class.step_kind(position) {
                return Err(PeError::Predicate("a step is not of its class kind"));
            }
            if read.is_some_and(|read| read >= class.attribute_length) {
                return Err(PeError::Predicate("a read is not an attribute index"));
            }
            let bound = if last { 2 } else { class.state_width };
            if table.len() != class.table_entries(position) ||
                table.iter().any(|value| *value >= bound)
            {
                return Err(PeError::Predicate("a transition table is out of range"));
            }
            reads.push(read.unwrap_or(0));
            tables.push(table);
        }
        Ok(Self { class, reads, tables })
    }

    pub const fn class(&self) -> &PeClass {
        &self.class
    }

    /// The program length `T`.
    pub fn length(&self) -> usize {
        self.reads.len()
    }

    pub fn reads(&self) -> &[usize] {
        &self.reads
    }

    pub fn tables(&self) -> &[Vec<usize>] {
        &self.tables
    }

    /// The kind of step `index` of this program.
    fn kind(&self, index: usize) -> StepKind {
        self.class.step_kind(self.class.max_steps - self.length() + index)
    }

    /// `F(x)`, evaluated in the clear: a key for the program decrypts a
    /// ciphertext of `x` when it is zero.
    pub fn evaluate(&self, attribute: &[usize]) -> Result<usize, PeError> {
        self.class.validate_attribute(attribute)?;
        let mut state = 0;
        for (index, (read, table)) in self.reads.iter().zip(&self.tables).enumerate() {
            let symbol = attribute[*read];
            state = match self.kind(index) {
                StepKind::Lookup => table[state + self.class.state_width * symbol],
                StepKind::Map => table[state],
            };
        }
        Ok(state)
    }

    /// A program of the class's maximal length whose every table is
    /// nonzero on its range: parameter selection simulates its structure,
    /// which bounds every program of the class (the bounds depend on the
    /// length, the width, and the output payload, not on table values).
    pub fn longest(class: PeClass) -> Self {
        let tables = (0..class.max_steps)
            .map(|step| {
                (0..class.table_entries(step))
                    .map(|index| {
                        let state = index % class.state_width;
                        if step + 1 == class.max_steps { usize::from(state == 0) } else { state }
                    })
                    .collect()
            })
            .collect();
        Self { class, reads: vec![0; class.max_steps], tables }
    }

    /// The Program data of this program ([`PeClass::compile`]): the reads and
    /// the tables, each padded to the lookup width `W` with zeros, of the `T_max`
    /// steps, the first `T_max - T` of them identity steps.
    pub fn data(&self, compiled: &CompiledPredicate) -> ProgramData {
        let class = &self.class;
        let width = class.lookup_width();
        let padding = class.max_steps - self.length();
        let identity = (0..class.state_width * class.alphabet_size)
            .map(|index| index % class.state_width)
            .collect::<Vec<_>>();
        let reads = std::iter::repeat_n(0, padding).chain(self.reads.iter().copied()).collect();
        let tables = std::iter::repeat_n(&identity, padding)
            .chain(&self.tables)
            .flat_map(|table| (0..width).map(move |index| table.get(index).copied().unwrap_or(0)))
            .collect();
        ProgramData::new().with(compiled.reads, reads).with(compiled.tables, tables)
    }
}

/// A compiled branching program: the Exponent-LUT Program and its mapping to
/// the setup coordinates.
#[derive(Clone, Debug)]
pub struct CompiledPredicate {
    pub(crate) program: ExponentLutProgram,
    pub(crate) binding: RnsBggDynamicBinding,
    /// The data families of the reads and of the tables.
    reads: DataId,
    tables: DataId,
}

impl CompiledPredicate {
    pub fn program(&self) -> &ExponentLutProgram {
        &self.program
    }

    /// The data family of the reads: entry `i` is the attribute step `i` reads.
    pub const fn reads(&self) -> DataId {
        self.reads
    }

    /// The data family of the tables: window `i` of `W` entries is the table of step `i`.
    pub const fn tables(&self) -> DataId {
        self.tables
    }

    pub fn binding(&self) -> &RnsBggDynamicBinding {
        &self.binding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::greater_than;

    fn class() -> PeClass {
        PeClass::new(3, 2, 3, 4).unwrap()
    }

    #[test]
    fn programs_are_checked_against_their_class() {
        let class = class();
        assert_eq!(class.lookup_width(), 8);
        assert_eq!(class.levels(), 10);
        assert!(BranchingProgram::new(class, vec![], vec![]).is_err());
        assert!(BranchingProgram::new(class, vec![3], vec![vec![0; 6]]).is_err());
        assert!(BranchingProgram::new(class, vec![0], vec![vec![2; 6]]).is_err());
        assert!(BranchingProgram::new(class, vec![0, 1], vec![vec![3; 6], vec![0; 6]]).is_err());
        assert!(BranchingProgram::new(class, vec![0; 5], vec![vec![0; 6]; 5]).is_err());
        assert!(BranchingProgram::new(class, vec![0, 1], vec![vec![2; 6], vec![1; 6]]).is_ok());
        assert!(class.validate_ring_dimension(4).is_err());
    }

    /// The default radix switches a branch 3 or 2 times when that divides `log2 W`, and
    /// directly otherwise.
    #[test]
    fn the_default_lut_radix_takes_three_or_two_switches() {
        let radix =
            |alphabet, state| PeClass::new(1, alphabet, state, 1).unwrap().default_lut_radix();
        assert_eq!(radix(2, 1), None);
        assert_eq!(radix(2, 2), Some(2));
        assert_eq!(radix(4, 2), Some(2));
        assert_eq!(radix(16, 1), Some(4));
        assert_eq!(radix(32, 1), None);
        assert_eq!(radix(256, 16), Some(16));
    }

    #[test]
    fn evaluation_follows_the_read_map_and_tables() {
        let class = class();
        let program = greater_than(class, &[1, 0, 1]);
        for value in 0..8usize {
            let bits = [value >> 2 & 1, value >> 1 & 1, value & 1];
            assert_eq!(program.evaluate(&bits).unwrap(), usize::from(value <= 5), "{bits:?}");
        }
        assert!(program.evaluate(&[0, 1]).is_err());
        assert!(program.evaluate(&[0, 2, 1]).is_err());
    }

    #[test]
    fn the_class_program_takes_every_program_as_data() {
        let compiled = class().compile().unwrap();
        let output = match compiled.program().outputs() {
            [Operand::Scalar(output)] => *output,
            _ => panic!("one scalar output"),
        };
        assert_eq!(compiled.program().wire_coefficient_bound(output).unwrap(), Some(1));
        assert_eq!(compiled.program().encryption_inputs().luts().len(), class().max_steps());
        // A program of 3 of the 4 steps starts with one identity step, which reads attribute
        // 0 and keeps the state, and its data validates against the class Program.
        let program = greater_than(class(), &[1, 0, 1]);
        let data = program.data(&compiled);
        data.validate(compiled.program()).unwrap();
        let width = class().lookup_width();
        assert_eq!(data.values(compiled.reads()).unwrap(), [0, 0, 1, 2]);
        let tables = data.values(compiled.tables()).unwrap();
        assert_eq!(&tables[..6], [0, 1, 2, 0, 1, 2]);
        for (step, table) in program.tables().iter().enumerate() {
            assert_eq!(&tables[(step + 1) * width..][..table.len()], table.as_slice());
        }
        // Every program of the class compiles to the same Program.
        let longest = BranchingProgram::longest(class());
        assert_eq!(longest.class().compile().unwrap().program().id(), compiled.program().id());
    }

    /// A map step applies its table to the state without reading an attribute, and a program
    /// follows its class's kinds and table sizes.
    #[test]
    fn map_steps_apply_a_table_to_the_state() {
        let class = PeClass::with_steps(2, 2, 3, &[StepKind::Lookup, StepKind::Map]).unwrap();
        assert_eq!((class.table_entries(0), class.table_entries(1)), (6, 3));
        // Step 0 stores the attribute bit, plus one; step 1 decrypts on state 2.
        let lookup = ProgramStep::Lookup { read: 1, table: vec![1, 1, 1, 2, 2, 2] };
        let map = ProgramStep::Map { table: vec![1, 1, 0] };
        let program = BranchingProgram::from_steps(class, vec![lookup.clone(), map]).unwrap();
        assert_eq!(program.evaluate(&[0, 1]).unwrap(), 0);
        assert_eq!(program.evaluate(&[1, 0]).unwrap(), 1);
        let compiled = class.compile().unwrap();
        let data = program.data(&compiled);
        data.validate(compiled.program()).unwrap();
        assert_eq!(data.values(compiled.reads()).unwrap(), [1, 0]);
        // A map table of the lookup size, a lookup where the class maps, and a short program
        // are rejected.
        let wide = ProgramStep::Map { table: vec![0; 6] };
        assert!(BranchingProgram::from_steps(class, vec![lookup.clone(), wide]).is_err());
        assert!(BranchingProgram::from_steps(class, vec![lookup.clone(), lookup.clone()]).is_err());
        assert!(BranchingProgram::from_steps(class, vec![lookup]).is_err());
    }

    /// The decimal range program of the specification's test vectors (`D = 4`, `[2000, 2025]`),
    /// evaluated in the clear and through the Program data: every step's table window of `W = 128`
    /// entries at `u = z + 8 x`.
    #[test]
    fn decimal_range_programs_check_the_notation_and_the_range() {
        let program = BranchingProgram::decimal_range(4, 2000, 2025).unwrap();
        let class = *program.class();
        assert_eq!(class, PeClass::decimal_range(4).unwrap());
        assert_eq!(
            (class.state_width(), class.alphabet_size(), class.lookup_width()),
            (8, 16, 128)
        );
        let compiled = class.compile().unwrap();
        let data = program.data(&compiled);
        data.validate(compiled.program()).unwrap();
        let tables = data.values(compiled.tables()).unwrap();
        let through_luts = |attribute: &[usize]| {
            attribute.iter().enumerate().fold(0, |z, (i, x)| tables[i * 128 + z + 8 * x])
        };
        for (attribute, expected) in [
            ([2, 0, 0, 0], 0),
            ([2, 0, 2, 5], 0),
            ([2, 0, 1, 7], 0),
            ([1, 9, 9, 9], 1),
            ([2, 0, 2, 6], 1),
            ([3, 0, 0, 0], 1),
            ([0, 9, 9, 9], 1),
            ([2, 0, 10, 5], 1),
        ] {
            assert_eq!(program.evaluate(&attribute).unwrap(), expected, "{attribute:?}");
            assert_eq!(through_luts(&attribute), expected, "{attribute:?}");
        }
        // Every 4-digit string over the alphabet agrees with the predicate.
        for value in 0..1 << 16 {
            let attribute = (0..4).map(|i| value >> (12 - 4 * i) & 15).collect::<Vec<usize>>();
            let number = attribute.iter().fold(0, |number, x| 10 * number + *x as u64);
            let accepted = attribute.iter().all(|x| *x < 10) &&
                attribute[0] != 0 &&
                (2000..=2025).contains(&number);
            assert_eq!(program.evaluate(&attribute).unwrap(), usize::from(!accepted));
        }
        // One digit: the leading zero and the range in one final step.
        let single = BranchingProgram::decimal_range(1, 3, 5).unwrap();
        let accepted = (0..16).filter(|x| single.evaluate(&[*x]).unwrap() == 0).collect::<Vec<_>>();
        assert_eq!(accepted, [3, 4, 5]);
        for (low, high) in [(999, 2000), (2000, 1999), (2000, 10_000)] {
            assert!(BranchingProgram::decimal_range(4, low, high).is_err(), "{low} {high}");
        }
    }
}
