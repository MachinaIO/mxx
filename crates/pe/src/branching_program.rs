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
//! lookup step above, an *add* step only multiplies the state by the package of
//! `X^x` or of its complement `X^(Bx - 1 - x)`, adding `x` or `Bx - 1 - x` to the
//! exponent without a table, and a *map* step only applies a table to the state,
//! reading no attribute. Each kind is fixed by the class because the Program is
//! one graph for every program; an add step thus saves the noise and the work of
//! a table. Encryption then publishes the add packages `X^(x_t)` and
//! `X^(Bx - 1 - x_t)` of every attribute, and the lookup packages only if a
//! lookup step reads them. A program of such a class has exactly `T_max` steps.
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
/// Setup coordinate of the add packages: a family of `2 l` members, `X^(x_t)`
/// then `X^(Bx - 1 - x_t)`.
pub const ADD_ATTRIBUTE_COORDINATE: u32 = 1;

/// The kind of one step of a class ([`PeClass::with_steps`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum StepKind {
    /// `z <- f(z + Bz x)`: the product by the package of `X^(Bz x)`, then a table.
    Lookup,
    /// `z <- z + x` or `z <- z + Bx - 1 - x`: the product by an add package only.
    Add,
    /// `z <- f(z)`: a table only.
    Map,
}

/// One step of a program of a class ([`BranchingProgram::from_steps`]).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramStep {
    /// Read attribute `read` and apply `table` at `z + Bz x`.
    Lookup { read: usize, table: Vec<usize> },
    /// Add attribute `read`, or its complement `Bx - 1 - x` with `complement`.
    Add { read: usize, complement: bool },
    /// Apply `table` at `z`.
    Map { table: Vec<usize> },
}

/// The function class of a setup: attribute length `l`, alphabet size `Bx`,
/// state width `Bz`, the largest program length `T_max`, and the kind of every
/// step (all lookups unless [`Self::with_steps`] fixes them, as bit masks of
/// the add and map steps). One lookup width `W` and automorphism group serve
/// every program of the class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeClass {
    attribute_length: usize,
    alphabet_size: usize,
    state_width: usize,
    max_steps: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    add_steps: u64,
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
        Ok(Self {
            attribute_length,
            alphabet_size,
            state_width,
            max_steps,
            add_steps: 0,
            map_steps: 0,
        })
    }

    /// A class whose step `i` has kind `steps[i]`, `T_max = steps.len() <= 64`.
    /// The last step applies a table (a lookup or a map step), whose coefficient
    /// is the output.
    pub fn with_steps(
        attribute_length: usize,
        alphabet_size: usize,
        state_width: usize,
        steps: &[StepKind],
    ) -> Result<Self, PeError> {
        if steps.len() > 64 {
            return Err(PeError::Class("a class of step kinds has at most 64 steps"));
        }
        if steps.last() == Some(&StepKind::Add) {
            return Err(PeError::Class("the last step must apply a table"));
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
            add_steps: mask(StepKind::Add),
            map_steps: mask(StepKind::Map),
            ..Self::new(attribute_length, alphabet_size, state_width, steps.len())?
        })
    }

    /// The kind of step `step`.
    pub const fn step_kind(&self, step: usize) -> StepKind {
        if step < 64 && self.add_steps >> step & 1 == 1 {
            StepKind::Add
        } else if step < 64 && self.map_steps >> step & 1 == 1 {
            StepKind::Map
        } else {
            StepKind::Lookup
        }
    }

    /// The class of the Hamming threshold programs over `bits` binary attributes
    /// ([`BranchingProgram::hamming_threshold`]): `bits` add steps and one map
    /// step, of state width `bits + 1`.
    pub fn hamming(bits: usize) -> Result<Self, PeError> {
        let mut steps = vec![StepKind::Add; bits];
        steps.push(StepKind::Map);
        Self::with_steps(bits, 2, bits + 1, &steps)
    }

    /// Whether every step is a lookup.
    pub const fn lookups_only(&self) -> bool {
        self.add_steps == 0 && self.map_steps == 0
    }

    fn has(&self, kind: StepKind) -> bool {
        (0..self.max_steps).any(|step| self.step_kind(step) == kind)
    }

    /// The RHS packages of a ciphertext: `l` lookup packages if a lookup step
    /// reads them, and `2 l` add packages if an add step reads them.
    pub fn rhs_packages(&self) -> usize {
        self.attribute_length *
            (usize::from(self.has(StepKind::Lookup)) + 2 * usize::from(self.has(StepKind::Add)))
    }

    /// Whether encryption publishes the lookup packages `X^(Bz x_t)`.
    pub fn has_lookup_packages(&self) -> bool {
        self.has(StepKind::Lookup)
    }

    /// Whether encryption publishes the add packages `X^(x_t)` and `X^(Bx - 1 - x_t)`.
    pub fn has_add_packages(&self) -> bool {
        self.has(StepKind::Add)
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

    /// The lookup width `W = next_power_of_two(Bz Bx)`, or `next_power_of_two(Bz)` when no
    /// lookup step reads an attribute into its table's input.
    pub fn lookup_width(&self) -> usize {
        if self.has(StepKind::Lookup) {
            (self.state_width * self.alphabet_size).next_power_of_two()
        } else {
            self.state_width.next_power_of_two()
        }
    }

    /// The entries of a table of step `step`: `Bz Bx` for a lookup, `Bz` for a map.
    pub fn table_entries(&self, step: usize) -> usize {
        match self.step_kind(step) {
            StepKind::Lookup => self.state_width * self.alphabet_size,
            StepKind::Map => self.state_width,
            StepKind::Add => 0,
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
        let adds = self
            .has_add_packages()
            .then(|| builder.rhs_family(IntExpr::constant(2 * self.attribute_length)))
            .transpose()
            .map_err(invalid)?;
        rhs.extend(attributes.map(|id| (id, ATTRIBUTE_COORDINATE)));
        rhs.extend(adds.map(|id| (id, ADD_ATTRIBUTE_COORDINATE)));
        let reads = builder.data_family(self.max_steps).map_err(invalid)?;
        let tables = builder.data_family(self.max_steps * width).map_err(invalid)?;
        let mut state = initial;
        for step in 0..self.max_steps {
            let selected = |family| Operand::Selected {
                family,
                data: reads,
                position: IntExpr::constant(step),
            };
            let input = match (self.step_kind(step), attributes, adds) {
                (StepKind::Add, _, Some(adds)) => {
                    state = builder
                        .mul_private(Operand::Scalar(state), selected(adds))
                        .map_err(invalid)?;
                    continue;
                }
                (StepKind::Lookup, Some(attributes), _) => builder
                    .mul_private(Operand::Scalar(state), selected(attributes))
                    .map_err(invalid)?,
                (StepKind::Map, _, _) => state,
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
/// step), and for an add step its complement flag (and an empty table).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BranchingProgram {
    class: PeClass,
    reads: Vec<usize>,
    tables: Vec<Vec<usize>>,
    #[serde(default)]
    complements: Vec<bool>,
}

impl BranchingProgram {
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
    /// `{0, 1}`, and every table input reachable below `Bz` (an add step adds up
    /// to `Bx - 1` to the state).
    pub fn from_steps(class: PeClass, steps: Vec<ProgramStep>) -> Result<Self, PeError> {
        let length = steps.len();
        let full = class.lookups_only() || length == class.max_steps;
        if length == 0 || length > class.max_steps || !full {
            return Err(PeError::Predicate("the program length is outside 1..=T_max"));
        }
        let offset = class.max_steps - length;
        let (mut reads, mut tables, mut complements) = (vec![], vec![], vec![]);
        // The largest reachable state.
        let mut state = 0;
        for (index, step) in steps.into_iter().enumerate() {
            let position = offset + index;
            let last = index + 1 == length;
            let (read, table, complement) = match step {
                ProgramStep::Lookup { read, table } => (Some(read), table, false),
                ProgramStep::Add { read, complement } => (Some(read), vec![], complement),
                ProgramStep::Map { table } => (None, table, false),
            };
            let kind = match (&read, table.is_empty()) {
                (Some(_), false) => StepKind::Lookup,
                (Some(_), true) => StepKind::Add,
                (None, _) => StepKind::Map,
            };
            if kind != class.step_kind(position) {
                return Err(PeError::Predicate("a step is not of its class kind"));
            }
            if read.is_some_and(|read| read >= class.attribute_length) {
                return Err(PeError::Predicate("a read is not an attribute index"));
            }
            if kind == StepKind::Add {
                state += class.alphabet_size - 1;
            } else {
                let bound = if last { 2 } else { class.state_width };
                if state >= class.state_width ||
                    table.len() != class.table_entries(position) ||
                    table.iter().any(|value| *value >= bound)
                {
                    return Err(PeError::Predicate("a transition table is out of range"));
                }
                state = table.iter().copied().max().unwrap_or(0);
            }
            reads.push(read.unwrap_or(0));
            tables.push(table);
            complements.push(complement);
        }
        Ok(Self { class, reads, tables, complements })
    }

    /// The threshold program of the Hamming distance to `template` over the class
    /// [`PeClass::hamming`] of `template.len()` bits: `F(x) = 0`, which decrypts,
    /// exactly when `x` differs from `template` in fewer than `threshold` bits.
    /// Step `i` adds `x_i` if `template_i = 0` and `1 - x_i` otherwise, and the
    /// final map step compares the sum with `threshold`.
    pub fn hamming_threshold(template: &[usize], threshold: usize) -> Result<Self, PeError> {
        let class = PeClass::hamming(template.len())?;
        if template.iter().any(|bit| *bit > 1) {
            return Err(PeError::Predicate("a template bit is not binary"));
        }
        let mut steps = template
            .iter()
            .enumerate()
            .map(|(read, bit)| ProgramStep::Add { read, complement: *bit == 1 })
            .collect::<Vec<_>>();
        steps.push(ProgramStep::Map {
            table: (0..class.state_width)
                .map(|distance| usize::from(distance >= threshold))
                .collect(),
        });
        Self::from_steps(class, steps)
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

    /// The complement flags of the steps (set only on add steps).
    pub fn complements(&self) -> &[bool] {
        &self.complements
    }

    fn complement(&self, index: usize) -> bool {
        self.complements.get(index).copied().unwrap_or(false)
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
                StepKind::Add if self.complement(index) => {
                    state + self.class.alphabet_size - 1 - symbol
                }
                StepKind::Add => state + symbol,
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
        Self {
            class,
            reads: vec![0; class.max_steps],
            tables,
            complements: vec![false; class.max_steps],
        }
    }

    /// The Program data of this program ([`PeClass::compile`]): the reads (an
    /// add step's read `t` of the add family, `t + l` for a complement) and
    /// the tables, each padded to the lookup width `W` with zeros (an add step's
    /// all zeros), of the `T_max` steps, the first `T_max - T` of them identity
    /// steps.
    pub fn data(&self, compiled: &CompiledPredicate) -> ProgramData {
        let class = &self.class;
        let width = class.lookup_width();
        let padding = class.max_steps - self.length();
        let identity = (0..class.state_width * class.alphabet_size)
            .map(|index| index % class.state_width)
            .collect::<Vec<_>>();
        let reads = std::iter::repeat_n(0, padding)
            .chain(self.reads.iter().enumerate().map(|(index, read)| {
                read + class.attribute_length * usize::from(self.complement(index))
            }))
            .collect();
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

    /// The Hamming threshold program decrypts exactly the attributes within distance
    /// `threshold - 1` of its template: its add steps add `x_i` or `1 - x_i`, and its map step
    /// compares the sum, which stays below the state width `bits + 1`.
    #[test]
    fn hamming_threshold_programs_compare_the_distance() {
        let template = [1, 0, 1, 1];
        let program = BranchingProgram::hamming_threshold(&template, 2).unwrap();
        let class = *program.class();
        assert_eq!(class, PeClass::hamming(4).unwrap());
        assert_eq!(class.step_kind(0), StepKind::Add);
        assert_eq!(class.step_kind(4), StepKind::Map);
        assert_eq!(class.lookup_width(), 8);
        assert_eq!(class.rhs_packages(), 8);
        assert!(!class.has_lookup_packages());
        for value in 0..16usize {
            let bits = (0..4).map(|i| value >> i & 1).collect::<Vec<_>>();
            let distance = bits.iter().zip(&template).filter(|(x, z)| x != z).count();
            assert_eq!(program.evaluate(&bits).unwrap(), usize::from(distance >= 2), "{bits:?}");
        }
        // A sum that can reach the state width, a lookup where the class adds, and a short
        // program are rejected.
        let narrow = PeClass::with_steps(
            4,
            2,
            4,
            &[StepKind::Add; 4].iter().copied().chain([StepKind::Map]).collect::<Vec<_>>(),
        )
        .unwrap();
        let adds = (0..4).map(|read| ProgramStep::Add { read, complement: false });
        let map = ProgramStep::Map { table: vec![0; 4] };
        assert!(BranchingProgram::from_steps(narrow, adds.clone().chain([map]).collect()).is_err());
        let lookup = ProgramStep::Lookup { read: 0, table: vec![0; 10] };
        let mut steps = adds.collect::<Vec<_>>();
        steps[0] = lookup;
        steps.push(ProgramStep::Map { table: vec![0; 5] });
        assert!(BranchingProgram::from_steps(class, steps).is_err());
        assert!(
            BranchingProgram::from_steps(class, vec![ProgramStep::Map { table: vec![0; 5] }])
                .is_err()
        );
        assert!(PeClass::with_steps(1, 2, 2, &[StepKind::Map, StepKind::Add]).is_err());
    }

    /// The Hamming class compiles to products by the add family and one table, and a program's
    /// reads select `X^(1 - x_i)` (member `i + l`) for the template bits that are one.
    #[test]
    fn the_hamming_class_reads_the_add_family() {
        let template = [1, 0, 1];
        let program = BranchingProgram::hamming_threshold(&template, 1).unwrap();
        let compiled = program.class().compile().unwrap();
        assert_eq!(compiled.program().rhs_inputs().len(), 1);
        assert_eq!(compiled.program().encryption_inputs().luts().len(), 1);
        let data = program.data(&compiled);
        data.validate(compiled.program()).unwrap();
        assert_eq!(data.values(compiled.reads()).unwrap(), [3, 1, 5, 0]);
        let width = program.class().lookup_width();
        let tables = data.values(compiled.tables()).unwrap();
        assert_eq!(&tables[3 * width..][..4], [0, 1, 1, 1]);
        assert!(tables[..3 * width].iter().all(|value| *value == 0));
        let longest = BranchingProgram::longest(*program.class());
        assert_eq!(longest.class().compile().unwrap().program().id(), compiled.program().id());
    }
}
