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

/// The function class of a setup: attribute length `l`, alphabet size `Bx`,
/// state width `Bz`, and the largest program length `T_max`. One lookup
/// width `W` and automorphism group serve every program of the class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeClass {
    attribute_length: usize,
    alphabet_size: usize,
    state_width: usize,
    max_steps: usize,
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
        Ok(Self { attribute_length, alphabet_size, state_width, max_steps })
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
        let attributes =
            builder.rhs_family(IntExpr::constant(self.attribute_length)).map_err(invalid)?;
        let reads = builder.data_family(self.max_steps).map_err(invalid)?;
        let tables = builder.data_family(self.max_steps * width).map_err(invalid)?;
        let mut state = initial;
        for step in 0..self.max_steps {
            let product = builder
                .mul_private(
                    Operand::Scalar(state),
                    Operand::Selected {
                        family: attributes,
                        data: reads,
                        position: IntExpr::constant(step),
                    },
                )
                .map_err(invalid)?;
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
            state = builder.lut(Operand::Scalar(product), table).map_err(invalid)?;
        }
        let program = builder.finish(vec![Operand::Scalar(state)]).map_err(invalid)?;
        let binding = RnsBggDynamicBinding::new(
            BTreeMap::from([(initial, INITIAL_STATE_COORDINATE)]),
            BTreeMap::from([(attributes, ATTRIBUTE_COORDINATE)]),
        );
        Ok(CompiledPredicate { program, binding, reads, tables })
    }
}

/// A checked branching program of a class: its read map and transition
/// tables, table `i` at index `z + Bz x` holding `f_i(z, x)`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BranchingProgram {
    class: PeClass,
    reads: Vec<usize>,
    tables: Vec<Vec<usize>>,
}

impl BranchingProgram {
    /// Check a program of `class`: `1 <= T <= T_max` steps, every read an
    /// attribute index, every table of `Bz Bx` entries, intermediate outputs
    /// below `Bz`, and final outputs in `{0, 1}`.
    pub fn new(
        class: PeClass,
        reads: Vec<usize>,
        tables: Vec<Vec<usize>>,
    ) -> Result<Self, PeError> {
        let length = reads.len();
        if length == 0 || length > class.max_steps || tables.len() != length {
            return Err(PeError::Predicate("the program length is outside 1..=T_max"));
        }
        if reads.iter().any(|read| *read >= class.attribute_length) {
            return Err(PeError::Predicate("a read is not an attribute index"));
        }
        let entries = class.state_width * class.alphabet_size;
        for (step, table) in tables.iter().enumerate() {
            let bound = if step + 1 == length { 2 } else { class.state_width };
            if table.len() != entries || table.iter().any(|value| *value >= bound) {
                return Err(PeError::Predicate("a transition table is out of range"));
            }
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

    /// `F(x)`, evaluated in the clear: a key for the program decrypts a
    /// ciphertext of `x` when it is zero.
    pub fn evaluate(&self, attribute: &[usize]) -> Result<usize, PeError> {
        self.class.validate_attribute(attribute)?;
        let mut state = 0;
        for (read, table) in self.reads.iter().zip(&self.tables) {
            state = table[state + self.class.state_width * attribute[*read]];
        }
        Ok(state)
    }

    /// A program of the class's maximal length whose every table is
    /// nonzero on its range: parameter selection simulates its structure,
    /// which bounds every program of the class (the bounds depend on the
    /// length, the width, and the output payload, not on table values).
    pub fn longest(class: PeClass) -> Self {
        let entries = class.state_width * class.alphabet_size;
        let tables = (0..class.max_steps)
            .map(|step| {
                (0..entries)
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
    /// the tables, each padded to the lookup width `W` with zeros, of the
    /// `T_max` steps, the first `T_max - T` of them identity steps.
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
}
