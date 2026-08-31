//! A validated, host-side Power-LUT computation program.
//!
//! [`PowerLutProgram`] is the shared description consumed by both private
//! encoding and public-key lowerers. It contains only public scheme metadata:
//! input widths, LUT tables, gate wiring, and explicit RHS/family identifiers.
//! It does not add an IR node and does not carry private vectors, GSW
//! ciphertexts, sparse support, selected slots, or refresh state.
//!
//! A binary gate always names an [`RhsInputId`]. An in-flight computed wire is
//! therefore never accidentally reused as an RHS package. One-hot gates are
//! represented and validated independently of any sparse schedule. Lowerers
//! receive explicit selector-family packages and public weighting values, so
//! the program itself never stores private support or selected-slot data.

use std::collections::BTreeMap;

use mxx_ir_core::IntExpr;
use serde::{Deserialize, Serialize};
use thiserror::Error;

macro_rules! id_type {
    ($name:ident, $doc:literal) => {
        #[derive(
            Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
        )]
        #[doc = $doc]
        pub struct $name(usize);
        impl $name {
            /// Creates an identifier from a builder-local index.
            pub const fn from_index(index: usize) -> Self {
                Self(index)
            }
            /// Returns the builder-local index.
            pub const fn index(self) -> usize {
                self.0
            }
        }
    };
}

id_type!(ProgramInputId, "Identifier of a declared program input.");
id_type!(ProgramWireId, "Identifier of a program wire, including inputs and gate outputs.");
id_type!(RhsInputId, "Identifier of an explicit RHS input package.");
id_type!(RhsFamilyId, "Identifier of a selector/RHS family declaration.");
id_type!(PublicValueFamilyId, "Identifier of a public value family declaration.");
id_type!(LutId, "Identifier of a LUT table in a Power-LUT program.");
id_type!(ProgramGateId, "Identifier of a gate in declaration order.");

/// Canonical SHA-256 identity of a complete Power-LUT program.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct PowerLutProgramId(pub(crate) [u8; 32]);

impl PowerLutProgramId {
    /// Creates an identity from a caller-owned canonical digest.
    pub(crate) const fn from_digest(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the raw canonical digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
    /// Returns a lower-case hexadecimal identity suitable for artifact names.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// A LUT table with statically declared input and output widths.
///
/// Entries are output exponents, indexed by the encoded input domain. A unary
/// table has one entry for each primary input value. A binary table uses the
/// explicit mapping `index = lhs + lhs_width * rhs`; stating the formula here
/// avoids relying on an ambiguous meaning of “row-major” across callers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LutTable {
    input_width: usize,
    rhs_width: Option<usize>,
    output_width: usize,
    values: Vec<usize>,
}

impl LutTable {
    /// Creates a unary table with one entry per input value.
    ///
    /// `output_width` describes the output exponent domain; table construction
    /// validates dimensions and entry count, while the lowerer validates the
    /// operation-specific matrix shapes.
    pub fn unary(
        input_width: usize,
        output_width: usize,
        values: Vec<usize>,
    ) -> Result<Self, ProgramValidationError> {
        let table = Self { input_width, rhs_width: None, output_width, values };
        validate_table(&table)?;
        Ok(table)
    }

    /// Creates a binary table with `lhs_width * rhs_width` entries.
    ///
    /// The canonical entry for `(lhs, rhs)` is at
    /// `lhs + lhs_width * rhs`.
    pub fn binary(
        lhs_width: usize,
        rhs_width: usize,
        output_width: usize,
        values: Vec<usize>,
    ) -> Result<Self, ProgramValidationError> {
        let table =
            Self { input_width: lhs_width, rhs_width: Some(rhs_width), output_width, values };
        validate_table(&table)?;
        Ok(table)
    }

    /// Returns the primary (unary or left) input width.
    pub const fn input_width(&self) -> usize {
        self.input_width
    }
    /// Returns the right-hand input width for a binary table.
    pub const fn rhs_width(&self) -> Option<usize> {
        self.rhs_width
    }
    /// Returns the declared width of the gate output.
    pub const fn output_width(&self) -> usize {
        self.output_width
    }
    /// Returns table entries in canonical order.
    pub fn values(&self) -> &[usize] {
        &self.values
    }
}

/// Metadata for one declared explicit RHS input.
///
/// A binary gate binds its `RhsInputId` to an externally prepared RHS package.
/// The RHS is never a computed [`ProgramWireId`]; `lhs_width` and `rhs_width`
/// describe the two input domains that the selected binary table must accept.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RhsInputDeclaration {
    family: RhsFamilyId,
    lhs_width: usize,
    rhs_width: usize,
}

/// Metadata for one selector/RHS family.
///
/// The width is the per-element encoded selector domain. Runtime cardinality
/// is supplied separately by a structural family binding, so one declaration
/// can be reused for different bucket ranges.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RhsFamilyDeclaration {
    width: usize,
}

/// Metadata for one public value family.
///
/// The width is the per-element public value domain paired with a selector
/// family. The number of elements is runtime data and is not part of the
/// value-agnostic program identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PublicValueFamilyDeclaration {
    width: usize,
}

/// A contiguous view into a flattened runtime family.
///
/// `start` and `count` are DSL integer expressions rather than host-side
/// indices. This lets a structural bucket loop select its own range from one
/// flattened selector/value family without constructing a separate Rust
/// graph body for every bucket. The view is runtime metadata; it is not part
/// of the program identity or IR schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyRange {
    start: IntExpr,
    count: IntExpr,
    capacity: IntExpr,
}

impl FamilyRange {
    /// Creates a non-empty range. Constant negative or zero bounds are
    /// rejected immediately; symbolic bounds are checked by graph
    /// validation when the enclosing DSL graph is frozen.
    pub fn new(
        start: impl Into<IntExpr>,
        count: impl Into<IntExpr>,
    ) -> Result<Self, ProgramValidationError> {
        let start = start.into();
        let count = count.into();
        let valid = match (
            start.evaluate(&mxx_ir_core::ParamEnv::default()),
            count.evaluate(&mxx_ir_core::ParamEnv::default()),
        ) {
            (Ok(start), Ok(count)) => start >= 0.into() && count > 0.into(),
            _ => true,
        };
        if !valid {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        Ok(Self { start: start.clone(), count: count.clone(), capacity: count })
    }

    /// Creates a non-empty view with a fixed structural capacity.
    ///
    /// The logical `count` may depend on a surrounding loop, but `capacity`
    /// is concrete. Backends can mask the inactive tail while keeping indexed
    /// family wire types independent of parent loop binders.
    pub fn bounded(
        start: impl Into<IntExpr>,
        count: impl Into<IntExpr>,
        capacity: usize,
    ) -> Result<Self, ProgramValidationError> {
        let start = start.into();
        let count = count.into();
        if capacity == 0 {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        let valid = match count.evaluate(&mxx_ir_core::ParamEnv::default()) {
            Ok(count) => count > 0.into() && count <= capacity.into(),
            _ => true,
        };
        if !valid {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        Ok(Self { start, count, capacity: IntExpr::constant(capacity) })
    }

    /// Creates a view spanning an entire family with the supplied count.
    pub fn full(count: impl Into<IntExpr>) -> Result<Self, ProgramValidationError> {
        Self::new(0usize, count)
    }

    /// Returns the first flattened family index.
    pub fn start(&self) -> &IntExpr {
        &self.start
    }

    /// Returns the number of elements in the view.
    pub fn count(&self) -> &IntExpr {
        &self.count
    }

    /// Returns the fixed structural capacity of this range.
    pub fn capacity(&self) -> &IntExpr {
        &self.capacity
    }

    /// Checks a range against a family count whenever all bounds are static.
    pub(crate) fn is_within(&self, family_count: &IntExpr) -> bool {
        match (
            self.start.evaluate(&mxx_ir_core::ParamEnv::default()),
            self.count.evaluate(&mxx_ir_core::ParamEnv::default()),
            family_count.evaluate(&mxx_ir_core::ParamEnv::default()),
        ) {
            (Ok(start), Ok(count), Ok(total)) => {
                let capacity = self.capacity.evaluate(&mxx_ir_core::ParamEnv::default()).ok();
                start >= 0.into() &&
                    count > 0.into() &&
                    capacity.is_none_or(|capacity| start + capacity <= total)
            }
            _ => true,
        }
    }
}

/// Runtime range bindings for one-hot selector and public-value families.
///
/// A range must be supplied for both sides of a one-hot gate. The generic
/// traversal requires them to be identical, preventing a selector family
/// from being paired with values from a different bucket. Callers that need
/// the complete family can use [`FamilyRange::full`].
#[derive(Clone, Debug, Default)]
pub struct ProgramFamilyRanges {
    selector: BTreeMap<RhsFamilyId, FamilyRange>,
    public_values: BTreeMap<PublicValueFamilyId, FamilyRange>,
}

impl ProgramFamilyRanges {
    /// Creates empty range bindings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds a range to a selector family declaration.
    pub fn selector(&mut self, id: RhsFamilyId, range: FamilyRange) -> Option<FamilyRange> {
        self.selector.insert(id, range)
    }

    /// Binds a corresponding range to a public-value family declaration.
    pub fn public_values(
        &mut self,
        id: PublicValueFamilyId,
        range: FamilyRange,
    ) -> Option<FamilyRange> {
        self.public_values.insert(id, range)
    }

    pub(crate) fn selector_range(&self, id: RhsFamilyId) -> Option<&FamilyRange> {
        self.selector.get(&id)
    }

    pub(crate) fn public_value_range(&self, id: PublicValueFamilyId) -> Option<&FamilyRange> {
        self.public_values.get(&id)
    }
}

/// One program gate. RHS material is always referenced by `RhsInputId`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramGate {
    /// Applies a unary LUT to a previously defined wire.
    Unary {
        /// Input wire consumed by the gate.
        input: ProgramWireId,
        /// Unary LUT selected by the gate.
        lut: LutId,
        /// Newly defined output wire.
        output: ProgramWireId,
    },
    /// Applies a binary LUT using an explicit, separately declared RHS input.
    Binary {
        /// Left-hand input wire.
        lhs: ProgramWireId,
        /// Explicit RHS package input; this is never a computed wire.
        rhs: RhsInputId,
        /// Binary LUT selected by the gate.
        lut: LutId,
        /// Newly defined output wire.
        output: ProgramWireId,
    },
    /// Applies one selector-family element at a time to a paired public-value
    /// family, then evaluates the resulting value through a unary LUT.
    ///
    /// The selector family is an explicit runtime input, not a private support
    /// list stored in the program. Lowerers implement this gate with one
    /// reusable structural loop body.
    OneHot {
        /// Input wire consumed by the gate.
        lhs: ProgramWireId,
        /// Declared selector/RHS family.
        selector_family: RhsFamilyId,
        /// Declared public value family.
        public_value_family: PublicValueFamilyId,
        /// Unary LUT selected by the gate.
        lut: LutId,
        /// Newly defined output wire.
        output: ProgramWireId,
    },
}

/// Errors returned while declaring or validating a program.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ProgramValidationError {
    #[error("program identifier has already been declared")]
    /// An automatically allocated identifier would collide with an existing declaration.
    DuplicateIdentifier,
    #[error("program references an undefined wire: {0:?}")]
    /// A gate refers to a wire that has not been defined yet.
    UndefinedWire(ProgramWireId),
    #[error("program references an undefined LUT: {0:?}")]
    /// A gate refers to a LUT that has not been declared.
    UndefinedLut(LutId),
    #[error("program references an undefined RHS input: {0:?}")]
    /// A binary gate refers to an undeclared explicit RHS input.
    UndefinedRhsInput(RhsInputId),
    #[error("program references an undefined RHS family: {0:?}")]
    /// A one-hot gate refers to an undeclared selector family.
    UndefinedRhsFamily(RhsFamilyId),
    #[error("program references an undefined public value family: {0:?}")]
    /// A one-hot gate refers to an undeclared public value family.
    UndefinedPublicValueFamily(PublicValueFamilyId),
    #[error("program has incompatible widths or reuses a gate output")]
    /// A LUT shape does not match its declared input or output family.
    WidthMismatch,
    #[error("program LUT has an invalid width or entry count")]
    /// A LUT has an empty or incorrectly sized table.
    InvalidLutTable,
    #[error("program output is duplicated or is not defined")]
    /// An output was not defined by an input or gate, or was marked twice.
    InvalidOutput,
    #[error("program builder reached an inconsistent internal state")]
    /// Internal builder maps are inconsistent.
    InvalidBuilderState,
    #[error("one-hot family range is empty, negative, or outside its family")]
    /// A runtime family view has invalid statically known bounds.
    InvalidFamilyRange,
    #[error("runtime value for program input is missing: {0:?}")]
    /// The private or public runtime input map omitted a declared input.
    MissingRuntimeInput(ProgramInputId),
    #[error("runtime value for RHS input is missing: {0:?}")]
    /// The runtime RHS map omitted a declared explicit RHS input.
    MissingRuntimeRhs(RhsInputId),
}

/// A validated immutable Power-LUT program.
///
/// The dataflow is `declared inputs -> wires -> ordered gates -> outputs`.
/// Its canonical identity covers declarations, tables, gate wiring, and output
/// order, allowing independent encoding and public-key lowerers to consume the
/// same graph description. Builder-created values are validated before being
/// returned; callers deserializing this type should still treat untrusted data
/// as requiring validation at its artifact boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerLutProgram {
    id: PowerLutProgramId,
    inputs: BTreeMap<ProgramInputId, usize>,
    input_wires: BTreeMap<ProgramInputId, ProgramWireId>,
    rhs_inputs: BTreeMap<RhsInputId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    public_value_families: BTreeMap<PublicValueFamilyId, PublicValueFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    gates: Vec<ProgramGate>,
    outputs: Vec<ProgramWireId>,
}

impl PowerLutProgram {
    /// Returns the canonical identity of the declarations and gate dataflow.
    pub const fn id(&self) -> PowerLutProgramId {
        self.id
    }
    /// Returns declared input widths keyed by input identifier.
    pub fn inputs(&self) -> &BTreeMap<ProgramInputId, usize> {
        &self.inputs
    }
    /// Returns the canonical mapping from each declared input to its wire.
    pub fn input_wires(&self) -> &BTreeMap<ProgramInputId, ProgramWireId> {
        &self.input_wires
    }
    /// Returns gates in deterministic declaration order.
    pub fn gates(&self) -> &[ProgramGate] {
        &self.gates
    }
    /// Returns output wires in the order committed by the builder.
    pub fn outputs(&self) -> &[ProgramWireId] {
        &self.outputs
    }
    /// Returns a declared LUT by identifier.
    pub fn lut(&self, id: LutId) -> Option<&LutTable> {
        self.luts.get(&id)
    }
    /// Returns explicit RHS declarations, never computed-wire declarations.
    pub fn rhs_inputs(&self) -> &BTreeMap<RhsInputId, RhsInputDeclaration> {
        &self.rhs_inputs
    }
    /// Returns the declared selector-family width, if present.
    pub fn rhs_family_width(&self, id: RhsFamilyId) -> Option<usize> {
        self.rhs_families.get(&id).map(|family| family.width)
    }
    /// Returns the declared public-value-family width, if present.
    pub fn public_value_family_width(&self, id: PublicValueFamilyId) -> Option<usize> {
        self.public_value_families.get(&id).map(|family| family.width)
    }
}

/// Mutable builder for a validated immutable [`PowerLutProgram`].
#[derive(Default)]
pub struct PowerLutProgramBuilder {
    inputs: BTreeMap<ProgramInputId, usize>,
    input_wires: BTreeMap<ProgramInputId, ProgramWireId>,
    rhs_inputs: BTreeMap<RhsInputId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    public_value_families: BTreeMap<PublicValueFamilyId, PublicValueFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    gates: Vec<ProgramGate>,
    wires: BTreeMap<ProgramWireId, usize>,
    outputs: Vec<ProgramWireId>,
}

impl PowerLutProgramBuilder {
    /// Creates an empty program builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares an input and returns its input identifier.
    pub fn input(&mut self, width: usize) -> Result<ProgramInputId, ProgramValidationError> {
        let id = ProgramInputId::from_index(self.inputs.len());
        let wire = ProgramWireId::from_index(self.wires.len());
        if width == 0 ||
            self.inputs.insert(id, width).is_some() ||
            self.input_wires.insert(id, wire).is_some() ||
            self.wires.insert(wire, width).is_some()
        {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Returns the wire associated with a declared input.
    pub fn input_wire(&self, id: ProgramInputId) -> Result<ProgramWireId, ProgramValidationError> {
        self.input_wires
            .get(&id)
            .copied()
            .ok_or(ProgramValidationError::UndefinedWire(ProgramWireId::from_index(id.index())))
    }

    /// Declares a selector/RHS family with a known public width.
    pub fn rhs_family(&mut self, width: usize) -> Result<RhsFamilyId, ProgramValidationError> {
        if width == 0 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let id = RhsFamilyId::from_index(self.rhs_families.len());
        if self.rhs_families.insert(id, RhsFamilyDeclaration { width }).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Declares a public value family.
    pub fn public_value_family(
        &mut self,
        width: usize,
    ) -> Result<PublicValueFamilyId, ProgramValidationError> {
        if width == 0 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let id = PublicValueFamilyId::from_index(self.public_value_families.len());
        if self.public_value_families.insert(id, PublicValueFamilyDeclaration { width }).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Declares an explicit RHS package input for a binary gate.
    pub fn rhs_input(
        &mut self,
        family: RhsFamilyId,
        lhs_width: usize,
        rhs_width: usize,
    ) -> Result<RhsInputId, ProgramValidationError> {
        if self.rhs_families.get(&family).map(|f| f.width) != Some(rhs_width) {
            return Err(ProgramValidationError::UndefinedRhsFamily(family));
        }
        let id = RhsInputId::from_index(self.rhs_inputs.len());
        if self
            .rhs_inputs
            .insert(id, RhsInputDeclaration { family, lhs_width, rhs_width })
            .is_some()
        {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Adds a LUT table and returns its identifier.
    pub fn lut(&mut self, table: LutTable) -> Result<LutId, ProgramValidationError> {
        validate_table(&table)?;
        let id = LutId::from_index(self.luts.len());
        if self.luts.insert(id, table).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Adds a unary gate and returns its output wire.
    pub fn unary(
        &mut self,
        input: ProgramWireId,
        lut: LutId,
    ) -> Result<ProgramWireId, ProgramValidationError> {
        let table = self.luts.get(&lut).ok_or(ProgramValidationError::UndefinedLut(lut))?;
        if table.rhs_width().is_some() || self.wire_width(input)? != table.input_width() {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = ProgramWireId::from_index(self.wires.len());
        self.wires.insert(output, table.output_width());
        self.gates.push(ProgramGate::Unary { input, lut, output });
        Ok(output)
    }

    /// Adds a binary gate whose RHS is an explicit declared input package.
    pub fn binary(
        &mut self,
        lhs: ProgramWireId,
        rhs: RhsInputId,
        lut: LutId,
    ) -> Result<ProgramWireId, ProgramValidationError> {
        let table = self.luts.get(&lut).ok_or(ProgramValidationError::UndefinedLut(lut))?;
        let rhs_decl =
            self.rhs_inputs.get(&rhs).ok_or(ProgramValidationError::UndefinedRhsInput(rhs))?;
        if table.rhs_width() != Some(rhs_decl.rhs_width) ||
            table.input_width() != rhs_decl.lhs_width ||
            self.wire_width(lhs)? != rhs_decl.lhs_width
        {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = ProgramWireId::from_index(self.wires.len());
        self.wires.insert(output, table.output_width());
        self.gates.push(ProgramGate::Binary { lhs, rhs, lut, output });
        Ok(output)
    }

    /// Adds a validated one-hot gate for a selector-family lowerer.
    pub fn one_hot(
        &mut self,
        lhs: ProgramWireId,
        selector_family: RhsFamilyId,
        public_value_family: PublicValueFamilyId,
        lut: LutId,
    ) -> Result<ProgramWireId, ProgramValidationError> {
        let selector = self
            .rhs_families
            .get(&selector_family)
            .ok_or(ProgramValidationError::UndefinedRhsFamily(selector_family))?;
        let values = self
            .public_value_families
            .get(&public_value_family)
            .ok_or(ProgramValidationError::UndefinedPublicValueFamily(public_value_family))?;
        let table = self.luts.get(&lut).ok_or(ProgramValidationError::UndefinedLut(lut))?;
        if table.rhs_width().is_some() ||
            self.wire_width(lhs)? != selector.width ||
            table.input_width() != values.width
        {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = ProgramWireId::from_index(self.wires.len());
        self.wires.insert(output, table.output_width());
        self.gates.push(ProgramGate::OneHot {
            lhs,
            selector_family,
            public_value_family,
            lut,
            output,
        });
        Ok(output)
    }

    /// Marks a defined wire as a program output.
    pub fn output(&mut self, wire: ProgramWireId) -> Result<(), ProgramValidationError> {
        if !self.wires.contains_key(&wire) || !self.outputs.iter().all(|existing| *existing != wire)
        {
            return Err(ProgramValidationError::InvalidOutput);
        }
        self.outputs.push(wire);
        Ok(())
    }

    /// Finalizes and canonically identifies the program.
    pub fn build(self) -> Result<PowerLutProgram, ProgramValidationError> {
        if self.outputs.is_empty() {
            return Err(ProgramValidationError::InvalidOutput);
        }
        validate_builder(&self)?;
        let canonical = serde_json::to_vec(&(
            &self.inputs,
            &self.input_wires,
            &self.rhs_inputs,
            &self.rhs_families,
            &self.public_value_families,
            &self.luts,
            &self.gates,
            &self.outputs,
        ))
        .map_err(|_| ProgramValidationError::InvalidBuilderState)?;
        let program = PowerLutProgram {
            id: canonical_program_id(&canonical),
            inputs: self.inputs,
            input_wires: self.input_wires,
            rhs_inputs: self.rhs_inputs,
            rhs_families: self.rhs_families,
            public_value_families: self.public_value_families,
            luts: self.luts,
            gates: self.gates,
            outputs: self.outputs,
        };
        Ok(program)
    }

    fn wire_width(&self, wire: ProgramWireId) -> Result<usize, ProgramValidationError> {
        self.wires.get(&wire).copied().ok_or(ProgramValidationError::UndefinedWire(wire))
    }

    fn lut_definition(&self, id: LutId) -> Result<&LutTable, ProgramValidationError> {
        self.luts.get(&id).ok_or(ProgramValidationError::UndefinedLut(id))
    }
}

// Program shape validation.

pub(crate) fn validate_builder(
    builder: &PowerLutProgramBuilder,
) -> Result<(), ProgramValidationError> {
    if builder.inputs.len() != builder.input_wires.len() {
        return Err(ProgramValidationError::InvalidBuilderState);
    }
    for gate in &builder.gates {
        match gate {
            ProgramGate::Unary { input, lut, output } => {
                let width = builder.wire_width(*input)?;
                let table = builder.lut_definition(*lut)?;
                if table.input_width() != width || !builder.wires.contains_key(output) {
                    return Err(ProgramValidationError::WidthMismatch);
                }
            }
            ProgramGate::Binary { lhs, rhs, lut, output } => {
                let lhs_width = builder.wire_width(*lhs)?;
                let rhs_input = builder
                    .rhs_inputs
                    .get(rhs)
                    .ok_or(ProgramValidationError::UndefinedRhsInput(*rhs))?;
                let table = builder.lut_definition(*lut)?;
                if table.input_width() != rhs_input.lhs_width ||
                    table.rhs_width() != Some(rhs_input.rhs_width) ||
                    lhs_width != rhs_input.lhs_width ||
                    !builder.wires.contains_key(output)
                {
                    return Err(ProgramValidationError::WidthMismatch);
                }
            }
            ProgramGate::OneHot { lhs, selector_family, public_value_family, lut, output } => {
                let lhs_width = builder.wire_width(*lhs)?;
                let selector = builder
                    .rhs_families
                    .get(selector_family)
                    .ok_or(ProgramValidationError::UndefinedRhsFamily(*selector_family))?;
                let values = builder.public_value_families.get(public_value_family).ok_or(
                    ProgramValidationError::UndefinedPublicValueFamily(*public_value_family),
                )?;
                let table = builder.lut_definition(*lut)?;
                if lhs_width != selector.width ||
                    table.input_width() != values.width ||
                    !builder.wires.contains_key(output)
                {
                    return Err(ProgramValidationError::WidthMismatch);
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_table(table: &LutTable) -> Result<(), ProgramValidationError> {
    if table.input_width() == 0 || table.output_width() == 0 || table.values.is_empty() {
        return Err(ProgramValidationError::InvalidLutTable);
    }
    let expected = match table.rhs_width() {
        Some(rhs) => {
            table.input_width().checked_mul(rhs).ok_or(ProgramValidationError::InvalidLutTable)?
        }
        None => table.input_width(),
    };
    if expected != table.values.len() {
        return Err(ProgramValidationError::InvalidLutTable);
    }
    Ok(())
}

// Shared runtime binding and traversal.

use crate::PowerLutError;

/// Runtime values bound to the declarations of one [`PowerLutProgram`].
///
/// The type parameters keep the private encoding and public-key paths
/// separate. In particular, a public binding can use only public key wires,
/// public RHS projections, and public value matrices. This container is
/// crate-private because application modules should expose purpose-specific
/// constructors rather than make callers assemble untyped maps by hand.
pub(crate) struct ProgramBindings<'a, W, R, SF, VF, H> {
    pub(crate) inputs: &'a BTreeMap<ProgramInputId, W>,
    pub(crate) rhs_inputs: &'a BTreeMap<RhsInputId, R>,
    pub(crate) one_hot_selectors: &'a BTreeMap<RhsFamilyId, SF>,
    pub(crate) public_values: &'a BTreeMap<PublicValueFamilyId, VF>,
    pub(crate) helpers: &'a [H],
}

impl<'a, W, R, SF, VF, H> ProgramBindings<'a, W, R, SF, VF, H> {
    pub(crate) fn new(
        inputs: &'a BTreeMap<ProgramInputId, W>,
        rhs_inputs: &'a BTreeMap<RhsInputId, R>,
        one_hot_selectors: &'a BTreeMap<RhsFamilyId, SF>,
        public_values: &'a BTreeMap<PublicValueFamilyId, VF>,
        helpers: &'a [H],
    ) -> Self {
        Self { inputs, rhs_inputs, one_hot_selectors, public_values, helpers }
    }
}

/// Cryptographic callbacks used by shared program traversal.
///
/// This deliberately contains no table lookup or runtime binding logic. The
/// caller receives a validated table and the exact bound RHS/family values;
/// implementations remain independent between encoding and public-key paths.
pub(crate) trait ProgramLoweringBackend {
    type Wire: Clone;
    type Rhs;
    /// Structural selector-family binding consumed by a one-hot gate.
    ///
    /// This is deliberately a family-level value, rather than an element or
    /// `Vec`.  A backend can therefore construct one reusable loop body and
    /// perform dynamic family access inside that body.
    type SelectorFamily;
    /// Structural public-value family paired with [`Self::SelectorFamily`].
    type PublicValueFamily;
    type Helper;

    fn unary(
        &self,
        input: Self::Wire,
        table: &LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError>;

    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError>;

    fn one_hot(
        &self,
        lhs: Self::Wire,
        selectors: &Self::SelectorFamily,
        public_values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        public_value_range: &FamilyRange,
        table: &LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, PowerLutError>;
}

/// Resolves declarations and runtime bindings once, then traverses every gate
/// in canonical order through a cryptographic backend.
///
/// This function owns wire availability, LUT/RHS/family lookup, range pairing,
/// and output collection. It deliberately does not own a cryptographic
/// formula: encoding and public-key backends implement those formulas
/// independently. Family bindings remain DSL families so OneHot lowering can
/// use one structural loop body rather than host-unrolling each cell.
pub(crate) fn lower_program<B: ProgramLoweringBackend>(
    program: &PowerLutProgram,
    bindings: &ProgramBindings<
        '_,
        B::Wire,
        B::Rhs,
        B::SelectorFamily,
        B::PublicValueFamily,
        B::Helper,
    >,
    family_ranges: &ProgramFamilyRanges,
    backend: &B,
) -> Result<BTreeMap<ProgramWireId, B::Wire>, PowerLutError> {
    let mut wires = BTreeMap::new();
    for (input_id, wire_id) in program.input_wires() {
        let value = bindings
            .inputs
            .get(input_id)
            .cloned()
            .ok_or(ProgramValidationError::MissingRuntimeInput(*input_id))?;
        wires.insert(*wire_id, value);
    }

    for gate in program.gates() {
        let (output, value) = match gate {
            ProgramGate::Unary { input, lut, output } => {
                let input = wires
                    .get(input)
                    .cloned()
                    .ok_or(ProgramValidationError::UndefinedWire(*input))?;
                let table = program.lut(*lut).ok_or(ProgramValidationError::UndefinedLut(*lut))?;
                validate_unary_table(table)?;
                (*output, backend.unary(input, table, bindings.helpers)?)
            }
            ProgramGate::Binary { lhs, rhs, lut, output } => {
                let lhs_value =
                    wires.get(lhs).cloned().ok_or(ProgramValidationError::UndefinedWire(*lhs))?;
                let table = program.lut(*lut).ok_or(ProgramValidationError::UndefinedLut(*lut))?;
                let declaration = program
                    .rhs_inputs()
                    .get(rhs)
                    .ok_or(ProgramValidationError::UndefinedRhsInput(*rhs))?;
                validate_binary_table(table, declaration.lhs_width, declaration.rhs_width)?;
                let rhs_value = bindings
                    .rhs_inputs
                    .get(rhs)
                    .ok_or(ProgramValidationError::MissingRuntimeRhs(*rhs))?;
                (*output, backend.binary(lhs_value, rhs_value, table, bindings.helpers)?)
            }
            ProgramGate::OneHot { lhs, selector_family, public_value_family, lut, output } => {
                let lhs_value =
                    wires.get(lhs).cloned().ok_or(ProgramValidationError::UndefinedWire(*lhs))?;
                let table = program.lut(*lut).ok_or(ProgramValidationError::UndefinedLut(*lut))?;
                validate_unary_table(table)?;
                // The declarations establish the per-element algebraic
                // widths. Their family cardinality is intentionally a
                // separate runtime shape: PBC may bind one flattened family
                // spanning several buckets. Resolve both declarations here;
                // concrete family bindings validate their paired cardinality.
                program
                    .rhs_family_width(*selector_family)
                    .ok_or(ProgramValidationError::UndefinedRhsFamily(*selector_family))?;
                program.public_value_family_width(*public_value_family).ok_or(
                    ProgramValidationError::UndefinedPublicValueFamily(*public_value_family),
                )?;
                let selectors = bindings
                    .one_hot_selectors
                    .get(selector_family)
                    .ok_or(ProgramValidationError::WidthMismatch)?;
                let public_values = bindings
                    .public_values
                    .get(public_value_family)
                    .ok_or(ProgramValidationError::WidthMismatch)?;
                let selector_range = family_ranges
                    .selector_range(*selector_family)
                    .ok_or(ProgramValidationError::WidthMismatch)?;
                let public_value_range = family_ranges
                    .public_value_range(*public_value_family)
                    .ok_or(ProgramValidationError::WidthMismatch)?;
                if selector_range != public_value_range {
                    return Err(ProgramValidationError::WidthMismatch.into());
                }
                (
                    *output,
                    backend.one_hot(
                        lhs_value,
                        selectors,
                        public_values,
                        selector_range,
                        public_value_range,
                        table,
                        bindings.helpers,
                    )?,
                )
            }
        };
        wires.insert(output, value);
    }
    Ok(wires)
}

fn validate_unary_table(table: &LutTable) -> Result<(), PowerLutError> {
    if table.rhs_width().is_some() {
        return Err(ProgramValidationError::InvalidLutTable.into());
    }
    Ok(())
}

fn validate_binary_table(
    table: &LutTable,
    lhs_width: usize,
    rhs_width: usize,
) -> Result<(), PowerLutError> {
    if table.rhs_width() != Some(rhs_width) || table.input_width() != lhs_width {
        return Err(ProgramValidationError::WidthMismatch.into());
    }
    Ok(())
}

// Public artifact namespace.

/// Returns the public namespace used when naming artifacts for `program`.
///
/// The namespace includes only the caller-provided public namespace and the
/// canonical program identifier. Private artifact names are deliberately not
/// accepted here.
pub fn artifact_namespace(program: &PowerLutProgram, public_namespace: &str) -> String {
    format!("mxx.power-lut.program.{}.{}", public_namespace, program.id().hex())
}

/// Computes the canonical identifier for a serialized program description.
pub(crate) fn canonical_program_id(bytes: &[u8]) -> PowerLutProgramId {
    PowerLutProgramId(crate::utils::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_rhs_is_explicit_and_identity_is_canonical() {
        let mut builder = PowerLutProgramBuilder::new();
        let lhs = builder.input(2).unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let lut = builder.lut(LutTable::binary(2, 2, 2, vec![0, 1, 2, 3]).unwrap()).unwrap();
        let output = builder.binary(builder.input_wire(lhs).unwrap(), rhs, lut).unwrap();
        builder.output(output).unwrap();
        let first = builder.build().unwrap();

        let mut second_builder = PowerLutProgramBuilder::new();
        let lhs = second_builder.input(2).unwrap();
        let family = second_builder.rhs_family(2).unwrap();
        let rhs = second_builder.rhs_input(family, 2, 2).unwrap();
        let lut = second_builder.lut(LutTable::binary(2, 2, 2, vec![0, 1, 2, 3]).unwrap()).unwrap();
        let output =
            second_builder.binary(second_builder.input_wire(lhs).unwrap(), rhs, lut).unwrap();
        second_builder.output(output).unwrap();
        assert_eq!(first.id(), second_builder.build().unwrap().id());
        assert!(matches!(first.gates()[0], ProgramGate::Binary { rhs: RhsInputId(0), .. }));
    }
}
