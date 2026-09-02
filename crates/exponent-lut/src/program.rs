//! A validated, host-side Exponent-LUT computation program.
//!
//! [`ExponentLutProgram`] is the shared description consumed by both private
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
//!
//! The gate equations are: unary `y=f(x)`; binary `y=f(x,r)` after explicit
//! RHS fusion with package `r`; and one-hot selection
//! `y=sum_i m_i Fuse(x,C_i) v_i`, where `m_i` is derived from selector
//! packages and `v_i` is the public value family. The builder validates wire,
//! table, family-range, and gate-order obligations before either backend
//! performs matrix lowering.

use std::collections::BTreeMap;

use mxx_dsl::{Family, Mat, Ring};
use mxx_ir_core::IntExpr;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use sha2::{Digest, Sha256};
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
id_type!(LutId, "Identifier of a LUT table in an Exponent-LUT program.");
id_type!(ProgramGateId, "Identifier of a gate in declaration order.");

/// Provenance descriptor for a trusted monomial public-value family.
pub(crate) const PBC_MONOMIAL_FAMILY_PROVENANCE: u8 = 1;

/// Opaque, validated family of ring monomials consumed by one-hot selection.
///
/// The wrapper deliberately exposes no public constructor. Trusted PBC
/// derivation paths create it after checking the concrete ring type and
/// non-empty family cardinality; lowerers additionally require the PBC
/// provenance descriptor before accepting it.
#[derive(Clone)]
pub(crate) struct ExponentLutMonomialFamily {
    family: Family<Mat>,
    ring_type: mxx_ir_core::types::MatrixType,
    count: IntExpr,
    provenance: u8,
}

impl ExponentLutMonomialFamily {
    pub(crate) fn from_trusted(
        family: Family<Mat>,
        ring: &Ring,
        provenance: u8,
    ) -> Result<Self, ProgramValidationError> {
        let ring_type = ring.matrix_type((1, 1));
        let count = family
            .count()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| ProgramValidationError::InvalidMonomialFamily)?;
        if *family.element_type() != ring_type || count <= 0.into() {
            return Err(ProgramValidationError::InvalidMonomialFamily);
        }
        Ok(Self { count: family.count().clone(), family, ring_type, provenance })
    }

    pub(crate) fn as_family(&self) -> &Family<Mat> {
        &self.family
    }

    pub(crate) fn into_family(self) -> Family<Mat> {
        self.family
    }

    pub(crate) fn count(&self) -> &IntExpr {
        &self.count
    }

    pub(crate) fn element_type(&self) -> &mxx_ir_core::types::MatrixType {
        &self.ring_type
    }

    pub(crate) fn has_provenance(&self, provenance: u8) -> bool {
        self.provenance == provenance
    }
}

/// Canonical SHA-256 identity of a complete Exponent-LUT program.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ExponentLutProgramId(pub(crate) [u8; 32]);

impl ExponentLutProgramId {
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

/// The algebraic representation of a LUT output.
///
/// Ordinary LUTs return a monomial `X^v`, while a terminal scalar LUT returns
/// the constant polynomial `v`.  The distinction is serialized and therefore
/// participates in the program identity and helper-artifact commitment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LutOutputForm {
    /// Return the monomial `X^v` in the ring.
    Monomial,
    /// Return the constant polynomial `v`; only terminal unary gates may use it.
    Scalar,
}

/// A LUT table with statically declared input and output widths.
///
/// Entries are public output values, interpreted as monomial exponents or
/// constant coefficients according to [`LutOutputForm`], indexed by the
/// encoded input domain. A unary table has one entry for each primary input
/// value. A binary table uses the explicit mapping `index = lhs + lhs_width *
/// rhs`; stating the formula here avoids relying on an ambiguous meaning of
/// “row-major” across callers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LutTable {
    input_width: usize,
    rhs_width: Option<usize>,
    output_width: usize,
    values: Vec<usize>,
    output_form: LutOutputForm,
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
        let table = Self {
            input_width,
            rhs_width: None,
            output_width,
            values,
            output_form: LutOutputForm::Monomial,
        };
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
        let table = Self {
            input_width: lhs_width,
            rhs_width: Some(rhs_width),
            output_width,
            values,
            output_form: LutOutputForm::Monomial,
        };
        validate_table(&table)?;
        Ok(table)
    }

    /// Creates a unary table whose outputs are constant ring polynomials.
    ///
    /// Scalar outputs are intended for a terminal application operation such
    /// as LWR rounding.  Program validation prevents such a wire from feeding
    /// another gate, so ordinary unary and binary LUTs remain monomial-valued.
    pub fn unary_scalar(
        input_width: usize,
        output_width: usize,
        values: Vec<usize>,
    ) -> Result<Self, ProgramValidationError> {
        let table = Self {
            input_width,
            rhs_width: None,
            output_width,
            values,
            output_form: LutOutputForm::Scalar,
        };
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
    /// Returns whether outputs are monomials or constant polynomials.
    pub const fn output_form(&self) -> LutOutputForm {
        self.output_form
    }
    /// Returns table entries in canonical order.
    pub fn values(&self) -> &[usize] {
        &self.values
    }

    /// Returns a stable commitment to this table's public contents and shape.
    /// Helper artifacts bind to this value, preventing accidental reuse across
    /// different tables with the same local LUT identifier or width.
    pub(crate) fn commitment(&self) -> [u8; 32] {
        let bytes = serde_json::to_vec(self).expect("LutTable is serializable");
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/lut-table/v1");
        digest.update(bytes);
        digest.finalize().into()
    }
}

impl<'de> Deserialize<'de> for LutTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LutTableRepr {
            input_width: usize,
            rhs_width: Option<usize>,
            output_width: usize,
            values: Vec<usize>,
            output_form: LutOutputForm,
        }
        let repr = LutTableRepr::deserialize(deserializer)?;
        let table = Self {
            input_width: repr.input_width,
            rhs_width: repr.rhs_width,
            output_width: repr.output_width,
            values: repr.values,
            output_form: repr.output_form,
        };
        validate_table(&table).map_err(D::Error::custom)?;
        if table.output_form == LutOutputForm::Scalar && table.rhs_width.is_some() {
            return Err(D::Error::custom("scalar LUT outputs must be unary"));
        }
        Ok(table)
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
    /// Selects one paired selector/public-value family element at a time.
    ///
    /// The selector family is an explicit runtime input, not a private support
    /// list stored in the program. Lowerers implement this gate with one
    /// reusable structural loop body.
    OneHot {
        /// Input wire consumed by the gate.
        input: ProgramWireId,
        /// Declared selector/RHS family.
        selector_family: RhsFamilyId,
        /// Declared public value family.
        public_value_family: PublicValueFamilyId,
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
    #[error("serialized program identity does not match its contents")]
    /// A serialized program was altered without recomputing its canonical identity.
    ProgramIdentityMismatch,
    #[error("one-hot family range is empty, negative, or outside its family")]
    /// A runtime family view has invalid statically known bounds.
    InvalidFamilyRange,
    #[error("one-hot public values are not a trusted monomial family")]
    /// A one-hot public-value family failed ring, cardinality, or provenance validation.
    InvalidMonomialFamily,
    #[error("runtime value for program input is missing: {0:?}")]
    /// The private or public runtime input map omitted a declared input.
    MissingRuntimeInput(ProgramInputId),
    #[error("runtime value for RHS input is missing: {0:?}")]
    /// The runtime RHS map omitted a declared explicit RHS input.
    MissingRuntimeRhs(RhsInputId),
}

/// A validated immutable Exponent-LUT program.
///
/// The dataflow is `declared inputs -> wires -> ordered gates -> outputs`.
/// Its canonical identity covers declarations, tables, gate wiring, and output
/// order, allowing independent encoding and public-key lowerers to consume the
/// same graph description. Builder-created values are validated before being
/// returned; callers deserializing this type should still treat untrusted data
/// as requiring validation at its artifact boundary.
#[derive(Clone, Debug, Serialize)]
pub struct ExponentLutProgram {
    id: ExponentLutProgramId,
    inputs: BTreeMap<ProgramInputId, usize>,
    input_wires: BTreeMap<ProgramInputId, ProgramWireId>,
    rhs_inputs: BTreeMap<RhsInputId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    public_value_families: BTreeMap<PublicValueFamilyId, PublicValueFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    gates: Vec<ProgramGate>,
    outputs: Vec<ProgramWireId>,
}

#[derive(Deserialize)]
struct ExponentLutProgramRepr {
    id: ExponentLutProgramId,
    inputs: BTreeMap<ProgramInputId, usize>,
    input_wires: BTreeMap<ProgramInputId, ProgramWireId>,
    rhs_inputs: BTreeMap<RhsInputId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    public_value_families: BTreeMap<PublicValueFamilyId, PublicValueFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    gates: Vec<ProgramGate>,
    outputs: Vec<ProgramWireId>,
}

impl<'de> Deserialize<'de> for ExponentLutProgram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ExponentLutProgramRepr::deserialize(deserializer)?;
        let builder = builder_from_program_parts(
            repr.inputs.clone(),
            repr.input_wires.clone(),
            repr.rhs_inputs.clone(),
            repr.rhs_families.clone(),
            repr.public_value_families.clone(),
            repr.luts.clone(),
            repr.gates.clone(),
            repr.outputs.clone(),
        )
        .map_err(D::Error::custom)?;
        if repr.id != canonical_program_id(&canonical_program_bytes(&builder)) {
            return Err(D::Error::custom(ProgramValidationError::ProgramIdentityMismatch));
        }
        Ok(Self {
            id: repr.id,
            inputs: repr.inputs,
            input_wires: repr.input_wires,
            rhs_inputs: repr.rhs_inputs,
            rhs_families: repr.rhs_families,
            public_value_families: repr.public_value_families,
            luts: repr.luts,
            gates: repr.gates,
            outputs: repr.outputs,
        })
    }
}

impl ExponentLutProgram {
    /// Returns the canonical identity of the declarations and gate dataflow.
    pub const fn id(&self) -> ExponentLutProgramId {
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

/// Mutable builder for a validated immutable [`ExponentLutProgram`].
#[derive(Default)]
pub struct ExponentLutProgramBuilder {
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

impl ExponentLutProgramBuilder {
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

    /// Adds a validated selection-only one-hot gate for a selector-family
    /// lowerer.
    pub fn one_hot_select(
        &mut self,
        input: ProgramWireId,
        selector_family: RhsFamilyId,
        public_value_family: PublicValueFamilyId,
    ) -> Result<ProgramWireId, ProgramValidationError> {
        let selector = self
            .rhs_families
            .get(&selector_family)
            .ok_or(ProgramValidationError::UndefinedRhsFamily(selector_family))?;
        let values = self
            .public_value_families
            .get(&public_value_family)
            .ok_or(ProgramValidationError::UndefinedPublicValueFamily(public_value_family))?;
        if self.wire_width(input)? != selector.width || selector.width != values.width {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = ProgramWireId::from_index(self.wires.len());
        self.wires.insert(output, selector.width);
        self.gates.push(ProgramGate::OneHot {
            input,
            selector_family,
            public_value_family,
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
    pub fn build(self) -> Result<ExponentLutProgram, ProgramValidationError> {
        if self.outputs.is_empty() {
            return Err(ProgramValidationError::InvalidOutput);
        }
        validate_builder(&self)?;
        let canonical = canonical_program_bytes(&self);
        let program = ExponentLutProgram {
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

fn canonical_program_bytes(builder: &ExponentLutProgramBuilder) -> Vec<u8> {
    serde_json::to_vec(&(
        &builder.inputs,
        &builder.input_wires,
        &builder.rhs_inputs,
        &builder.rhs_families,
        &builder.public_value_families,
        &builder.luts,
        &builder.gates,
        &builder.outputs,
    ))
    .expect("ExponentLutProgram declarations are serializable")
}

fn builder_from_program_parts(
    inputs: BTreeMap<ProgramInputId, usize>,
    input_wires: BTreeMap<ProgramInputId, ProgramWireId>,
    rhs_inputs: BTreeMap<RhsInputId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    public_value_families: BTreeMap<PublicValueFamilyId, PublicValueFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    gates: Vec<ProgramGate>,
    outputs: Vec<ProgramWireId>,
) -> Result<ExponentLutProgramBuilder, ProgramValidationError> {
    if inputs.len() != input_wires.len() || outputs.is_empty() {
        return Err(ProgramValidationError::InvalidBuilderState);
    }
    let mut wires = BTreeMap::new();
    for (input, width) in &inputs {
        if *width == 0 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let wire =
            input_wires.get(input).copied().ok_or(ProgramValidationError::InvalidBuilderState)?;
        if wires.insert(wire, *width).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
    }
    for gate in &gates {
        let (output, width) = match gate {
            ProgramGate::Unary { output, lut, .. } | ProgramGate::Binary { output, lut, .. } => (
                *output,
                luts.get(lut).ok_or(ProgramValidationError::UndefinedLut(*lut))?.output_width(),
            ),
            ProgramGate::OneHot { output, input, .. } => {
                (*output, *wires.get(input).ok_or(ProgramValidationError::UndefinedWire(*input))?)
            }
        };
        if wires.insert(output, width).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
    }
    let builder = ExponentLutProgramBuilder {
        inputs,
        input_wires,
        rhs_inputs,
        rhs_families,
        public_value_families,
        luts,
        gates,
        wires,
        outputs,
    };
    validate_builder(&builder)?;
    Ok(builder)
}

// Program shape validation.

pub(crate) fn validate_builder(
    builder: &ExponentLutProgramBuilder,
) -> Result<(), ProgramValidationError> {
    if builder.inputs.len() != builder.input_wires.len() {
        return Err(ProgramValidationError::InvalidBuilderState);
    }
    if builder.outputs.is_empty() ||
        builder.outputs.iter().any(|wire| !builder.wires.contains_key(wire)) ||
        builder
            .outputs
            .iter()
            .enumerate()
            .any(|(index, wire)| builder.outputs[..index].contains(wire))
    {
        return Err(ProgramValidationError::InvalidOutput);
    }
    for table in builder.luts.values() {
        validate_table(table)?;
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
            ProgramGate::OneHot { input, selector_family, public_value_family, output } => {
                let input_width = builder.wire_width(*input)?;
                let selector = builder
                    .rhs_families
                    .get(selector_family)
                    .ok_or(ProgramValidationError::UndefinedRhsFamily(*selector_family))?;
                let values = builder.public_value_families.get(public_value_family).ok_or(
                    ProgramValidationError::UndefinedPublicValueFamily(*public_value_family),
                )?;
                if input_width != selector.width ||
                    selector.width != values.width ||
                    !builder.wires.contains_key(output)
                {
                    return Err(ProgramValidationError::WidthMismatch);
                }
            }
        }
    }
    for gate in &builder.gates {
        let (output, table, is_unary) = match gate {
            ProgramGate::Unary { output, lut, .. } => {
                (*output, builder.lut_definition(*lut)?, true)
            }
            ProgramGate::Binary { output, lut, .. } => {
                (*output, builder.lut_definition(*lut)?, false)
            }
            ProgramGate::OneHot { .. } => {
                continue;
            }
        };
        if table.output_form() == LutOutputForm::Scalar {
            let consumed_later = builder.gates.iter().any(|consumer| match consumer {
                ProgramGate::Unary { input, .. } => *input == output,
                ProgramGate::Binary { lhs, .. } => *lhs == output,
                ProgramGate::OneHot { input, .. } => *input == output,
            });
            if !is_unary || consumed_later || !builder.outputs.contains(&output) {
                return Err(ProgramValidationError::InvalidLutTable);
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
    if table.values.iter().any(|value| *value >= table.output_width) {
        return Err(ProgramValidationError::InvalidLutTable);
    }
    if table.output_form == LutOutputForm::Scalar && table.rhs_width.is_some() {
        return Err(ProgramValidationError::InvalidLutTable);
    }
    Ok(())
}

// Shared runtime binding and traversal.

use crate::ExponentLutError;

/// Runtime values bound to the declarations of one [`ExponentLutProgram`].
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
    pub(crate) helpers: &'a BTreeMap<LutId, H>,
}

impl<'a, W, R, SF, VF, H> ProgramBindings<'a, W, R, SF, VF, H> {
    pub(crate) fn new(
        inputs: &'a BTreeMap<ProgramInputId, W>,
        rhs_inputs: &'a BTreeMap<RhsInputId, R>,
        one_hot_selectors: &'a BTreeMap<RhsFamilyId, SF>,
        public_values: &'a BTreeMap<PublicValueFamilyId, VF>,
        helpers: &'a BTreeMap<LutId, H>,
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
    /// Setup helper container validated against the concrete LUT table.
    type HelperSet;

    fn resolve_helpers<'a>(
        &self,
        helpers: &'a Self::HelperSet,
        table: &LutTable,
    ) -> Result<&'a [Self::Helper], ExponentLutError>;

    fn unary(
        &self,
        input: Self::Wire,
        table: &LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, ExponentLutError>;

    fn binary(
        &self,
        lhs: Self::Wire,
        rhs: &Self::Rhs,
        table: &LutTable,
        helpers: &[Self::Helper],
    ) -> Result<Self::Wire, ExponentLutError>;

    /// Selects from paired runtime families and returns their balanced sum.
    /// Selection has no LUT/helper inputs; callers append a unary gate when a
    /// LUT is required after this operation.
    fn one_hot_select(
        &self,
        input: Self::Wire,
        selectors: &Self::SelectorFamily,
        public_values: &Self::PublicValueFamily,
        selector_range: &FamilyRange,
        public_value_range: &FamilyRange,
    ) -> Result<Self::Wire, ExponentLutError>;
}

/// Resolves declarations and runtime bindings once, then traverses every gate
/// in canonical order through a cryptographic backend.
///
/// This function owns wire availability, LUT/RHS/family lookup, range pairing,
/// and output collection. It deliberately does not own a cryptographic
/// formula: encoding and public-key backends implement those formulas
/// independently. Family bindings remain DSL families so OneHot lowering can
/// use one structural loop body rather than host-unrolling each cell.
///
/// For each gate, `wires` is the partial map from a declared wire id to the
/// backend value representing that wire. Unary dispatch applies `f` to the
/// mapped input; binary dispatch supplies the separately declared RHS `r`;
/// OneHot gathers matching selector/value family ranges and evaluates
/// `sum_i m_i Fuse(wire,C_i) v_i`. The output is inserted only after the
/// backend has accepted the gate's shape. Any following LUT is represented by
/// an ordinary [`ProgramGate::Unary`] gate.
pub(crate) fn lower_program<B: ProgramLoweringBackend>(
    program: &ExponentLutProgram,
    bindings: &ProgramBindings<
        '_,
        B::Wire,
        B::Rhs,
        B::SelectorFamily,
        B::PublicValueFamily,
        B::HelperSet,
    >,
    family_ranges: &ProgramFamilyRanges,
    backend: &B,
) -> Result<BTreeMap<ProgramWireId, B::Wire>, ExponentLutError> {
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
                let helper_set = bindings.helpers.get(lut).ok_or(ExponentLutError::InvalidLut)?;
                let helpers = backend.resolve_helpers(helper_set, table)?;
                (*output, backend.unary(input, table, helpers)?)
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
                let helper_set = bindings.helpers.get(lut).ok_or(ExponentLutError::InvalidLut)?;
                let helpers = backend.resolve_helpers(helper_set, table)?;
                (*output, backend.binary(lhs_value, rhs_value, table, helpers)?)
            }
            ProgramGate::OneHot { input, selector_family, public_value_family, output } => {
                let input_value = wires
                    .get(input)
                    .cloned()
                    .ok_or(ProgramValidationError::UndefinedWire(*input))?;
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
                    backend.one_hot_select(
                        input_value,
                        selectors,
                        public_values,
                        selector_range,
                        public_value_range,
                    )?,
                )
            }
        };
        wires.insert(output, value);
    }
    Ok(wires)
}

fn validate_unary_table(table: &LutTable) -> Result<(), ExponentLutError> {
    if table.rhs_width().is_some() {
        return Err(ProgramValidationError::InvalidLutTable.into());
    }
    Ok(())
}

fn validate_binary_table(
    table: &LutTable,
    lhs_width: usize,
    rhs_width: usize,
) -> Result<(), ExponentLutError> {
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
pub fn artifact_namespace(program: &ExponentLutProgram, public_namespace: &str) -> String {
    format!("mxx.exponent-lut.program.{}.{}", public_namespace, program.id().hex())
}

/// Computes the canonical identifier for a serialized program description.
pub(crate) fn canonical_program_id(bytes: &[u8]) -> ExponentLutProgramId {
    ExponentLutProgramId(Sha256::digest(bytes).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_rhs_is_explicit_and_identity_is_canonical() {
        let mut builder = ExponentLutProgramBuilder::new();
        let lhs = builder.input(2).unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let lut = builder.lut(LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap()).unwrap();
        let output = builder.binary(builder.input_wire(lhs).unwrap(), rhs, lut).unwrap();
        builder.output(output).unwrap();
        let first = builder.build().unwrap();

        let mut second_builder = ExponentLutProgramBuilder::new();
        let lhs = second_builder.input(2).unwrap();
        let family = second_builder.rhs_family(2).unwrap();
        let rhs = second_builder.rhs_input(family, 2, 2).unwrap();
        let lut = second_builder.lut(LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap()).unwrap();
        let output =
            second_builder.binary(second_builder.input_wire(lhs).unwrap(), rhs, lut).unwrap();
        second_builder.output(output).unwrap();
        assert_eq!(first.id(), second_builder.build().unwrap().id());
        assert!(matches!(first.gates()[0], ProgramGate::Binary { rhs: RhsInputId(0), .. }));
    }

    #[test]
    fn one_hot_selection_has_no_lut_and_round_trips_canonical_identity() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let public_value_family = builder.public_value_family(2).unwrap();
        let output = builder
            .one_hot_select(
                builder.input_wire(input).unwrap(),
                selector_family,
                public_value_family,
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        assert!(program.luts.is_empty());
        assert!(matches!(
            program.gates()[0],
            ProgramGate::OneHot {
                input: ProgramWireId(0),
                selector_family: RhsFamilyId(0),
                public_value_family: PublicValueFamilyId(0),
                output: ProgramWireId(1),
            }
        ));
        let encoded = serde_json::to_value(&program).unwrap();
        assert!(encoded["gates"][0].get("lut").is_none());
        let decoded: ExponentLutProgram = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.id(), program.id());
    }

    #[test]
    fn one_hot_selection_requires_matching_input_and_value_widths() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let public_value_family = builder.public_value_family(4).unwrap();
        assert_eq!(
            builder.one_hot_select(
                builder.input_wire(input).unwrap(),
                selector_family,
                public_value_family,
            ),
            Err(ProgramValidationError::WidthMismatch)
        );
    }

    #[test]
    fn one_hot_selection_traversal_does_not_resolve_lut_helpers() {
        struct SelectionOnlyBackend;
        impl ProgramLoweringBackend for SelectionOnlyBackend {
            type Wire = usize;
            type Rhs = ();
            type SelectorFamily = ();
            type PublicValueFamily = ();
            type Helper = ();
            type HelperSet = ();

            fn resolve_helpers<'a>(
                &self,
                _helpers: &'a Self::HelperSet,
                _table: &LutTable,
            ) -> Result<&'a [Self::Helper], ExponentLutError> {
                panic!("selection-only lowering must not resolve LUT helpers")
            }

            fn unary(
                &self,
                _input: Self::Wire,
                _table: &LutTable,
                _helpers: &[Self::Helper],
            ) -> Result<Self::Wire, ExponentLutError> {
                panic!("selection-only test has no unary gates")
            }

            fn binary(
                &self,
                _lhs: Self::Wire,
                _rhs: &Self::Rhs,
                _table: &LutTable,
                _helpers: &[Self::Helper],
            ) -> Result<Self::Wire, ExponentLutError> {
                panic!("selection-only test has no binary gates")
            }

            fn one_hot_select(
                &self,
                input: Self::Wire,
                _selectors: &Self::SelectorFamily,
                _public_values: &Self::PublicValueFamily,
                _selector_range: &FamilyRange,
                _public_value_range: &FamilyRange,
            ) -> Result<Self::Wire, ExponentLutError> {
                Ok(input + 1)
            }
        }

        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let public_value_family = builder.public_value_family(2).unwrap();
        let output = builder
            .one_hot_select(
                builder.input_wire(input).unwrap(),
                selector_family,
                public_value_family,
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let inputs = BTreeMap::from([(input, 41usize)]);
        let rhs_inputs = BTreeMap::new();
        let selectors = BTreeMap::from([(selector_family, ())]);
        let public_values = BTreeMap::from([(public_value_family, ())]);
        let helpers = BTreeMap::new();
        let bindings =
            ProgramBindings::new(&inputs, &rhs_inputs, &selectors, &public_values, &helpers);
        let mut ranges = ProgramFamilyRanges::new();
        ranges.selector(selector_family, FamilyRange::full(2).unwrap());
        ranges.public_values(public_value_family, FamilyRange::full(2).unwrap());
        let wires = lower_program(&program, &bindings, &ranges, &SelectionOnlyBackend).unwrap();
        assert_eq!(wires[&output], 42);
    }

    #[test]
    fn trusted_monomial_family_rejects_wrong_ring_and_empty_count() {
        let ring = Ring::new(97, 4);
        let valid = Family::pack(vec![ring.zero((1, 1))]).unwrap();
        assert!(
            ExponentLutMonomialFamily::from_trusted(valid, &ring, PBC_MONOMIAL_FAMILY_PROVENANCE,)
                .is_ok()
        );
        let wrong_ring = Ring::new(97, 8);
        let wrong_ring_family = Family::pack(vec![ring.zero((1, 1))]).unwrap();
        assert!(
            ExponentLutMonomialFamily::from_trusted(
                wrong_ring_family,
                &wrong_ring,
                PBC_MONOMIAL_FAMILY_PROVENANCE,
            )
            .is_err()
        );
        let empty = Family::pack(Vec::<Mat>::new());
        assert!(
            empty.is_err() ||
                ExponentLutMonomialFamily::from_trusted(
                    empty.unwrap(),
                    &ring,
                    PBC_MONOMIAL_FAMILY_PROVENANCE,
                )
                .is_err()
        );
    }

    #[test]
    fn lut_rejects_values_outside_the_declared_output_domain() {
        assert!(LutTable::unary(2, 2, vec![0, 2]).is_err());
        assert!(LutTable::binary(2, 2, 2, vec![0, 1, 2, 0]).is_err());
        assert!(LutTable::unary_scalar(2, 2, vec![1, 2]).is_err());
    }

    #[test]
    fn deserialization_revalidates_lut_output_domain() {
        let table = LutTable::unary(2, 3, vec![0, 2]).unwrap();
        let mut value = serde_json::to_value(&table).unwrap();
        value["values"][0] = serde_json::json!(3);
        let result = serde_json::from_value::<LutTable>(value);
        assert!(result.is_err());
    }

    #[test]
    fn scalar_lut_is_terminal_and_cannot_feed_a_later_gate() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let scalar = builder.lut(LutTable::unary_scalar(2, 2, vec![0, 1]).unwrap()).unwrap();
        let scalar_wire = builder.unary(builder.input_wire(input).unwrap(), scalar).unwrap();
        builder.output(scalar_wire).unwrap();
        let ordinary = builder.lut(LutTable::unary(2, 2, vec![1, 0]).unwrap()).unwrap();
        let later_wire = builder.unary(scalar_wire, ordinary).unwrap();
        builder.output(later_wire).unwrap();
        assert!(matches!(builder.build(), Err(ProgramValidationError::InvalidLutTable)));
    }

    #[test]
    fn scalar_lut_is_accepted_when_it_is_the_terminal_output() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let scalar = builder.lut(LutTable::unary_scalar(2, 2, vec![0, 1]).unwrap()).unwrap();
        let output = builder.unary(builder.input_wire(input).unwrap(), scalar).unwrap();
        builder.output(output).unwrap();
        assert!(builder.build().is_ok());
    }

    #[test]
    fn serialized_program_rejects_identity_or_output_form_tampering() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let lut = builder.lut(LutTable::unary(2, 2, vec![0, 1]).unwrap()).unwrap();
        let output = builder.unary(builder.input_wire(input).unwrap(), lut).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        let mut wrong_id = serde_json::to_value(&program).unwrap();
        wrong_id["id"] =
            serde_json::Value::Array((0..32).map(|_| serde_json::Value::from(0u8)).collect());
        assert!(serde_json::from_value::<ExponentLutProgram>(wrong_id).is_err());

        let mut wrong_form = serde_json::to_value(&program).unwrap();
        wrong_form["luts"]["0"]["output_form"] = serde_json::json!("Scalar");
        let error = serde_json::from_value::<ExponentLutProgram>(wrong_form).unwrap_err();
        assert!(error.to_string().contains("serialized program identity"));
    }
}
