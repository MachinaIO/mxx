//! A validated, host-side Exponent-LUT computation program.
//!
//! [`ExponentLutProgram`] is the shared description consumed by both private
//! encoding and public-key lowerers. It contains only public scheme metadata:
//! input widths, LUT tables, gate wiring, and explicit RHS/family identifiers.
//! It does not add an IR node and does not carry private vectors, GSW
//! ciphertexts, sparse support, selected slots, or refresh state.
//!
//! A private multiplication always names an [`RhsId`]. An in-flight
//! computed wire is therefore never accidentally reused as an RHS package.
//! Family operations are structural steps, not cryptographic gates. Concrete
//! family element types and values belong entirely to each interpreter.
//!
//! The gate equations are private/public multiplication, addition, and a
//! single-layer LUT. Selection is lowered structurally to those primitives.
//! The builder validates wire, table, family-range, and gate-order obligations
//! before either backend performs matrix lowering.

use std::collections::BTreeMap;

use mxx_ir_core::IntExpr;
use num_traits::ToPrimitive;
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
id_type!(WireId, "Identifier of a program wire, including inputs and gate outputs.");
id_type!(RhsId, "Identifier of an explicit RHS input package.");
id_type!(RhsFamilyId, "Identifier of an indexed RHS family declaration.");
id_type!(WireFamilyId, "Identifier of an indexed program-wire family declaration.");
id_type!(ProgramFamilyId, "Identifier of a structural family invocation.");
id_type!(LutId, "Identifier of a LUT table in an Exponent-LUT program.");

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

/// A LUT table with statically declared input and output widths.
///
/// Entries are public native exponent values, indexed by the encoded input
/// domain. A unary table has one entry for each primary input
/// value. A binary table uses the explicit mapping `index = lhs + lhs_width *
/// rhs`; stating the formula here avoids relying on an ambiguous meaning of
/// “row-major” across callers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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
        }
        let repr = LutTableRepr::deserialize(deserializer)?;
        let table = Self {
            input_width: repr.input_width,
            rhs_width: repr.rhs_width,
            output_width: repr.output_width,
            values: repr.values,
        };
        validate_table(&table).map_err(D::Error::custom)?;
        Ok(table)
    }
}

/// Metadata for one declared explicit RHS input.
///
/// A binary gate binds its `RhsId` to an externally prepared RHS package.
/// The RHS is never a computed [`WireId`]; `lhs_width` and `rhs_width`
/// describe the two input domains that the selected binary table must accept.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RhsInputDeclaration {
    family: RhsFamilyId,
    lhs_width: usize,
    rhs_width: usize,
    exponent_scale_log2: u32,
    /// Whether the RHS relation exposes its plaintext attribute publicly.
    plaintext_knowledge: AttributeKnowledge,
}

/// Concrete exponent layout and ring dimension committed by a program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramLayout {
    pub exponent: crate::layout::ExpLayout,
    pub ring_dimension_log2: u32,
}

impl ProgramLayout {
    fn private_product_width(self, lhs: usize, rhs: usize) -> usize {
        let native_width = 1usize << (self.ring_dimension_log2 - self.exponent.stride_log2);
        lhs.saturating_mul(rhs).min(native_width)
    }
}

/// Metadata for one externally bound RHS family.
///
/// The width is the per-element domain. Runtime cardinality is supplied by a
/// structural family binding, so one declaration can serve multiple ranges.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct RhsFamilyDeclaration {
    width: usize,
    plaintext_knowledge: AttributeKnowledge,
}

/// Metadata for one externally bound wire family.
///
/// The width is the per-element domain. Cardinality is runtime data and is
/// not part of the value-agnostic program identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct WireFamilyDeclaration {
    width: usize,
}

/// A contiguous view into a flattened runtime family.
///
/// `start` and `count` are DSL integer expressions rather than host-side
/// indices. This lets a structural invocation select a range from one
/// flattened family without constructing a separate Rust graph body for each
/// member. The view is runtime metadata; it is not part
/// of the program identity or IR schema.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FamilyRange {
    start: IntExpr,
    count: IntExpr,
    capacity: IntExpr,
}

impl FamilyRange {
    /// Creates a non-empty range. Constant negative or zero bounds are
    /// rejected immediately; symbolic bounds are checked by graph
    /// validation when the enclosing DSL graph is frozen.
    pub(crate) fn new(
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
    /// is concrete. Backends operate on active members only while keeping
    /// indexed family wire types independent of parent loop binders.
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

/// Runtime ranges for structural family invocations.
///
/// Each invocation receives one contiguous active range. Callers that need
/// the complete family can use [`FamilyRange::full`].
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramFamilyBindings {
    ranges: BTreeMap<ProgramFamilyId, FamilyRange>,
}

impl ProgramFamilyBindings {
    /// Creates empty range bindings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds an active range to one structural family invocation.
    pub fn bind(&mut self, id: ProgramFamilyId, range: FamilyRange) -> Option<FamilyRange> {
        self.ranges.insert(id, range)
    }

    pub(crate) fn range(&self, id: ProgramFamilyId) -> Option<&FamilyRange> {
        self.ranges.get(&id)
    }
}

/// Visibility of a wire's represented attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AttributeKnowledge {
    Private,
    Public,
}

/// Type information carried by a program wire.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramWireType {
    pub stride_log2: u32,
    pub knowledge: AttributeKnowledge,
}

/// Source accepted by the compound LUT builder operation.
pub trait LutSpec {
    fn resolve(
        self,
        builder: &mut ExponentLutProgramBuilder,
    ) -> Result<LutId, ProgramValidationError>;
}

impl LutSpec for LutId {
    fn resolve(
        self,
        _builder: &mut ExponentLutProgramBuilder,
    ) -> Result<LutId, ProgramValidationError> {
        Ok(self)
    }
}

impl LutSpec for LutTable {
    fn resolve(
        self,
        builder: &mut ExponentLutProgramBuilder,
    ) -> Result<LutId, ProgramValidationError> {
        builder.register_lut(self)
    }
}

/// One cryptographic program gate. Structural family control is represented by
/// [`ProgramStep`], not by an additional cryptographic gate variant.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Gate {
    MulPublic { private: WireId, public: WireId, output: WireId },
    Add { lhs: WireId, rhs: WireId, output: WireId },
    MulPrivate { lhs: WireId, rhs: RhsId, output: WireId },
    Lut { input: WireId, lut_id: LutId, output: WireId },
}

/// Source of an input wire in one indexed family body.
///
/// `Broadcast` is one structural scalar shared at every logical index; it
/// never requests `capacity` Rust clones. `Member` is the index-dependent
/// element of an externally bound family at the current logical body index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FamilySource<S, F> {
    Broadcast(S),
    Member(F),
}

pub type FamilyWireSource = FamilySource<WireId, WireFamilyId>;
pub type FamilyRhsSource = FamilySource<RhsId, RhsFamilyId>;

/// Validated local SSA body evaluated once per active family member.
///
/// Its `WireId` and `RhsId` values belong to this local scope and are
/// independent of the same identifier types used by the enclosing program.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FamilyProgram {
    wire_inputs: BTreeMap<WireId, FamilyWireSource>,
    rhs_inputs: BTreeMap<RhsId, FamilyRhsSource>,
    gates: Vec<Gate>,
    output: WireId,
}

impl FamilyProgram {
    pub fn new(
        wire_inputs: BTreeMap<WireId, FamilyWireSource>,
        rhs_inputs: BTreeMap<RhsId, FamilyRhsSource>,
        gates: Vec<Gate>,
        output: WireId,
    ) -> Result<Self, ProgramValidationError> {
        if wire_inputs.is_empty() || gates.is_empty() {
            return Err(ProgramValidationError::WidthMismatch);
        }
        Ok(Self { wire_inputs, rhs_inputs, gates, output })
    }
    pub fn wire_inputs(&self) -> &BTreeMap<WireId, FamilyWireSource> {
        &self.wire_inputs
    }
    pub fn rhs_inputs(&self) -> &BTreeMap<RhsId, FamilyRhsSource> {
        &self.rhs_inputs
    }
    /// Resolve local ports from enclosing captures and indexed member bindings.
    /// Graph callers index members inside their single body scope; symbolic
    /// callers retain their source identities. Neither needs a second traversal
    /// of the body's reference declarations in a scheme adapter.
    pub(crate) fn bind_inputs<W, R, E>(
        &self,
        mut capture_wire: impl FnMut(WireId) -> Result<W, E>,
        mut member_wire: impl FnMut(WireFamilyId) -> Result<W, E>,
        mut capture_rhs: impl FnMut(RhsId) -> Result<R, E>,
        mut member_rhs: impl FnMut(RhsFamilyId) -> Result<R, E>,
    ) -> Result<(BTreeMap<WireId, W>, BTreeMap<RhsId, R>), E> {
        let wires = self
            .wire_inputs
            .iter()
            .map(|(local, source)| {
                let value = match source {
                    FamilySource::Broadcast(id) => capture_wire(*id),
                    FamilySource::Member(id) => member_wire(*id),
                }?;
                Ok((*local, value))
            })
            .collect::<Result<_, E>>()?;
        let rhs = self
            .rhs_inputs
            .iter()
            .map(|(local, source)| {
                let value = match source {
                    FamilySource::Broadcast(id) => capture_rhs(*id),
                    FamilySource::Member(id) => member_rhs(*id),
                }?;
                Ok((*local, value))
            })
            .collect::<Result<_, E>>()?;
        Ok((wires, rhs))
    }
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }
    pub const fn output(&self) -> WireId {
        self.output
    }
}

/// Aggregation applied after evaluating every active family member.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FamilyAggregation {
    Add,
}

/// Structural family control used by the compound selection API.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramStep {
    Gate(Gate),
    Family {
        id: ProgramFamilyId,
        capacity: usize,
        body: FamilyProgram,
        aggregation: FamilyAggregation,
        output: WireId,
    },
}

/// Errors returned while declaring or validating a program.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ProgramValidationError {
    #[error("Program artifact: {0}")]
    Artifact(String),
    #[error("program identifier has already been declared")]
    /// An automatically allocated identifier would collide with an existing declaration.
    DuplicateIdentifier,
    #[error("program references an undefined wire: {0:?}")]
    /// A gate refers to a wire that has not been defined yet.
    UndefinedWire(WireId),
    #[error("program references an undefined LUT: {0:?}")]
    /// A gate refers to a LUT that has not been declared.
    UndefinedLut(LutId),
    #[error("program references an undefined RHS input: {0:?}")]
    /// A binary gate refers to an undeclared explicit RHS input.
    UndefinedRhsInput(RhsId),
    #[error("program references an undefined RHS family: {0:?}")]
    /// A family body refers to an undeclared RHS family.
    UndefinedRhsFamily(RhsFamilyId),
    #[error("program references an undefined wire family: {0:?}")]
    /// A family body refers to an undeclared wire family.
    UndefinedWireFamily(WireFamilyId),
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
    #[error("family range is empty, negative, or outside its family")]
    /// A runtime family view has invalid statically known bounds.
    InvalidFamilyRange,
    #[error("runtime wire family is invalid")]
    /// A runtime wire family failed shape, cardinality, or value validation.
    InvalidWireFamily,
    #[error("runtime value for program input is missing: {0:?}")]
    /// The private or public runtime input map omitted a declared input.
    MissingRuntimeInput(ProgramInputId),
    #[error("runtime value for RHS input is missing: {0:?}")]
    /// The runtime RHS map omitted a declared explicit RHS input.
    MissingRuntimeRhs(RhsId),
    #[error("runtime helper set for LUT is missing: {0:?}")]
    MissingRuntimeLutHelpers(LutId),
    #[error("program public multiplication requires a public attribute wire")]
    PublicAttributeRequired(WireId),
    #[error("program layout is required before declaring RHS inputs or building")]
    MissingProgramLayout,
}

/// A validated immutable Exponent-LUT program.
///
/// The dataflow is `declared inputs -> wires -> ordered gates -> outputs`.
/// Its canonical identity covers declarations, tables, gate wiring, and output
/// order, allowing independent encoding and public-key lowerers to consume the
/// same graph description. Builder-created values are validated before being
/// returned.
#[derive(Clone, Debug)]
pub struct ExponentLutProgram {
    id: ExponentLutProgramId,
    inputs: BTreeMap<ProgramInputId, usize>,
    wire_types: BTreeMap<WireId, ProgramWireType>,
    input_wires: BTreeMap<ProgramInputId, WireId>,
    rhs_inputs: BTreeMap<RhsId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    wire_families: BTreeMap<WireFamilyId, WireFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    layout: Option<ProgramLayout>,
    steps: Vec<ProgramStep>,
    outputs: Vec<WireId>,
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
    /// Returns the frozen type of a declared or computed wire.
    pub fn wire_type(&self, wire: WireId) -> Option<ProgramWireType> {
        self.wire_types.get(&wire).copied()
    }
    /// Returns all frozen wire types in canonical wire-id order.
    pub fn wire_types(&self) -> &BTreeMap<WireId, ProgramWireType> {
        &self.wire_types
    }
    /// Returns the canonical mapping from each declared input to its wire.
    pub fn input_wires(&self) -> &BTreeMap<ProgramInputId, WireId> {
        &self.input_wires
    }
    /// Returns gates in deterministic declaration order.
    pub fn gates(&self) -> Vec<Gate> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                ProgramStep::Gate(gate) => Some(gate.clone()),
                _ => None,
            })
            .collect()
    }
    /// Visits scalar gates and structural family body gates in declaration order.
    /// Each family body is visited once; members and reduction operations are not expanded.
    pub fn gate_instances(&self) -> impl Iterator<Item = (ProgramGateInstance, &Gate)> {
        self.steps.iter().enumerate().flat_map(|(step, item)| {
            let (gates, family) = match item {
                ProgramStep::Gate(gate) => (std::slice::from_ref(gate), None),
                ProgramStep::Family { id, body, .. } => (body.gates(), Some(*id)),
            };
            gates.iter().enumerate().map(move |(gate, value)| {
                let instance = if let Some(family) = family {
                    ProgramGateInstance::FamilyGate { step, gate, family }
                } else {
                    ProgramGateInstance::Gate { step }
                };
                (instance, value)
            })
        })
    }
    /// Returns structural family steps in deterministic declaration order.
    pub fn steps(&self) -> &[ProgramStep] {
        &self.steps
    }
    /// Returns output wires in the order committed by the builder.
    pub fn outputs(&self) -> &[WireId] {
        &self.outputs
    }
    /// Returns a declared LUT by identifier.
    pub fn lut(&self, id: LutId) -> Option<&LutTable> {
        self.luts.get(&id)
    }
    /// Returns the layout/ring pair committed to this program, if configured.
    pub fn layout(&self) -> Option<ProgramLayout> {
        self.layout
    }
    /// Returns explicit RHS declarations, never computed-wire declarations.
    pub fn rhs_inputs(&self) -> &BTreeMap<RhsId, RhsInputDeclaration> {
        &self.rhs_inputs
    }
    /// Returns the native exponent scale declared for an explicit RHS.
    pub fn rhs_exponent_scale_log2(&self, id: RhsId) -> Option<u32> {
        self.rhs_inputs.get(&id).map(|rhs| rhs.exponent_scale_log2)
    }
    /// Returns the declared RHS-family element width, if present.
    pub fn rhs_family_width(&self, id: RhsFamilyId) -> Option<usize> {
        self.rhs_families.get(&id).map(|family| family.width)
    }
    /// Returns the declared wire-family element width, if present.
    pub fn wire_family_width(&self, id: WireFamilyId) -> Option<usize> {
        self.wire_families.get(&id).map(|family| family.width)
    }
}

/// Mutable builder for a validated immutable [`ExponentLutProgram`].
#[derive(Default)]
pub struct ExponentLutProgramBuilder {
    inputs: BTreeMap<ProgramInputId, usize>,
    wire_types: BTreeMap<WireId, ProgramWireType>,
    input_wires: BTreeMap<ProgramInputId, WireId>,
    rhs_inputs: BTreeMap<RhsId, RhsInputDeclaration>,
    rhs_families: BTreeMap<RhsFamilyId, RhsFamilyDeclaration>,
    wire_families: BTreeMap<WireFamilyId, WireFamilyDeclaration>,
    luts: BTreeMap<LutId, LutTable>,
    layout: Option<ProgramLayout>,
    steps: Vec<ProgramStep>,
    wires: BTreeMap<WireId, usize>,
    outputs: Vec<WireId>,
}

impl ExponentLutProgramBuilder {
    /// Creates an empty program builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Commits the exponent layout and concrete ring dimension to the program
    /// identity and validates their native-domain relationship.
    pub fn set_layout(
        &mut self,
        exponent: crate::layout::ExpLayout,
        ring_dimension: usize,
    ) -> Result<(), ProgramValidationError> {
        exponent.validate(ring_dimension).map_err(|_| ProgramValidationError::WidthMismatch)?;
        self.layout =
            Some(ProgramLayout { exponent, ring_dimension_log2: ring_dimension.trailing_zeros() });
        Ok(())
    }

    /// Declares an input with its frozen stride and attribute knowledge.
    pub fn input(
        &mut self,
        width: usize,
        wire_type: ProgramWireType,
    ) -> Result<ProgramInputId, ProgramValidationError> {
        let id = ProgramInputId::from_index(self.inputs.len());
        let wire = WireId::from_index(self.wires.len());
        if width == 0 ||
            self.inputs.insert(id, width).is_some() ||
            self.input_wires.insert(id, wire).is_some() ||
            self.wires.insert(wire, width).is_some()
        {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        self.wire_types.insert(wire, wire_type);
        Ok(id)
    }

    /// Returns the wire associated with a declared input.
    pub fn input_wire(&self, id: ProgramInputId) -> Result<WireId, ProgramValidationError> {
        self.input_wires
            .get(&id)
            .copied()
            .ok_or(ProgramValidationError::UndefinedWire(WireId::from_index(id.index())))
    }

    /// Declares an RHS family with a known element width.
    pub fn rhs_family(&mut self, width: usize) -> Result<RhsFamilyId, ProgramValidationError> {
        self.rhs_family_with_knowledge(width, AttributeKnowledge::Private)
    }

    /// Declares an RHS family and its plaintext visibility.
    pub fn rhs_family_with_knowledge(
        &mut self,
        width: usize,
        plaintext_knowledge: AttributeKnowledge,
    ) -> Result<RhsFamilyId, ProgramValidationError> {
        if width == 0 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let id = RhsFamilyId::from_index(self.rhs_families.len());
        if self
            .rhs_families
            .insert(id, RhsFamilyDeclaration { width, plaintext_knowledge })
            .is_some()
        {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Declares an externally bound public wire family.
    pub fn wire_family(&mut self, width: usize) -> Result<WireFamilyId, ProgramValidationError> {
        if width == 0 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let id = WireFamilyId::from_index(self.wire_families.len());
        if self.wire_families.insert(id, WireFamilyDeclaration { width }).is_some() {
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
    ) -> Result<RhsId, ProgramValidationError> {
        let exponent_scale_log2 =
            self.layout.ok_or(ProgramValidationError::MissingProgramLayout)?.exponent.stride_log2;
        self.rhs_input_derived(family, lhs_width, rhs_width, exponent_scale_log2)
    }

    /// Declares a RHS using the native exponent scale committed by the
    /// program layout. The scale is intentionally not caller-configurable.
    fn rhs_input_derived(
        &mut self,
        family: RhsFamilyId,
        lhs_width: usize,
        rhs_width: usize,
        exponent_scale_log2: u32,
    ) -> Result<RhsId, ProgramValidationError> {
        if lhs_width == 0 || rhs_width == 0 || !rhs_width.is_power_of_two() {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let rhs_width_log2 = rhs_width.trailing_zeros();
        let ring_log2 =
            self.layout.ok_or(ProgramValidationError::MissingProgramLayout)?.ring_dimension_log2;
        if exponent_scale_log2.saturating_add(rhs_width_log2) > ring_log2 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        if self.rhs_families.get(&family).map(|f| f.width) != Some(rhs_width) {
            return Err(ProgramValidationError::UndefinedRhsFamily(family));
        }
        let id = RhsId::from_index(self.rhs_inputs.len());
        if self
            .rhs_inputs
            .insert(
                id,
                RhsInputDeclaration {
                    family,
                    lhs_width,
                    rhs_width,
                    exponent_scale_log2,
                    plaintext_knowledge: self
                        .rhs_families
                        .get(&family)
                        .map(|declaration| declaration.plaintext_knowledge)
                        .unwrap_or(AttributeKnowledge::Private),
                },
            )
            .is_some()
        {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Adds a LUT table and returns its identifier.
    pub fn register_lut(&mut self, table: LutTable) -> Result<LutId, ProgramValidationError> {
        validate_table(&table)?;
        let id = LutId::from_index(self.luts.len());
        if self.luts.insert(id, table).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        Ok(id)
    }

    /// Adds a unary LUT, expanding a binary declaration into private multiplication
    /// followed by a unary table on the packed index `lhs + lhs_width * rhs`.
    /// The supplied RHS package must already encrypt the packed RHS exponent.
    pub fn lut<S: LutSpec>(
        &mut self,
        input: WireId,
        table: S,
        rhs: Option<RhsId>,
    ) -> Result<WireId, ProgramValidationError> {
        let lut = table.resolve(self)?;
        let table = self.luts.get(&lut).ok_or(ProgramValidationError::UndefinedLut(lut))?.clone();
        validate_lut_layout(self.layout, &table)?;
        if self.wire_width(input)? != table.input_width() {
            return Err(ProgramValidationError::WidthMismatch);
        }
        self.wire_types.get(&input).ok_or(ProgramValidationError::UndefinedWire(input))?;
        let (input, lut) = match rhs {
            Some(rhs) => {
                let declaration = self
                    .rhs_inputs
                    .get(&rhs)
                    .ok_or(ProgramValidationError::UndefinedRhsInput(rhs))?;
                if declaration.lhs_width != table.input_width() ||
                    table.rhs_width() != Some(declaration.rhs_width)
                {
                    return Err(ProgramValidationError::WidthMismatch);
                }
                let unary =
                    LutTable::unary(table.values.len(), table.output_width, table.values.clone())?;
                let existing =
                    self.luts.iter().find_map(|(id, value)| (value == &unary).then_some(*id));
                let unary_id = match existing {
                    Some(id) => id,
                    None => self.register_lut(unary)?,
                };
                (self.private_mul(input, rhs)?, unary_id)
            }
            None if table.rhs_width().is_some() => {
                return Err(ProgramValidationError::WidthMismatch)
            }
            None => (input, lut),
        };
        let output_type =
            *self.wire_types.get(&input).ok_or(ProgramValidationError::UndefinedWire(input))?;
        let output = WireId::from_index(self.wires.len());
        self.wires.insert(output, table.output_width());
        self.wire_types.insert(output, output_type);
        self.steps.push(ProgramStep::Gate(Gate::Lut { input, lut_id: lut, output }));
        Ok(output)
    }

    /// Adds a private multiplication with an explicitly declared RHS package.
    pub fn private_mul(
        &mut self,
        lhs: WireId,
        rhs: RhsId,
    ) -> Result<WireId, ProgramValidationError> {
        let lhs_width = self.wire_width(lhs)?;
        let rhs_decl = self
            .rhs_inputs
            .get(&rhs)
            .ok_or(ProgramValidationError::UndefinedRhsInput(rhs))?
            .clone();
        if lhs_width != rhs_decl.lhs_width {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let layout = self.layout.ok_or(ProgramValidationError::MissingProgramLayout)?;
        // Direct ring multiplication may wrap in the signed native domain.
        // Packed binary LUTs validate their separate table extent at the LUT gate.
        let output_width = layout.private_product_width(lhs_width, rhs_decl.rhs_width);
        let output = WireId::from_index(self.wires.len());
        self.wires.insert(output, output_width);
        let lhs_type =
            self.wire_types.get(&lhs).copied().ok_or(ProgramValidationError::UndefinedWire(lhs))?;
        let rhs_knowledge = rhs_decl.plaintext_knowledge;
        self.wire_types.insert(
            output,
            ProgramWireType {
                knowledge: if lhs_type.knowledge == AttributeKnowledge::Public &&
                    rhs_knowledge == AttributeKnowledge::Public
                {
                    AttributeKnowledge::Public
                } else {
                    AttributeKnowledge::Private
                },
                ..lhs_type
            },
        );
        self.steps.push(ProgramStep::Gate(Gate::MulPrivate { lhs, rhs, output }));
        Ok(output)
    }

    /// Adds multiplication by a public-attribute wire.
    pub fn mul_public(
        &mut self,
        private: WireId,
        public: WireId,
    ) -> Result<WireId, ProgramValidationError> {
        if self.wire_width(private)? != self.wire_width(public)? {
            return Err(ProgramValidationError::WidthMismatch);
        }
        if self.wire_types.get(&public).map(|ty| ty.knowledge) != Some(AttributeKnowledge::Public) {
            return Err(ProgramValidationError::PublicAttributeRequired(public));
        }
        if self.wire_types.get(&private).map(|ty| ty.stride_log2) !=
            self.wire_types.get(&public).map(|ty| ty.stride_log2)
        {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = WireId::from_index(self.wires.len());
        self.wires.insert(output, self.wire_width(private)?);
        let private_type = self
            .wire_types
            .get(&private)
            .copied()
            .ok_or(ProgramValidationError::UndefinedWire(private))?;
        self.wire_types.insert(
            output,
            ProgramWireType {
                knowledge: (private_type.knowledge == AttributeKnowledge::Public)
                    .then_some(AttributeKnowledge::Public)
                    .unwrap_or(AttributeKnowledge::Private),
                ..private_type
            },
        );
        self.steps.push(ProgramStep::Gate(Gate::MulPublic { private, public, output }));
        Ok(output)
    }

    /// Adds ordinary relation addition.
    pub fn add(&mut self, lhs: WireId, rhs: WireId) -> Result<WireId, ProgramValidationError> {
        if self.wire_width(lhs)? != self.wire_width(rhs)? {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let lhs_type =
            self.wire_types.get(&lhs).copied().ok_or(ProgramValidationError::UndefinedWire(lhs))?;
        let rhs_type =
            self.wire_types.get(&rhs).copied().ok_or(ProgramValidationError::UndefinedWire(rhs))?;
        if lhs_type.stride_log2 != rhs_type.stride_log2 {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let output = WireId::from_index(self.wires.len());
        self.wires.insert(output, self.wire_width(lhs)?);
        self.wire_types.insert(
            output,
            ProgramWireType {
                knowledge: (lhs_type.knowledge == AttributeKnowledge::Public &&
                    rhs_type.knowledge == AttributeKnowledge::Public)
                    .then_some(AttributeKnowledge::Public)
                    .unwrap_or(AttributeKnowledge::Private),
                ..lhs_type
            },
        );
        self.steps.push(ProgramStep::Gate(Gate::Add { lhs, rhs, output }));
        Ok(output)
    }

    /// Adds a validated selection-only one-hot gate for a selector-family
    /// lowerer.
    pub fn select(
        &mut self,
        input: WireId,
        selector_family: RhsFamilyId,
        wire_family: WireFamilyId,
        capacity: usize,
    ) -> Result<WireId, ProgramValidationError> {
        let selector = self
            .rhs_families
            .get(&selector_family)
            .ok_or(ProgramValidationError::UndefinedRhsFamily(selector_family))?;
        let values = self
            .wire_families
            .get(&wire_family)
            .ok_or(ProgramValidationError::UndefinedWireFamily(wire_family))?;
        if capacity == 0 ||
            self.wire_width(input)? != selector.width ||
            selector.width != values.width
        {
            return Err(ProgramValidationError::WidthMismatch);
        }
        let input_wire = WireId::from_index(0);
        let public_wire = WireId::from_index(1);
        let multiplied = WireId::from_index(2);
        let weighted = WireId::from_index(3);
        let selector_rhs = RhsId::from_index(0);
        self.family(
            capacity,
            FamilyProgram::new(
                BTreeMap::from([
                    (input_wire, FamilyWireSource::Broadcast(input)),
                    (public_wire, FamilyWireSource::Member(wire_family)),
                ]),
                BTreeMap::from([(selector_rhs, FamilyRhsSource::Member(selector_family))]),
                vec![
                    Gate::MulPrivate { lhs: input_wire, rhs: selector_rhs, output: multiplied },
                    Gate::MulPublic { private: multiplied, public: public_wire, output: weighted },
                ],
                weighted,
            )?,
        )
    }

    /// Adds one structurally shared family body followed by Add aggregation.
    pub(crate) fn family(
        &mut self,
        capacity: usize,
        body: FamilyProgram,
    ) -> Result<WireId, ProgramValidationError> {
        if capacity == 0 {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        let (width, wire_type) = family_program_output_type(self, &body)?;
        let output = WireId::from_index(self.wires.len());
        let family_id = ProgramFamilyId::from_index(
            self.steps.iter().filter(|step| matches!(step, ProgramStep::Family { .. })).count(),
        );
        self.wires.insert(output, width);
        self.wire_types.insert(output, wire_type);
        self.steps.push(ProgramStep::Family {
            id: family_id,
            capacity,
            body,
            aggregation: FamilyAggregation::Add,
            output,
        });
        Ok(output)
    }

    /// Marks a defined wire as a program output.
    pub fn output(&mut self, wire: WireId) -> Result<(), ProgramValidationError> {
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
            wire_types: self.wire_types,
            input_wires: self.input_wires,
            rhs_inputs: self.rhs_inputs,
            rhs_families: self.rhs_families,
            wire_families: self.wire_families,
            luts: self.luts,
            layout: self.layout,
            steps: self.steps,
            outputs: self.outputs,
        };
        Ok(program)
    }

    fn wire_width(&self, wire: WireId) -> Result<usize, ProgramValidationError> {
        self.wires.get(&wire).copied().ok_or(ProgramValidationError::UndefinedWire(wire))
    }

    fn lut_definition(&self, id: LutId) -> Result<&LutTable, ProgramValidationError> {
        self.luts.get(&id).ok_or(ProgramValidationError::UndefinedLut(id))
    }
}

fn canonical_program_bytes(builder: &ExponentLutProgramBuilder) -> Vec<u8> {
    serde_json::to_vec(&(
        &builder.inputs,
        &builder.wire_types,
        &builder.input_wires,
        &builder.rhs_inputs,
        &builder.rhs_families,
        &builder.wire_families,
        &builder.luts,
        &builder.layout,
        &builder.steps,
        &builder.outputs,
    ))
    .expect("ExponentLutProgram declarations are serializable")
}

// Program shape validation.

pub(crate) fn validate_builder(
    builder: &ExponentLutProgramBuilder,
) -> Result<(), ProgramValidationError> {
    let layout = builder.layout.ok_or(ProgramValidationError::MissingProgramLayout)?;
    let ring_dimension = 1usize
        .checked_shl(layout.ring_dimension_log2)
        .ok_or(ProgramValidationError::WidthMismatch)?;
    layout.exponent.validate(ring_dimension).map_err(|_| ProgramValidationError::WidthMismatch)?;
    if builder.wire_types.values().any(|wire| wire.stride_log2 != layout.exponent.stride_log2) {
        return Err(ProgramValidationError::WidthMismatch);
    }
    // RHS declarations are committed when they are created, but the builder
    // permits setting the layout before the final build.  Re-check every
    // stored declaration against that final layout so a declaration cannot
    // retain a stale exponent scale after a later layout change (or after
    // deserialization of untrusted data).
    for declaration in builder.rhs_inputs.values() {
        if declaration.lhs_width == 0 ||
            declaration.rhs_width == 0 ||
            !declaration.rhs_width.is_power_of_two() ||
            builder.rhs_families.get(&declaration.family).map(|family| family.width) !=
                Some(declaration.rhs_width) ||
            declaration.exponent_scale_log2 != layout.exponent.stride_log2 ||
            declaration
                .exponent_scale_log2
                .checked_add(declaration.rhs_width.trailing_zeros())
                .map_or(true, |bound| bound > layout.ring_dimension_log2)
        {
            return Err(ProgramValidationError::WidthMismatch);
        }
    }
    if builder.inputs.len() != builder.input_wires.len() {
        return Err(ProgramValidationError::InvalidBuilderState);
    }
    if builder.wires.len() != builder.wire_types.len() ||
        builder.wires.keys().any(|wire| !builder.wire_types.contains_key(wire))
    {
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
    for step in &builder.steps {
        match step {
            ProgramStep::Gate(gate) => validate_gate_output_type(builder, gate)?,
            ProgramStep::Family { capacity, body, output, .. } => {
                if *capacity == 0 {
                    return Err(ProgramValidationError::WidthMismatch);
                }
                validate_family_program(builder, body, *output)?;
            }
        }
    }
    Ok(())
}

/// Recomputes the derived output wire type of a gate. This is used both for
/// builder-created graphs and for deserialized graphs, so changing a wire's
/// width/knowledge/stride without changing the gate is rejected.
fn validate_gate_output_type(
    builder: &ExponentLutProgramBuilder,
    gate: &Gate,
) -> Result<(), ProgramValidationError> {
    let (output, expected_width, expected_type) = match gate {
        Gate::MulPrivate { lhs, rhs, output } => {
            let width = builder.wire_width(*lhs)?;
            let mut ty =
                *builder.wire_types.get(lhs).ok_or(ProgramValidationError::UndefinedWire(*lhs))?;
            let rhs = builder
                .rhs_inputs
                .get(rhs)
                .ok_or(ProgramValidationError::UndefinedRhsInput(*rhs))?;
            if rhs.lhs_width != width {
                return Err(ProgramValidationError::WidthMismatch);
            }
            if rhs.plaintext_knowledge == AttributeKnowledge::Private {
                ty.knowledge = AttributeKnowledge::Private;
            }
            let layout = builder.layout.ok_or(ProgramValidationError::MissingProgramLayout)?;
            let width = layout.private_product_width(width, rhs.rhs_width);
            (*output, width, ty)
        }
        Gate::Lut { input, lut_id, output } => {
            let width = builder.wire_width(*input)?;
            let ty = *builder
                .wire_types
                .get(input)
                .ok_or(ProgramValidationError::UndefinedWire(*input))?;
            let table = builder.lut_definition(*lut_id)?;
            validate_lut_layout(builder.layout, table)?;
            if table.input_width() != width || table.rhs_width().is_some() {
                return Err(ProgramValidationError::WidthMismatch);
            }
            (*output, table.output_width(), ty)
        }

        Gate::MulPublic { private, public, output } => {
            let private_type = *builder
                .wire_types
                .get(private)
                .ok_or(ProgramValidationError::UndefinedWire(*private))?;
            let public_type = *builder
                .wire_types
                .get(public)
                .ok_or(ProgramValidationError::UndefinedWire(*public))?;
            if builder.wire_width(*private)? != builder.wire_width(*public)? ||
                private_type.stride_log2 != public_type.stride_log2
            {
                return Err(ProgramValidationError::WidthMismatch);
            }
            if public_type.knowledge != AttributeKnowledge::Public {
                return Err(ProgramValidationError::PublicAttributeRequired(*public));
            }
            (
                *output,
                builder.wire_width(*private)?,
                ProgramWireType {
                    stride_log2: private_type.stride_log2,
                    knowledge: if private_type.knowledge == AttributeKnowledge::Public &&
                        public_type.knowledge == AttributeKnowledge::Public
                    {
                        AttributeKnowledge::Public
                    } else {
                        AttributeKnowledge::Private
                    },
                },
            )
        }
        Gate::Add { lhs, rhs, output } => {
            let lhs_type =
                *builder.wire_types.get(lhs).ok_or(ProgramValidationError::UndefinedWire(*lhs))?;
            let rhs_type =
                *builder.wire_types.get(rhs).ok_or(ProgramValidationError::UndefinedWire(*rhs))?;
            if builder.wire_width(*lhs)? != builder.wire_width(*rhs)? ||
                lhs_type.stride_log2 != rhs_type.stride_log2
            {
                return Err(ProgramValidationError::WidthMismatch);
            }
            (
                *output,
                builder.wire_width(*lhs)?,
                ProgramWireType {
                    stride_log2: lhs_type.stride_log2,
                    knowledge: if lhs_type.knowledge == AttributeKnowledge::Public &&
                        rhs_type.knowledge == AttributeKnowledge::Public
                    {
                        AttributeKnowledge::Public
                    } else {
                        AttributeKnowledge::Private
                    },
                },
            )
        }
    };
    if builder.wires.get(&output).copied() != Some(expected_width) ||
        builder.wire_types.get(&output).copied() != Some(expected_type)
    {
        return Err(ProgramValidationError::WidthMismatch);
    }
    Ok(())
}

fn validate_family_program(
    builder: &ExponentLutProgramBuilder,
    body: &FamilyProgram,
    outer_output: WireId,
) -> Result<(), ProgramValidationError> {
    let (output_width, output_type) = family_program_output_type(builder, body)?;
    if builder.wires.get(&outer_output).copied() != Some(output_width) ||
        builder.wire_types.get(&outer_output).copied() != Some(output_type)
    {
        return Err(ProgramValidationError::WidthMismatch);
    }
    Ok(())
}

fn family_program_output_type(
    builder: &ExponentLutProgramBuilder,
    body: &FamilyProgram,
) -> Result<(usize, ProgramWireType), ProgramValidationError> {
    let mut widths = BTreeMap::new();
    let mut types = BTreeMap::new();
    for (wire, source) in body.wire_inputs() {
        let (width, wire_type) = match source {
            FamilyWireSource::Broadcast(source) => (
                builder.wire_width(*source)?,
                *builder
                    .wire_types
                    .get(source)
                    .ok_or(ProgramValidationError::UndefinedWire(*source))?,
            ),
            FamilyWireSource::Member(family) => {
                let declaration = builder
                    .wire_families
                    .get(family)
                    .ok_or(ProgramValidationError::UndefinedWireFamily(*family))?;
                (
                    declaration.width,
                    ProgramWireType {
                        stride_log2: builder
                            .layout
                            .ok_or(ProgramValidationError::MissingProgramLayout)?
                            .exponent
                            .stride_log2,
                        knowledge: AttributeKnowledge::Public,
                    },
                )
            }
        };
        if widths.insert(*wire, width).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        types.insert(*wire, wire_type);
    }
    let rhs_attributes = |source: &FamilyRhsSource| match source {
        FamilyRhsSource::Broadcast(id) => {
            let declaration =
                builder.rhs_inputs.get(id).ok_or(ProgramValidationError::UndefinedRhsInput(*id))?;
            Ok((
                Some(declaration.lhs_width),
                declaration.rhs_width,
                declaration.plaintext_knowledge,
            ))
        }
        FamilyRhsSource::Member(id) => {
            let declaration = builder
                .rhs_families
                .get(id)
                .ok_or(ProgramValidationError::UndefinedRhsFamily(*id))?;
            Ok((None, declaration.width, declaration.plaintext_knowledge))
        }
    };
    for source in body.rhs_inputs().values() {
        rhs_attributes(source)?;
    }
    for gate in body.gates() {
        let (output, width, wire_type) = match gate {
            Gate::MulPrivate { lhs, rhs, output } => {
                let lhs_width = *widths.get(lhs).ok_or(ProgramValidationError::WidthMismatch)?;
                let source =
                    body.rhs_inputs().get(rhs).ok_or(ProgramValidationError::WidthMismatch)?;
                let (declared_lhs, rhs_width, rhs_knowledge) = rhs_attributes(source)?;
                let output_width = if let Some(declared_lhs) = declared_lhs {
                    if lhs_width != declared_lhs {
                        return Err(ProgramValidationError::WidthMismatch);
                    }
                    builder
                        .layout
                        .ok_or(ProgramValidationError::MissingProgramLayout)?
                        .private_product_width(lhs_width, rhs_width)
                } else {
                    if lhs_width != rhs_width {
                        return Err(ProgramValidationError::WidthMismatch);
                    }
                    lhs_width
                };
                let lhs_type = *types.get(lhs).ok_or(ProgramValidationError::WidthMismatch)?;
                (
                    *output,
                    output_width,
                    ProgramWireType {
                        knowledge: if lhs_type.knowledge == AttributeKnowledge::Public &&
                            rhs_knowledge == AttributeKnowledge::Public
                        {
                            AttributeKnowledge::Public
                        } else {
                            AttributeKnowledge::Private
                        },
                        ..lhs_type
                    },
                )
            }
            Gate::MulPublic { private, public, output } => {
                let private_width =
                    *widths.get(private).ok_or(ProgramValidationError::WidthMismatch)?;
                let public_width =
                    *widths.get(public).ok_or(ProgramValidationError::WidthMismatch)?;
                let private_type =
                    *types.get(private).ok_or(ProgramValidationError::WidthMismatch)?;
                let public_type =
                    *types.get(public).ok_or(ProgramValidationError::WidthMismatch)?;
                if private_width != public_width ||
                    public_type.knowledge != AttributeKnowledge::Public ||
                    private_type.stride_log2 != public_type.stride_log2
                {
                    return Err(ProgramValidationError::WidthMismatch);
                }
                (
                    *output,
                    private_width,
                    ProgramWireType {
                        knowledge: if private_type.knowledge == AttributeKnowledge::Public {
                            AttributeKnowledge::Public
                        } else {
                            AttributeKnowledge::Private
                        },
                        ..private_type
                    },
                )
            }
            Gate::Add { lhs, rhs, output } => {
                let lhs_width = *widths.get(lhs).ok_or(ProgramValidationError::WidthMismatch)?;
                let rhs_width = *widths.get(rhs).ok_or(ProgramValidationError::WidthMismatch)?;
                let lhs_type = *types.get(lhs).ok_or(ProgramValidationError::WidthMismatch)?;
                let rhs_type = *types.get(rhs).ok_or(ProgramValidationError::WidthMismatch)?;
                if lhs_width != rhs_width || lhs_type.stride_log2 != rhs_type.stride_log2 {
                    return Err(ProgramValidationError::WidthMismatch);
                }
                (
                    *output,
                    lhs_width,
                    ProgramWireType {
                        knowledge: if lhs_type.knowledge == AttributeKnowledge::Public &&
                            rhs_type.knowledge == AttributeKnowledge::Public
                        {
                            AttributeKnowledge::Public
                        } else {
                            AttributeKnowledge::Private
                        },
                        ..lhs_type
                    },
                )
            }
            Gate::Lut { input, lut_id: lut, output } => {
                let input_width =
                    *widths.get(input).ok_or(ProgramValidationError::WidthMismatch)?;
                let input_type = *types.get(input).ok_or(ProgramValidationError::WidthMismatch)?;
                let table =
                    builder.luts.get(lut).ok_or(ProgramValidationError::UndefinedLut(*lut))?;
                validate_lut_layout(builder.layout, table)?;
                if input_width != table.input_width() || table.rhs_width().is_some() {
                    return Err(ProgramValidationError::WidthMismatch);
                }
                (*output, table.output_width(), input_type)
            }
        };
        if widths.insert(output, width).is_some() {
            return Err(ProgramValidationError::DuplicateIdentifier);
        }
        types.insert(output, wire_type);
    }
    let output_width = *widths.get(&body.output()).ok_or(ProgramValidationError::WidthMismatch)?;
    let output_type = *types.get(&body.output()).ok_or(ProgramValidationError::WidthMismatch)?;
    Ok((output_width, output_type))
}

fn validate_lut_layout(
    layout: Option<ProgramLayout>,
    table: &LutTable,
) -> Result<(), ProgramValidationError> {
    let layout = layout.ok_or(ProgramValidationError::MissingProgramLayout)?;
    let native_width = 1usize
        .checked_shl(
            layout
                .ring_dimension_log2
                .checked_sub(layout.exponent.stride_log2)
                .ok_or(ProgramValidationError::WidthMismatch)?,
        )
        .ok_or(ProgramValidationError::WidthMismatch)?;
    if !table.values().len().is_power_of_two() || table.values().len() > native_width {
        return Err(ProgramValidationError::WidthMismatch);
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
    Ok(())
}

// Shared runtime binding and traversal.  This layer intentionally knows only
// the program's logical wires.  Concrete BGG matrices, encodings, preimages,
// and noise states belong to the interpreters that implement the traits below.

use std::error::Error;

/// Stable structural identity of one primitive-operation invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ProgramGateInstance {
    Gate { step: usize },
    FamilyGate { step: usize, gate: usize, family: ProgramFamilyId },
    FamilyReduction { step: usize },
}

/// Scalar or structural-family storage for a Program value.
///
/// Scalar and indexed storage are a Program concern, including for fixed RHS
/// operands. The payload types retain their cryptographic and phase identity;
/// a family is never expanded into scalar values on the host.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProgramValue<S, F> {
    Scalar(S),
    Family(F),
}

impl<S, F> ProgramValue<S, F> {
    pub fn scalar(&self) -> Option<&S> {
        match self {
            Self::Scalar(value) => Some(value),
            Self::Family(_) => None,
        }
    }

    pub fn as_ref(&self) -> ProgramValue<&S, &F> {
        match self {
            Self::Scalar(value) => ProgramValue::Scalar(value),
            Self::Family(values) => ProgramValue::Family(values),
        }
    }
}

impl<S: Clone, F: Clone> ProgramValue<&S, &F> {
    pub fn cloned(self) -> ProgramValue<S, F> {
        match self {
            Self::Scalar(value) => ProgramValue::Scalar(value.clone()),
            Self::Family(values) => ProgramValue::Family(values.clone()),
        }
    }
}

impl<T: mxx_dsl::GraphValue> ProgramValue<T, mxx_dsl::Family<T>> {
    /// Resolve an indexed operand without materializing a shared value.
    /// Cloning a scalar reuses its graph handles and therefore its source.
    pub fn at(&self, index: impl Into<mxx_dsl::Int>) -> T {
        match self {
            Self::Scalar(value) => value.clone(),
            Self::Family(values) => values.at(index),
        }
    }

    pub fn element_schema(&self) -> T::Schema {
        match self {
            Self::Scalar(value) => value.schema(),
            Self::Family(values) => mxx_dsl::GraphValue::schema(values).element,
        }
    }
}

impl<S: mxx_dsl::GraphValue, F: mxx_dsl::GraphValue> mxx_dsl::GraphValue for ProgramValue<S, F> {
    type Schema = ProgramValue<S::Schema, F::Schema>;

    fn flatten(&self) -> Vec<mxx_ir_core::ValueHandle> {
        match self {
            Self::Scalar(value) => value.flatten(),
            Self::Family(values) => values.flatten(),
        }
    }

    fn schema(&self) -> Self::Schema {
        match self {
            Self::Scalar(value) => ProgramValue::Scalar(value.schema()),
            Self::Family(values) => ProgramValue::Family(values.schema()),
        }
    }

    fn from_values(
        schema: &Self::Schema,
        values: &[mxx_ir_core::ValueHandle],
    ) -> Result<Self, mxx_dsl::DslError> {
        match schema {
            ProgramValue::Scalar(schema) => S::from_values(schema, values).map(Self::Scalar),
            ProgramValue::Family(schema) => F::from_values(schema, values).map(Self::Family),
        }
    }
}

impl<S: mxx_dsl::GraphValueSchema, F: mxx_dsl::GraphValueSchema> mxx_dsl::GraphValueSchema
    for ProgramValue<S, F>
{
    type Value = ProgramValue<S::Value, F::Value>;

    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        match self {
            Self::Scalar(schema) => ProgramValue::Scalar(schema.placeholders_from(next)),
            Self::Family(schema) => ProgramValue::Family(schema.placeholders_from(next)),
        }
    }

    fn wire_types(&self) -> Vec<mxx_ir_core::types::WireType> {
        match self {
            Self::Scalar(schema) => schema.wire_types(),
            Self::Family(schema) => schema.wire_types(),
        }
    }
}

/// Carrier and error types selected by an interpreter for one program scope.
pub trait ProgramLoweringTypes {
    type Wire: Clone;
    type Rhs: Clone;
    type HelperSet;
    type Error: Error + Send + Sync + 'static;
}

/// The four primitive exponent-program operations.
///
/// Preparation must pair the interpreter with a Program of the same exponent
/// layout and ring dimension. Bindings must provide public plaintext metadata
/// for public inputs; operations must preserve that metadata whenever their
/// output remains public. Lowering relies on these contracts without querying
/// the interpreter again at entry or at each gate.
pub trait ExponentProgramLowering: ProgramLoweringTypes {
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error>;

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error>;

    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        material: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error>;

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error>;
}

#[derive(Debug, Error)]
pub enum ProgramLowerError<E: Error + Send + Sync + 'static> {
    #[error(transparent)]
    Validation(#[from] ProgramValidationError),
    #[error(transparent)]
    Graph(#[from] mxx_dsl::DslError),
    #[error("operation {gate:?}: {source}")]
    Operation {
        gate: ProgramGateInstance,
        #[source]
        source: E,
    },
}

/// Runtime values bound to the declarations of one [`ExponentLutProgram`].
///
/// Each interpreter chooses its concrete wire, RHS, and helper carriers.
/// Family values arrive already wrapped in those carrier types, so traversal
/// never expands a family into host-side members. Inputs declared public must
/// carry their public plaintext metadata. All values and prepared helpers must
/// belong to the layout and scheme used to prepare this Program.
pub struct ProgramBindings<'a, W, R, H, WF = W, RF = R> {
    pub inputs: &'a BTreeMap<ProgramInputId, W>,
    pub rhs_inputs: &'a BTreeMap<RhsId, R>,
    pub rhs_families: &'a BTreeMap<RhsFamilyId, RF>,
    pub wire_values: &'a BTreeMap<WireFamilyId, WF>,
    pub helpers: &'a BTreeMap<LutId, H>,
    /// Instance material takes precedence over reusable table-only material.
    /// Fixed private-product material is bound to its operation instance here.
    pub gate_helpers: Option<&'a BTreeMap<ProgramGateInstance, H>>,
}

impl<'a, W, R, H, WF, RF> ProgramBindings<'a, W, R, H, WF, RF> {
    pub fn new(
        inputs: &'a BTreeMap<ProgramInputId, W>,
        rhs_inputs: &'a BTreeMap<RhsId, R>,
        rhs_families: &'a BTreeMap<RhsFamilyId, RF>,
        wire_values: &'a BTreeMap<WireFamilyId, WF>,
        helpers: &'a BTreeMap<LutId, H>,
        gate_helpers: Option<&'a BTreeMap<ProgramGateInstance, H>>,
    ) -> Self {
        Self { inputs, rhs_inputs, rhs_families, wire_values, helpers, gate_helpers }
    }
}

type LoweringWire<B> = <B as ProgramLoweringTypes>::Wire;
type LoweringRhs<B> = <B as ProgramLoweringTypes>::Rhs;
type LoweringHelpers<B> = <B as ProgramLoweringTypes>::HelperSet;
type LoweringError<B> = <B as ProgramLoweringTypes>::Error;

/// One gate dispatch for root and indexed scopes; family construction remains
/// structural in `lower_program`, and scheme operations own only arithmetic.
pub(crate) fn lower_gate<B: ExponentProgramLowering>(
    program: &ExponentLutProgram,
    gate: &Gate,
    at: ProgramGateInstance,
    wires: &BTreeMap<WireId, B::Wire>,
    rhs_inputs: &BTreeMap<RhsId, B::Rhs>,
    helpers: (&BTreeMap<LutId, B::HelperSet>, Option<&BTreeMap<ProgramGateInstance, B::HelperSet>>),
    backend: &mut B,
) -> Result<(WireId, B::Wire), ProgramLowerError<B::Error>> {
    let wire = |id: &WireId| wires.get(id).ok_or(ProgramValidationError::UndefinedWire(*id));
    let rhs = |id: RhsId| rhs_inputs.get(&id).ok_or(ProgramValidationError::MissingRuntimeRhs(id));
    let (output, value) = match gate {
        Gate::MulPublic { private, public, output } => {
            let public_value = wire(public)?;
            (*output, backend.mul_public(wire(private)?, public_value, at))
        }
        Gate::Add { lhs, rhs: right, output } => {
            (*output, backend.add(wire(lhs)?, wire(right)?, at))
        }
        Gate::MulPrivate { lhs, rhs: right, output } => {
            let material = helpers.1.and_then(|helpers| helpers.get(&at));
            (*output, backend.mul_private(wire(lhs)?, rhs(*right)?, material, at))
        }
        Gate::Lut { input, lut_id, output } => {
            let helper = helpers
                .1
                .and_then(|helpers| helpers.get(&at))
                .or_else(|| helpers.0.get(lut_id))
                .ok_or(ProgramValidationError::MissingRuntimeLutHelpers(*lut_id))?;
            (*output, backend.lut(program, *lut_id, wire(input)?, Some(helper), at))
        }
    };
    value
        .map(|value| (output, value))
        .map_err(|source| ProgramLowerError::Operation { gate: at, source })
}

/// Evaluate one SSA body after its captures and indexed inputs are resolved.
/// Graph interpreters call this inside their structural scope; symbolic noise
/// interpreters use the same gates without constructing graph values.
pub(crate) fn lower_body<B: ExponentProgramLowering>(
    program: &ExponentLutProgram,
    step_index: usize,
    family: ProgramFamilyId,
    body: &FamilyProgram,
    mut wires: BTreeMap<WireId, B::Wire>,
    rhs_inputs: &BTreeMap<RhsId, B::Rhs>,
    helpers: (&BTreeMap<LutId, B::HelperSet>, Option<&BTreeMap<ProgramGateInstance, B::HelperSet>>),
    backend: &mut B,
) -> Result<B::Wire, ProgramLowerError<B::Error>> {
    for (gate_index, gate) in body.gates().iter().enumerate() {
        let at = ProgramGateInstance::FamilyGate { step: step_index, gate: gate_index, family };
        let (output, value) = lower_gate(program, gate, at, &wires, rhs_inputs, helpers, backend)?;
        wires.insert(output, value);
    }
    wires.remove(&body.output()).ok_or_else(|| ProgramValidationError::InvalidOutput.into())
}

/// Resolves bindings and evaluates a frozen program through an opaque
/// interpreter.  The program layer owns only SSA traversal and structural
/// family control; every carried value is selected by `B`.
pub(crate) fn lower_program<B>(
    program: &ExponentLutProgram,
    bindings: &ProgramBindings<'_, LoweringWire<B>, LoweringRhs<B>, LoweringHelpers<B>>,
    family_ranges: &ProgramFamilyBindings,
    backend: &mut B,
    mut reduce: impl FnMut(
        &mut B,
        &B::Wire,
        &FamilyRange,
        ProgramGateInstance,
    ) -> Result<B::Wire, B::Error>,
) -> Result<BTreeMap<WireId, LoweringWire<B>>, ProgramLowerError<LoweringError<B>>>
where
    B: ExponentProgramLowering,
{
    let mut wires = BTreeMap::new();
    for (input_id, wire_id) in program.input_wires() {
        let value = bindings
            .inputs
            .get(input_id)
            .cloned()
            .ok_or(ProgramValidationError::MissingRuntimeInput(*input_id))?;
        wires.insert(*wire_id, value);
    }

    for (step_index, step) in program.steps().iter().enumerate() {
        match step {
            ProgramStep::Gate(gate) => {
                let at = ProgramGateInstance::Gate { step: step_index };
                let (output, value) = lower_gate(
                    program,
                    gate,
                    at,
                    &wires,
                    bindings.rhs_inputs,
                    (bindings.helpers, bindings.gate_helpers),
                    backend,
                )?;
                wires.insert(output, value);
            }
            ProgramStep::Family { id, capacity, body, output, .. } => {
                let range =
                    family_ranges.range(*id).ok_or(ProgramValidationError::WidthMismatch)?;
                if range
                    .capacity()
                    .evaluate(&mxx_ir_core::ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize()) !=
                    Some(*capacity)
                {
                    return Err(ProgramValidationError::WidthMismatch.into());
                }

                let (family_wires, family_rhs) = body.bind_inputs(
                    |wire| {
                        let value =
                            wires.get(&wire).ok_or(ProgramValidationError::UndefinedWire(wire))?;
                        Ok::<_, ProgramLowerError<B::Error>>(value.clone())
                    },
                    |family| {
                        bindings
                            .wire_values
                            .get(&family)
                            .cloned()
                            .ok_or_else(|| ProgramValidationError::WidthMismatch.into())
                    },
                    |rhs| {
                        bindings
                            .rhs_inputs
                            .get(&rhs)
                            .cloned()
                            .ok_or_else(|| ProgramValidationError::MissingRuntimeRhs(rhs).into())
                    },
                    |family| {
                        bindings
                            .rhs_families
                            .get(&family)
                            .cloned()
                            .ok_or_else(|| ProgramValidationError::WidthMismatch.into())
                    },
                )?;

                let value = lower_body(
                    program,
                    step_index,
                    *id,
                    body,
                    family_wires,
                    &family_rhs,
                    (bindings.helpers, bindings.gate_helpers),
                    backend,
                )?;
                let at = ProgramGateInstance::FamilyReduction { step: step_index };
                let reduced = reduce(backend, &value, range, at)
                    .map_err(|source| ProgramLowerError::Operation { gate: at, source })?;
                wires.insert(*output, reduced);
            }
        }
    }
    Ok(wires)
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
    fn gate_instances_cover_each_multiplication_and_family_body() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 }, 16).unwrap();
        let private = builder
            .input(1, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let public = builder
            .input(4, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let family = builder.rhs_family(4).unwrap();
        let values = builder.wire_family(4).unwrap();
        let rhs = builder.rhs_input(family, 1, 4).unwrap();
        let first = builder.private_mul(builder.input_wire(private).unwrap(), rhs).unwrap();
        let second = builder.mul_public(first, builder.input_wire(public).unwrap()).unwrap();
        let selected = builder.select(second, family, values, 4).unwrap();
        builder.output(selected).unwrap();
        let program = builder.build().unwrap();
        let instances = program.gate_instances().collect::<Vec<_>>();
        assert_eq!(instances.len(), 4);
        assert!(instances.iter().all(|(_, gate)| matches!(
            gate,
            crate::program::Gate::MulPrivate { .. } | crate::program::Gate::MulPublic { .. }
        )));
        assert!(instances.iter().any(|(instance, _)| matches!(
            instance,
            crate::program::ProgramGateInstance::FamilyGate { gate: 0, .. }
        )));
        assert!(instances.iter().any(|(instance, _)| matches!(
            instance,
            crate::program::ProgramGateInstance::FamilyGate { gate: 1, .. }
        )));
    }

    #[test]
    fn binary_rhs_is_explicit_and_identity_is_canonical() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let lhs = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let lut = LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap();
        let output = builder.lut(builder.input_wire(lhs).unwrap(), lut, Some(rhs)).unwrap();
        builder.output(output).unwrap();
        let first = builder.build().unwrap();

        let mut second_builder = ExponentLutProgramBuilder::new();
        second_builder
            .set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16)
            .unwrap();
        let lhs = second_builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = second_builder.rhs_family(2).unwrap();
        let rhs = second_builder.rhs_input(family, 2, 2).unwrap();
        let lut = LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap();
        let output =
            second_builder.lut(second_builder.input_wire(lhs).unwrap(), lut, Some(rhs)).unwrap();
        second_builder.output(output).unwrap();
        assert_eq!(first.id(), second_builder.build().unwrap().id());
        let reopened: Vec<ProgramStep> =
            serde_json::from_slice(&serde_json::to_vec(first.steps()).unwrap()).unwrap();
        assert_eq!(first.steps(), reopened);
        assert!(matches!(first.gates()[0], Gate::MulPrivate { rhs: RhsId(0), .. }));
        assert!(matches!(first.gates()[1], Gate::Lut { .. }));
    }

    #[test]
    fn layout_is_mandatory_and_rhs_cannot_bypass_ring_bound() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        assert_eq!(
            builder.rhs_input(family, 2, 2),
            Err(ProgramValidationError::MissingProgramLayout)
        );
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let output = builder.lut(
            builder.input_wire(input).unwrap(),
            LutTable::unary(2, 2, vec![0, 1]).unwrap(),
            None,
        );
        let output = output.unwrap();
        builder.output(output).unwrap();
        assert!(builder.build().is_ok());
    }

    #[test]
    fn native_private_product_preserves_the_full_signed_domain() {
        for stride_log2 in [0, 1, 3] {
            let width = 16 >> stride_log2;
            let mut builder = ExponentLutProgramBuilder::new();
            builder
                .set_layout(
                    crate::layout::ExpLayout { stride_log2, width_log2: 4 - stride_log2 },
                    16,
                )
                .unwrap();
            let input = builder
                .input(
                    width,
                    ProgramWireType { stride_log2, knowledge: AttributeKnowledge::Private },
                )
                .unwrap();
            let family = builder.rhs_family(width).unwrap();
            let rhs = builder.rhs_input(family, width, width).unwrap();
            let first = builder.private_mul(builder.input_wire(input).unwrap(), rhs).unwrap();
            let second = builder.private_mul(first, rhs).unwrap();
            assert_eq!(builder.wire_width(first).unwrap(), width);
            assert_eq!(builder.wire_width(second).unwrap(), width);
            // A direct native product is not an injective two-axis LUT packing.
            assert_eq!(
                builder.lut(
                    first,
                    LutTable::binary(width, width, width, vec![0; width * width]).unwrap(),
                    Some(rhs),
                ),
                Err(ProgramValidationError::WidthMismatch)
            );
            builder.output(second).unwrap();
            let program = builder.build().unwrap();
            assert!(program.gates().iter().all(|gate| matches!(gate, Gate::MulPrivate { .. })));
        }
    }

    #[test]
    fn rhs_scale_and_layout_are_validated_and_committed_to_identity() {
        let mut first = ExponentLutProgramBuilder::new();
        first.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = first
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = first.rhs_family(2).unwrap();
        let rhs = first.rhs_input(family, 2, 2).unwrap();
        assert_eq!(first.rhs_inputs.get(&rhs).map(|decl| decl.exponent_scale_log2), Some(0));
        let output = first.private_mul(first.input_wire(input).unwrap(), rhs).unwrap();
        first.output(output).unwrap();
        let first = first.build().unwrap();

        let mut second = ExponentLutProgramBuilder::new();
        second.set_layout(crate::layout::ExpLayout { stride_log2: 1, width_log2: 1 }, 16).unwrap();
        let input = second
            .input(2, ProgramWireType { stride_log2: 1, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = second.rhs_family(2).unwrap();
        let rhs = second.rhs_input(family, 2, 2).unwrap();
        let output = second.private_mul(second.input_wire(input).unwrap(), rhs).unwrap();
        second.output(output).unwrap();
        let second = second.build().unwrap();
        assert_ne!(first.id(), second.id());

        let mut rejected = ExponentLutProgramBuilder::new();
        rejected
            .set_layout(crate::layout::ExpLayout { stride_log2: 3, width_log2: 0 }, 16)
            .unwrap();
        let family = rejected.rhs_family(4).unwrap();
        assert_eq!(rejected.rhs_input(family, 2, 4), Err(ProgramValidationError::WidthMismatch));

        // A declaration made under m=0 must not survive a later final
        // layout change to m=3.  The declaration stores its committed scale,
        // while all wires below are intentionally declared for the final
        // layout so this isolates RHS revalidation.
        let mut changed = ExponentLutProgramBuilder::new();
        changed.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 0 }, 16).unwrap();
        let family = changed.rhs_family(2).unwrap();
        let rhs = changed.rhs_input(family, 2, 2).unwrap();
        changed.set_layout(crate::layout::ExpLayout { stride_log2: 3, width_log2: 0 }, 16).unwrap();
        let input = changed
            .input(2, ProgramWireType { stride_log2: 3, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let output = changed.private_mul(changed.input_wire(input).unwrap(), rhs).unwrap();
        changed.output(output).unwrap();
        assert!(matches!(changed.build(), Err(ProgramValidationError::WidthMismatch)));
    }

    #[test]
    fn frozen_gate_expansions_and_knowledge_propagation_are_explicit() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let private = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let public = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let private_wire = builder.input_wire(private).unwrap();
        let public_wire = builder.input_wire(public).unwrap();
        let _fused = builder.private_mul(private_wire, rhs).unwrap();
        let rejected = builder.mul_public(private_wire, private_wire);
        assert_eq!(rejected, Err(ProgramValidationError::PublicAttributeRequired(private_wire)));
        let product = builder.mul_public(public_wire, public_wire).unwrap();
        let sum = builder.add(private_wire, public_wire).unwrap();
        builder.output(product).unwrap();
        builder.output(sum).unwrap();
        let program = builder.build().unwrap();
        assert!(matches!(program.gates()[0], Gate::MulPrivate { .. }));
        assert!(matches!(program.gates()[1], Gate::MulPublic { .. }));
        assert!(matches!(program.gates()[2], Gate::Add { .. }));
        assert_eq!(program.wire_type(product).unwrap().knowledge, AttributeKnowledge::Public);
        assert_eq!(program.wire_type(sum).unwrap().knowledge, AttributeKnowledge::Private);
        assert_eq!(program.wire_type(product).unwrap().stride_log2, 0);
    }

    #[test]
    fn mul_private_exposes_public_only_with_public_rhs_relation() {
        let mut public_builder = ExponentLutProgramBuilder::new();
        public_builder
            .set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16)
            .unwrap();
        let lhs = public_builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let public_family =
            public_builder.rhs_family_with_knowledge(2, AttributeKnowledge::Public).unwrap();
        let public_rhs = public_builder.rhs_input(public_family, 2, 2).unwrap();
        let public_product = public_builder
            .private_mul(public_builder.input_wire(lhs).unwrap(), public_rhs)
            .unwrap();
        public_builder.output(public_product).unwrap();
        let public_program = public_builder.build().unwrap();
        assert_eq!(
            public_program.wire_type(public_product).unwrap().knowledge,
            AttributeKnowledge::Public
        );

        let mut private_builder = ExponentLutProgramBuilder::new();
        private_builder
            .set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16)
            .unwrap();
        let lhs = private_builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Public })
            .unwrap();
        let private_family = private_builder.rhs_family(2).unwrap();
        let private_rhs = private_builder.rhs_input(private_family, 2, 2).unwrap();
        let private_product = private_builder
            .private_mul(private_builder.input_wire(lhs).unwrap(), private_rhs)
            .unwrap();
        private_builder.output(private_product).unwrap();
        let private_program = private_builder.build().unwrap();
        assert_eq!(
            private_program.wire_type(private_product).unwrap().knowledge,
            AttributeKnowledge::Private
        );
    }

    #[test]
    fn binary_lut_expands_to_private_multiplication_and_unary_lut() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let output = builder
            .lut(
                builder.input_wire(input).unwrap(),
                LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap(),
                Some(rhs),
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        assert_eq!(program.gates().len(), 2);
        let Gate::Lut { input: packed, lut_id, .. } = program.gates()[1] else {
            panic!("unary LUT")
        };
        assert_eq!(program.lut(lut_id).unwrap().rhs_width(), None);
        assert_eq!(program.lut(lut_id).unwrap().input_width(), 4);
        assert!(matches!(program.gates()[0], Gate::MulPrivate { output, .. } if output == packed));
        assert!(matches!(program.gates()[0], Gate::MulPrivate { rhs: id, .. } if id == rhs));
    }

    #[test]
    fn binary_lut_reuses_unary_table_and_rejects_truncation() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 1, width_log2: 3 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 1, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(4).unwrap();
        let first_rhs = builder.rhs_input(family, 2, 4).unwrap();
        let second_rhs = builder.rhs_input(family, 2, 4).unwrap();
        let input = builder.input_wire(input).unwrap();
        let table =
            builder.register_lut(LutTable::binary(2, 4, 8, (0..8).collect()).unwrap()).unwrap();
        builder.lut(input, table, Some(first_rhs)).unwrap();
        builder.lut(input, table, Some(second_rhs)).unwrap();
        let ProgramStep::Gate(Gate::Lut { lut_id: first, .. }) = builder.steps[1] else {
            panic!("LUT")
        };
        let ProgramStep::Gate(Gate::Lut { lut_id: second, .. }) = builder.steps[3] else {
            panic!("LUT")
        };
        assert_eq!(first, second);
        let unary = &builder.luts[&first];
        assert_eq!(unary, &LutTable::unary(8, 8, (0..8).collect()).unwrap());
        for y in 0..4 {
            for x in 0..2 {
                assert_eq!(unary.values()[x + 2 * y], x + 2 * y);
            }
        }
        let family = builder.rhs_family(8).unwrap();
        let rhs = builder.rhs_input(family, 2, 8).unwrap();
        let too_wide =
            builder.register_lut(LutTable::binary(2, 8, 1, vec![0; 16]).unwrap()).unwrap();
        let before = (builder.steps.clone(), builder.wires.clone(), builder.wire_types.clone());
        assert!(builder.lut(input, too_wide, Some(rhs)).is_err());
        assert_eq!(
            before,
            (builder.steps.clone(), builder.wires.clone(), builder.wire_types.clone())
        );
        assert!(builder.lut(input, table, None).is_err());
    }

    #[test]
    fn lut_gate_rejects_removed_rhs_field() {
        let gate = Gate::Lut { input: WireId(0), lut_id: LutId(0), output: WireId(1) };
        let mut value = serde_json::to_value(&gate).unwrap();
        value["Lut"]["rhs"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<Gate>(value).is_err());
        assert_eq!(
            serde_json::from_slice::<Gate>(&serde_json::to_vec(&gate).unwrap()).unwrap(),
            gate
        );
    }

    #[test]
    fn one_hot_selection_uses_mul_private_without_registered_table() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let wire_family = builder.wire_family(2).unwrap();
        let output = builder
            .select(builder.input_wire(input).unwrap(), selector_family, wire_family, 2)
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        assert!(program.luts.is_empty());
        assert!(program.gates().is_empty());
        assert!(matches!(
            &program.steps()[0],
            ProgramStep::Family { capacity: 2, body, .. }
                if matches!(body.gates(), [Gate::MulPrivate { .. }, Gate::MulPublic { .. }])
        ));
    }

    #[test]
    fn one_hot_selection_requires_matching_input_and_value_widths() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let wire_family = builder.wire_family(4).unwrap();
        assert_eq!(
            builder.select(builder.input_wire(input).unwrap(), selector_family, wire_family, 2),
            Err(ProgramValidationError::WidthMismatch)
        );
    }

    #[test]
    fn structural_family_reduction_uses_only_the_active_partial_range() {
        #[derive(Clone)]
        enum TestWire {
            Scalar(i32),
            Members(Vec<i32>),
            Active { values: Vec<i32>, start: usize },
        }
        #[derive(Clone)]
        enum TestRhs {
            Members,
        }
        struct TestBackend {
            start: usize,
            count: usize,
        }
        impl ProgramLoweringTypes for TestBackend {
            type Wire = TestWire;
            type Rhs = TestRhs;
            type HelperSet = ();
            type Error = crate::ExponentLutError;
        }
        impl ExponentProgramLowering for TestBackend {
            fn mul_public(
                &mut self,
                private: &Self::Wire,
                public: &Self::Wire,
                _gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                let (TestWire::Active { values: private, start }, TestWire::Members(public)) =
                    (private, public)
                else {
                    return Err(crate::ExponentLutError::InvalidLut);
                };
                Ok(TestWire::Active {
                    values: private
                        .iter()
                        .zip(public.iter().skip(*start))
                        .map(|(left, right)| left + right)
                        .collect(),
                    start: *start,
                })
            }
            fn add(
                &mut self,
                _lhs: &Self::Wire,
                _rhs: &Self::Wire,
                _gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                unreachable!()
            }
            fn mul_private(
                &mut self,
                input: &Self::Wire,
                _: &Self::Rhs,
                _: Option<&Self::HelperSet>,
                _: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                let TestWire::Scalar(value) = input else {
                    return Err(crate::ExponentLutError::InvalidLut);
                };
                Ok(TestWire::Active { values: vec![*value; self.count], start: self.start })
            }
            fn lut(
                &mut self,
                _program: &ExponentLutProgram,
                _lut: LutId,
                _input: &Self::Wire,
                _helpers: Option<&Self::HelperSet>,
                _gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                unreachable!()
            }
        }
        impl TestBackend {
            fn reduce_add(
                &mut self,
                value: &TestWire,
                _range: &FamilyRange,
                _gate: ProgramGateInstance,
            ) -> Result<TestWire, crate::ExponentLutError> {
                let TestWire::Active { values, .. } = value else {
                    return Err(crate::ExponentLutError::InvalidLut);
                };
                Ok(TestWire::Scalar(values.iter().sum()))
            }
        }

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selectors = builder.rhs_family(2).unwrap();
        let values = builder.wire_family(2).unwrap();
        let output =
            builder.select(builder.input_wire(input).unwrap(), selectors, values, 4).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let inputs = BTreeMap::from([(input, TestWire::Scalar(1))]);
        let rhs = BTreeMap::new();
        let rhs_families = BTreeMap::from([(selectors, TestRhs::Members)]);
        let wire_families = BTreeMap::from([(values, TestWire::Members(vec![10, 20, 30, 40, 50]))]);
        let helpers = BTreeMap::new();
        let bindings =
            ProgramBindings::new(&inputs, &rhs, &rhs_families, &wire_families, &helpers, None);
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::bounded(3, 2, 4).unwrap());
        let lowered = lower_program(
            &program,
            &bindings,
            &ranges,
            &mut TestBackend { start: 3, count: 2 },
            TestBackend::reduce_add,
        )
        .unwrap();
        assert!(matches!(lowered[&output], TestWire::Scalar(92)));
    }

    #[test]
    fn family_lut_is_validated_with_the_same_table_rules_as_a_scalar_lut() {
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let members = builder.wire_family(2).unwrap();
        let lut = builder.register_lut(LutTable::unary(2, 4, vec![1, 3]).unwrap()).unwrap();
        let input = WireId::from_index(0);
        let transformed = WireId::from_index(1);
        let output = builder
            .family(
                3,
                FamilyProgram::new(
                    BTreeMap::from([(input, FamilyWireSource::Member(members))]),
                    BTreeMap::new(),
                    vec![Gate::Lut { input, lut_id: lut, output: transformed }],
                    transformed,
                )
                .unwrap(),
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        assert_eq!(program.wire_type(output).unwrap().knowledge, AttributeKnowledge::Public);
        assert!(matches!(
            program.steps(),
            [ProgramStep::Family { body, .. }]
                if matches!(body.gates(), [Gate::Lut { lut_id: gate_lut, .. }] if *gate_lut == lut)
        ));
    }

    #[test]
    fn shared_lowering_dispatches_family_lut_to_the_backend() {
        #[derive(Clone)]
        enum TestWire {
            Members(Vec<i32>),
            Scalar(i32),
        }
        struct TestBackend {
            lut_gate: Option<ProgramGateInstance>,
        }
        impl ProgramLoweringTypes for TestBackend {
            type Wire = TestWire;
            type Rhs = ();
            type HelperSet = i32;
            type Error = crate::ExponentLutError;
        }
        impl ExponentProgramLowering for TestBackend {
            fn mul_public(
                &mut self,
                _private: &Self::Wire,
                _public: &Self::Wire,
                _gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                unreachable!()
            }
            fn add(
                &mut self,
                _lhs: &Self::Wire,
                _rhs: &Self::Wire,
                _gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                unreachable!()
            }
            fn mul_private(
                &mut self,
                _: &Self::Wire,
                _: &Self::Rhs,
                _: Option<&Self::HelperSet>,
                _: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                unreachable!()
            }
            fn lut(
                &mut self,
                _program: &ExponentLutProgram,
                _lut: LutId,
                input: &Self::Wire,
                helper: Option<&Self::HelperSet>,
                gate: ProgramGateInstance,
            ) -> Result<Self::Wire, Self::Error> {
                let helper = helper.ok_or(crate::ExponentLutError::InvalidLut)?;

                let TestWire::Members(values) = input else {
                    return Err(crate::ExponentLutError::InvalidLut);
                };
                self.lut_gate = Some(gate);
                Ok(TestWire::Members(values.iter().map(|value| value * helper).collect()))
            }
        }
        impl TestBackend {
            fn reduce_add(
                &mut self,
                wire: &TestWire,
                _range: &FamilyRange,
                _gate: ProgramGateInstance,
            ) -> Result<TestWire, crate::ExponentLutError> {
                let TestWire::Members(values) = wire else {
                    return Err(crate::ExponentLutError::InvalidLut);
                };
                Ok(TestWire::Scalar(values.iter().sum()))
            }
        }

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let members = builder.wire_family(2).unwrap();
        let lut = builder.register_lut(LutTable::unary(2, 2, vec![0, 1]).unwrap()).unwrap();
        let input = WireId::from_index(0);
        let transformed = WireId::from_index(1);
        let output = builder
            .family(
                3,
                FamilyProgram::new(
                    BTreeMap::from([(input, FamilyWireSource::Member(members))]),
                    BTreeMap::new(),
                    vec![Gate::Lut { input, lut_id: lut, output: transformed }],
                    transformed,
                )
                .unwrap(),
            )
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let inputs = BTreeMap::new();
        let rhs = BTreeMap::new();
        let rhs_families = BTreeMap::new();
        let wire_families = BTreeMap::from([(members, TestWire::Members(vec![2, 3, 5]))]);
        let helpers = BTreeMap::from([(lut, 4)]);
        let bindings =
            ProgramBindings::new(&inputs, &rhs, &rhs_families, &wire_families, &helpers, None);
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), FamilyRange::full(3).unwrap());
        let mut backend = TestBackend { lut_gate: None };
        let lowered =
            lower_program(&program, &bindings, &ranges, &mut backend, TestBackend::reduce_add)
                .unwrap();

        assert!(matches!(lowered[&output], TestWire::Scalar(40)));
        assert_eq!(
            backend.lut_gate,
            Some(ProgramGateInstance::FamilyGate { step: 0, gate: 0, family: ProgramFamilyId(0) })
        );
    }

    #[test]
    fn lut_rejects_values_outside_the_declared_output_domain() {
        assert!(LutTable::unary(2, 2, vec![0, 2]).is_err());
        assert!(LutTable::binary(2, 2, 2, vec![0, 1, 2, 0]).is_err());
        assert!(LutTable::unary(2, 2, vec![1, 2]).is_err());
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
    fn family_map_has_explicit_body_and_no_label_or_public_identity() {
        let mut first = ExponentLutProgramBuilder::new();
        first.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = first
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selectors = first.rhs_family(2).unwrap();
        let values = first.wire_family(2).unwrap();
        let output = first.select(first.input_wire(input).unwrap(), selectors, values, 2).unwrap();
        first.output(output).unwrap();
        let first = first.build().unwrap();
        assert!(
            matches!(first.steps(), [ProgramStep::Family { id, capacity: 2, body, output, .. }]
            if *id == ProgramFamilyId::from_index(0) &&
                *output == WireId::from_index(1) &&
                matches!(body.gates(), [Gate::MulPrivate { .. }, Gate::MulPublic { .. }]))
        );
    }
}
