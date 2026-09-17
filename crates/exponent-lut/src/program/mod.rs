//! Immutable Exponent-LUT programs and their structural execution model.
//!
//! This module deliberately contains no scheme-specific carrier types.  A
//! program describes scalar operations, indexed bodies, and literal LUTs;
//! [`lower_program`](lowering::lower_program) supplies the only execution
//! path and delegates each scalar operation to a scheme evaluator.

mod builder;
mod graph;
mod lowering;

pub use lowering::ExponentLUTProgramLowering;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub(crate) use builder::ExponentLutProgramBuilder;
pub(crate) use graph::GraphParallelLowering;
pub use graph::{GraphFamilyExecution, RhsRef};
pub(crate) use lowering::lower_program;
pub use lowering::{
    Environment, FamilyExecution, LoweringSite, ProgramLowerError, ProgramLoweringContext,
};

/// A deterministic identifier for a program.  The body and declarations are
/// committed by the canonical digest; runtime state is intentionally absent.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ExponentLutProgramId(pub(crate) [u8; 32]);

impl ExponentLutProgramId {
    pub(crate) const fn from_digest(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Native exponent layout used by a scheme when materialising an exponent
/// LUT.  This is pure layout arithmetic; graph-dependent helpers live in
/// `program::graph`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ExpLayout {
    pub stride_log2: u32,
    pub width_log2: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum LayoutError {
    #[error("ring dimension {0} is not a positive power of two")]
    InvalidRingDimension(usize),
    #[error("stride log2 {stride_log2} must be smaller than ring log2 {ring_log2}")]
    StrideOutOfRange { stride_log2: u32, ring_log2: u32 },
    #[error("stride log2 {stride_log2} plus width log2 {width_log2} exceeds ring log2 {ring_log2}")]
    WidthOutOfRange { stride_log2: u32, width_log2: u32, ring_log2: u32 },
    #[error("ring dimension expression is not concrete")]
    NonConcreteRingDimension,
    #[error("ring dimension {0} is too large")]
    RingDimensionOverflow(usize),
    #[error("native automorphism enumeration produced an invalid set")]
    InvalidAutomorphismSet,
    #[error("exponent modulus is not representable for ring dimension {0}")]
    ExponentModulusOverflow(usize),
}

impl ExpLayout {
    pub fn validate(&self, ring_dimension: usize) -> Result<(), LayoutError> {
        if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
            return Err(LayoutError::InvalidRingDimension(ring_dimension));
        }
        let ring_log2 = ring_dimension.trailing_zeros();
        if self.stride_log2 >= ring_log2 {
            return Err(LayoutError::StrideOutOfRange { stride_log2: self.stride_log2, ring_log2 });
        }
        let total =
            self.stride_log2.checked_add(self.width_log2).ok_or(LayoutError::WidthOutOfRange {
                stride_log2: self.stride_log2,
                width_log2: self.width_log2,
                ring_log2,
            })?;
        if total > ring_log2 {
            return Err(LayoutError::WidthOutOfRange {
                stride_log2: self.stride_log2,
                width_log2: self.width_log2,
                ring_log2,
            });
        }
        Ok(())
    }

    pub fn stride(&self) -> usize {
        1usize.checked_shl(self.stride_log2).expect("exponent stride does not fit in usize")
    }

    pub fn width(&self) -> usize {
        1usize.checked_shl(self.width_log2).expect("exponent width does not fit in usize")
    }

    pub fn effective_modulus(&self, ring_dimension: usize) -> usize {
        self.validate(ring_dimension).expect("invalid exponent layout");
        ring_dimension.checked_mul(2).expect("2n does not fit in usize") / self.stride()
    }

    pub fn mono(
        &self,
        ring: &mxx_dsl::Ring,
        exponent: &BigInt,
    ) -> Result<mxx_dsl::Mat, LayoutError> {
        use mxx_ir_core::node::ConstantMatrix;
        let dimension = ring
            .matrix_type((1, 1))
            .ring_dimension
            .evaluate(&ParamEnv::default())
            .map_err(|_| LayoutError::NonConcreteRingDimension)?
            .to_usize()
            .ok_or(LayoutError::NonConcreteRingDimension)?;
        self.validate(dimension)?;
        let doubled =
            dimension.checked_mul(2).ok_or(LayoutError::ExponentModulusOverflow(dimension))?;
        let modulus = BigInt::from(doubled);
        let mut reduced = (exponent * BigInt::from(self.stride())) % &modulus;
        if reduced.sign() == Sign::Minus {
            reduced += &modulus;
        }
        let reduced = reduced.to_usize().ok_or(LayoutError::NonConcreteRingDimension)?;
        let (rotation, negate) =
            if reduced < dimension { (reduced, false) } else { (reduced - dimension, true) };
        let matrix = ring
            .constant((1, 1), ConstantMatrix::Rotation { exponent: IntExpr::constant(rotation) });
        Ok(if negate { -matrix } else { matrix })
    }

    pub fn automorphism_indices(&self, ring_dimension: usize) -> Result<Vec<usize>, LayoutError> {
        self.validate(ring_dimension)?;
        let modulus = ring_dimension
            .checked_mul(2)
            .ok_or(LayoutError::RingDimensionOverflow(ring_dimension))?;
        let mut indices = BTreeSet::new();
        for subset in 0..self.width() {
            let mut product = 1u128;
            for bit in 0..self.width_log2 {
                if subset & (1usize << bit) == 0 {
                    continue;
                }
                let shift = self
                    .stride_log2
                    .checked_add(bit)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(LayoutError::RingDimensionOverflow(ring_dimension))?;
                let denominator = 1usize
                    .checked_shl(shift)
                    .ok_or(LayoutError::RingDimensionOverflow(ring_dimension))?;
                product = product * (modulus / denominator + 1) as u128 % modulus as u128;
            }
            indices.insert(product as usize);
        }
        if indices.len() != self.width() ||
            indices.iter().any(|i| *i == 0 || i % 2 == 0) ||
            !indices.contains(&1)
        {
            return Err(LayoutError::InvalidAutomorphismSet);
        }
        Ok(indices.into_iter().collect())
    }
}

/// Identifier namespaces used by a Program.  These remain crate-private so
/// callers cannot construct unchecked references to a body instruction.
macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub(crate) u32);
    };
}
id_type!(WireId);
id_type!(RhsId);
id_type!(LutId);

/// A scalar or an indexed member of a family in a validated Program body.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Operand<I> {
    Scalar(I),
    Member { family: I, index: IntExpr },
}

/// Reused shape representation for declarations and runtime values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputBinding<S, F = S> {
    Scalar(S),
    Family(F),
}

pub type InputShape = InputBinding<(), IntExpr>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AttributeKnowledge {
    PrivatePayload,
    PublicPayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramInputDeclaration {
    shape: InputShape,
    knowledge: AttributeKnowledge,
}

impl ProgramInputDeclaration {
    pub fn shape(&self) -> &InputShape {
        &self.shape
    }
    pub fn knowledge(&self) -> AttributeKnowledge {
        self.knowledge
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct FamilyRange {
    start: IntExpr,
    count: IntExpr,
    capacity: IntExpr,
}

/// A validated lexical family scope and the concrete (possibly symbolic)
/// member indices selected in that scope.
///
/// The index order is lexical: the outermost family axis comes first and the
/// innermost (including a branch axis) comes last.  Keeping the ranges with
/// the indices is important because an inner index alone is not a qualified
/// family member when two outer scopes share it.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct FamilyScope {
    ranges: Vec<FamilyRange>,
    indices: Vec<IntExpr>,
}

impl FamilyScope {
    pub fn new(
        ranges: impl Into<Vec<FamilyRange>>,
        indices: impl Into<Vec<IntExpr>>,
    ) -> Result<Self, ProgramValidationError> {
        let ranges = ranges.into();
        let indices = indices.into();
        if ranges.len() != indices.len() {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        for range in &ranges {
            range.validate()?;
        }
        let scope = Self { ranges, indices };
        scope.validate_indices()?;
        Ok(scope)
    }

    pub fn ranges(&self) -> &[FamilyRange] {
        &self.ranges
    }

    pub fn indices(&self) -> &[IntExpr] {
        &self.indices
    }

    /// Return the checked row-major member index.
    ///
    /// The last lexical axis is the fastest-changing axis.  Symbolic
    /// expressions are retained, while concrete expressions are checked
    /// against their range before the arithmetic is constructed.
    pub fn flattened_index(&self) -> Result<IntExpr, ProgramValidationError> {
        self.validate_indices()?;
        let mut flattened = IntExpr::constant(0);
        for (range, index) in self.ranges.iter().zip(&self.indices) {
            let local = IntExpr::Sub(Box::new(index.clone()), Box::new(range.start().clone()))
                .canonicalize();
            flattened = IntExpr::Add(
                Box::new(IntExpr::Mul(Box::new(flattened), Box::new(range.count().clone()))),
                Box::new(local),
            )
            .canonicalize();
        }
        Ok(flattened)
    }

    fn validate_indices(&self) -> Result<(), ProgramValidationError> {
        let env = ParamEnv::default();
        for (range, index) in self.ranges.iter().zip(&self.indices) {
            let (Ok(index), Ok(start), Ok(count)) =
                (index.evaluate(&env), range.start().evaluate(&env), range.count().evaluate(&env))
            else {
                continue;
            };
            if index < start || index >= start + count {
                return Err(ProgramValidationError::MemberOutOfBounds);
            }
        }
        Ok(())
    }
}

impl FamilyRange {
    pub fn full(count: IntExpr) -> Result<Self, ProgramValidationError> {
        Self::bounded(IntExpr::constant(0), count.clone(), count)
    }

    pub fn bounded(
        start: IntExpr,
        count: IntExpr,
        capacity: IntExpr,
    ) -> Result<Self, ProgramValidationError> {
        let env = ParamEnv::default();
        if let Ok(start_value) = start.evaluate(&env) &&
            start_value < 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let Ok(count_value) = count.evaluate(&env) &&
            count_value <= 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let Ok(capacity_value) = capacity.evaluate(&env) &&
            capacity_value <= 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let (Ok(count_value), Ok(capacity_value)) =
            (count.evaluate(&env), capacity.evaluate(&env)) &&
            count_value > capacity_value
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        Ok(Self { start, count, capacity })
    }

    pub fn start(&self) -> &IntExpr {
        &self.start
    }
    pub fn count(&self) -> &IntExpr {
        &self.count
    }
    pub fn capacity(&self) -> &IntExpr {
        &self.capacity
    }

    /// Validate the lexical range independently of a builder.  This is used
    /// when a lowering site is turned into an artifact scope.
    pub fn validate(&self) -> Result<(), ProgramValidationError> {
        let env = ParamEnv::default();
        if let Ok(start_value) = self.start.evaluate(&env) &&
            start_value < 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let Ok(count_value) = self.count.evaluate(&env) &&
            count_value <= 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let Ok(capacity_value) = self.capacity.evaluate(&env) &&
            capacity_value <= 0.into()
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        if let (Ok(count_value), Ok(capacity_value)) =
            (self.count.evaluate(&env), self.capacity.evaluate(&env)) &&
            count_value > capacity_value
        {
            return Err(ProgramValidationError::InvalidFamilyRange);
        }
        Ok(())
    }
}

/// Explicit output semantics for a LUT.  Every table must carry this value;
/// no program layout is consulted as a fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LutOutputRepresentation {
    Exponent(ExpLayout),
    Coefficient,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LutTable {
    input_width: usize,
    output_width: usize,
    values: Vec<usize>,
    output_representation: LutOutputRepresentation,
}

impl LutTable {
    pub fn unary(
        input_width: usize,
        output_width: usize,
        values: Vec<usize>,
        output_representation: LutOutputRepresentation,
    ) -> Result<Self, ProgramValidationError> {
        Self::new(input_width, output_width, values, output_representation)
    }

    fn new(
        input_width: usize,
        output_width: usize,
        values: Vec<usize>,
        output_representation: LutOutputRepresentation,
    ) -> Result<Self, ProgramValidationError> {
        let table = Self { input_width, output_width, values, output_representation };
        validate_table(&table)?;
        Ok(table)
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }
    pub fn output_width(&self) -> usize {
        self.output_width
    }
    pub fn values(&self) -> &[usize] {
        &self.values
    }
    pub fn output_representation(&self) -> LutOutputRepresentation {
        self.output_representation
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Step {
    Add { lhs: Operand<WireId>, rhs: Operand<WireId>, output: WireId },
    MulPublic { private: Operand<WireId>, public: Operand<WireId>, output: WireId },
    MulPrivate { input: Operand<WireId>, rhs: Operand<RhsId>, output: WireId },
    Lut { input: Operand<WireId>, table: LutId, output: WireId },
    Parallel { range: FamilyRange, body: Box<Body>, output: WireId },
    Sum { family: WireId, output: WireId },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct Body {
    steps: Vec<Step>,
    outputs: Vec<Operand<WireId>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ProgramValidationError {
    #[error("undefined wire {0:?}")]
    UndefinedWire(WireId),
    #[error("undefined RHS {0:?}")]
    UndefinedRhs(RhsId),
    #[error("duplicate wire {0:?}")]
    DuplicateWire(WireId),
    #[error("duplicate RHS {0:?}")]
    DuplicateRhs(RhsId),
    #[error("invalid LUT id {0:?}")]
    InvalidLutId(LutId),
    #[error("expected scalar wire {0:?}")]
    ExpectedScalarWire(WireId),
    #[error("expected family wire {0:?}")]
    ExpectedFamilyWire(WireId),
    #[error("expected scalar RHS {0:?}")]
    ExpectedScalarRhs(RhsId),
    #[error("expected family RHS {0:?}")]
    ExpectedFamilyRhs(RhsId),
    #[error("invalid output count")]
    InvalidOutputCount,
    #[error("invalid loop index {0}")]
    InvalidLoopIndex(u32),
    #[error("input declaration in a nested body")]
    InputInNestedBody,
    #[error("invalid family range")]
    InvalidFamilyRange,
    #[error("family member is out of bounds")]
    MemberOutOfBounds,
    #[error("invalid LUT")]
    InvalidLut,
    #[error("program identifier allocation exhausted")]
    IdExhausted,
    #[error("duplicate binding")]
    DuplicateBinding,
    #[error("missing binding")]
    MissingBinding,
    #[error("invalid structure: {0}")]
    InvalidStructure(String),
}

/// Input-only projection used by the encryption stage.  It intentionally
/// excludes body, RHS declarations, and the program identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncryptionInputs {
    inputs: BTreeMap<WireId, ProgramInputDeclaration>,
    luts: Vec<LutTable>,
}

impl EncryptionInputs {
    pub fn inputs(&self) -> &BTreeMap<WireId, ProgramInputDeclaration> {
        &self.inputs
    }
    pub fn luts(&self) -> &[LutTable] {
        &self.luts
    }
}

/// Opaque immutable computation description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExponentLutProgram {
    id: ExponentLutProgramId,
    inputs: BTreeMap<WireId, ProgramInputDeclaration>,
    rhs_inputs: BTreeMap<RhsId, InputShape>,
    luts: Vec<LutTable>,
    body: Body,
    metadata_cache: OnceLock<BTreeMap<WireId, WireMetadata>>,
}

impl ExponentLutProgram {
    pub fn id(&self) -> ExponentLutProgramId {
        self.id
    }
    pub(crate) fn inputs(&self) -> &BTreeMap<WireId, ProgramInputDeclaration> {
        &self.inputs
    }
    pub(crate) fn rhs_inputs(&self) -> &BTreeMap<RhsId, InputShape> {
        &self.rhs_inputs
    }
    #[cfg(test)]
    pub(crate) fn luts(&self) -> &[LutTable] {
        &self.luts
    }
    /// Returns the declared scalar output operands in program order.
    ///
    /// The returned operands are part of the validated immutable Program and
    /// can be used to bind a refresh declaration to a declared exponent
    /// output. No new wire or operand can be constructed through this view.
    pub fn outputs(&self) -> &[Operand<WireId>] {
        &self.body.outputs
    }
    pub fn encryption_inputs(&self) -> EncryptionInputs {
        EncryptionInputs { inputs: self.inputs.clone(), luts: self.luts.clone() }
    }

    /// Return the payload knowledge of a validated wire, including wires
    /// local to nested parallel bodies and their family/sum aliases.
    pub fn wire_knowledge(
        &self,
        wire: WireId,
    ) -> Result<AttributeKnowledge, ProgramValidationError> {
        let metadata = self.metadata()?;
        metadata
            .get(&wire)
            .map(|metadata| metadata.knowledge)
            .ok_or(ProgramValidationError::UndefinedWire(wire))
    }

    /// Return whether a validated wire is the raw output of a coefficient LUT
    /// lineage.  Input declarations intentionally have no inferred layout.
    pub fn wire_is_coefficient_output(&self, wire: WireId) -> Result<bool, ProgramValidationError> {
        let metadata = self.metadata()?;
        metadata
            .get(&wire)
            .map(|metadata| metadata.coefficient)
            .ok_or(ProgramValidationError::UndefinedWire(wire))
    }

    fn metadata(&self) -> Result<BTreeMap<WireId, WireMetadata>, ProgramValidationError> {
        validate_program_structure(self)?;
        if let Some(metadata) = self.metadata_cache.get() {
            return Ok(metadata.clone());
        }
        let metadata = derive_wire_metadata(self)?;
        match self.metadata_cache.set(metadata) {
            Ok(()) => Ok(self.metadata_cache.get().expect("metadata was just initialized").clone()),
            Err(metadata) => Ok(self.metadata_cache.get().cloned().unwrap_or(metadata)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WireMetadata {
    knowledge: AttributeKnowledge,
    coefficient: bool,
}

pub(crate) fn validate_table(table: &LutTable) -> Result<(), ProgramValidationError> {
    if table.input_width == 0 || table.output_width == 0 {
        return Err(ProgramValidationError::InvalidLut);
    }
    if table.values.len() != table.input_width ||
        table.values.iter().any(|value| *value >= table.output_width)
    {
        return Err(ProgramValidationError::InvalidLut);
    }
    Ok(())
}

pub(crate) fn canonical_program_id(program: &ExponentLutProgram) -> ExponentLutProgramId {
    let bytes =
        serde_json::to_vec(&(&program.inputs, &program.rhs_inputs, &program.luts, &program.body))
            .expect("program canonical encoding");
    let mut digest = Sha256::new();
    digest.update(b"mxx-exponent-lut/program/v1");
    digest.update(bytes);
    ExponentLutProgramId::from_digest(digest.finalize().into())
}

fn validate_program_structure(program: &ExponentLutProgram) -> Result<(), ProgramValidationError> {
    let mut visible = program
        .inputs
        .iter()
        .map(|(id, declaration)| (*id, VisibleKind::from_shape(&declaration.shape)))
        .collect::<BTreeMap<_, _>>();
    let mut all_wires = program.inputs.keys().copied().collect::<BTreeSet<_>>();
    let mut rhs = program
        .rhs_inputs
        .iter()
        .map(|(id, shape)| (*id, VisibleKind::from_shape(shape)))
        .collect::<BTreeMap<_, _>>();
    // Validate nested operations before validating the root output contract.
    // This makes a malformed member used inside a nested body observable even
    // when the fixture's eventual root output is itself not a scalar.
    validate_body(
        &program.body,
        &program.luts,
        &mut visible,
        &mut all_wires,
        &mut rhs,
        0,
        &BTreeMap::new(),
        true,
        false,
    )?;
    if program.body.outputs.is_empty() ||
        program.body.outputs.iter().any(|output| {
            !matches!(output, Operand::Scalar(id) if matches!(visible.get(id), Some(VisibleKind::Scalar)))
        })
    {
        return Err(ProgramValidationError::InvalidOutputCount);
    }
    Ok(())
}

pub(crate) fn validate_program(program: &ExponentLutProgram) -> Result<(), ProgramValidationError> {
    validate_program_structure(program)?;
    derive_wire_metadata(program).map(|_| ())
}

fn derive_wire_metadata(
    program: &ExponentLutProgram,
) -> Result<BTreeMap<WireId, WireMetadata>, ProgramValidationError> {
    let mut metadata = program
        .inputs
        .iter()
        .map(|(wire, declaration)| {
            (*wire, WireMetadata { knowledge: declaration.knowledge, coefficient: false })
        })
        .collect::<BTreeMap<_, _>>();
    derive_body_metadata(&program.body, &program.luts, &mut metadata)?;
    Ok(metadata)
}

fn derive_body_metadata(
    body: &Body,
    luts: &[LutTable],
    metadata: &mut BTreeMap<WireId, WireMetadata>,
) -> Result<(), ProgramValidationError> {
    for step in &body.steps {
        let output = match step {
            Step::Add { lhs, rhs, .. } => {
                let lhs = metadata_for_operand(lhs, metadata)?;
                let rhs = metadata_for_operand(rhs, metadata)?;
                if lhs != rhs {
                    return Err(ProgramValidationError::InvalidStructure(
                        "Add operands must have matching knowledge and representation".into(),
                    ));
                }
                lhs
            }
            Step::MulPublic { private, public, .. } => {
                let lhs = metadata_for_operand(private, metadata)?;
                let rhs = metadata_for_operand(public, metadata)?;
                if lhs.knowledge != AttributeKnowledge::PrivatePayload ||
                    rhs.knowledge != AttributeKnowledge::PublicPayload ||
                    lhs.coefficient ||
                    rhs.coefficient
                {
                    return Err(ProgramValidationError::InvalidStructure(
                        "MulPublic requires a private exponent LHS and public exponent RHS".into(),
                    ));
                }
                lhs
            }
            Step::MulPrivate { input, .. } => {
                let input = metadata_for_operand(input, metadata)?;
                if input.coefficient {
                    return Err(ProgramValidationError::InvalidStructure(
                        "MulPrivate cannot consume a coefficient output".into(),
                    ));
                }
                WireMetadata { knowledge: input.knowledge, coefficient: false }
            }
            Step::Lut { input, table, .. } => {
                let input = metadata_for_operand(input, metadata)?;
                if input.coefficient {
                    return Err(ProgramValidationError::InvalidStructure(
                        "LUT input cannot be a coefficient output".into(),
                    ));
                }
                let table = luts
                    .get(table.0 as usize)
                    .ok_or(ProgramValidationError::InvalidLutId(*table))?;
                WireMetadata {
                    knowledge: input.knowledge,
                    coefficient: matches!(
                        table.output_representation,
                        LutOutputRepresentation::Coefficient
                    ),
                }
            }
            Step::Parallel { body, .. } => {
                derive_body_metadata(body, luts, metadata)?;
                body.outputs
                    .first()
                    .ok_or(ProgramValidationError::InvalidOutputCount)
                    .and_then(|operand| metadata_for_operand(operand, metadata))?
            }
            Step::Sum { family, .. } => metadata
                .get(family)
                .copied()
                .ok_or(ProgramValidationError::UndefinedWire(*family))?,
        };
        let wire = match step {
            Step::Add { output, .. } |
            Step::MulPublic { output, .. } |
            Step::MulPrivate { output, .. } |
            Step::Lut { output, .. } |
            Step::Parallel { output, .. } |
            Step::Sum { output, .. } => *output,
        };
        metadata.insert(wire, output);
    }
    Ok(())
}

fn metadata_for_operand(
    operand: &Operand<WireId>,
    metadata: &BTreeMap<WireId, WireMetadata>,
) -> Result<WireMetadata, ProgramValidationError> {
    let wire = match operand {
        Operand::Scalar(wire) | Operand::Member { family: wire, .. } => wire,
    };
    metadata.get(wire).copied().ok_or(ProgramValidationError::UndefinedWire(*wire))
}

#[derive(Clone, Debug)]
enum VisibleKind {
    Scalar,
    Family(IntExpr),
}

impl VisibleKind {
    fn from_shape(shape: &InputShape) -> Self {
        match shape {
            InputBinding::Scalar(()) => Self::Scalar,
            InputBinding::Family(count) => Self::Family(count.clone()),
        }
    }
}

fn validate_body(
    body: &Body,
    luts: &[LutTable],
    visible: &mut BTreeMap<WireId, VisibleKind>,
    all_wires: &mut BTreeSet<WireId>,
    rhs: &mut BTreeMap<RhsId, VisibleKind>,
    depth: u32,
    loop_bounds: &BTreeMap<u32, IntInterval>,
    root: bool,
    check_outputs: bool,
) -> Result<(), ProgramValidationError> {
    let mut defined = BTreeSet::new();
    for step in &body.steps {
        let (output, kind) = match step {
            Step::Add { lhs, rhs: right, output } => {
                check_wire_operand(lhs, visible, false, depth, loop_bounds)?;
                check_wire_operand(right, visible, false, depth, loop_bounds)?;
                (*output, VisibleKind::Scalar)
            }
            Step::MulPublic { private, public, output } => {
                check_wire_operand(private, visible, false, depth, loop_bounds)?;
                check_wire_operand(public, visible, false, depth, loop_bounds)?;
                (*output, VisibleKind::Scalar)
            }
            Step::MulPrivate { input, rhs: right, output } => {
                check_wire_operand(input, visible, false, depth, loop_bounds)?;
                check_rhs_operand(right, rhs, false, depth, loop_bounds)?;
                (*output, VisibleKind::Scalar)
            }
            Step::Lut { input, table, output } => {
                check_wire_operand(input, visible, false, depth, loop_bounds)?;
                if (*table).0 as usize >= luts.len() {
                    return Err(ProgramValidationError::InvalidLutId(*table));
                }
                (*output, VisibleKind::Scalar)
            }
            Step::Parallel { range, body, output } => {
                validate_range_with_bounds(range, loop_bounds)?;
                let mut child_visible = visible.clone();
                let mut child_loop_bounds = loop_bounds.clone();
                let start = interval_of(range.start(), loop_bounds);
                let count = interval_of(range.count(), loop_bounds);
                let member_count = interval_sub(count, IntInterval::exact(1.into()));
                child_loop_bounds.insert(depth, interval_add(start, member_count));
                validate_body(
                    body,
                    luts,
                    &mut child_visible,
                    all_wires,
                    rhs,
                    depth + 1,
                    &child_loop_bounds,
                    false,
                    true,
                )?;
                if body.outputs.len() != 1 || matches!(body.outputs[0], Operand::Member { .. }) {
                    return Err(ProgramValidationError::InvalidStructure(
                        "parallel body must return one scalar".into(),
                    ));
                }
                (*output, VisibleKind::Family(range.count.clone()))
            }
            Step::Sum { family, output } => {
                check_wire_operand(&Operand::Scalar(*family), visible, true, depth, loop_bounds)?;
                (*output, VisibleKind::Scalar)
            }
        };
        if !defined.insert(output) || visible.contains_key(&output) || !all_wires.insert(output) {
            return Err(ProgramValidationError::DuplicateWire(output));
        }
        visible.insert(output, kind);
    }
    if check_outputs {
        for output in &body.outputs {
            check_wire_operand(output, visible, false, depth, loop_bounds)?;
        }
    }
    if !root {
        // Child-local definitions are intentionally scoped; callers receive
        // only the body output and never the child's visible map.
        for output in &body.outputs {
            if let Operand::Scalar(id) = output {
                visible.remove(id);
            }
        }
    }
    Ok(())
}

fn validate_range(range: &FamilyRange) -> Result<(), ProgramValidationError> {
    let env = ParamEnv::default();
    if let Ok(start) = range.start.evaluate(&env) &&
        start < 0.into()
    {
        return Err(ProgramValidationError::InvalidFamilyRange);
    }
    if let Ok(count) = range.count.evaluate(&env) &&
        count <= 0.into()
    {
        return Err(ProgramValidationError::InvalidFamilyRange);
    }
    if let Ok(capacity) = range.capacity.evaluate(&env) &&
        capacity <= 0.into()
    {
        return Err(ProgramValidationError::InvalidFamilyRange);
    }
    if let (Ok(count), Ok(capacity)) = (range.count.evaluate(&env), range.capacity.evaluate(&env)) &&
        count > capacity
    {
        return Err(ProgramValidationError::InvalidFamilyRange);
    }
    Ok(())
}

fn validate_range_with_bounds(
    range: &FamilyRange,
    loop_bounds: &BTreeMap<u32, IntInterval>,
) -> Result<(), ProgramValidationError> {
    validate_range(range)?;
    let start = interval_of(range.start(), loop_bounds);
    let count = interval_of(range.count(), loop_bounds);
    let capacity = interval_of(range.capacity(), loop_bounds);
    if start.minimum.as_ref().is_some_and(|value| value < &0.into()) ||
        count.maximum.as_ref().is_some_and(|value| value <= &0.into()) ||
        capacity.maximum.as_ref().is_some_and(|value| value <= &0.into()) ||
        count
            .minimum
            .as_ref()
            .zip(capacity.maximum.as_ref())
            .is_some_and(|(count, capacity)| count > capacity)
    {
        return Err(ProgramValidationError::InvalidFamilyRange);
    }
    Ok(())
}

fn check_wire_operand(
    operand: &Operand<WireId>,
    visible: &BTreeMap<WireId, VisibleKind>,
    family: bool,
    depth: u32,
    loop_bounds: &BTreeMap<u32, IntInterval>,
) -> Result<(), ProgramValidationError> {
    match operand {
        Operand::Scalar(id) => match visible.get(id) {
            Some(VisibleKind::Family(_)) if family => Ok(()),
            Some(VisibleKind::Scalar) if family => {
                Err(ProgramValidationError::ExpectedFamilyWire(*id))
            }
            Some(VisibleKind::Family(_)) => Err(ProgramValidationError::ExpectedScalarWire(*id)),
            Some(VisibleKind::Scalar) => Ok(()),
            None => Err(ProgramValidationError::UndefinedWire(*id)),
        },
        Operand::Member { family: id, index } => {
            if family {
                return Err(ProgramValidationError::ExpectedFamilyWire(*id));
            }
            let Some(VisibleKind::Family(count)) = visible.get(id) else {
                return Err(ProgramValidationError::ExpectedFamilyWire(*id));
            };
            if let Some(slot) = invalid_loop_slot(index, depth) {
                return Err(ProgramValidationError::InvalidLoopIndex(slot));
            }
            check_member_bounds(index, count, loop_bounds)?;
            Ok(())
        }
    }
}

fn check_rhs_operand(
    operand: &Operand<RhsId>,
    rhs: &BTreeMap<RhsId, VisibleKind>,
    family: bool,
    depth: u32,
    loop_bounds: &BTreeMap<u32, IntInterval>,
) -> Result<(), ProgramValidationError> {
    match operand {
        Operand::Scalar(id) => match rhs.get(id) {
            Some(VisibleKind::Family(_)) if family => Ok(()),
            Some(VisibleKind::Scalar) if family => {
                Err(ProgramValidationError::ExpectedFamilyRhs(*id))
            }
            Some(VisibleKind::Family(_)) => Err(ProgramValidationError::ExpectedScalarRhs(*id)),
            Some(VisibleKind::Scalar) => Ok(()),
            None => Err(ProgramValidationError::UndefinedRhs(*id)),
        },
        Operand::Member { family: id, index } => {
            if family {
                return Err(ProgramValidationError::ExpectedFamilyRhs(*id));
            }
            let Some(VisibleKind::Family(count)) = rhs.get(id) else {
                return Err(ProgramValidationError::ExpectedFamilyRhs(*id));
            };
            if let Some(slot) = invalid_loop_slot(index, depth) {
                return Err(ProgramValidationError::InvalidLoopIndex(slot));
            }
            check_member_bounds(index, count, loop_bounds)?;
            Ok(())
        }
    }
}

fn check_member_bounds(
    index: &IntExpr,
    count: &IntExpr,
    loop_bounds: &BTreeMap<u32, IntInterval>,
) -> Result<(), ProgramValidationError> {
    let index = interval_of(index, loop_bounds);
    let count = interval_of(count, loop_bounds);
    if index.minimum.as_ref().is_some_and(|value| value < &0.into()) {
        return Err(ProgramValidationError::MemberOutOfBounds);
    }
    if let (Some(index_max), Some(count_min)) = (&index.maximum, &count.minimum) &&
        index_max >= count_min
    {
        return Err(ProgramValidationError::MemberOutOfBounds);
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct IntInterval {
    minimum: Option<BigInt>,
    maximum: Option<BigInt>,
}

impl IntInterval {
    fn exact(value: BigInt) -> Self {
        Self { minimum: Some(value.clone()), maximum: Some(value) }
    }

    fn unknown() -> Self {
        Self::default()
    }
}

fn interval_of(expression: &IntExpr, loop_bounds: &BTreeMap<u32, IntInterval>) -> IntInterval {
    if let Ok(value) = expression.evaluate(&ParamEnv::default()) {
        return IntInterval::exact(value);
    }
    match expression {
        IntExpr::Const(value) => IntInterval::exact(value.clone()),
        IntExpr::Var(_) => IntInterval::unknown(),
        IntExpr::LoopIndex(slot) => loop_bounds.get(slot).cloned().unwrap_or_default(),
        IntExpr::Add(lhs, rhs) => {
            interval_add(interval_of(lhs, loop_bounds), interval_of(rhs, loop_bounds))
        }
        IntExpr::Sub(lhs, rhs) => {
            interval_sub(interval_of(lhs, loop_bounds), interval_of(rhs, loop_bounds))
        }
        IntExpr::Mul(lhs, rhs) => {
            interval_mul(interval_of(lhs, loop_bounds), interval_of(rhs, loop_bounds))
        }
        IntExpr::Div(lhs, rhs) | IntExpr::FloorDiv(lhs, rhs) | IntExpr::RoundDiv(lhs, rhs) => {
            interval_div(interval_of(lhs, loop_bounds), interval_of(rhs, loop_bounds))
        }
        IntExpr::Rem(_, rhs) => {
            let denominator = interval_of(rhs, loop_bounds);
            match (denominator.minimum, denominator.maximum) {
                (Some(minimum), Some(maximum)) if minimum == maximum && !minimum.is_zero() => {
                    let bound = maximum.abs() - 1;
                    if maximum.is_positive() {
                        IntInterval { minimum: Some(0.into()), maximum: Some(bound) }
                    } else {
                        IntInterval { minimum: Some(-bound), maximum: Some(0.into()) }
                    }
                }
                _ => IntInterval::unknown(),
            }
        }
        IntExpr::Log2Ceil(value) => {
            let interval = interval_of(value, loop_bounds);
            match (interval.minimum, interval.maximum) {
                (Some(minimum), Some(maximum)) if minimum >= 1.into() && maximum >= minimum => {
                    let low = log2_ceil(&minimum);
                    let high = log2_ceil(&maximum);
                    IntInterval { minimum: Some(low), maximum: Some(high) }
                }
                _ => IntInterval::unknown(),
            }
        }
        IntExpr::Select { branches, .. } => branches
            .iter()
            .map(|branch| interval_of(branch, loop_bounds))
            .reduce(interval_union)
            .unwrap_or_default(),
    }
}

fn interval_add(lhs: IntInterval, rhs: IntInterval) -> IntInterval {
    IntInterval {
        minimum: lhs.minimum.zip(rhs.minimum).map(|(lhs, rhs)| lhs + rhs),
        maximum: lhs.maximum.zip(rhs.maximum).map(|(lhs, rhs)| lhs + rhs),
    }
}

fn interval_sub(lhs: IntInterval, rhs: IntInterval) -> IntInterval {
    IntInterval {
        minimum: lhs.minimum.zip(rhs.maximum).map(|(lhs, rhs)| lhs - rhs),
        maximum: lhs.maximum.zip(rhs.minimum).map(|(lhs, rhs)| lhs - rhs),
    }
}

fn interval_mul(lhs: IntInterval, rhs: IntInterval) -> IntInterval {
    let Some(lhs_minimum) = lhs.minimum else { return IntInterval::unknown() };
    let Some(lhs_maximum) = lhs.maximum else { return IntInterval::unknown() };
    let Some(rhs_minimum) = rhs.minimum else { return IntInterval::unknown() };
    let Some(rhs_maximum) = rhs.maximum else { return IntInterval::unknown() };
    let products = [
        &lhs_minimum * &rhs_minimum,
        &lhs_minimum * &rhs_maximum,
        &lhs_maximum * &rhs_minimum,
        &lhs_maximum * &rhs_maximum,
    ];
    IntInterval { minimum: products.iter().min().cloned(), maximum: products.iter().max().cloned() }
}

fn interval_div(lhs: IntInterval, rhs: IntInterval) -> IntInterval {
    let Some(lhs_minimum) = lhs.minimum else { return IntInterval::unknown() };
    let Some(lhs_maximum) = lhs.maximum else { return IntInterval::unknown() };
    let Some(rhs_minimum) = rhs.minimum else { return IntInterval::unknown() };
    let Some(rhs_maximum) = rhs.maximum else { return IntInterval::unknown() };
    if rhs_minimum <= 0.into() && rhs_maximum >= 0.into() {
        return IntInterval::unknown();
    }
    let quotients = [
        conservative_quotient(&lhs_minimum, &rhs_minimum),
        conservative_quotient(&lhs_minimum, &rhs_maximum),
        conservative_quotient(&lhs_maximum, &rhs_minimum),
        conservative_quotient(&lhs_maximum, &rhs_maximum),
    ];
    IntInterval {
        minimum: quotients.iter().min().cloned(),
        maximum: quotients.iter().max().cloned(),
    }
}

fn conservative_quotient(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if !remainder.is_zero() && quotient.is_negative() { quotient - 1 } else { quotient }
}

fn interval_union(lhs: IntInterval, rhs: IntInterval) -> IntInterval {
    IntInterval {
        minimum: match (lhs.minimum, rhs.minimum) {
            (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
            _ => None,
        },
        maximum: match (lhs.maximum, rhs.maximum) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            _ => None,
        },
    }
}

fn log2_ceil(value: &BigInt) -> BigInt {
    let value = value.to_biguint().expect("positive log2-ceil input");
    let floor = value.bits() - 1;
    let power = num_bigint::BigUint::from(1u8) << floor as usize;
    BigInt::from(if value == power { floor } else { floor + 1 })
}

fn invalid_loop_slot(expression: &IntExpr, depth: u32) -> Option<u32> {
    match expression {
        IntExpr::LoopIndex(slot) if *slot >= depth => Some(*slot),
        IntExpr::Add(lhs, rhs) |
        IntExpr::Sub(lhs, rhs) |
        IntExpr::Mul(lhs, rhs) |
        IntExpr::Div(lhs, rhs) |
        IntExpr::FloorDiv(lhs, rhs) |
        IntExpr::Rem(lhs, rhs) |
        IntExpr::RoundDiv(lhs, rhs) => {
            invalid_loop_slot(lhs, depth).or_else(|| invalid_loop_slot(rhs, depth))
        }
        IntExpr::Log2Ceil(value) => invalid_loop_slot(value, depth),
        IntExpr::Select { selector, branches } => invalid_loop_slot(selector, depth)
            .or_else(|| branches.iter().find_map(|b| invalid_loop_slot(b, depth))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_native_indices_are_deterministic() {
        let layout = ExpLayout { stride_log2: 0, width_log2: 4 };
        assert_eq!(
            layout.automorphism_indices(64).unwrap(),
            (0..16).map(|j| 1 + j * 8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn lut_representation_is_mandatory() {
        let table =
            LutTable::unary(2, 2, vec![0, 1], LutOutputRepresentation::Coefficient).unwrap();
        assert_eq!(table.output_representation(), LutOutputRepresentation::Coefficient);
    }

    #[test]
    fn public_and_private_input_knowledge_is_preserved_through_public_multiplication() {
        let mut builder = ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let product =
            builder.mul_public(Operand::Scalar(private), Operand::Scalar(public)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(product)]).unwrap();

        assert_eq!(program.inputs[&private].knowledge(), AttributeKnowledge::PrivatePayload);
        assert_eq!(program.inputs[&public].knowledge(), AttributeKnowledge::PublicPayload);
        assert!(matches!(
            program.body.steps.as_slice(),
            [Step::MulPublic { private: Operand::Scalar(lhs), public: Operand::Scalar(rhs), .. }]
                if *lhs == private && *rhs == public
        ));
    }

    #[test]
    fn public_multiplication_rejects_public_first_ordering() {
        let mut builder = ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let output = builder.mul_public(Operand::Scalar(public), Operand::Scalar(private)).unwrap();

        assert!(matches!(
            builder.finish(vec![Operand::Scalar(output)]),
            Err(ProgramValidationError::InvalidStructure(message))
                if message.contains("MulPublic requires a private exponent LHS")
        ));
    }

    #[test]
    fn public_multiplication_requires_exact_operand_knowledge() {
        fn rejects(lhs_knowledge: AttributeKnowledge, rhs_knowledge: AttributeKnowledge) -> bool {
            let mut builder = ExponentLutProgramBuilder::new();
            let lhs = builder.input(lhs_knowledge).unwrap();
            let rhs = builder.input(rhs_knowledge).unwrap();
            let output = builder.mul_public(Operand::Scalar(lhs), Operand::Scalar(rhs)).unwrap();
            builder.finish(vec![Operand::Scalar(output)]).is_err()
        }

        assert!(rejects(AttributeKnowledge::PublicPayload, AttributeKnowledge::PublicPayload));
        assert!(rejects(AttributeKnowledge::PrivatePayload, AttributeKnowledge::PrivatePayload));
        assert!(rejects(AttributeKnowledge::PublicPayload, AttributeKnowledge::PrivatePayload));
        assert!(!rejects(AttributeKnowledge::PrivatePayload, AttributeKnowledge::PublicPayload));
    }

    #[test]
    fn public_multiplication_rejects_invalid_family_knowledge_in_nested_scopes() {
        let mut builder = ExponentLutProgramBuilder::new();
        let lhs =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PublicPayload).unwrap();
        let rhs =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PublicPayload).unwrap();
        let products = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, index| {
                builder
                    .mul_public(
                        Operand::Member { family: lhs, index: index.clone() },
                        Operand::Member { family: rhs, index },
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(products).unwrap();

        assert!(builder.finish(vec![Operand::Scalar(output)]).is_err());
    }

    #[test]
    fn metadata_queries_follow_nested_parallel_and_sum_aliases() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let family = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, _| {
                builder.add(Operand::Scalar(input), Operand::Scalar(input)).map(Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(family).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        assert_eq!(program.wire_knowledge(input).unwrap(), AttributeKnowledge::PrivatePayload);
        assert_eq!(program.wire_knowledge(family).unwrap(), AttributeKnowledge::PrivatePayload);
        assert_eq!(program.wire_knowledge(output).unwrap(), AttributeKnowledge::PrivatePayload);
        assert!(!program.wire_is_coefficient_output(family).unwrap());
        assert!(!program.wire_is_coefficient_output(output).unwrap());
    }

    #[test]
    fn metadata_queries_keep_explicit_coefficient_lut_lineage() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = builder
            .register_lut(
                LutTable::unary(2, 2, vec![0, 1], LutOutputRepresentation::Coefficient).unwrap(),
            )
            .unwrap();
        let output = builder.lut(Operand::Scalar(input), table).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        assert!(!program.wire_is_coefficient_output(input).unwrap());
        assert!(program.wire_is_coefficient_output(output).unwrap());
        assert_eq!(program.wire_knowledge(output).unwrap(), AttributeKnowledge::PrivatePayload);
        assert_eq!(
            program.wire_knowledge(WireId(999)),
            Err(ProgramValidationError::UndefinedWire(WireId(999)))
        );
    }

    #[test]
    fn metadata_validation_rejects_mixed_add_operands() {
        let mut builder = ExponentLutProgramBuilder::new();
        let private = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let sum = builder.add(Operand::Scalar(private), Operand::Scalar(public)).unwrap();
        assert!(matches!(
            builder.finish(vec![Operand::Scalar(sum)]),
            Err(ProgramValidationError::InvalidStructure(_))
        ));
    }

    #[test]
    fn family_range_rejects_zero_and_over_capacity() {
        assert!(FamilyRange::full(IntExpr::constant(0)).is_err());
        assert!(
            FamilyRange::bounded(IntExpr::constant(0), IntExpr::constant(3), IntExpr::constant(2))
                .is_err()
        );
    }

    #[test]
    fn identical_builder_programs_have_identical_ids() {
        fn build() -> ExponentLutProgram {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let output = builder.add(Operand::Scalar(input), Operand::Scalar(input)).unwrap();
            builder.finish(vec![Operand::Scalar(output)]).unwrap()
        }
        assert_eq!(build().id(), build().id());
    }

    #[test]
    fn failed_parallel_body_rolls_back_ids_and_luts() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let range = FamilyRange::full(IntExpr::constant(2)).unwrap();
        let error = builder.parallel(range, |builder, index| {
            let table = LutTable::unary(1, 1, vec![0], LutOutputRepresentation::Coefficient)?;
            let lut = builder.register_lut(table)?;
            let value = builder.lut(Operand::Member { family: input, index }, lut)?;
            Err(ProgramValidationError::InvalidStructure(format!("discard {value:?}")))
        });
        assert!(error.is_err());
        let output = builder.add(Operand::Scalar(input), Operand::Scalar(input)).unwrap();
        assert_eq!(output, WireId(1));
    }

    #[test]
    fn parallel_members_support_broadcast_nonzero_start_and_rhs_families() {
        let mut builder = ExponentLutProgramBuilder::new();
        let scalar = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let wires =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PrivatePayload).unwrap();
        let public_wires =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PublicPayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(4)).unwrap();
        let private_range =
            FamilyRange::bounded(IntExpr::constant(1), IntExpr::constant(2), IntExpr::constant(2))
                .unwrap();
        let selected = builder
            .parallel(private_range, |builder, index| {
                let product = builder.mul_private(
                    Operand::Member { family: wires, index },
                    Operand::Member { family: rhs, index: IntExpr::constant(1) },
                )?;
                builder.add(Operand::Scalar(product), Operand::Scalar(scalar)).map(Operand::Scalar)
            })
            .unwrap();
        let private_sum = builder.sum(selected).unwrap();
        let public_range =
            FamilyRange::bounded(IntExpr::constant(1), IntExpr::constant(2), IntExpr::constant(2))
                .unwrap();
        let public_products = builder
            .parallel(public_range, |builder, index| {
                builder
                    .mul_public(
                        Operand::Member { family: wires, index: index.clone() },
                        Operand::Member { family: public_wires, index },
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let public_sum = builder.sum(public_products).unwrap();
        let output =
            builder.add(Operand::Scalar(private_sum), Operand::Scalar(public_sum)).unwrap();
        assert!(builder.finish(vec![Operand::Scalar(output)]).is_ok());
    }

    #[test]
    fn nested_parallel_body_has_distinct_lexical_indices() {
        let mut builder = ExponentLutProgramBuilder::new();
        let wires =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let other_wires =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let outer = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, outer_index| {
                let inner = builder.parallel(
                    FamilyRange::full(IntExpr::constant(2)).unwrap(),
                    |builder, inner_index| {
                        builder
                            .add(
                                Operand::Member { family: wires, index: outer_index.clone() },
                                Operand::Member { family: other_wires, index: inner_index },
                            )
                            .map(Operand::Scalar)
                    },
                )?;
                builder.sum(inner).map(Operand::Scalar)
            })
            .unwrap();
        let total = builder.sum(outer).unwrap();
        assert!(builder.finish(vec![Operand::Scalar(total)]).is_ok());
    }

    #[test]
    fn nested_dynamic_member_bounds_include_each_nonzero_loop_start() {
        let mut builder = ExponentLutProgramBuilder::new();
        let source =
            builder.input_family(IntExpr::constant(4), AttributeKnowledge::PrivatePayload).unwrap();
        let outer = builder
            .parallel(
                FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(2),
                    IntExpr::constant(2),
                )
                .unwrap(),
                |builder, outer_index| {
                    let inner = builder.parallel(
                        FamilyRange::bounded(
                            IntExpr::constant(1),
                            IntExpr::constant(2),
                            IntExpr::constant(2),
                        )
                        .unwrap(),
                        |builder, inner_index| {
                            builder
                                .add(
                                    Operand::Member { family: source, index: outer_index.clone() },
                                    Operand::Member { family: source, index: inner_index },
                                )
                                .map(Operand::Scalar)
                        },
                    )?;
                    builder.sum(inner).map(Operand::Scalar)
                },
            )
            .unwrap();
        let output = builder.sum(outer).unwrap();
        assert!(builder.finish(vec![Operand::Scalar(output)]).is_ok());
    }

    #[test]
    fn nested_dynamic_member_bounds_reject_an_out_of_range_combination() {
        let mut builder = ExponentLutProgramBuilder::new();
        let source =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let outer = builder
            .parallel(
                FamilyRange::bounded(
                    IntExpr::constant(1),
                    IntExpr::constant(1),
                    IntExpr::constant(1),
                )
                .unwrap(),
                |builder, outer_index| {
                    let inner = builder.parallel(
                        FamilyRange::full(IntExpr::constant(1)).unwrap(),
                        |builder, inner_index| {
                            let index = outer_index.clone() + inner_index + 1;
                            builder
                                .add(
                                    Operand::Member { family: source, index },
                                    Operand::Member { family: source, index: outer_index.clone() },
                                )
                                .map(Operand::Scalar)
                        },
                    )?;
                    builder.sum(inner).map(Operand::Scalar)
                },
            )
            .unwrap();
        let output = builder.sum(outer).unwrap();
        assert_eq!(
            builder.finish(vec![Operand::Scalar(output)]),
            Err(ProgramValidationError::MemberOutOfBounds)
        );
    }

    #[test]
    fn finish_rejects_a_static_member_outside_its_source_family() {
        let mut builder = ExponentLutProgramBuilder::new();
        let wires =
            builder.input_family(IntExpr::constant(2), AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let family = builder
            .parallel(FamilyRange::full(IntExpr::constant(1)).unwrap(), |builder, _| {
                builder
                    .mul_private(
                        Operand::Member { family: wires, index: IntExpr::constant(2) },
                        Operand::Scalar(rhs),
                    )
                    .map(Operand::Scalar)
            })
            .unwrap();
        let error = builder.finish(vec![Operand::Scalar(family)]).unwrap_err();
        assert_eq!(error, ProgramValidationError::MemberOutOfBounds);
    }

    #[test]
    fn parallel_child_wires_cannot_escape_lexical_scope() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let family = builder
            .parallel(FamilyRange::full(IntExpr::constant(1)).unwrap(), |builder, _| {
                builder.add(Operand::Scalar(input), Operand::Scalar(input)).map(Operand::Scalar)
            })
            .unwrap();
        assert_eq!(family, WireId(2));
        assert_eq!(
            builder.add(Operand::Scalar(WireId(1)), Operand::Scalar(input)),
            Err(ProgramValidationError::UndefinedWire(WireId(1)))
        );
    }

    #[test]
    fn binary_lut_keeps_one_rhs_and_one_table_reference_per_gate() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let lut_id = builder
            .register_binary_lut(2, 2, 2, vec![0, 1, 1, 0], LutOutputRepresentation::Coefficient)
            .unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let output = builder.lut(Operand::Scalar(product), lut_id).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();

        assert_eq!(program.rhs_inputs.len(), 1);
        assert_eq!(program.luts.len(), 1);
        assert!(matches!(
            program.body.steps.as_slice(),
            [
                Step::MulPrivate { input: Operand::Scalar(lhs), rhs: Operand::Scalar(actual_rhs), .. },
                Step::Lut { input: Operand::Scalar(actual_input), table: actual_lut, .. },
            ] if *lhs == input && *actual_rhs == rhs && *actual_input == product && *actual_lut == lut_id
        ));
    }

    #[test]
    fn binary_lut_flattens_rhs_domain_into_one_checked_input_width() {
        let mut builder = ExponentLutProgramBuilder::new();
        let lut = builder
            .register_binary_lut(
                2,
                3,
                2,
                vec![0, 1, 1, 0, 0, 1],
                LutOutputRepresentation::Coefficient,
            )
            .unwrap();
        assert_eq!(builder.luts[lut.0 as usize].input_width(), 6);
        assert_eq!(builder.luts[lut.0 as usize].values(), &[0, 1, 1, 0, 0, 1]);
        assert!(
            builder
                .register_binary_lut(2, 3, 2, vec![0; 5], LutOutputRepresentation::Coefficient,)
                .is_err()
        );
    }

    #[test]
    fn unknown_rhs_and_lut_are_rejected_without_expanding_public_types() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        assert_eq!(
            builder.mul_private(Operand::Scalar(input), Operand::Scalar(RhsId(99))),
            Err(ProgramValidationError::UndefinedRhs(RhsId(99)))
        );
        assert_eq!(
            builder.lut(Operand::Scalar(input), LutId(99)),
            Err(ProgramValidationError::InvalidLutId(LutId(99)))
        );
    }

    #[test]
    fn input_projection_excludes_rhs_and_computation_body() {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let product = builder.mul_private(Operand::Scalar(input), Operand::Scalar(rhs)).unwrap();
        let program = builder.finish(vec![Operand::Scalar(product)]).unwrap();
        let projection = program.encryption_inputs();

        assert_eq!(projection.inputs().len(), 1);
        assert!(projection.inputs().contains_key(&input));
        assert!(projection.luts().is_empty());
        assert_eq!(program.rhs_inputs.len(), 1);
        assert_eq!(program.body.steps.len(), 1);
    }

    #[test]
    fn canonical_identity_is_stable_and_commits_lut_contents() {
        fn build(values: Vec<usize>) -> ExponentLutProgram {
            let mut builder = ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let table =
                LutTable::unary(2, 2, values, LutOutputRepresentation::Coefficient).unwrap();
            let lut = builder.register_lut(table).unwrap();
            let output = builder.lut(Operand::Scalar(input), lut).unwrap();
            builder.finish(vec![Operand::Scalar(output)]).unwrap()
        }

        let first = build(vec![0, 1]);
        let same = build(vec![0, 1]);
        let changed = build(vec![1, 0]);
        assert_eq!(first.id(), same.id());
        assert_ne!(first.id(), changed.id());
        assert_eq!(first.body, same.body);
    }

    #[test]
    fn validation_rejects_duplicate_wire_ids_in_sibling_parallel_bodies() {
        let input = WireId(0);
        let duplicate = WireId(1);
        let child = |output| Body {
            steps: vec![Step::Add {
                lhs: Operand::Scalar(input),
                rhs: Operand::Scalar(input),
                output,
            }],
            outputs: vec![Operand::Scalar(output)],
        };
        let program = ExponentLutProgram {
            id: ExponentLutProgramId::from_digest([0; 32]),
            inputs: std::collections::BTreeMap::from([(
                input,
                ProgramInputDeclaration {
                    shape: InputBinding::Scalar(()),
                    knowledge: AttributeKnowledge::PrivatePayload,
                },
            )]),
            rhs_inputs: std::collections::BTreeMap::new(),
            luts: Vec::new(),
            body: Body {
                steps: vec![
                    Step::Parallel {
                        range: FamilyRange::full(IntExpr::constant(1)).unwrap(),
                        body: Box::new(child(duplicate)),
                        output: WireId(2),
                    },
                    Step::Parallel {
                        range: FamilyRange::full(IntExpr::constant(1)).unwrap(),
                        body: Box::new(child(duplicate)),
                        output: WireId(3),
                    },
                ],
                outputs: vec![Operand::Scalar(WireId(2))],
            },
            metadata_cache: OnceLock::new(),
        };

        assert_eq!(
            validate_program(&program),
            Err(ProgramValidationError::DuplicateWire(duplicate))
        );
    }

    #[test]
    fn validation_rejects_duplicate_wire_ids_across_nested_parallel_bodies() {
        let input = WireId(0);
        let duplicate = WireId(1);
        let nested_body = Body {
            steps: vec![
                Step::Parallel {
                    range: FamilyRange::full(IntExpr::constant(1)).unwrap(),
                    body: Box::new(Body {
                        steps: vec![Step::Add {
                            lhs: Operand::Scalar(input),
                            rhs: Operand::Scalar(input),
                            output: duplicate,
                        }],
                        outputs: vec![Operand::Scalar(duplicate)],
                    }),
                    output: WireId(2),
                },
                Step::Sum { family: WireId(2), output: WireId(3) },
            ],
            outputs: vec![Operand::Scalar(WireId(3))],
        };
        let program = ExponentLutProgram {
            id: ExponentLutProgramId::from_digest([0; 32]),
            inputs: std::collections::BTreeMap::from([(
                input,
                ProgramInputDeclaration {
                    shape: InputBinding::Scalar(()),
                    knowledge: AttributeKnowledge::PrivatePayload,
                },
            )]),
            rhs_inputs: std::collections::BTreeMap::new(),
            luts: Vec::new(),
            body: Body {
                steps: vec![
                    Step::Parallel {
                        range: FamilyRange::full(IntExpr::constant(1)).unwrap(),
                        body: Box::new(nested_body),
                        output: WireId(4),
                    },
                    Step::Parallel {
                        range: FamilyRange::full(IntExpr::constant(1)).unwrap(),
                        body: Box::new(Body {
                            steps: vec![Step::Add {
                                lhs: Operand::Scalar(input),
                                rhs: Operand::Scalar(input),
                                output: duplicate,
                            }],
                            outputs: vec![Operand::Scalar(duplicate)],
                        }),
                        output: WireId(5),
                    },
                ],
                outputs: vec![Operand::Scalar(WireId(4))],
            },
            metadata_cache: OnceLock::new(),
        };

        assert_eq!(
            validate_program(&program),
            Err(ProgramValidationError::DuplicateWire(duplicate))
        );
    }

    #[test]
    fn family_scope_flattens_nonzero_nested_ranges_row_major() {
        let outer =
            FamilyRange::bounded(IntExpr::constant(1), IntExpr::constant(2), IntExpr::constant(4))
                .unwrap();
        let inner =
            FamilyRange::bounded(IntExpr::constant(2), IntExpr::constant(2), IntExpr::constant(3))
                .unwrap();
        let scope =
            FamilyScope::new(vec![outer, inner], vec![IntExpr::constant(2), IntExpr::constant(3)])
                .unwrap();
        // (2 - 1) * inner_count + (3 - 2) = 3.  The inner/branch axis
        // is last and therefore the fastest-changing axis.
        assert_eq!(scope.flattened_index().unwrap(), IntExpr::constant(3));
    }

    #[test]
    fn family_scope_rejects_member_outside_its_lexical_range() {
        let range =
            FamilyRange::bounded(IntExpr::constant(3), IntExpr::constant(2), IntExpr::constant(4))
                .unwrap();
        assert_eq!(
            FamilyScope::new(vec![range], vec![IntExpr::constant(2)]),
            Err(ProgramValidationError::MemberOutOfBounds)
        );
    }
}
