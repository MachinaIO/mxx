//! Job-local expression and program arena.
//!
//! It owns no lowering state. IDs in this file are job-local values: they are
//! never suitable for serialization or comparison with IDs from another arena.

use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, Zero};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_ARENA_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Identifies one job-local arena.  The token is part of every ID's equality
/// and ordering relation, so a slot from a different arena can never alias a
/// local slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ArenaToken(pub(crate) u32);

impl ArenaToken {
    pub(crate) fn fresh() -> Self {
        let token = NEXT_ARENA_TOKEN.fetch_add(1, Ordering::Relaxed);
        Self(u32::try_from(token).expect("job-local arena token space exhausted"))
    }
}

/// Compact identity of one immutable expression node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ExprId {
    pub(crate) arena: ArenaToken,
    pub(crate) slot: u32,
}

impl ExprId {
    pub(crate) const fn new(arena: ArenaToken, slot: u32) -> Self {
        Self { arena, slot }
    }

    pub(crate) const fn arena(self) -> ArenaToken {
        self.arena
    }

    pub(crate) const fn slot(self) -> u32 {
        self.slot
    }
}

/// Compact identity of one finalized value program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ValueProgramId {
    pub(crate) arena: ArenaToken,
    pub(crate) slot: u32,
}

impl ValueProgramId {
    pub(crate) const fn new(arena: ArenaToken, slot: u32) -> Self {
        Self { arena, slot }
    }
}

/// Matrix shape and ring contract carried by every matrix expression.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedMatrixType {
    pub modulus: BigUint,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
}

impl ResolvedMatrixType {
    pub fn new(
        modulus: BigUint,
        ring_dimension: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self, ArenaError> {
        if modulus.is_zero() || ring_dimension == 0 || rows == 0 || columns == 0 {
            return Err(ArenaError::InvalidMatrixType);
        }
        Ok(Self { modulus, ring_dimension, rows, columns })
    }

    fn same_ring(&self, other: &Self) -> bool {
        self.modulus == other.modulus && self.ring_dimension == other.ring_dimension
    }
}

/// The resolved type of an expression.  This is intentionally smaller than
/// the wire-type vocabulary: it describes semantic values, not graph ports.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResolvedValueType {
    Bool,
    Int,
    Real,
    Bytes,
    Matrix(ResolvedMatrixType),
    Trapdoor,
}

/// An exact half-open family domain.  Empty domains are representable here;
/// family constructors that require a stored family must additionally call
/// [`FamilyDomain::nonempty`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FamilyDomain {
    pub minimum: u64,
    pub maximum_exclusive: u64,
}

impl FamilyDomain {
    pub fn new(minimum: u64, maximum_exclusive: u64) -> Result<Self, ArenaError> {
        if minimum > maximum_exclusive {
            return Err(ArenaError::InvalidRange { minimum, maximum_exclusive });
        }
        Ok(Self { minimum, maximum_exclusive })
    }

    pub fn nonempty(self) -> Result<Self, ArenaError> {
        if self.minimum == self.maximum_exclusive {
            return Err(ArenaError::EmptyFamilyDomain);
        }
        Ok(self)
    }

    pub fn contains(self, range: TrustedIndexRange) -> bool {
        self.minimum <= range.minimum && range.maximum_exclusive <= self.maximum_exclusive
    }
}

/// A trusted, caller-declared half-open range for one integer expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TrustedIndexRange {
    pub minimum: u64,
    pub maximum_exclusive: u64,
}

#[cfg(test)]
impl TrustedIndexRange {
    pub fn new(minimum: u64, maximum_exclusive: u64) -> Result<Self, ArenaError> {
        if minimum > maximum_exclusive {
            return Err(ArenaError::InvalidRange { minimum, maximum_exclusive });
        }
        Ok(Self { minimum, maximum_exclusive })
    }
}

/// A typed literal.  Bytes and real values are represented canonically by the
/// caller; they are not interpreted by Stage 1.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TypedConstant {
    pub value_type: ResolvedValueType,
    pub value: ConstantValue,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ConstantValue {
    Bool(bool),
    Int(BigInt),
    Real(String),
    Bytes(Box<[u8]>),
}

impl TypedConstant {
    pub fn bool(value: bool) -> Self {
        Self { value_type: ResolvedValueType::Bool, value: ConstantValue::Bool(value) }
    }

    pub fn int(value: impl Into<BigInt>) -> Self {
        Self { value_type: ResolvedValueType::Int, value: ConstantValue::Int(value.into()) }
    }

    pub fn real(value: impl Into<String>) -> Self {
        Self { value_type: ResolvedValueType::Real, value: ConstantValue::Real(value.into()) }
    }

    pub fn bytes(value: impl Into<Box<[u8]>>) -> Self {
        Self { value_type: ResolvedValueType::Bytes, value: ConstantValue::Bytes(value.into()) }
    }
}

/// Stable event identity for independent random samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SampleEventId(pub u64);

/// Complete sampler descriptor.  Parameters are resolved before entering the
/// arena, so no temporary loop owner or display path participates in identity.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SampleDescriptor {
    pub definition: String,
    pub parameters: Box<[u64]>,
    pub output_type: ResolvedValueType,
    pub gadget_base: Option<BigUint>,
    pub digit_count: Option<u32>,
    pub decomposition: Option<String>,
}

impl SampleDescriptor {
    pub fn new(definition: impl Into<String>, output_type: ResolvedValueType) -> Self {
        Self {
            definition: definition.into(),
            parameters: Box::new([]),
            output_type,
            gadget_base: None,
            digit_count: None,
            decomposition: None,
        }
    }
}

/// Artifact identity used by source/family leaves.  It is intentionally a
/// semantic descriptor rather than a file path.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ArtifactIdentity {
    pub definition: String,
    pub version: u32,
    pub confidentiality: u8,
    pub value_type: ResolvedValueType,
    pub layout: String,
    pub domain: Option<FamilyDomain>,
}

/// Complete identity of a deterministic source value.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SemanticSourceIdentity {
    pub stable_definition: String,
    pub invocation: String,
    pub sample_event: Option<SampleEventId>,
    pub output_role: String,
    pub sampler: Option<SampleDescriptor>,
    pub artifact: Option<ArtifactIdentity>,
    pub value_type: ResolvedValueType,
    pub coordinates: Box<[u64]>,
    pub matrix_constant: Option<MatrixConstantKind>,
}

/// Typed identity for a matrix constant whose algebraic meaning is used by the checker.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MatrixConstantKind {
    Zero,
    Identity,
    UnitRow { index: u64 },
    UnitColumn { index: u64 },
    Gadget { base: u64, small: bool },
    PowerOfBase { base: BigInt, exponent: BigUint },
    Rotation { exponent: u64 },
    Polynomial { coefficients: Box<[BigInt]> },
}

/// Complete identity of a generated/source family element.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SemanticFamilySourceIdentity {
    pub stable_definition: String,
    pub invocation: String,
    pub element_type: ResolvedValueType,
    pub domain: FamilyDomain,
    pub artifact: Option<ArtifactIdentity>,
}

/// Stable identity of a registered pure index function.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IndexFunctionDefinitionId(pub u64);

/// A registered index-function signature.  Evaluation is intentionally not a
/// Stage 1 concern; the signature is enough to validate arity and output type.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IndexFunctionDefinition {
    pub id: IndexFunctionDefinitionId,
    pub arity: usize,
    pub output_type: ResolvedValueType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IndexEvaluatorId {
    arena: ArenaToken,
    definition: IndexFunctionDefinitionId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuiltinIndexEvaluator {
    Add,
    Subtract,
    Modulo,
    Divide,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexEvaluationInput {
    pub value: i64,
    pub value_type: ResolvedValueType,
    pub trusted_range: TrustedIndexRange,
}

type IndexEvaluator =
    dyn Fn(&[i64], &[u64]) -> Result<i64, String> + Send + Sync + std::panic::RefUnwindSafe;

/// Layout descriptor used by view/slice/concat/tensor operators.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixLayout {
    pub name: String,
    pub row_stride: usize,
    pub column_stride: usize,
}

impl MatrixLayout {
    pub fn row_major(rows: usize, columns: usize) -> Self {
        Self { name: format!("row-major-{rows}x{columns}"), row_stride: columns, column_stride: 1 }
    }

    fn validate_for(&self, matrix: &ResolvedMatrixType) -> Result<(), ArenaError> {
        if self.name.is_empty() || self.row_stride == 0 || self.column_stride == 0 {
            return Err(ArenaError::InvalidLayout);
        }
        // Distinct logical cells must not map to one physical address.  For positive
        // row/column strides, the smallest collision has row delta
        // `column_stride / gcd` and column delta `row_stride / gcd`; reject it when both
        // deltas fit inside the matrix.  This keeps malformed descriptors from silently
        // aliasing values while retaining valid padded/strided layouts.
        let gcd = |mut left: usize, mut right: usize| {
            while right != 0 {
                let remainder = left % right;
                left = right;
                right = remainder;
            }
            left
        };
        let divisor = gcd(self.row_stride, self.column_stride);
        let row_period = self.column_stride / divisor;
        let column_period = self.row_stride / divisor;
        if row_period < matrix.rows && column_period < matrix.columns {
            return Err(ArenaError::InvalidLayout);
        }
        // A layout is a descriptor for a dense, finite matrix.  Check the
        // largest addressed element without allowing a malformed descriptor
        // to wrap usize.  The name is intentionally not interpreted: custom
        // layouts are valid as long as their strides describe the value.
        let row_offset = matrix
            .rows
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.row_stride))
            .ok_or(ArenaError::InvalidLayout)?;
        let column_offset = matrix
            .columns
            .checked_sub(1)
            .and_then(|columns| columns.checked_mul(self.column_stride))
            .ok_or(ArenaError::InvalidLayout)?;
        row_offset.checked_add(column_offset).ok_or(ArenaError::InvalidLayout)?;
        Ok(())
    }
}

/// Scalar operators retain all value-affecting descriptor fields.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ScalarOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Negate,
    Equal,
    Less,
    LessEqual,
    BoolToInt,
    IntToReal,
    RealAdd,
    RealSubtract,
    RealMultiply,
    RealDivide,
    RealSqrt,
    ThresholdDecode { plaintext_modulus: BigUint, length: u64, output_bool: bool },
    Bit { position: u32 },
    Slice { start: u64, end_exclusive: u64 },
    Hash { tag: String, dynamic_tags: Box<[u64]> },
    ExtractCoefficient { row: u64, column: u64 },
    LiftConstantPolynomial { output: ResolvedMatrixType, coefficient_bits: u32 },
}

/// Matrix operators retain complete shape, ring, layout, and decomposition
/// descriptors.  Operations that return an integer are kept in this namespace
/// so the top-level operator vocabulary remains exhaustive.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MatrixOperation {
    Add,
    Subtract,
    Multiply,
    Negate,
    Scale,
    /// Norm-preserving signed coefficient permutation in the negacyclic ring.
    RingAutomorphism {
        index: u64,
    },
    Transpose,
    Slice {
        row_start: usize,
        row_end_exclusive: usize,
        column_start: usize,
        column_end_exclusive: usize,
        layout: MatrixLayout,
    },
    /// A slice whose coordinates are binder-open integer children. The coordinate expressions
    /// are stored as operator inputs (matrix, row start/end, column start/end), so lowering never
    /// evaluates a loop binder in the closed parameter environment.
    IndexedSlice {
        output: ResolvedMatrixType,
        layout: MatrixLayout,
    },
    View {
        output: ResolvedMatrixType,
        layout: MatrixLayout,
    },
    Concat {
        axis: u8,
        output: ResolvedMatrixType,
        layout: MatrixLayout,
    },
    Tensor {
        output: ResolvedMatrixType,
        left_layout: MatrixLayout,
        right_layout: MatrixLayout,
        output_layout: MatrixLayout,
    },
    CrtRecompose {
        plaintext_moduli: Box<[BigUint]>,
        reconstruction_coefficients: Box<[BigInt]>,
        output: ResolvedMatrixType,
    },
    ExtractCoefficient {
        row: u64,
        column: u64,
    },
    LiftConstantPolynomial {
        output: ResolvedMatrixType,
        coefficient_bits: u32,
    },
}

/// Operations whose output is a sampled matrix.  Keeping these descriptors
/// explicit prevents a sampler's value-affecting contract from being hidden in
/// a display string or an incomplete generic `SampleDescriptor`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SamplerOperation {
    UniformResidue {
        output: ResolvedMatrixType,
    },
    UniformInterval {
        output: ResolvedMatrixType,
        minimum: BigInt,
        maximum: BigInt,
    },
    Gaussian {
        output: ResolvedMatrixType,
        sigma: String,
        max_coefficient_bound: BigInt,
    },
    Hash {
        output: ResolvedMatrixType,
        variant: HashVariant,
        tag_prefix: Box<[u8]>,
        tag_expressions: Box<[u64]>,
        tag_decimal_expressions: Box<[u64]>,
        tag_u64_le_expressions: Box<[u64]>,
        base: Option<u64>,
        digit_count: Option<u32>,
    },
    Trapdoor {
        output: ResolvedMatrixType,
        sigma: String,
        gadget_base: u64,
        digit_count: u32,
        preimage_max_coefficient_bound: BigInt,
    },
    Preimage {
        output: ResolvedMatrixType,
        max_coefficient_bound: BigInt,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HashVariant {
    Plain,
    Decomposed,
    SmallDecomposed,
}

/// Stable definition of the deterministic polynomial hash used by Graph IR.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DeterministicHashDefinition {
    MxxPolynomialHash,
}

/// Complete typed encoding of a deterministic hash invocation. The expression inputs are,
/// in order, the 32-byte key, binary tag expressions, decimal tag expressions, little-endian
/// u64 tag expressions, and dynamic integer tag wires.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DeterministicHashDescriptor {
    pub definition: DeterministicHashDefinition,
    pub version: u32,
    pub key_byte_length: u32,
    pub output: ResolvedMatrixType,
    pub tag_prefix: Box<[u8]>,
    pub binary_tag_count: u32,
    pub decimal_tag_count: u32,
    pub u64_le_tag_count: u32,
    pub dynamic_tag_count: u32,
}

/// Exact decomposition and polynomial-packing descriptors.  Inputs carry the
/// source value (and, for packing, the ordered boolean bits) exactly once.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ValueTransformOperation {
    GadgetDecompose { output: ResolvedMatrixType, base: u64, small: bool, digit_count: u32 },
    PackPolynomialCoefficients { output: ResolvedMatrixType, coefficient_bits: u32 },
}

/// Trapdoor operations are typed atoms/transformations, not public matrices.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TrapdoorOperation {
    Generate {
        descriptor: String,
        parameters: Box<[u64]>,
        paired_public_event: SampleEventId,
        paired_public_output_role: String,
    },
    Transform {
        descriptor: String,
        output: ResolvedValueType,
        parameters: Box<[u64]>,
    },
}

/// Semantic operator namespace. A descriptor is never inferred from
/// facts or display provenance after interning.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ValueOperator {
    /// A positional binder carries its complete type in the expression key.  The same position
    /// may therefore be used by independent programs with different signatures in one arena.
    Argument {
        position: u32,
        value_type: ResolvedValueType,
    },
    Constant(TypedConstant),
    Source(SemanticSourceIdentity),
    Sample {
        event: SampleEventId,
        descriptor: SampleDescriptor,
    },
    Sampler {
        event: SampleEventId,
        operation: SamplerOperation,
    },
    DeterministicHash(DeterministicHashDescriptor),
    OpaqueFamilyElement {
        source: SemanticFamilySourceIdentity,
    },
    IndexMap {
        definition: IndexFunctionDefinitionId,
        parameters: Box<[u64]>,
    },
    ExplicitElement {
        domain: FamilyDomain,
        element_type: ResolvedValueType,
    },
    ProgramCall {
        program: ValueProgramId,
    },
    Transform(ValueTransformOperation),
    ExtractCoefficient {
        position: u64,
        canonical_input_exclusive_upper: Option<BigUint>,
    },
    Scalar(ScalarOperation),
    Matrix(MatrixOperation),
    Trapdoor(TrapdoorOperation),
}

impl ValueOperator {
    pub(crate) fn source_matrix_constant(&self) -> Option<&MatrixConstantKind> {
        match self {
            Self::Source(source) => source.matrix_constant.as_ref(),
            _ => None,
        }
    }
}

/// One immutable DAG node.  Inputs are IDs, never recursively-owned nodes.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ExprNode {
    pub operator: ValueOperator,
    pub inputs: Box<[ExprId]>,
}

/// Compact immutable summary of the exact free binders below one expression slot.
///
/// `Unknown` is used only for slots which have not been summarized yet.  Once a slot is
/// visited, its entry is replaced permanently by one of the other variants because expression
/// nodes are immutable and the arena is append-only.  The `Many` representation is sorted and
/// deduplicated by exact [`ExprId`], rather than by the displayed binder position.
#[derive(Clone, Debug, Eq, PartialEq)]
enum FreeArgumentSummary {
    Unknown,
    Closed,
    One(ExprId),
    Many(Box<[ExprId]>),
}

impl FreeArgumentSummary {
    fn from_sorted_ids(ids: Vec<ExprId>) -> Self {
        match ids.as_slice() {
            [] => Self::Closed,
            [id] => Self::One(*id),
            _ => Self::Many(ids.into_boxed_slice()),
        }
    }

    fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown)
    }

    fn into_ids(self) -> BTreeSet<ExprId> {
        match self {
            Self::Unknown | Self::Closed => BTreeSet::new(),
            Self::One(id) => BTreeSet::from([id]),
            Self::Many(ids) => ids.into_vec().into_iter().collect(),
        }
    }
}

/// Counters for the append-only free-binder summary cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FreeArgumentCounters {
    pub queries: u64,
    pub nodes_visited: u64,
    pub root_hits: u64,
    pub subtree_hits: u64,
    pub newly_summarized: u64,
}

/// A completed exact substitution key.  The replacement pairs are canonical because callers
/// construct them from an ordered map after validating every ID and exact source/replacement
/// type pair.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ExactSubstitutionKey {
    root: ExprId,
    replacements: Box<[(ExprId, ExprId)]>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SubstitutionScratchEntry {
    epoch: u64,
    value: Option<ExprId>,
}

/// Candidate-local state for exact expression substitution.
///
/// The completed-result cache is deliberately owned by the caller's candidate (the program
/// arena or production lowerer), while this engine supplies the common validation, traversal,
/// and slot-indexed scratch implementation.  Scratch entries are tagged with an invocation
/// epoch, so repeated substitutions never clear a per-call tree map.
pub(crate) struct ExactSubstitutionEngine {
    completed: BTreeMap<ExactSubstitutionKey, ExprId>,
    scratch: Vec<SubstitutionScratchEntry>,
    epoch: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExactSubstitutionResult {
    pub root: ExprId,
    pub cache_hit: bool,
    pub binder_shortcut: bool,
    pub skipped_subtrees: u64,
    pub nodes_visited: u64,
    pub nodes_reinterned: u64,
}

/// Failure information from the common substitution engine.  Keeping the failing source node,
/// operator, output type, and rewritten input types allows production lowering to attach its
/// existing wire/operation/type context without reimplementing the traversal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExactSubstitutionFailure {
    pub node: ExprId,
    pub operator: Option<ValueOperator>,
    pub expected_output: Option<ResolvedValueType>,
    pub actual_inputs: Box<[ResolvedValueType]>,
    pub source: ArenaError,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArenaError {
    ForeignExpression {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    ForeignProgram {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    InvalidSlot {
        slot: u32,
    },
    InvalidRange {
        minimum: u64,
        maximum_exclusive: u64,
    },
    IndexRangeRequired {
        id: ExprId,
    },
    EmptyFamilyDomain,
    InvalidMatrixType,
    InvalidLayout,
    InvalidCoefficientBits {
        coefficient_bits: u32,
    },
    DimensionOverflow {
        operator: String,
    },
    InvalidSamplerRange,
    InvalidDomainWidth {
        minimum: u64,
        maximum_exclusive: u64,
    },
    InvalidArity {
        operator: String,
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        operator: String,
        position: usize,
        expected: ResolvedValueType,
        actual: ResolvedValueType,
    },
    IncompatibleMatrixTypes,
    UnknownIndexFunction(IndexFunctionDefinitionId),
    MissingIndexEvaluator(IndexFunctionDefinitionId),
    ForeignIndexEvaluator {
        expected: ArenaToken,
        actual: ArenaToken,
    },
    IndexEvaluatorPanicked(IndexFunctionDefinitionId),
    IndexEvaluatorImplementation {
        definition: IndexFunctionDefinitionId,
        message: String,
    },
    IndexValueOutOfRange {
        value: i64,
        minimum: u64,
        maximum_exclusive: u64,
    },
    IndexArity {
        expected: usize,
        actual: usize,
    },
    ArgumentTypeRequired {
        position: u32,
    },
    ConflictingArgumentType {
        position: u32,
        first: ResolvedValueType,
        second: ResolvedValueType,
    },
    /// A value-fact transfer failed after the expression itself was constructed. Keep the
    /// expression and fact-store reason visible; collapsing this into a matrix-shape error
    /// makes production Graph failures impossible to localize.
    FactTransferRejected {
        expression: ExprId,
        reason: String,
    },
    UnknownProgram(ValueProgramId),
    ProgramSignatureMismatch,
    ProgramOutputMismatch,
    ScopeMismatch {
        expected: ValueProgramId,
        actual: ValueProgramId,
    },
    InvalidScopeProof,
    FreeArgumentEscapes {
        position: u32,
    },
    InvalidClosedExpression,
    ProgramArenaExhausted,
    ExpressionArenaExhausted,
    ExpressionAllocationFailed,
}

impl fmt::Display for ArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ArenaError {}

fn validate_index_value(value: i64, range: TrustedIndexRange) -> Result<(), ArenaError> {
    let in_range = u64::try_from(value)
        .ok()
        .is_some_and(|value| value >= range.minimum && value < range.maximum_exclusive);
    if in_range {
        Ok(())
    } else {
        Err(ArenaError::IndexValueOutOfRange {
            value,
            minimum: range.minimum,
            maximum_exclusive: range.maximum_exclusive,
        })
    }
}

/// An arena-owned expression with no recursive semantic storage.
pub struct ExprArena {
    token: ArenaToken,
    nodes: Vec<Arc<ExprNode>>,
    types: Vec<ResolvedValueType>,
    interner: BTreeMap<Arc<ExprNode>, u32>,
    index_definitions: BTreeMap<IndexFunctionDefinitionId, IndexFunctionDefinition>,
    index_evaluators: BTreeMap<IndexFunctionDefinitionId, Arc<IndexEvaluator>>,
    program_signatures: BTreeMap<ValueProgramId, ProgramSignature>,
    scoped_derivations: BTreeMap<ValueProgramId, HashSet<u32>>,
    /// Free-argument summaries are immutable once an expression is interned. The slot-indexed
    /// cache therefore grows append-only and preserves exact [`ExprId`] binder identity.
    free_argument_cache: RefCell<Vec<FreeArgumentSummary>>,
    free_argument_queries: Cell<u64>,
    free_argument_nodes_visited: Cell<u64>,
    free_argument_root_hits: Cell<u64>,
    free_argument_subtree_hits: Cell<u64>,
    free_argument_newly_summarized: Cell<u64>,
    #[cfg(test)]
    scope_proof_builds: std::cell::Cell<u64>,
}

impl Default for ExprArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprArena {
    pub fn new() -> Self {
        Self {
            token: ArenaToken::fresh(),
            nodes: Vec::new(),
            types: Vec::new(),
            interner: BTreeMap::new(),
            index_definitions: BTreeMap::new(),
            index_evaluators: BTreeMap::new(),
            program_signatures: BTreeMap::new(),
            scoped_derivations: BTreeMap::new(),
            free_argument_cache: RefCell::new(Vec::new()),
            free_argument_queries: Cell::new(0),
            free_argument_nodes_visited: Cell::new(0),
            free_argument_root_hits: Cell::new(0),
            free_argument_subtree_hits: Cell::new(0),
            free_argument_newly_summarized: Cell::new(0),
            #[cfg(test)]
            scope_proof_builds: std::cell::Cell::new(0),
        }
    }

    pub fn token(&self) -> ArenaToken {
        self.token
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn free_argument_counters(&self) -> (u64, u64) {
        (self.free_argument_queries.get(), self.free_argument_nodes_visited.get())
    }

    pub(crate) fn free_argument_counter_details(&self) -> FreeArgumentCounters {
        FreeArgumentCounters {
            queries: self.free_argument_queries.get(),
            nodes_visited: self.free_argument_nodes_visited.get(),
            root_hits: self.free_argument_root_hits.get(),
            subtree_hits: self.free_argument_subtree_hits.get(),
            newly_summarized: self.free_argument_newly_summarized.get(),
        }
    }

    pub fn register_index_definition(
        &mut self,
        definition: IndexFunctionDefinition,
    ) -> Result<(), ArenaError> {
        if let Some(existing) = self.index_definitions.get(&definition.id) {
            if existing != &definition {
                return Err(ArenaError::ProgramSignatureMismatch);
            }
        } else {
            self.index_definitions.insert(definition.id, definition);
        }
        Ok(())
    }

    pub fn register_index_evaluator<F>(
        &mut self,
        definition: IndexFunctionDefinitionId,
        evaluator: F,
    ) -> Result<IndexEvaluatorId, ArenaError>
    where
        F: Fn(&[i64], &[u64]) -> Result<i64, String>
            + Send
            + Sync
            + std::panic::RefUnwindSafe
            + 'static,
    {
        self.index_definitions
            .get(&definition)
            .ok_or(ArenaError::UnknownIndexFunction(definition))?;
        self.index_evaluators.entry(definition).or_insert_with(|| Arc::new(evaluator));
        Ok(IndexEvaluatorId { arena: self.token, definition })
    }

    pub fn register_builtin_index_evaluator(
        &mut self,
        definition: IndexFunctionDefinitionId,
        builtin: BuiltinIndexEvaluator,
    ) -> Result<IndexEvaluatorId, ArenaError> {
        self.register_index_evaluator(definition, move |inputs, parameters| {
            let input = *inputs.first().ok_or_else(|| "missing input".to_owned())?;
            let parameter =
                i64::try_from(*parameters.first().ok_or_else(|| "missing parameter".to_owned())?)
                    .map_err(|_| "parameter does not fit i64".to_owned())?;
            match builtin {
                BuiltinIndexEvaluator::Add => input.checked_add(parameter),
                BuiltinIndexEvaluator::Subtract => input.checked_sub(parameter),
                BuiltinIndexEvaluator::Modulo => input.checked_rem(parameter),
                BuiltinIndexEvaluator::Divide => input.checked_div(parameter),
            }
            .ok_or_else(|| "invalid or overflowing index arithmetic".to_owned())
        })
    }

    pub fn index_evaluator_id(
        &self,
        definition: IndexFunctionDefinitionId,
    ) -> Result<IndexEvaluatorId, ArenaError> {
        self.index_definitions
            .contains_key(&definition)
            .then_some(IndexEvaluatorId { arena: self.token, definition })
            .ok_or(ArenaError::UnknownIndexFunction(definition))
    }

    pub fn evaluate_index_map(
        &self,
        evaluator: IndexEvaluatorId,
        expression: ExprId,
        inputs: &[IndexEvaluationInput],
        output_range: TrustedIndexRange,
    ) -> Result<i64, ArenaError> {
        if evaluator.arena != self.token {
            return Err(ArenaError::ForeignIndexEvaluator {
                expected: self.token,
                actual: evaluator.arena,
            });
        }
        let node = self.node(expression)?;
        let ValueOperator::IndexMap { definition, parameters } = &node.operator else {
            return Err(ArenaError::ProgramSignatureMismatch);
        };
        if *definition != evaluator.definition {
            return Err(ArenaError::ProgramSignatureMismatch);
        }
        let signature = self
            .index_definitions
            .get(definition)
            .ok_or(ArenaError::UnknownIndexFunction(*definition))?;
        if signature.output_type != ResolvedValueType::Int || inputs.len() != signature.arity {
            return Err(ArenaError::ProgramSignatureMismatch);
        }
        let mut values = Vec::with_capacity(inputs.len());
        for (position, input) in inputs.iter().enumerate() {
            if input.value_type != ResolvedValueType::Int {
                return Err(ArenaError::TypeMismatch {
                    operator: "IndexEvaluator".to_owned(),
                    position,
                    expected: ResolvedValueType::Int,
                    actual: input.value_type.clone(),
                });
            }
            validate_index_value(input.value, input.trusted_range)?;
            values.push(input.value);
        }
        let implementation = self
            .index_evaluators
            .get(definition)
            .ok_or(ArenaError::MissingIndexEvaluator(*definition))?;
        let output = std::panic::catch_unwind(|| implementation(&values, parameters))
            .map_err(|_| ArenaError::IndexEvaluatorPanicked(*definition))?
            .map_err(|message| ArenaError::IndexEvaluatorImplementation {
                definition: *definition,
                message,
            })?;
        validate_index_value(output, output_range)?;
        Ok(output)
    }

    pub(crate) fn register_program_signature(
        &mut self,
        id: ValueProgramId,
        signature: ProgramSignature,
    ) -> Result<(), ArenaError> {
        if let Some(existing) = self.program_signatures.get(&id) {
            if existing != &signature {
                return Err(ArenaError::ProgramSignatureMismatch);
            }
        } else {
            self.program_signatures.insert(id, signature);
        }
        Ok(())
    }

    pub(crate) fn check_id(&self, id: ExprId) -> Result<usize, ArenaError> {
        if id.arena != self.token {
            return Err(ArenaError::ForeignExpression { expected: self.token, actual: id.arena });
        }
        let slot = id.slot as usize;
        (slot < self.nodes.len()).then_some(slot).ok_or(ArenaError::InvalidSlot { slot: id.slot })
    }

    pub fn node(&self, id: ExprId) -> Result<&ExprNode, ArenaError> {
        let slot = self.check_id(id)?;
        Ok(self.nodes[slot].as_ref())
    }

    pub(crate) fn node_arc(&self, id: ExprId) -> Result<Arc<ExprNode>, ArenaError> {
        let slot = self.check_id(id)?;
        Ok(Arc::clone(&self.nodes[slot]))
    }

    pub fn value_type(&self, id: ExprId) -> Result<&ResolvedValueType, ArenaError> {
        let slot = self.check_id(id)?;
        Ok(&self.types[slot])
    }

    pub fn intern_argument(
        &mut self,
        position: u32,
        value_type: ResolvedValueType,
    ) -> Result<ExprId, ArenaError> {
        self.intern(ValueOperator::Argument { position, value_type }, Box::new([]))
    }

    /// Allocate an argument which is deliberately not hash-consed.
    ///
    /// Ordinary expressions use positional arguments and therefore may be interned.  A
    /// generated lexical closure needs a distinct binder identity even when it reuses the same
    /// position as an enclosing loop, so its argument is kept out of the interner.
    pub(crate) fn fresh_argument(
        &mut self,
        position: u32,
        value_type: ResolvedValueType,
    ) -> Result<ExprId, ArenaError> {
        let slot =
            u32::try_from(self.nodes.len()).map_err(|_| ArenaError::ExpressionArenaExhausted)?;
        self.nodes.try_reserve(1).map_err(|_| ArenaError::ExpressionAllocationFailed)?;
        self.types.try_reserve(1).map_err(|_| ArenaError::ExpressionAllocationFailed)?;
        self.nodes.push(Arc::new(ExprNode {
            operator: ValueOperator::Argument { position, value_type: value_type.clone() },
            inputs: Box::new([]),
        }));
        self.types.push(value_type);
        self.free_argument_cache.borrow_mut().push(FreeArgumentSummary::Unknown);
        Ok(ExprId::new(self.token, slot))
    }

    /// Report whether an expression is the canonical hash-consed node. Generated lexical
    /// binders are deliberately absent from the interner; opaque programs cannot carry those
    /// binders because they have no explicit capture metadata.
    pub(crate) fn is_interned(&self, id: ExprId) -> Result<bool, ArenaError> {
        self.check_id(id)?;
        let slot = id.slot();
        Ok(self.interner.get(self.nodes[slot as usize].as_ref()).copied() == Some(slot))
    }

    pub fn intern(
        &mut self,
        operator: ValueOperator,
        inputs: Box<[ExprId]>,
    ) -> Result<ExprId, ArenaError> {
        for input in &inputs {
            self.check_id(*input)?;
        }
        let output_type = self.validate_operator(&operator, &inputs)?;
        let node = Arc::new(ExprNode { operator, inputs });
        if let Some(slot) = self.interner.get(node.as_ref()).copied() {
            return Ok(ExprId::new(self.token, slot));
        }
        let slot =
            u32::try_from(self.nodes.len()).map_err(|_| ArenaError::ExpressionArenaExhausted)?;
        self.nodes.try_reserve(1).map_err(|_| ArenaError::ExpressionAllocationFailed)?;
        self.types.try_reserve(1).map_err(|_| ArenaError::ExpressionAllocationFailed)?;
        self.nodes.push(Arc::clone(&node));
        self.types.push(output_type);
        self.free_argument_cache.borrow_mut().push(FreeArgumentSummary::Unknown);
        self.interner.insert(node, slot);
        Ok(ExprId::new(self.token, slot))
    }

    pub fn intern_slice(
        &mut self,
        operator: ValueOperator,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.intern(operator, inputs.to_vec().into_boxed_slice())
    }

    /// Intern one typed transformation in the ordinary expression arena.
    ///
    /// This is intentionally only a thin entry point over [`Self::intern`]:
    /// transformations do not get a second identity table or an owned tree.
    /// The descriptor and ordered inputs form the complete canonical `BTreeMap`
    /// key, and all foreign/type/layout checks therefore use the same validator
    /// as ordinary leaves and operations.
    pub(crate) fn intern_transform(
        &mut self,
        descriptor: ValueOperator,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.intern_slice(descriptor, inputs)
    }

    pub(crate) fn intern_matrix_transform(
        &mut self,
        operation: MatrixOperation,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.intern_transform(ValueOperator::Matrix(operation), inputs)
    }

    pub(crate) fn intern_scalar_transform(
        &mut self,
        operation: ScalarOperation,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.intern_transform(ValueOperator::Scalar(operation), inputs)
    }

    pub(crate) fn intern_value_transform(
        &mut self,
        operation: ValueTransformOperation,
        inputs: &[ExprId],
    ) -> Result<ExprId, ArenaError> {
        self.intern_transform(ValueOperator::Transform(operation), inputs)
    }

    pub(crate) fn intern_extract_coefficient(
        &mut self,
        input: ExprId,
        position: u64,
        canonical_input_exclusive_upper: Option<BigUint>,
    ) -> Result<ExprId, ArenaError> {
        self.intern_transform(
            ValueOperator::ExtractCoefficient { position, canonical_input_exclusive_upper },
            &[input],
        )
    }

    /// Scope-check and intern a transformation whose inputs are already
    /// validated views under one exact program.  The returned scoped view is
    /// O(1): the capability carries the program/signature proof, while the
    /// expression arena only checks the new compact ID's arena token.
    pub(crate) fn intern_scoped_transform(
        &mut self,
        proof: &mut ScopeProof,
        descriptor: ValueOperator,
        inputs: &[ScopedExprId],
    ) -> Result<ScopedExprId, ArenaError> {
        self.validate_scope_proof(proof)?;
        let mut expressions = Vec::with_capacity(inputs.len());
        for input in inputs {
            if input.program != proof.program {
                return Err(ArenaError::ScopeMismatch {
                    expected: proof.program,
                    actual: input.program,
                });
            }
            if !proof.reachable.contains(&input.expression.slot) {
                return Err(ArenaError::InvalidScopeProof);
            }
            expressions.push(input.expression);
        }
        let expression = self.intern_transform(descriptor, &expressions)?;
        proof.reachable.insert(expression.slot);
        self.scoped_derivations.entry(proof.program).or_default().insert(expression.slot);
        self.scoped_from_proof(proof, expression)
    }

    /// Issue a non-forgeable crate-private capability after validating one
    /// finalized program root.  The capability is deliberately not `Clone` or
    /// `Copy`; callers can borrow it for O(1) scoped projections and typed
    /// transforms, but cannot manufacture one from a raw program ID.
    pub(crate) fn scope_proof(
        &self,
        program: ValueProgramId,
        root: ExprId,
    ) -> Result<ScopeProof, ArenaError> {
        self.check_id(root)?;
        let signature = self
            .program_signatures
            .get(&program)
            .cloned()
            .ok_or(ArenaError::UnknownProgram(program))?;
        self.validate_free_arguments_for_signature(&signature, root)?;
        let mut reachable = HashSet::new();
        let mut work = vec![root];
        while let Some(expression) = work.pop() {
            if !reachable.insert(expression.slot) {
                continue;
            }
            work.extend(self.node(expression)?.inputs.iter().copied());
        }
        if let Some(derived) = self.scoped_derivations.get(&program) {
            reachable.extend(derived.iter().copied());
        }
        let proof = ScopeProof { arena: self.token, program, signature, root, reachable };
        #[cfg(test)]
        self.scope_proof_builds.set(self.scope_proof_builds.get().saturating_add(1));
        Ok(proof)
    }

    #[cfg(test)]
    pub(crate) fn reset_scope_proof_build_count(&self) {
        self.scope_proof_builds.set(0);
    }

    #[cfg(test)]
    pub(crate) fn scope_proof_build_count(&self) -> u64 {
        self.scope_proof_builds.get()
    }

    /// Convert an expression produced under an already validated capability to
    /// a scoped view without traversing the expression DAG.  This method is
    /// crate-private and accepts no raw program/signature fields.
    pub(crate) fn scoped_from_proof(
        &self,
        proof: &ScopeProof,
        expression: ExprId,
    ) -> Result<ScopedExprId, ArenaError> {
        self.validate_scope_proof(proof)?;
        self.check_id(expression)?;
        if !proof.reachable.contains(&expression.slot) {
            return Err(ArenaError::InvalidScopeProof);
        }
        Ok(ScopedExprId { program: proof.program, expression })
    }

    /// Absorb the reachable slots of a completed detached proof into an active proof for the
    /// same finalized program.  Detached normalization may intern scoped transforms while
    /// proving a selected branch; registering exactly those slots keeps subsequent parent-local
    /// projections valid without broadening the program's raw expression reachability.
    pub(crate) fn absorb_scope_proof(
        &mut self,
        target: &mut ScopeProof,
        detached: ScopeProof,
    ) -> Result<(), ArenaError> {
        self.validate_scope_proof(target)?;
        self.validate_scope_proof(&detached)?;
        if target.arena != detached.arena {
            return Err(ArenaError::InvalidScopeProof);
        }
        if target.program != detached.program || target.signature != detached.signature {
            return Err(ArenaError::InvalidScopeProof);
        }
        target.reachable.extend(detached.reachable.iter().copied());
        self.scoped_derivations.entry(detached.program).or_default().extend(detached.reachable);
        Ok(())
    }

    /// Project the sole immutable input edge of an already-scoped parent into the same program.
    ///
    /// The caller supplies no child ID or signature. Authority comes only from the arena-owned
    /// parent edge and its registered program, so a node created after an older whole-root proof
    /// can expose its direct child without rebuilding or extending that proof.
    pub(crate) fn scoped_only_input(
        &self,
        parent: ScopedExprId,
    ) -> Result<ScopedExprId, ArenaError> {
        self.check_id(parent.expression)?;
        if !self.program_signatures.contains_key(&parent.program) {
            return Err(ArenaError::UnknownProgram(parent.program));
        }
        let node = self.node(parent.expression)?;
        let [child] = node.inputs.as_ref() else {
            return Err(ArenaError::InvalidArity {
                operator: "ScopedOnlyInput".to_owned(),
                expected: 1,
                actual: node.inputs.len(),
            });
        };
        self.check_id(*child)?;
        Ok(ScopedExprId { program: parent.program, expression: *child })
    }

    pub(crate) fn validate_scoped_from_proof(
        &self,
        proof: &ScopeProof,
        scoped: ScopedExprId,
    ) -> Result<(), ArenaError> {
        self.validate_scope_proof(proof)?;
        if scoped.program != proof.program {
            return Err(ArenaError::ScopeMismatch {
                expected: proof.program,
                actual: scoped.program,
            });
        }
        self.check_id(scoped.expression)?;
        if !proof.reachable.contains(&scoped.expression.slot) {
            return Err(ArenaError::InvalidScopeProof);
        }
        Ok(())
    }

    /// Validate that an owned capability is authority for this exact normalization root before
    /// it is moved into the normalizer. This deliberately rechecks every stable component rather
    /// than accepting program equality alone: foreign arenas, replaced signatures, a proof for a
    /// different root, and a root absent from the proven reachable set all fail closed.
    pub(crate) fn validate_scope_proof_for_root(
        &self,
        proof: &ScopeProof,
        program: ValueProgramId,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        self.validate_scope_proof(proof)?;
        self.check_id(root)?;
        if proof.program != program {
            return Err(ArenaError::ScopeMismatch { expected: proof.program, actual: program });
        }
        if proof.root != root || !proof.reachable.contains(&root.slot) {
            return Err(ArenaError::InvalidScopeProof);
        }
        Ok(())
    }

    fn validate_scope_proof(&self, proof: &ScopeProof) -> Result<(), ArenaError> {
        if proof.arena != self.token {
            return Err(ArenaError::ForeignExpression { expected: self.token, actual: proof.arena });
        }
        self.check_id(proof.root)?;
        let Some(signature) = self.program_signatures.get(&proof.program) else {
            return Err(ArenaError::InvalidScopeProof);
        };
        if signature != &proof.signature {
            return Err(ArenaError::InvalidScopeProof);
        }
        Ok(())
    }

    fn validate_free_arguments_for_signature(
        &self,
        signature: &ProgramSignature,
        root: ExprId,
    ) -> Result<(), ArenaError> {
        for (position, actual) in self.free_arguments(root)? {
            let Some(input) = signature.inputs.get(position as usize) else {
                return Err(ArenaError::FreeArgumentEscapes { position });
            };
            if actual != input.value_type {
                return Err(ArenaError::TypeMismatch {
                    operator: "ProgramSignature".to_owned(),
                    position: position as usize,
                    expected: input.value_type.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }

    pub fn close(&self, expression: ExprId) -> Result<ClosedExprId, ArenaError> {
        self.check_id(expression)?;
        if let Some((position, _)) = self.free_arguments(expression)?.first().cloned() {
            return Err(ArenaError::FreeArgumentEscapes { position });
        }
        Ok(ClosedExprId { expression })
    }

    pub fn free_arguments(
        &self,
        root: ExprId,
    ) -> Result<BTreeSet<(u32, ResolvedValueType)>, ArenaError> {
        Ok(self
            .free_argument_ids(root)?
            .into_iter()
            .map(|id| match self.node(id).expect("validated free argument") {
                ExprNode { operator: ValueOperator::Argument { position, value_type }, .. } => {
                    (*position, value_type.clone())
                }
                _ => unreachable!("free_argument_ids returns only arguments"),
            })
            .collect())
    }

    /// Return the exact argument nodes reachable from an expression.
    ///
    /// The positional view in [`Self::free_arguments`] remains useful for closed/manual
    /// programs.  Closure validation and beta reduction use these IDs so nested binders with the
    /// same displayed position cannot alias one another.
    pub(crate) fn free_argument_ids(&self, root: ExprId) -> Result<BTreeSet<ExprId>, ArenaError> {
        self.free_argument_queries.set(self.free_argument_queries.get().saturating_add(1));
        self.ensure_free_argument_summary(root)?;
        let slot = self.check_id(root)?;
        let summary = self
            .free_argument_cache
            .borrow()
            .get(slot)
            .cloned()
            .ok_or(ArenaError::InvalidSlot { slot: root.slot() })?;
        Ok(summary.into_ids())
    }

    /// Test whether a sorted exact-binder list intersects one cached free-argument summary.
    /// The compact `Closed`/`One`/`Many` representation is inspected directly, so this query
    /// performs no per-node set allocation. An unknown summary is populated by the same
    /// bottom-up cache builder used by [`Self::free_argument_ids`].
    pub(crate) fn free_argument_summary_intersects(
        &self,
        root: ExprId,
        sorted_sources: &[ExprId],
    ) -> Result<bool, ArenaError> {
        self.ensure_free_argument_summary(root)?;
        let slot = self.check_id(root)?;
        let cache = self.free_argument_cache.borrow();
        let summary = cache.get(slot).ok_or(ArenaError::InvalidSlot { slot: root.slot() })?;
        Ok(match summary {
            FreeArgumentSummary::Unknown | FreeArgumentSummary::Closed => false,
            FreeArgumentSummary::One(id) => sorted_sources.binary_search(id).is_ok(),
            FreeArgumentSummary::Many(ids) => {
                let mut summary_index = 0;
                let mut source_index = 0;
                while summary_index < ids.len() && source_index < sorted_sources.len() {
                    match ids[summary_index].cmp(&sorted_sources[source_index]) {
                        std::cmp::Ordering::Less => summary_index += 1,
                        std::cmp::Ordering::Greater => source_index += 1,
                        std::cmp::Ordering::Equal => return Ok(true),
                    }
                }
                false
            }
        })
    }

    fn ensure_free_argument_summary(&self, root: ExprId) -> Result<(), ArenaError> {
        let slot = self.check_id(root)?;
        let root_known =
            self.free_argument_cache.borrow().get(slot).is_some_and(FreeArgumentSummary::is_known);
        if root_known {
            self.free_argument_root_hits.set(self.free_argument_root_hits.get().saturating_add(1));
            return Ok(());
        }

        // The arena is a DAG, so a post-order work stack computes each unknown slot from already
        // summarized children.  A child which was summarized by an earlier query is consumed
        // directly, while every newly visited slot receives its own permanent cache entry.
        let mut work = vec![(root, false)];
        while let Some((id, expanded)) = work.pop() {
            let id_slot = self.check_id(id)?;
            let known = self
                .free_argument_cache
                .borrow()
                .get(id_slot)
                .is_some_and(FreeArgumentSummary::is_known);
            if known {
                if id != root {
                    self.free_argument_subtree_hits
                        .set(self.free_argument_subtree_hits.get().saturating_add(1));
                }
                continue;
            }

            let (is_argument, inputs) = {
                let node = self.node(id)?;
                (matches!(node.operator, ValueOperator::Argument { .. }), node.inputs.clone())
            };
            if !expanded {
                self.free_argument_nodes_visited
                    .set(self.free_argument_nodes_visited.get().saturating_add(1));
                if is_argument {
                    self.store_free_argument_summary(id_slot, FreeArgumentSummary::One(id));
                    continue;
                }
                work.push((id, true));
                for child in inputs.iter().rev().copied() {
                    let child_slot = self.check_id(child)?;
                    if self
                        .free_argument_cache
                        .borrow()
                        .get(child_slot)
                        .is_some_and(FreeArgumentSummary::is_known)
                    {
                        self.free_argument_subtree_hits
                            .set(self.free_argument_subtree_hits.get().saturating_add(1));
                    } else {
                        work.push((child, false));
                    }
                }
                continue;
            }

            let mut ids = Vec::new();
            for child in inputs {
                let child_slot = self.check_id(child)?;
                let summary = self
                    .free_argument_cache
                    .borrow()
                    .get(child_slot)
                    .cloned()
                    .ok_or(ArenaError::InvalidSlot { slot: child.slot() })?;
                if !summary.is_known() {
                    return Err(ArenaError::InvalidSlot { slot: child.slot() });
                }
                ids.extend(summary.into_ids());
            }
            ids.sort_unstable();
            ids.dedup();
            self.store_free_argument_summary(id_slot, FreeArgumentSummary::from_sorted_ids(ids));
        }
        if self.free_argument_cache.borrow().get(slot).is_some_and(FreeArgumentSummary::is_known) {
            Ok(())
        } else {
            Err(ArenaError::InvalidSlot { slot: root.slot() })
        }
    }

    fn store_free_argument_summary(&self, slot: usize, summary: FreeArgumentSummary) {
        debug_assert!(summary.is_known());
        let mut cache = self.free_argument_cache.borrow_mut();
        debug_assert!(matches!(cache.get(slot), Some(FreeArgumentSummary::Unknown)));
        cache[slot] = summary;
        self.free_argument_newly_summarized
            .set(self.free_argument_newly_summarized.get().saturating_add(1));
    }

    pub fn reachable_node_count(&self, root: ExprId) -> Result<usize, ArenaError> {
        self.check_id(root)?;
        let mut seen = BTreeSet::new();
        let mut work = vec![root];
        while let Some(id) = work.pop() {
            if !seen.insert(id.slot) {
                continue;
            }
            work.extend(self.node(id)?.inputs.iter().copied());
        }
        Ok(seen.len())
    }

    fn validate_operator(
        &self,
        operator: &ValueOperator,
        inputs: &[ExprId],
    ) -> Result<ResolvedValueType, ArenaError> {
        let types =
            inputs.iter().map(|id| self.value_type(*id).cloned()).collect::<Result<Vec<_>, _>>()?;
        let arity = |expected: usize| {
            if inputs.len() == expected {
                Ok(())
            } else {
                Err(ArenaError::InvalidArity {
                    operator: format!("{operator:?}"),
                    expected,
                    actual: inputs.len(),
                })
            }
        };
        let same = |position: usize, expected: &ResolvedValueType, actual: &ResolvedValueType| {
            if expected == actual {
                Ok(())
            } else {
                Err(ArenaError::TypeMismatch {
                    operator: format!("{operator:?}"),
                    position,
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
        };
        match operator {
            ValueOperator::Argument { position: _, value_type } => {
                arity(0)?;
                Ok(value_type.clone())
            }
            ValueOperator::Constant(constant) => {
                arity(0)?;
                let valid = matches!(
                    (&constant.value_type, &constant.value),
                    (ResolvedValueType::Bool, ConstantValue::Bool(_)) |
                        (ResolvedValueType::Int, ConstantValue::Int(_)) |
                        (ResolvedValueType::Real, ConstantValue::Real(_)) |
                        (ResolvedValueType::Bytes, ConstantValue::Bytes(_))
                );
                if valid {
                    Ok(constant.value_type.clone())
                } else {
                    Err(ArenaError::ProgramOutputMismatch)
                }
            }
            ValueOperator::Source(source) => {
                arity(0)?;
                Ok(source.value_type.clone())
            }
            ValueOperator::Sample { event: _, descriptor } => {
                arity(0)?;
                Ok(descriptor.output_type.clone())
            }
            ValueOperator::Sampler { event: _, operation } => {
                arity(0)?;
                self.validate_sampler(operation)
            }
            ValueOperator::DeterministicHash(descriptor) => {
                let counts = [
                    descriptor.binary_tag_count,
                    descriptor.decimal_tag_count,
                    descriptor.u64_le_tag_count,
                    descriptor.dynamic_tag_count,
                ];
                let expected = counts.into_iter().try_fold(1usize, |total, count| {
                    total.checked_add(count as usize).ok_or(ArenaError::InvalidArity {
                        operator: "deterministic-hash".to_owned(),
                        expected: usize::MAX,
                        actual: inputs.len(),
                    })
                })?;
                arity(expected)?;
                same(0, &ResolvedValueType::Bytes, &types[0])?;
                for (position, actual) in types.iter().enumerate().skip(1) {
                    same(position, &ResolvedValueType::Int, actual)?;
                }
                if descriptor.version == 0 || descriptor.key_byte_length != 32 {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                Ok(ResolvedValueType::Matrix(descriptor.output.clone()))
            }
            ValueOperator::OpaqueFamilyElement { source } => {
                arity(1)?;
                same(0, &ResolvedValueType::Int, &types[0])?;
                Ok(source.element_type.clone())
            }
            ValueOperator::IndexMap { definition, parameters: _ } => {
                let definition = self
                    .index_definitions
                    .get(definition)
                    .ok_or(ArenaError::UnknownIndexFunction(*definition))?;
                if inputs.len() != definition.arity {
                    return Err(ArenaError::IndexArity {
                        expected: definition.arity,
                        actual: inputs.len(),
                    });
                }
                for (position, input) in types.iter().enumerate() {
                    same(position, &ResolvedValueType::Int, input)?;
                }
                Ok(definition.output_type.clone())
            }
            ValueOperator::ExplicitElement { domain, element_type } => {
                domain.nonempty()?;
                let width = domain.maximum_exclusive.checked_sub(domain.minimum).ok_or(
                    ArenaError::InvalidDomainWidth {
                        minimum: domain.minimum,
                        maximum_exclusive: domain.maximum_exclusive,
                    },
                )?;
                let width = usize::try_from(width).map_err(|_| ArenaError::InvalidDomainWidth {
                    minimum: domain.minimum,
                    maximum_exclusive: domain.maximum_exclusive,
                })?;
                let expected = width.checked_add(1).ok_or(ArenaError::InvalidDomainWidth {
                    minimum: domain.minimum,
                    maximum_exclusive: domain.maximum_exclusive,
                })?;
                arity(expected)?;
                same(0, &ResolvedValueType::Int, &types[0])?;
                for (position, actual) in types.iter().enumerate().skip(1) {
                    same(position, element_type, actual)?;
                }
                Ok(element_type.clone())
            }
            ValueOperator::ProgramCall { program } => {
                let signature = self
                    .program_signatures
                    .get(program)
                    .ok_or(ArenaError::UnknownProgram(*program))?;
                if inputs.len() != signature.inputs.len() {
                    return Err(ArenaError::InvalidArity {
                        operator: format!("{operator:?}"),
                        expected: signature.inputs.len(),
                        actual: inputs.len(),
                    });
                }
                for (position, (input, expected)) in types.iter().zip(&signature.inputs).enumerate()
                {
                    same(position, &expected.value_type, input)?;
                }
                Ok(signature.output.clone())
            }
            ValueOperator::Transform(operation) => self.validate_transform(operation, &types),
            ValueOperator::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
                arity(1)?;
                let Some(ResolvedValueType::Matrix(matrix)) = types.first() else {
                    return Err(ArenaError::TypeMismatch {
                        operator: format!("{operator:?}"),
                        position: 0,
                        expected: ResolvedValueType::Matrix(ResolvedMatrixType::new(
                            BigUint::from(1_u8),
                            1,
                            1,
                            1,
                        )?),
                        actual: types.first().cloned().unwrap_or(ResolvedValueType::Bool),
                    });
                };
                // The scalar matrix stores one polynomial, so `position` addresses a
                // coefficient in that polynomial rather than a flattened matrix cell.
                // Keep this contract aligned with the IR validator: extraction is valid only
                // for a 1x1 matrix and only within the ring dimension.
                let position =
                    usize::try_from(*position).map_err(|_| ArenaError::ProgramOutputMismatch)?;
                if matrix.rows != 1 || matrix.columns != 1 || position >= matrix.ring_dimension {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                if let Some(upper) = canonical_input_exclusive_upper {
                    if upper.is_zero() || upper > &matrix.modulus {
                        return Err(ArenaError::ProgramOutputMismatch);
                    }
                }
                Ok(ResolvedValueType::Int)
            }
            ValueOperator::Scalar(operation) => self.validate_scalar(operation, &types),
            ValueOperator::Matrix(operation) => self.validate_matrix(operation, &types),
            ValueOperator::Trapdoor(operation) => self.validate_trapdoor(operation, &types),
        }
    }

    fn validate_sampler(
        &self,
        operation: &SamplerOperation,
    ) -> Result<ResolvedValueType, ArenaError> {
        let output = match operation {
            SamplerOperation::UniformResidue { output } => output,
            SamplerOperation::UniformInterval { output, minimum, maximum } => {
                if minimum > maximum {
                    return Err(ArenaError::InvalidSamplerRange);
                }
                output
            }
            SamplerOperation::Gaussian { output, sigma, .. } => {
                if sigma.is_empty() {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                output
            }
            SamplerOperation::Hash { output, base, digit_count, .. } => {
                if base.is_some_and(|base| base < 2) || digit_count.is_some_and(|count| count == 0)
                {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                output
            }
            SamplerOperation::Trapdoor { output, sigma, gadget_base, digit_count, .. } => {
                if sigma.is_empty() || *gadget_base < 2 || *digit_count == 0 {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                output
            }
            SamplerOperation::Preimage { output, .. } => output,
        };
        Ok(ResolvedValueType::Matrix(output.clone()))
    }

    fn validate_transform(
        &self,
        operation: &ValueTransformOperation,
        types: &[ResolvedValueType],
    ) -> Result<ResolvedValueType, ArenaError> {
        match operation {
            ValueTransformOperation::GadgetDecompose { output, base, digit_count, .. } => {
                if types.len() != 1 {
                    return Err(ArenaError::InvalidArity {
                        operator: format!("{operation:?}"),
                        expected: 1,
                        actual: types.len(),
                    });
                }
                if *base < 2 || *digit_count == 0 {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                let ResolvedValueType::Matrix(input) = &types[0] else {
                    return Err(ArenaError::TypeMismatch {
                        operator: format!("{operation:?}"),
                        position: 0,
                        expected: ResolvedValueType::Matrix(output.clone()),
                        actual: types[0].clone(),
                    });
                };
                if input.same_ring(output) == false || input.columns != output.columns {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                let Some(rows) = input.rows.checked_mul(*digit_count as usize) else {
                    return Err(ArenaError::InvalidMatrixType);
                };
                if rows != output.rows {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
            ValueTransformOperation::PackPolynomialCoefficients { output, coefficient_bits } => {
                if *coefficient_bits == 0 {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                let expected = output
                    .rows
                    .checked_mul(output.columns)
                    .and_then(|count| count.checked_mul(*coefficient_bits as usize))
                    .ok_or(ArenaError::InvalidMatrixType)?;
                if types.len() != expected {
                    return Err(ArenaError::InvalidArity {
                        operator: format!("{operation:?}"),
                        expected,
                        actual: types.len(),
                    });
                }
                for (position, actual) in types.iter().enumerate() {
                    if *actual != ResolvedValueType::Bool {
                        return Err(ArenaError::TypeMismatch {
                            operator: format!("{operation:?}"),
                            position,
                            expected: ResolvedValueType::Bool,
                            actual: actual.clone(),
                        });
                    }
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
        }
    }

    fn validate_scalar(
        &self,
        operation: &ScalarOperation,
        types: &[ResolvedValueType],
    ) -> Result<ResolvedValueType, ArenaError> {
        let operator = format!("{operation:?}");
        let arity = |expected| {
            if types.len() == expected {
                Ok(())
            } else {
                Err(ArenaError::InvalidArity {
                    operator: operator.clone(),
                    expected,
                    actual: types.len(),
                })
            }
        };
        match operation {
            ScalarOperation::Add |
            ScalarOperation::Subtract |
            ScalarOperation::Multiply |
            ScalarOperation::Divide => {
                arity(2)?;
                if types[0] != types[1] ||
                    !matches!(types[0], ResolvedValueType::Int | ResolvedValueType::Real)
                {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                Ok(types[0].clone())
            }
            ScalarOperation::Remainder => {
                arity(2)?;
                if !matches!(types[0], ResolvedValueType::Int) || types[0] != types[1] {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 1,
                        expected: ResolvedValueType::Int,
                        actual: types[1].clone(),
                    });
                }
                Ok(ResolvedValueType::Int)
            }
            ScalarOperation::Negate => {
                arity(1)?;
                if !matches!(types[0], ResolvedValueType::Int | ResolvedValueType::Real) {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types[0].clone(),
                    });
                }
                Ok(types[0].clone())
            }
            ScalarOperation::Bit { .. } | ScalarOperation::Slice { .. } => {
                arity(1)?;
                if types[0] != ResolvedValueType::Int {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types[0].clone(),
                    });
                }
                if let ScalarOperation::Slice { start, end_exclusive } = operation {
                    if start > end_exclusive {
                        return Err(ArenaError::InvalidRange {
                            minimum: *start,
                            maximum_exclusive: *end_exclusive,
                        });
                    }
                }
                Ok(ResolvedValueType::Int)
            }
            ScalarOperation::Equal | ScalarOperation::Less | ScalarOperation::LessEqual => {
                arity(2)?;
                if types[0] != ResolvedValueType::Int || types[0] != types[1] {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Bool)
            }
            ScalarOperation::BoolToInt => {
                arity(1)?;
                if types[0] != ResolvedValueType::Bool {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Bool,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Int)
            }
            ScalarOperation::IntToReal => {
                arity(1)?;
                if types[0] != ResolvedValueType::Int {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Real)
            }
            ScalarOperation::RealAdd |
            ScalarOperation::RealSubtract |
            ScalarOperation::RealMultiply |
            ScalarOperation::RealDivide => {
                arity(2)?;
                if types[0] != ResolvedValueType::Real || types[0] != types[1] {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Real,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Real)
            }
            ScalarOperation::RealSqrt => {
                arity(1)?;
                if types[0] != ResolvedValueType::Real {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Real,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Real)
            }
            ScalarOperation::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                arity(1)?;
                if plaintext_modulus.is_zero() || *length == 0 || types[0] != ResolvedValueType::Int
                {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                Ok(if *output_bool { ResolvedValueType::Bool } else { ResolvedValueType::Int })
            }
            ScalarOperation::Hash { .. } => {
                if types.iter().any(|value_type| {
                    !matches!(value_type, ResolvedValueType::Int | ResolvedValueType::Bytes)
                }) {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types.first().cloned().unwrap_or(ResolvedValueType::Bool),
                    });
                }
                Ok(ResolvedValueType::Int)
            }
            ScalarOperation::ExtractCoefficient { .. } => {
                arity(1)?;
                let Some(ResolvedValueType::Matrix(matrix)) = types.first() else {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Matrix(ResolvedMatrixType::new(
                            BigUint::from(1_u8),
                            1,
                            1,
                            1,
                        )?),
                        actual: types.first().cloned().unwrap_or(ResolvedValueType::Bool),
                    });
                };
                let ScalarOperation::ExtractCoefficient { row, column } = operation else {
                    unreachable!()
                };
                if *row as usize >= matrix.rows || *column as usize >= matrix.columns {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                Ok(ResolvedValueType::Int)
            }
            ScalarOperation::LiftConstantPolynomial { output, coefficient_bits } => {
                arity(1)?;
                if *coefficient_bits == 0 {
                    return Err(ArenaError::InvalidCoefficientBits {
                        coefficient_bits: *coefficient_bits,
                    });
                }
                if types[0] != ResolvedValueType::Int {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 0,
                        expected: ResolvedValueType::Int,
                        actual: types[0].clone(),
                    });
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
        }
    }

    fn validate_matrix(
        &self,
        operation: &MatrixOperation,
        types: &[ResolvedValueType],
    ) -> Result<ResolvedValueType, ArenaError> {
        let operator = format!("{operation:?}");
        // Lift consumes an integer and produces a matrix.  Dispatch it before
        // collecting matrix-only inputs; otherwise a valid lift is rejected
        // by the generic matrix conversion before its scalar contract runs.
        if let MatrixOperation::LiftConstantPolynomial { output, coefficient_bits } = operation {
            if types.len() != 1 {
                return Err(ArenaError::InvalidArity { operator, expected: 1, actual: types.len() });
            }
            if *coefficient_bits == 0 {
                return Err(ArenaError::InvalidCoefficientBits {
                    coefficient_bits: *coefficient_bits,
                });
            }
            if types[0] != ResolvedValueType::Int {
                return Err(ArenaError::TypeMismatch {
                    operator,
                    position: 0,
                    expected: ResolvedValueType::Int,
                    actual: types[0].clone(),
                });
            }
            return Ok(ResolvedValueType::Matrix(output.clone()));
        }
        if matches!(operation, MatrixOperation::Scale) {
            if types.len() != 2 {
                return Err(ArenaError::InvalidArity { operator, expected: 2, actual: types.len() });
            }
            let ResolvedValueType::Matrix(matrix) = &types[0] else {
                return Err(ArenaError::TypeMismatch {
                    operator,
                    position: 0,
                    expected: ResolvedValueType::Matrix(ResolvedMatrixType::new(
                        BigUint::from(1_u8),
                        1,
                        1,
                        1,
                    )?),
                    actual: types[0].clone(),
                });
            };
            if types[1] != ResolvedValueType::Int {
                return Err(ArenaError::TypeMismatch {
                    operator,
                    position: 1,
                    expected: ResolvedValueType::Int,
                    actual: types[1].clone(),
                });
            }
            return Ok(ResolvedValueType::Matrix(matrix.clone()));
        }
        if let MatrixOperation::IndexedSlice { output, layout } = operation {
            if types.len() != 5 {
                return Err(ArenaError::InvalidArity { operator, expected: 5, actual: types.len() });
            }
            let ResolvedValueType::Matrix(input) = &types[0] else {
                return Err(ArenaError::TypeMismatch {
                    operator,
                    position: 0,
                    expected: ResolvedValueType::Matrix(output.clone()),
                    actual: types[0].clone(),
                });
            };
            if types[1..].iter().any(|ty| ty != &ResolvedValueType::Int) {
                let position = types[1..]
                    .iter()
                    .position(|ty| ty != &ResolvedValueType::Int)
                    .map_or(1, |position| position + 1);
                return Err(ArenaError::TypeMismatch {
                    operator,
                    position,
                    expected: ResolvedValueType::Int,
                    actual: types[position].clone(),
                });
            }
            if !input.same_ring(output) {
                return Err(ArenaError::IncompatibleMatrixTypes);
            }
            layout.validate_for(output)?;
            return Ok(ResolvedValueType::Matrix(output.clone()));
        }
        let matrices = types
            .iter()
            .map(|value_type| match value_type {
                ResolvedValueType::Matrix(matrix) => Ok(matrix),
                actual => Err(ArenaError::TypeMismatch {
                    operator: operator.clone(),
                    position: 0,
                    expected: ResolvedValueType::Matrix(ResolvedMatrixType::new(
                        BigUint::from(1_u8),
                        1,
                        1,
                        1,
                    )?),
                    actual: actual.clone(),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arity = |expected| {
            if types.len() == expected {
                Ok(())
            } else {
                Err(ArenaError::InvalidArity {
                    operator: operator.clone(),
                    expected,
                    actual: types.len(),
                })
            }
        };
        match operation {
            MatrixOperation::Add | MatrixOperation::Subtract => {
                arity(2)?;
                if matrices[0] != matrices[1] {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                Ok(ResolvedValueType::Matrix(matrices[0].clone().clone()))
            }
            MatrixOperation::Negate => {
                arity(1)?;
                Ok(ResolvedValueType::Matrix(matrices[0].clone().clone()))
            }
            MatrixOperation::RingAutomorphism { index } => {
                arity(1)?;
                let matrix = matrices[0];
                if !matrix.ring_dimension.is_power_of_two() ||
                    *index == 0 ||
                    *index >= 2 * matrix.ring_dimension as u64 ||
                    index % 2 == 0
                {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                Ok(ResolvedValueType::Matrix(matrix.clone()))
            }
            MatrixOperation::Scale => {
                arity(2)?;
                if types[1] != ResolvedValueType::Int {
                    return Err(ArenaError::TypeMismatch {
                        operator,
                        position: 1,
                        expected: ResolvedValueType::Int,
                        actual: types[1].clone(),
                    });
                }
                Ok(ResolvedValueType::Matrix(matrices[0].clone().clone()))
            }
            MatrixOperation::Multiply => {
                arity(2)?;
                let (left, right) = (matrices[0], matrices[1]);
                if !left.same_ring(right) {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                // Graph-IR matrix multiplication follows the runtime contract: a 1x1
                // matrix is a ring scalar and broadcasts over the other operand.  Only
                // non-scalar operands require the ordinary inner-dimension rule.  Keeping
                // this rule in the arena (rather than special-casing a protocol node) ensures
                // every production graph and normal-form operation shares one type transfer.
                let output = if left.rows == 1 && left.columns == 1 {
                    right.clone()
                } else if right.rows == 1 && right.columns == 1 {
                    left.clone()
                } else if left.columns == right.rows {
                    ResolvedMatrixType {
                        modulus: left.modulus.clone(),
                        ring_dimension: left.ring_dimension,
                        rows: left.rows,
                        columns: right.columns,
                    }
                } else {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                };
                Ok(ResolvedValueType::Matrix(output))
            }
            MatrixOperation::Transpose => {
                arity(1)?;
                let matrix = matrices[0];
                Ok(ResolvedValueType::Matrix(ResolvedMatrixType {
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension,
                    rows: matrix.columns,
                    columns: matrix.rows,
                }))
            }
            MatrixOperation::Slice {
                row_start,
                row_end_exclusive,
                column_start,
                column_end_exclusive,
                layout,
            } => {
                arity(1)?;
                let matrix = matrices[0];
                if row_start > row_end_exclusive ||
                    column_start > column_end_exclusive ||
                    *row_end_exclusive > matrix.rows ||
                    *column_end_exclusive > matrix.columns ||
                    row_start == row_end_exclusive ||
                    column_start == column_end_exclusive
                {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                let output = ResolvedMatrixType {
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension,
                    rows: row_end_exclusive - row_start,
                    columns: column_end_exclusive - column_start,
                };
                layout.validate_for(&output)?;
                Ok(ResolvedValueType::Matrix(output))
            }
            MatrixOperation::View { output, layout } => {
                arity(1)?;
                if !output.same_ring(matrices[0]) {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                let input_elements = matrices[0]
                    .rows
                    .checked_mul(matrices[0].columns)
                    .ok_or(ArenaError::InvalidMatrixType)?;
                let output_elements =
                    output.rows.checked_mul(output.columns).ok_or(ArenaError::InvalidMatrixType)?;
                if input_elements != output_elements {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                layout.validate_for(output)?;
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
            MatrixOperation::Concat { axis, output, layout } => {
                if types.is_empty() {
                    return Err(ArenaError::InvalidArity { operator, expected: 1, actual: 0 });
                }
                layout.validate_for(output)?;
                for matrix in &matrices {
                    if !output.same_ring(matrix) {
                        return Err(ArenaError::IncompatibleMatrixTypes);
                    }
                }
                if *axis > 2 {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                if *axis == 0 {
                    let total_rows = matrices.iter().try_fold(0_usize, |total, matrix| {
                        total.checked_add(matrix.rows).ok_or_else(|| {
                            ArenaError::DimensionOverflow { operator: operator.clone() }
                        })
                    })?;
                    if matrices.iter().any(|matrix| matrix.columns != matrices[0].columns) ||
                        total_rows != output.rows ||
                        matrices[0].columns != output.columns
                    {
                        return Err(ArenaError::IncompatibleMatrixTypes);
                    }
                } else if *axis == 1 {
                    let total_columns = matrices.iter().try_fold(0_usize, |total, matrix| {
                        total.checked_add(matrix.columns).ok_or_else(|| {
                            ArenaError::DimensionOverflow { operator: operator.clone() }
                        })
                    })?;
                    if matrices.iter().any(|matrix| matrix.rows != matrices[0].rows) ||
                        total_columns != output.columns ||
                        matrices[0].rows != output.rows
                    {
                        return Err(ArenaError::IncompatibleMatrixTypes);
                    }
                } else if *axis == 2 {
                    let (total_rows, total_columns) = matrices.iter().try_fold(
                        (0_usize, 0_usize),
                        |(rows, columns), matrix| {
                            Ok((
                                rows.checked_add(matrix.rows).ok_or_else(|| {
                                    ArenaError::DimensionOverflow { operator: operator.clone() }
                                })?,
                                columns.checked_add(matrix.columns).ok_or_else(|| {
                                    ArenaError::DimensionOverflow { operator: operator.clone() }
                                })?,
                            ))
                        },
                    )?;
                    if total_rows != output.rows || total_columns != output.columns {
                        return Err(ArenaError::IncompatibleMatrixTypes);
                    }
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
            MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } => {
                arity(2)?;
                if !output.same_ring(matrices[0]) || !output.same_ring(matrices[1]) {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                left_layout.validate_for(matrices[0])?;
                right_layout.validate_for(matrices[1])?;
                output_layout.validate_for(output)?;
                let Some(rows) = matrices[0].rows.checked_mul(matrices[1].rows) else {
                    return Err(ArenaError::ProgramOutputMismatch);
                };
                let Some(columns) = matrices[0].columns.checked_mul(matrices[1].columns) else {
                    return Err(ArenaError::ProgramOutputMismatch);
                };
                if output.rows != rows || output.columns != columns {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
            MatrixOperation::CrtRecompose {
                plaintext_moduli,
                reconstruction_coefficients,
                output,
            } => {
                if plaintext_moduli.is_empty() ||
                    plaintext_moduli.len() != types.len() ||
                    reconstruction_coefficients.len() != plaintext_moduli.len()
                {
                    return Err(ArenaError::InvalidArity {
                        operator,
                        expected: plaintext_moduli.len(),
                        actual: types.len(),
                    });
                }
                let first = matrices[0];
                if output != first {
                    return Err(ArenaError::IncompatibleMatrixTypes);
                }
                for (position, modulus) in plaintext_moduli.iter().enumerate() {
                    if *modulus <= BigUint::from(1_u8) || *modulus >= first.modulus {
                        return Err(ArenaError::ProgramOutputMismatch);
                    }
                    let matrix = matrices[position];
                    if matrix != first {
                        return Err(ArenaError::IncompatibleMatrixTypes);
                    }
                }
                for coefficient in reconstruction_coefficients {
                    if coefficient.is_negative() ||
                        coefficient.to_biguint().is_some_and(|value| value >= first.modulus)
                    {
                        return Err(ArenaError::ProgramOutputMismatch);
                    }
                }
                Ok(ResolvedValueType::Matrix(output.clone()))
            }
            MatrixOperation::ExtractCoefficient { row, column } => {
                arity(1)?;
                let matrix = matrices[0];
                if *row as usize >= matrix.rows || *column as usize >= matrix.columns {
                    return Err(ArenaError::ProgramOutputMismatch);
                }
                Ok(ResolvedValueType::Int)
            }
            MatrixOperation::LiftConstantPolynomial { .. } |
            MatrixOperation::IndexedSlice { .. } => unreachable!("handled above"),
        }
    }

    fn validate_trapdoor(
        &self,
        operation: &TrapdoorOperation,
        types: &[ResolvedValueType],
    ) -> Result<ResolvedValueType, ArenaError> {
        match operation {
            TrapdoorOperation::Generate { descriptor, paired_public_output_role, .. } => {
                if !types.is_empty() {
                    return Err(ArenaError::InvalidArity {
                        operator: "Trapdoor::Generate".to_owned(),
                        expected: 0,
                        actual: types.len(),
                    });
                }
                if descriptor.is_empty() || paired_public_output_role.is_empty() {
                    return Err(ArenaError::ProgramSignatureMismatch);
                }
                Ok(ResolvedValueType::Trapdoor)
            }
            TrapdoorOperation::Transform { output, .. } => {
                if types.len() != 1 || types[0] != ResolvedValueType::Trapdoor {
                    return Err(ArenaError::TypeMismatch {
                        operator: "Trapdoor::Transform".to_owned(),
                        position: 0,
                        expected: ResolvedValueType::Trapdoor,
                        actual: types.first().cloned().unwrap_or(ResolvedValueType::Bool),
                    });
                }
                Ok(output.clone())
            }
        }
    }
}

impl ExactSubstitutionEngine {
    pub(crate) fn new() -> Self {
        Self { completed: BTreeMap::new(), scratch: Vec::new(), epoch: 0 }
    }

    /// Apply an exact source-to-replacement map to an immutable expression DAG.
    ///
    /// All IDs and source/replacement types are checked before the completed-result cache is
    /// consulted. A cold traversal is iterative and follows child order deterministically;
    /// unmatched arguments retain their exact node identity. Only an all-argument map may use
    /// the free-binder disjointness shortcut, because an arbitrary source can occur below a
    /// non-argument operator without appearing in that summary.
    pub(crate) fn substitute(
        &mut self,
        expressions: &mut ExprArena,
        root: ExprId,
        replacements: &BTreeMap<ExprId, ExprId>,
    ) -> Result<ExactSubstitutionResult, ExactSubstitutionFailure> {
        expressions.check_id(root).map_err(|source| ExactSubstitutionFailure {
            node: root,
            operator: None,
            expected_output: None,
            actual_inputs: Box::new([]),
            source,
        })?;

        // Validate every ID first. In particular, a stale or foreign replacement cannot become
        // an accidental completed-cache hit. Exact type equality is part of this contract even
        // when a source is not reachable from this root.
        let mut canonical = Vec::with_capacity(replacements.len());
        for (source_id, replacement_id) in replacements {
            expressions
                .check_id(*source_id)
                .map_err(|source| self.failure_for_id(expressions, *source_id, source))?;
            expressions
                .check_id(*replacement_id)
                .map_err(|source| self.failure_for_id(expressions, *source_id, source))?;
        }
        for (source_id, replacement_id) in replacements {
            let source_type = expressions
                .value_type(*source_id)
                .map_err(|source| self.failure_for_id(expressions, *source_id, source))?;
            let replacement_type = expressions
                .value_type(*replacement_id)
                .map_err(|source| self.failure_for_id(expressions, *source_id, source))?;
            if source_type != replacement_type {
                let source_operator =
                    expressions.node(*source_id).ok().map(|node| node.operator.clone());
                let source_error = ArenaError::TypeMismatch {
                    operator: "ExactSubstitution".to_owned(),
                    position: 0,
                    expected: source_type.clone(),
                    actual: replacement_type.clone(),
                };
                return Err(ExactSubstitutionFailure {
                    node: *source_id,
                    operator: source_operator,
                    expected_output: Some(source_type.clone()),
                    actual_inputs: Box::new([replacement_type.clone()]),
                    source: source_error,
                });
            }
            canonical.push((*source_id, *replacement_id));
        }

        let key = ExactSubstitutionKey { root, replacements: canonical.clone().into_boxed_slice() };
        if let Some(result) = self.completed.get(&key).copied() {
            return Ok(ExactSubstitutionResult {
                root: result,
                cache_hit: true,
                binder_shortcut: false,
                skipped_subtrees: 0,
                nodes_visited: 0,
                nodes_reinterned: 0,
            });
        }

        let all_arguments = canonical.iter().all(|(source_id, _)| {
            matches!(
                expressions.node(*source_id).map(|node| &node.operator),
                Ok(ValueOperator::Argument { .. })
            )
        });
        let argument_sources = canonical.iter().map(|(source, _)| *source).collect::<Vec<_>>();
        if all_arguments {
            let free = expressions
                .free_argument_ids(root)
                .map_err(|source| self.failure_for_id(expressions, root, source))?;
            if canonical.iter().all(|(source_id, _)| !free.contains(source_id)) {
                self.completed.insert(key, root);
                return Ok(ExactSubstitutionResult {
                    root,
                    cache_hit: false,
                    binder_shortcut: true,
                    skipped_subtrees: 0,
                    nodes_visited: 0,
                    nodes_reinterned: 0,
                });
            }
        }

        self.start_invocation();
        let mut stack = vec![(root, false)];
        let mut nodes_visited = 0_u64;
        let mut nodes_reinterned = 0_u64;
        let mut skipped_subtrees = 0_u64;
        while let Some((id, expanded)) = stack.pop() {
            let slot = expressions
                .check_id(id)
                .map_err(|source| self.failure_for_id(expressions, id, source))?;
            if self.scratch_value(slot).is_some() {
                continue;
            }
            if !expanded {
                let (is_argument, inputs) = {
                    let node = expressions
                        .node(id)
                        .map_err(|source| self.failure_for_id(expressions, id, source))?;
                    (matches!(node.operator, ValueOperator::Argument { .. }), node.inputs.clone())
                };
                if let Some((_, replacement)) = canonical.iter().find(|(source, _)| *source == id) {
                    nodes_visited = nodes_visited.saturating_add(1);
                    self.set_scratch(slot, *replacement);
                } else if all_arguments &&
                    !expressions
                        .free_argument_summary_intersects(id, &argument_sources)
                        .map_err(|source| self.failure_for_id(expressions, id, source))?
                {
                    self.set_scratch(slot, id);
                    skipped_subtrees = skipped_subtrees.saturating_add(1);
                } else if is_argument {
                    nodes_visited = nodes_visited.saturating_add(1);
                    // This includes fresh lexical binders, which must never be reconstructed as
                    // canonical positional arguments.
                    self.set_scratch(slot, id);
                } else {
                    nodes_visited = nodes_visited.saturating_add(1);
                    stack.push((id, true));
                    for child in inputs.iter().rev().copied() {
                        let child_slot = expressions
                            .check_id(child)
                            .map_err(|source| self.failure_for_id(expressions, child, source))?;
                        if self.scratch_value(child_slot).is_none() {
                            stack.push((child, false));
                        }
                    }
                }
                continue;
            }

            let (operator, children) = {
                let node = expressions
                    .node(id)
                    .map_err(|source| self.failure_for_id(expressions, id, source))?;
                (node.operator.clone(), node.inputs.clone())
            };
            let mut inputs = Vec::with_capacity(children.len());
            for child in children {
                let child_slot = expressions
                    .check_id(child)
                    .map_err(|source| self.failure_for_id(expressions, child, source))?;
                let value = self.scratch_value(child_slot).ok_or_else(|| {
                    self.failure_for_id(
                        expressions,
                        id,
                        ArenaError::InvalidSlot { slot: child.slot() },
                    )
                })?;
                inputs.push(value);
            }
            let expected_output = expressions
                .value_type(id)
                .map_err(|source| self.failure_for_id(expressions, id, source))?
                .clone();
            let actual_inputs = inputs
                .iter()
                .map(|input| expressions.value_type(*input).cloned())
                .collect::<Result<Box<[_]>, _>>()
                .map_err(|source| self.failure_for_id(expressions, id, source))?;
            let rewritten = expressions
                .intern(operator.clone(), inputs.into_boxed_slice())
                .map_err(|source| ExactSubstitutionFailure {
                    node: id,
                    operator: Some(operator.clone()),
                    expected_output: Some(expected_output.clone()),
                    actual_inputs: actual_inputs.clone(),
                    source,
                })?;
            nodes_reinterned = nodes_reinterned.saturating_add(1);
            self.set_scratch(slot, rewritten);
        }

        let result = self
            .scratch_value(
                expressions
                    .check_id(root)
                    .map_err(|source| self.failure_for_id(expressions, root, source))?,
            )
            .ok_or_else(|| {
                self.failure_for_id(
                    expressions,
                    root,
                    ArenaError::InvalidSlot { slot: root.slot() },
                )
            })?;
        self.completed.insert(key, result);
        Ok(ExactSubstitutionResult {
            root: result,
            cache_hit: false,
            binder_shortcut: false,
            skipped_subtrees,
            nodes_visited,
            nodes_reinterned,
        })
    }

    fn failure_for_id(
        &self,
        expressions: &ExprArena,
        node: ExprId,
        source: ArenaError,
    ) -> ExactSubstitutionFailure {
        ExactSubstitutionFailure {
            node,
            operator: expressions.node(node).ok().map(|node| node.operator.clone()),
            expected_output: expressions.value_type(node).ok().cloned(),
            actual_inputs: Box::new([]),
            source,
        }
    }

    fn start_invocation(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            for entry in &mut self.scratch {
                entry.epoch = 0;
            }
        }
    }

    fn scratch_value(&self, slot: usize) -> Option<ExprId> {
        self.scratch
            .get(slot)
            .filter(|entry| entry.epoch == self.epoch)
            .and_then(|entry| entry.value)
    }

    fn set_scratch(&mut self, slot: usize, value: ExprId) {
        if self.scratch.len() <= slot {
            self.scratch.resize(slot + 1, SubstitutionScratchEntry::default());
        }
        self.scratch[slot] = SubstitutionScratchEntry { epoch: self.epoch, value: Some(value) };
    }
}

/// A validated closed expression view.  It is impossible to construct one
/// from an open expression without going through [`ExprArena::close`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClosedExprId {
    expression: ExprId,
}

impl ClosedExprId {
    pub(crate) const fn expression(self) -> ExprId {
        self.expression
    }
}

/// A crate-private proof that one expression root has been checked against
/// one registered program signature.  Its fields intentionally remain private
/// and it has no `Clone`/`Copy` implementation, so a raw ID cannot be promoted
/// into a scope from outside the arena API.
pub(crate) struct ScopeProof {
    arena: ArenaToken,
    program: ValueProgramId,
    signature: ProgramSignature,
    root: ExprId,
    reachable: HashSet<u32>,
}

/// An expression paired with the finalized program that binds its arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScopedExprId {
    program: ValueProgramId,
    expression: ExprId,
}

impl ScopedExprId {
    pub(crate) const fn program(self) -> ValueProgramId {
        self.program
    }

    pub(crate) const fn expression(self) -> ExprId {
        self.expression
    }

    pub(crate) const fn with_expression(self, expression: ExprId) -> Self {
        Self { program: self.program, expression }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProgramInput {
    pub value_type: ResolvedValueType,
    pub trusted_index_range: Option<TrustedIndexRange>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProgramSignature {
    pub inputs: Box<[ProgramInput]>,
    pub output: ResolvedValueType,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueProgram {
    pub signature: ProgramSignature,
    pub root: ExprId,
    /// Exact formal argument nodes for generated lexical closures.  Ordinary programs have no
    /// generated binder and keep this empty.
    pub(crate) formal_arguments: Box<[ExprId]>,
    pub(crate) generated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_local_ids_remain_compact() {
        assert_eq!(std::mem::size_of::<ArenaToken>(), 4);
        assert_eq!(std::mem::size_of::<ExprId>(), 8);
        assert_eq!(std::mem::size_of::<ValueProgramId>(), 8);
        assert_eq!(std::mem::size_of::<ScopedExprId>(), 16);
    }
    use crate::operational_noise::program::ProgramArena;

    fn matrix() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(17_u8), 8, 2, 2).unwrap()
    }

    fn scalar_ring_matrix() -> ResolvedMatrixType {
        ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 1).unwrap()
    }

    #[test]
    fn repeated_complete_keys_reuse_and_ordered_inputs_differ() {
        let mut arena = ExprArena::new();
        let left =
            arena.intern(ValueOperator::Constant(TypedConstant::int(2)), Box::new([])).unwrap();
        let right =
            arena.intern(ValueOperator::Constant(TypedConstant::int(3)), Box::new([])).unwrap();
        let first = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[left, right])
            .unwrap();
        let second = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[left, right])
            .unwrap();
        let swapped = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[right, left])
            .unwrap();
        assert_eq!(first, second);
        assert_ne!(first, swapped);
    }

    #[test]
    fn foreign_arena_is_rejected_before_slot_inspection() {
        let mut first = ExprArena::new();
        let mut second = ExprArena::new();
        let foreign =
            second.intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([])).unwrap();
        let error = first
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[foreign])
            .unwrap_err();
        assert!(matches!(error, ArenaError::ForeignExpression { .. }));
    }

    #[test]
    fn independent_samples_remain_distinct() {
        let mut arena = ExprArena::new();
        let descriptor = SampleDescriptor::new("uniform", ResolvedValueType::Int);
        let first = arena
            .intern(
                ValueOperator::Sample { event: SampleEventId(1), descriptor: descriptor.clone() },
                Box::new([]),
            )
            .unwrap();
        let second = arena
            .intern(ValueOperator::Sample { event: SampleEventId(2), descriptor }, Box::new([]))
            .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn argument_is_alpha_ready_and_scope_is_not_expression_identity() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let one = expressions
            .intern(ValueOperator::Constant(TypedConstant::int(1)), Box::new([]))
            .unwrap();
        let root = expressions
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[argument, one])
            .unwrap();
        let signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        let first = programs.finalize(&mut expressions, signature.clone(), root).unwrap();
        let second = programs.finalize(&mut expressions, signature, root).unwrap();
        assert_eq!(first, second);
        assert!(expressions.close(argument).is_err());
        assert!(programs.scoped(&expressions, first, argument).is_ok());
    }

    #[test]
    fn argument_position_is_scoped_by_type_and_same_signature_still_reuses() {
        let mut expressions = ExprArena::new();
        let int_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let matrix_argument =
            expressions.intern_argument(0, ResolvedValueType::Matrix(matrix())).unwrap();
        assert_ne!(int_argument, matrix_argument);

        let int_signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let matrix_signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Matrix(matrix()),
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Matrix(matrix()),
        };
        let mut programs = ProgramArena::new();
        let int_first =
            programs.finalize(&mut expressions, int_signature.clone(), int_argument).unwrap();
        let int_second = programs.finalize(&mut expressions, int_signature, int_argument).unwrap();
        let matrix_program =
            programs.finalize(&mut expressions, matrix_signature, matrix_argument).unwrap();
        assert_eq!(int_first, int_second);
        assert_ne!(int_first, matrix_program);
    }

    #[test]
    fn free_argument_validation_uses_only_reachable_position_and_type() {
        let mut expressions = ExprArena::new();
        let bool_argument = expressions.intern_argument(0, ResolvedValueType::Bool).unwrap();
        let int_argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let int_signature = ProgramSignature {
            inputs: Box::new([ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        assert!(programs.finalize(&mut expressions, int_signature.clone(), int_argument).is_ok());
        let bool_as_int = expressions
            .intern_slice(ValueOperator::Scalar(ScalarOperation::BoolToInt), &[bool_argument])
            .unwrap();
        assert!(matches!(
            programs.finalize(&mut expressions, int_signature, bool_as_int),
            Err(ArenaError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn deep_shared_dag_is_iterative_and_diamond_is_linear() {
        let mut arena = ExprArena::new();
        let mut current =
            arena.intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([])).unwrap();
        for _ in 0..4_096 {
            current = arena
                .intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[current])
                .unwrap();
        }
        assert_eq!(arena.reachable_node_count(current).unwrap(), 4_097);
        let shared = current;
        let left =
            arena.intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[shared]).unwrap();
        let right =
            arena.intern_slice(ValueOperator::Scalar(ScalarOperation::Negate), &[shared]).unwrap();
        let root = arena
            .intern_slice(ValueOperator::Scalar(ScalarOperation::Add), &[left, right])
            .unwrap();
        // The two equal negations are hash-consed into one node, so the
        // shared diamond has one root plus one shared branch and remains
        // linear in the depth rather than duplicating the branch.
        assert_eq!(arena.reachable_node_count(root).unwrap(), 4_099);
    }

    #[test]
    fn matrix_validation_checks_complete_ring_and_shape() {
        let mut arena = ExprArena::new();
        let value = |arena: &mut ExprArena| {
            arena
                .intern(
                    ValueOperator::Source(SemanticSourceIdentity {
                        stable_definition: "m".to_owned(),
                        invocation: "0".to_owned(),
                        sample_event: None,
                        output_role: "value".to_owned(),
                        sampler: None,
                        artifact: None,
                        value_type: ResolvedValueType::Matrix(matrix()),
                        coordinates: Box::new([]),
                        matrix_constant: None,
                    }),
                    Box::new([]),
                )
                .unwrap()
        };
        let left = value(&mut arena);
        let right = value(&mut arena);
        let product = arena
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Multiply), &[left, right])
            .unwrap();
        assert_eq!(arena.value_type(product).unwrap(), &ResolvedValueType::Matrix(matrix()));
    }

    #[test]
    fn value_coefficient_extraction_uses_ring_dimension_for_scalar_polynomials() {
        let mut arena = ExprArena::new();
        let scalar = arena
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: "scalar-ring".to_owned(),
                    invocation: "0".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(scalar_ring_matrix()),
                    coordinates: Box::new([]),
                    matrix_constant: None,
                }),
                Box::new([]),
            )
            .unwrap();

        let last = arena.intern_extract_coefficient(scalar, 7, Some(BigUint::from(17_u8)));
        assert!(last.is_ok(), "the final ring coefficient must be addressable");

        let past_end = arena.intern_extract_coefficient(scalar, 8, Some(BigUint::from(17_u8)));
        assert!(matches!(past_end, Err(ArenaError::ProgramOutputMismatch)));

        let non_scalar = arena
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: "non-scalar".to_owned(),
                    invocation: "0".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix()),
                    coordinates: Box::new([]),
                    matrix_constant: None,
                }),
                Box::new([]),
            )
            .unwrap();
        let rejected = arena.intern_extract_coefficient(non_scalar, 0, Some(BigUint::from(17_u8)));
        assert!(matches!(rejected, Err(ArenaError::ProgramOutputMismatch)));
    }

    #[test]
    fn family_element_identity_includes_exact_index_and_explicit_inputs() {
        let mut arena = ExprArena::new();
        let source = SemanticFamilySourceIdentity {
            stable_definition: "family".to_owned(),
            invocation: "call".to_owned(),
            element_type: ResolvedValueType::Int,
            domain: FamilyDomain::new(0, 2).unwrap(),
            artifact: None,
        };
        let x = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let y = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let fx = arena
            .intern(ValueOperator::OpaqueFamilyElement { source: source.clone() }, [x].into())
            .unwrap();
        let fy = arena.intern(ValueOperator::OpaqueFamilyElement { source }, [y].into()).unwrap();
        assert_ne!(fx, fy, "F(x) and F(y) must retain exact runtime identity");

        let zero = arena.intern(ValueOperator::Constant(TypedConstant::int(7)), [].into()).unwrap();
        let one = arena.intern(ValueOperator::Constant(TypedConstant::int(8)), [].into()).unwrap();
        let explicit = arena
            .intern(
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 2).unwrap(),
                    element_type: ResolvedValueType::Int,
                },
                [x, zero, one].into(),
            )
            .unwrap();
        assert_eq!(arena.value_type(explicit).unwrap(), &ResolvedValueType::Int);
        let wrong_count = arena.intern(
            ValueOperator::ExplicitElement {
                domain: FamilyDomain::new(0, 2).unwrap(),
                element_type: ResolvedValueType::Int,
            },
            [x, zero].into(),
        );
        assert!(matches!(wrong_count, Err(ArenaError::InvalidArity { .. })));
    }

    #[test]
    fn lift_dispatches_scalar_input_before_matrix_conversion() {
        let mut arena = ExprArena::new();
        let scalar =
            arena.intern(ValueOperator::Constant(TypedConstant::int(3)), [].into()).unwrap();
        let lifted = arena
            .intern(
                ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                    output: matrix(),
                    coefficient_bits: 4,
                }),
                [scalar].into(),
            )
            .unwrap();
        assert_eq!(arena.value_type(lifted).unwrap(), &ResolvedValueType::Matrix(matrix()));
    }

    #[test]
    fn program_arena_rejects_foreign_callee_and_wrong_free_argument_type() {
        let mut expressions = ExprArena::new();
        let argument = expressions.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let mut first = ProgramArena::new();
        let callee = first.finalize(&mut expressions, signature.clone(), argument).unwrap();
        let constant =
            expressions.intern(ValueOperator::Constant(TypedConstant::int(1)), [].into()).unwrap();
        let call = expressions
            .intern(ValueOperator::ProgramCall { program: callee }, [constant].into())
            .unwrap();
        let mut second = ProgramArena::new();
        let error = second.finalize(&mut expressions, signature.clone(), call).unwrap_err();
        assert!(matches!(error, ArenaError::ForeignProgram { .. }));

        let mut wrong = ProgramArena::new();
        let wrong_signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Bool,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let error = wrong.finalize(&mut expressions, wrong_signature, argument).unwrap_err();
        assert!(matches!(error, ArenaError::TypeMismatch { .. }));
    }

    #[test]
    fn layouts_and_shape_contracts_are_checked() {
        let mut arena = ExprArena::new();
        let source = |arena: &mut ExprArena, value_type: ResolvedValueType| {
            arena
                .intern(
                    ValueOperator::Source(SemanticSourceIdentity {
                        stable_definition: "layout".to_owned(),
                        invocation: format!("{}", arena.node_count()),
                        sample_event: None,
                        output_role: "value".to_owned(),
                        sampler: None,
                        artifact: None,
                        value_type,
                        coordinates: [].into(),
                        matrix_constant: None,
                    }),
                    [].into(),
                )
                .unwrap()
        };
        let input = source(&mut arena, ResolvedValueType::Matrix(matrix()));
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 4).unwrap();
        let viewed = arena
            .intern(
                ValueOperator::Matrix(MatrixOperation::View {
                    output: output.clone(),
                    layout: MatrixLayout::row_major(1, 4),
                }),
                [input].into(),
            )
            .unwrap();
        assert_eq!(arena.value_type(viewed).unwrap(), &ResolvedValueType::Matrix(output));
        let invalid = arena.intern(
            ValueOperator::Matrix(MatrixOperation::View {
                output: ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 3).unwrap(),
                layout: MatrixLayout { name: "bad".to_owned(), row_stride: 0, column_stride: 1 },
            }),
            [input].into(),
        );
        assert!(matches!(
            invalid,
            Err(ArenaError::IncompatibleMatrixTypes | ArenaError::InvalidLayout)
        ));

        let aliased = arena.intern(
            ValueOperator::Matrix(MatrixOperation::View {
                output: ResolvedMatrixType::new(BigUint::from(17_u8), 8, 2, 2).unwrap(),
                layout: MatrixLayout {
                    name: "aliased".to_owned(),
                    row_stride: 1,
                    column_stride: 1,
                },
            }),
            [input].into(),
        );
        assert!(matches!(aliased, Err(ArenaError::InvalidLayout)));
    }

    #[test]
    fn production_value_vocabulary_accepts_remaining_typed_operations() {
        let mut arena = ExprArena::new();
        let int = |arena: &mut ExprArena, value| {
            arena.intern(ValueOperator::Constant(TypedConstant::int(value)), [].into()).unwrap()
        };
        let a = int(&mut arena, 3);
        let b = int(&mut arena, 2);
        for operation in [
            ScalarOperation::Remainder,
            ScalarOperation::Equal,
            ScalarOperation::Less,
            ScalarOperation::LessEqual,
        ] {
            let id = arena.intern_slice(ValueOperator::Scalar(operation), &[a, b]).unwrap();
            assert!(matches!(
                arena.value_type(id).unwrap(),
                ResolvedValueType::Int | ResolvedValueType::Bool
            ));
        }
        let real =
            arena.intern(ValueOperator::Constant(TypedConstant::real("4")), [].into()).unwrap();
        let sqrt =
            arena.intern_slice(ValueOperator::Scalar(ScalarOperation::RealSqrt), &[real]).unwrap();
        assert_eq!(arena.value_type(sqrt).unwrap(), &ResolvedValueType::Real);
        let matrix_id = arena
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: "m".to_owned(),
                    invocation: "0".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix()),
                    coordinates: [].into(),
                    matrix_constant: None,
                }),
                [].into(),
            )
            .unwrap();
        let scaled = arena
            .intern_slice(ValueOperator::Matrix(MatrixOperation::Scale), &[matrix_id, a])
            .unwrap();
        assert_eq!(arena.value_type(scaled).unwrap(), &ResolvedValueType::Matrix(matrix()));
        let packed = arena.intern(
            ValueOperator::Transform(ValueTransformOperation::PackPolynomialCoefficients {
                output: matrix(),
                coefficient_bits: 1,
            }),
            [a].into(),
        );
        assert!(packed.is_err(), "packing must reject non-boolean inputs");
    }

    #[test]
    fn typed_transform_interning_reuses_full_keys_and_supports_all_matrix_transforms() {
        let mut arena = ExprArena::new();
        let source = |arena: &mut ExprArena, definition: &str, value_type: ResolvedValueType| {
            arena
                .intern(
                    ValueOperator::Source(SemanticSourceIdentity {
                        stable_definition: definition.to_owned(),
                        invocation: format!("{}-{}", definition, arena.node_count()),
                        sample_event: None,
                        output_role: "value".to_owned(),
                        sampler: None,
                        artifact: None,
                        value_type,
                        coordinates: [].into(),
                        matrix_constant: None,
                    }),
                    [].into(),
                )
                .unwrap()
        };
        let two_by_two = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 2, 2).unwrap();
        let one_by_four = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 4).unwrap();
        let input = source(&mut arena, "input", ResolvedValueType::Matrix(two_by_two.clone()));
        let other = source(&mut arena, "other", ResolvedValueType::Matrix(two_by_two.clone()));
        let slice = MatrixOperation::Slice {
            row_start: 0,
            row_end_exclusive: 1,
            column_start: 0,
            column_end_exclusive: 2,
            layout: MatrixLayout::row_major(1, 2),
        };
        let first = arena.intern_matrix_transform(slice.clone(), &[input]).unwrap();
        let repeated = arena.intern_matrix_transform(slice.clone(), &[input]).unwrap();
        assert_eq!(first, repeated, "the complete transform key is hash-consed");
        let different_range = arena
            .intern_matrix_transform(
                MatrixOperation::Slice {
                    row_start: 1,
                    row_end_exclusive: 2,
                    column_start: 0,
                    column_end_exclusive: 2,
                    layout: MatrixLayout::row_major(1, 2),
                },
                &[input],
            )
            .unwrap();
        let different_operand = arena.intern_matrix_transform(slice, &[other]).unwrap();
        assert_ne!(first, different_range);
        assert_ne!(first, different_operand);

        let viewed = arena
            .intern_matrix_transform(
                MatrixOperation::View {
                    output: one_by_four.clone(),
                    layout: MatrixLayout::row_major(1, 4),
                },
                &[input],
            )
            .unwrap();
        let concat = arena
            .intern_matrix_transform(
                MatrixOperation::Concat {
                    axis: 0,
                    output: two_by_two.clone(),
                    layout: MatrixLayout::row_major(2, 2),
                },
                &[first, first],
            )
            .unwrap();
        let tensor = arena
            .intern_matrix_transform(
                MatrixOperation::Tensor {
                    output: one_by_four,
                    left_layout: MatrixLayout::row_major(1, 2),
                    right_layout: MatrixLayout::row_major(1, 2),
                    output_layout: MatrixLayout::row_major(1, 4),
                },
                &[first, first],
            )
            .unwrap();
        let mod_fifteen = ResolvedMatrixType::new(BigUint::from(15_u8), 8, 1, 1).unwrap();
        let crt_left =
            source(&mut arena, "crt-left", ResolvedValueType::Matrix(mod_fifteen.clone()));
        let crt_right =
            source(&mut arena, "crt-right", ResolvedValueType::Matrix(mod_fifteen.clone()));
        let crt = arena
            .intern_matrix_transform(
                MatrixOperation::CrtRecompose {
                    plaintext_moduli: [BigUint::from(3_u8), BigUint::from(5_u8)].into(),
                    reconstruction_coefficients: [BigInt::from(2), BigInt::from(1)].into(),
                    output: mod_fifteen,
                },
                &[crt_left, crt_right],
            )
            .unwrap();
        let scalar_input =
            source(&mut arena, "scalar-input", ResolvedValueType::Matrix(scalar_ring_matrix()));
        let extracted =
            arena.intern_extract_coefficient(scalar_input, 0, Some(BigUint::from(17_u8))).unwrap();
        let scalar =
            arena.intern(ValueOperator::Constant(TypedConstant::int(3)), [].into()).unwrap();
        let lifted = arena
            .intern_matrix_transform(
                MatrixOperation::LiftConstantPolynomial { output: two_by_two, coefficient_bits: 4 },
                &[scalar],
            )
            .unwrap();
        for expression in [viewed, concat, tensor, crt, extracted, lifted] {
            assert!(arena.value_type(expression).is_ok());
        }
    }

    #[test]
    fn matrix_multiply_transfers_ring_scalar_broadcast_shapes() {
        let mut arena = ExprArena::new();
        let scalar_type = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 1).unwrap();
        let rectangular_type = ResolvedMatrixType::new(BigUint::from(17_u8), 8, 1, 2).unwrap();
        let source = |arena: &mut ExprArena, name: &str, matrix_type: ResolvedMatrixType| {
            arena
                .intern(
                    ValueOperator::Source(SemanticSourceIdentity {
                        stable_definition: name.to_owned(),
                        invocation: name.to_owned(),
                        sample_event: None,
                        output_role: "value".to_owned(),
                        sampler: None,
                        artifact: None,
                        value_type: ResolvedValueType::Matrix(matrix_type),
                        coordinates: [].into(),
                        matrix_constant: None,
                    }),
                    [].into(),
                )
                .unwrap()
        };
        let scalar = source(&mut arena, "scalar", scalar_type.clone());
        let rectangular = source(&mut arena, "rectangular", rectangular_type.clone());

        let scalar_left = arena
            .intern_matrix_transform(MatrixOperation::Multiply, &[scalar, rectangular])
            .unwrap();
        assert_eq!(
            arena.value_type(scalar_left).unwrap(),
            &ResolvedValueType::Matrix(rectangular_type.clone())
        );

        let scalar_right = arena
            .intern_matrix_transform(MatrixOperation::Multiply, &[rectangular, scalar])
            .unwrap();
        assert_eq!(
            arena.value_type(scalar_right).unwrap(),
            &ResolvedValueType::Matrix(rectangular_type)
        );
    }

    #[test]
    fn transform_and_scope_proof_reject_foreign_inputs_without_a_second_arena() {
        let mut arena = ExprArena::new();
        let mut foreign_arena = ExprArena::new();
        let input =
            arena.intern(ValueOperator::Constant(TypedConstant::int(1)), [].into()).unwrap();
        let foreign = foreign_arena
            .intern(ValueOperator::Constant(TypedConstant::int(1)), [].into())
            .unwrap();
        assert!(matches!(
            arena.intern_scalar_transform(ScalarOperation::Negate, &[foreign]),
            Err(ArenaError::ForeignExpression { .. })
        ));
        assert!(matches!(
            arena.intern_matrix_transform(
                MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 1,
                    layout: MatrixLayout::row_major(1, 1),
                },
                &[input],
            ),
            Err(ArenaError::TypeMismatch { .. })
        ));

        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let mut first_programs = ProgramArena::new();
        let first_program =
            first_programs.finalize(&mut arena, signature.clone(), argument).unwrap();
        let mut second_programs = ProgramArena::new();
        let second_program = second_programs.finalize(&mut arena, signature, argument).unwrap();
        let mut first_proof = arena.scope_proof(first_program, argument).unwrap();
        let second_proof = arena.scope_proof(second_program, argument).unwrap();
        let first_scoped = arena.scoped_from_proof(&first_proof, argument).unwrap();
        let second_scoped = arena.scoped_from_proof(&second_proof, argument).unwrap();
        let transformed = arena
            .intern_scoped_transform(
                &mut first_proof,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[first_scoped],
            )
            .unwrap();
        assert_eq!(transformed.program(), first_program);
        assert!(matches!(
            arena.intern_scoped_transform(
                &mut first_proof,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[second_scoped],
            ),
            Err(ArenaError::ScopeMismatch { .. })
        ));

        let unrelated =
            arena.intern(ValueOperator::Constant(TypedConstant::int(9)), Box::new([])).unwrap();
        assert_eq!(
            arena.scoped_from_proof(&first_proof, unrelated),
            Err(ArenaError::InvalidScopeProof)
        );
        let forged = ScopedExprId { program: first_program, expression: unrelated };
        assert_eq!(
            arena.intern_scoped_transform(
                &mut first_proof,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[forged],
            ),
            Err(ArenaError::InvalidScopeProof)
        );
    }

    #[test]
    fn absorb_scope_proof_requires_same_finalized_program_and_arena() {
        let mut arena = ExprArena::new();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        let program = programs.finalize(&mut arena, signature.clone(), argument).unwrap();
        let mut target = arena.scope_proof(program, argument).unwrap();
        let mut detached = arena.scope_proof(program, argument).unwrap();
        let root = arena.scoped_from_proof(&detached, argument).unwrap();
        let derived = arena
            .intern_scoped_transform(
                &mut detached,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[root],
            )
            .unwrap();
        assert_eq!(
            arena.scoped_from_proof(&target, derived.expression()),
            Err(ArenaError::InvalidScopeProof)
        );
        arena.absorb_scope_proof(&mut target, detached).unwrap();
        assert_eq!(arena.scoped_from_proof(&target, derived.expression()), Ok(derived));

        let mut other_programs = ProgramArena::new();
        let other = other_programs.finalize(&mut arena, signature, argument).unwrap();
        let other_proof = arena.scope_proof(other, argument).unwrap();
        assert_eq!(
            arena.absorb_scope_proof(&mut target, other_proof),
            Err(ArenaError::InvalidScopeProof)
        );

        let mut foreign_arena = ExprArena::new();
        let foreign_argument = foreign_arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut foreign_programs = ProgramArena::new();
        let foreign_program = foreign_programs
            .finalize(
                &mut foreign_arena,
                ProgramSignature {
                    inputs: [ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]
                    .into(),
                    output: ResolvedValueType::Int,
                },
                foreign_argument,
            )
            .unwrap();
        let foreign_proof = foreign_arena.scope_proof(foreign_program, foreign_argument).unwrap();
        assert!(matches!(
            arena.absorb_scope_proof(&mut target, foreign_proof),
            Err(ArenaError::ForeignExpression { .. })
        ));
    }

    #[test]
    fn scoped_only_input_uses_the_parent_edge_without_rebuilding_scope_authority() {
        let mut arena = ExprArena::new();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let mut programs = ProgramArena::new();
        let program = programs.finalize(&mut arena, signature, argument).unwrap();

        let outer_proof = arena.scope_proof(program, argument).unwrap();
        let mut construction_proof = arena.scope_proof(program, argument).unwrap();
        let root = arena.scoped_from_proof(&construction_proof, argument).unwrap();
        let detached_child = arena
            .intern_scoped_transform(
                &mut construction_proof,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[root],
            )
            .unwrap();
        let detached_parent = arena
            .intern_scoped_transform(
                &mut construction_proof,
                ValueOperator::Scalar(ScalarOperation::Negate),
                &[detached_child],
            )
            .unwrap();
        assert_eq!(
            arena.scoped_from_proof(&outer_proof, detached_child.expression()),
            Err(ArenaError::InvalidScopeProof),
            "an older whole-root proof does not retroactively gain detached derivations"
        );
        arena.reset_scope_proof_build_count();
        assert_eq!(arena.scoped_only_input(detached_parent), Ok(detached_child));
        assert_eq!(arena.scope_proof_build_count(), 0);
        let old_child =
            arena.scoped_from_proof(&construction_proof, detached_child.expression()).unwrap();
        assert_eq!(arena.scoped_only_input(detached_parent).unwrap(), old_child);
        assert_eq!(arena.scope_proof_build_count(), 0);

        let binary_parent = arena
            .intern_scoped_transform(
                &mut construction_proof,
                ValueOperator::Scalar(ScalarOperation::Add),
                &[root, root],
            )
            .unwrap();
        assert!(matches!(
            arena.scoped_only_input(root),
            Err(ArenaError::InvalidArity { expected: 1, actual: 0, .. })
        ));
        assert!(matches!(
            arena.scoped_only_input(binary_parent),
            Err(ArenaError::InvalidArity { expected: 1, actual: 2, .. })
        ));

        let unknown = ScopedExprId {
            program: ValueProgramId::new(programs.token(), u32::MAX),
            expression: detached_parent.expression(),
        };
        assert!(matches!(arena.scoped_only_input(unknown), Err(ArenaError::UnknownProgram(_))));

        let mut foreign_arena = ExprArena::new();
        let foreign_argument = foreign_arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut foreign_programs = ProgramArena::new();
        let foreign_program = foreign_programs
            .finalize(
                &mut foreign_arena,
                ProgramSignature {
                    inputs: [ProgramInput {
                        value_type: ResolvedValueType::Int,
                        trusted_index_range: None,
                    }]
                    .into(),
                    output: ResolvedValueType::Int,
                },
                foreign_argument,
            )
            .unwrap();
        let foreign_parent = foreign_programs.root(&foreign_arena, foreign_program).unwrap();
        assert!(matches!(
            arena.scoped_only_input(foreign_parent),
            Err(ArenaError::ForeignExpression { .. })
        ));

        arena.program_signatures.remove(&program);
        assert_eq!(
            arena.scoped_only_input(detached_parent),
            Err(ArenaError::UnknownProgram(program))
        );
    }

    #[test]
    fn exact_root_scope_proof_validation_rejects_every_mismatched_authority() {
        let mut arena = ExprArena::new();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let signature = ProgramSignature {
            inputs: [ProgramInput {
                value_type: ResolvedValueType::Int,
                trusted_index_range: None,
            }]
            .into(),
            output: ResolvedValueType::Int,
        };
        let mut first_programs = ProgramArena::new();
        let first = first_programs.finalize(&mut arena, signature.clone(), argument).unwrap();
        let mut second_programs = ProgramArena::new();
        let second = second_programs.finalize(&mut arena, signature, argument).unwrap();

        let proof = arena.scope_proof(first, argument).unwrap();
        assert_eq!(arena.validate_scope_proof_for_root(&proof, first, argument), Ok(()));
        assert!(matches!(
            arena.validate_scope_proof_for_root(&proof, second, argument),
            Err(ArenaError::ScopeMismatch { .. })
        ));

        let other =
            arena.intern(ValueOperator::Constant(TypedConstant::int(7)), Box::new([])).unwrap();
        assert_eq!(
            arena.validate_scope_proof_for_root(&proof, first, other),
            Err(ArenaError::InvalidScopeProof)
        );

        let mut wrong_signature = arena.scope_proof(first, argument).unwrap();
        wrong_signature.signature.output = ResolvedValueType::Bool;
        assert_eq!(
            arena.validate_scope_proof_for_root(&wrong_signature, first, argument),
            Err(ArenaError::InvalidScopeProof)
        );

        let mut unreachable = arena.scope_proof(first, argument).unwrap();
        unreachable.reachable.remove(&argument.slot);
        assert_eq!(
            arena.validate_scope_proof_for_root(&unreachable, first, argument),
            Err(ArenaError::InvalidScopeProof)
        );

        let mut foreign = ExprArena::new();
        let foreign_argument = foreign.intern_argument(0, ResolvedValueType::Int).unwrap();
        assert!(matches!(
            foreign.validate_scope_proof_for_root(&proof, first, foreign_argument),
            Err(ArenaError::ForeignExpression { .. })
        ));
    }

    #[test]
    fn lift_bits_concat_overflow_and_large_nodes_fail_or_share_exactly() {
        let mut arena = ExprArena::new();
        let integer =
            arena.intern(ValueOperator::Constant(TypedConstant::int(0)), Box::new([])).unwrap();
        let matrix = ResolvedMatrixType::new(BigUint::from(17_u8), 1, 1, 1).unwrap();
        assert_eq!(
            arena.intern_matrix_transform(
                MatrixOperation::LiftConstantPolynomial {
                    output: matrix.clone(),
                    coefficient_bits: 0,
                },
                &[integer],
            ),
            Err(ArenaError::InvalidCoefficientBits { coefficient_bits: 0 })
        );
        assert_eq!(
            arena.intern_scalar_transform(
                ScalarOperation::LiftConstantPolynomial {
                    output: matrix.clone(),
                    coefficient_bits: 0,
                },
                &[integer],
            ),
            Err(ArenaError::InvalidCoefficientBits { coefficient_bits: 0 })
        );

        let huge = ResolvedMatrixType::new(BigUint::from(17_u8), 1, usize::MAX, 1).unwrap();
        let huge_source = |arena: &mut ExprArena, event| {
            arena
                .intern(
                    ValueOperator::Sampler {
                        event: SampleEventId(event),
                        operation: SamplerOperation::UniformResidue { output: huge.clone() },
                    },
                    Box::new([]),
                )
                .unwrap()
        };
        let first = huge_source(&mut arena, 100);
        let second = huge_source(&mut arena, 101);
        assert!(matches!(
            arena.intern_matrix_transform(
                MatrixOperation::Concat {
                    axis: 0,
                    output: huge,
                    layout: MatrixLayout::row_major(usize::MAX, 1),
                },
                &[first, second],
            ),
            Err(ArenaError::DimensionOverflow { .. })
        ));

        let lane_count = 4_096;
        let output = ResolvedMatrixType::new(BigUint::from(1_u8) << lane_count, 1, 1, 1).unwrap();
        let lane = arena
            .intern(
                ValueOperator::Sampler {
                    event: SampleEventId(102),
                    operation: SamplerOperation::UniformResidue { output: output.clone() },
                },
                Box::new([]),
            )
            .unwrap();
        let crt = arena
            .intern(
                ValueOperator::Matrix(MatrixOperation::CrtRecompose {
                    plaintext_moduli: vec![BigUint::from(2_u8); lane_count].into_boxed_slice(),
                    reconstruction_coefficients: vec![BigInt::from(1_u8); lane_count]
                        .into_boxed_slice(),
                    output,
                }),
                vec![lane; lane_count].into_boxed_slice(),
            )
            .unwrap();
        let stored = &arena.nodes[crt.slot as usize];
        let (key, _) = arena.interner.get_key_value(stored.as_ref()).unwrap();
        assert!(Arc::ptr_eq(stored, key));
        assert_eq!(Arc::strong_count(stored), 2);
    }

    #[test]
    fn shared_prefix_transform_work_is_linear_and_reuses_the_existing_arena() {
        let mut arena = ExprArena::new();
        let source = arena
            .intern(
                ValueOperator::Source(SemanticSourceIdentity {
                    stable_definition: "shared".to_owned(),
                    invocation: "0".to_owned(),
                    sample_event: None,
                    output_role: "value".to_owned(),
                    sampler: None,
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix()),
                    coordinates: [].into(),
                    matrix_constant: None,
                }),
                [].into(),
            )
            .unwrap();
        let mut prefix = source;
        for _ in 0..4_096 {
            prefix = arena.intern_matrix_transform(MatrixOperation::Negate, &[prefix]).unwrap();
        }
        let left = arena.intern_matrix_transform(MatrixOperation::Negate, &[prefix]).unwrap();
        let right = arena.intern_matrix_transform(MatrixOperation::Transpose, &[prefix]).unwrap();
        assert_eq!(arena.reachable_node_count(left).unwrap(), 4_098);
        assert_eq!(arena.reachable_node_count(right).unwrap(), 4_098);
        assert_eq!(arena.node_count(), 4_099);
    }

    fn index_map(
        arena: &mut ExprArena,
        definition: IndexFunctionDefinitionId,
        parameter: u64,
    ) -> ExprId {
        arena
            .register_index_definition(IndexFunctionDefinition {
                id: definition,
                arity: 1,
                output_type: ResolvedValueType::Int,
            })
            .unwrap();
        let argument = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        arena
            .intern(
                ValueOperator::IndexMap { definition, parameters: Box::new([parameter]) },
                Box::new([argument]),
            )
            .unwrap()
    }

    fn index_input(value: i64) -> IndexEvaluationInput {
        IndexEvaluationInput {
            value,
            value_type: ResolvedValueType::Int,
            trusted_range: TrustedIndexRange::new(0, 100).unwrap(),
        }
    }

    #[test]
    fn index_evaluator_reuses_semantic_keys_and_preserves_parameters() {
        let mut arena = ExprArena::new();
        let definition = IndexFunctionDefinitionId(900);
        let plus_one = index_map(&mut arena, definition, 1);
        let plus_one_again = index_map(&mut arena, definition, 1);
        let plus_two = index_map(&mut arena, definition, 2);
        assert_eq!(plus_one, plus_one_again);
        assert_ne!(plus_one, plus_two);
        let evaluator =
            arena.register_builtin_index_evaluator(definition, BuiltinIndexEvaluator::Add).unwrap();
        let output_range = TrustedIndexRange::new(0, 100).unwrap();
        assert_eq!(
            arena.evaluate_index_map(evaluator, plus_one, &[index_input(4)], output_range),
            Ok(5)
        );
        assert_eq!(
            arena.evaluate_index_map(evaluator, plus_two, &[index_input(4)], output_range),
            Ok(6)
        );
    }

    #[test]
    fn index_evaluator_catches_panics_and_range_violations() {
        let mut arena = ExprArena::new();
        let panic_definition = IndexFunctionDefinitionId(901);
        let panic_map = index_map(&mut arena, panic_definition, 0);
        let panic_evaluator = arena
            .register_index_evaluator(panic_definition, |_, _| panic!("implementation panic"))
            .unwrap();
        assert_eq!(
            arena.evaluate_index_map(
                panic_evaluator,
                panic_map,
                &[index_input(1)],
                TrustedIndexRange::new(0, 2).unwrap(),
            ),
            Err(ArenaError::IndexEvaluatorPanicked(panic_definition))
        );

        let add_definition = IndexFunctionDefinitionId(902);
        let add_map = index_map(&mut arena, add_definition, 10);
        let add_evaluator = arena
            .register_builtin_index_evaluator(add_definition, BuiltinIndexEvaluator::Add)
            .unwrap();
        assert!(matches!(
            arena.evaluate_index_map(
                add_evaluator,
                add_map,
                &[index_input(5)],
                TrustedIndexRange::new(0, 10).unwrap(),
            ),
            Err(ArenaError::IndexValueOutOfRange { value: 15, .. })
        ));
    }

    #[test]
    fn index_evaluator_rejects_foreign_and_missing_implementations() {
        let definition = IndexFunctionDefinitionId(903);
        let mut first = ExprArena::new();
        let first_map = index_map(&mut first, definition, 1);
        let evaluator =
            first.register_builtin_index_evaluator(definition, BuiltinIndexEvaluator::Add).unwrap();
        let mut second = ExprArena::new();
        let second_map = index_map(&mut second, definition, 1);
        assert!(matches!(
            second.evaluate_index_map(
                evaluator,
                second_map,
                &[index_input(1)],
                TrustedIndexRange::new(0, 10).unwrap(),
            ),
            Err(ArenaError::ForeignIndexEvaluator { .. })
        ));
        let missing = second.index_evaluator_id(definition).unwrap();
        assert_eq!(
            second.evaluate_index_map(
                missing,
                second_map,
                &[index_input(1)],
                TrustedIndexRange::new(0, 10).unwrap(),
            ),
            Err(ArenaError::MissingIndexEvaluator(definition))
        );
        assert!(first.node(first_map).is_ok());
    }

    #[test]
    fn deterministic_hash_identity_is_binder_open_ordered_and_event_independent() {
        let mut arena = ExprArena::new();
        let key = arena
            .intern(
                ValueOperator::Argument { position: 0, value_type: ResolvedValueType::Bytes },
                Box::new([]),
            )
            .unwrap();
        let first_tag = arena
            .intern(
                ValueOperator::Argument { position: 1, value_type: ResolvedValueType::Int },
                Box::new([]),
            )
            .unwrap();
        let second_tag = arena
            .intern(
                ValueOperator::Argument { position: 2, value_type: ResolvedValueType::Int },
                Box::new([]),
            )
            .unwrap();
        let output = ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap();
        let descriptor = DeterministicHashDescriptor {
            definition: DeterministicHashDefinition::MxxPolynomialHash,
            version: 1,
            key_byte_length: 32,
            output,
            tag_prefix: b"domain".to_vec().into_boxed_slice(),
            binary_tag_count: 1,
            decimal_tag_count: 1,
            u64_le_tag_count: 0,
            dynamic_tag_count: 0,
        };
        let hash = arena
            .intern(
                ValueOperator::DeterministicHash(descriptor.clone()),
                Box::new([key, first_tag, second_tag]),
            )
            .unwrap();
        let reused = arena
            .intern(
                ValueOperator::DeterministicHash(descriptor.clone()),
                Box::new([key, first_tag, second_tag]),
            )
            .unwrap();
        assert_eq!(hash, reused);

        let reordered = arena
            .intern(
                ValueOperator::DeterministicHash(descriptor.clone()),
                Box::new([key, second_tag, first_tag]),
            )
            .unwrap();
        assert_ne!(hash, reordered);
        let mut changed_prefix = descriptor.clone();
        changed_prefix.tag_prefix = b"other-domain".to_vec().into_boxed_slice();
        assert_ne!(
            hash,
            arena
                .intern(
                    ValueOperator::DeterministicHash(changed_prefix),
                    Box::new([key, first_tag, second_tag]),
                )
                .unwrap()
        );
        let mut regrouped = descriptor.clone();
        regrouped.binary_tag_count = 0;
        regrouped.decimal_tag_count = 2;
        let regrouped = arena
            .intern(
                ValueOperator::DeterministicHash(regrouped),
                Box::new([key, first_tag, second_tag]),
            )
            .unwrap();
        assert_ne!(hash, regrouped);

        let mut bad_key_contract = descriptor;
        bad_key_contract.key_byte_length = 31;
        assert_eq!(
            arena.intern(
                ValueOperator::DeterministicHash(bad_key_contract),
                Box::new([key, first_tag, second_tag]),
            ),
            Err(ArenaError::ProgramOutputMismatch)
        );
        assert!(matches!(
            arena.intern(
                ValueOperator::DeterministicHash(DeterministicHashDescriptor {
                    definition: DeterministicHashDefinition::MxxPolynomialHash,
                    version: 1,
                    key_byte_length: 32,
                    output: ResolvedMatrixType::new(BigUint::from(17_u8), 4, 1, 2).unwrap(),
                    tag_prefix: b"domain".to_vec().into_boxed_slice(),
                    binary_tag_count: 1,
                    decimal_tag_count: 1,
                    u64_le_tag_count: 0,
                    dynamic_tag_count: 0,
                }),
                Box::new([first_tag, first_tag, second_tag]),
            ),
            Err(ArenaError::TypeMismatch { position: 0, .. })
        ));
    }

    #[test]
    fn free_argument_summaries_are_cached_by_exact_expression_slot() {
        let mut arena = ExprArena::new();
        let argument =
            arena.intern_argument(0, ResolvedValueType::Int).expect("argument allocation");
        let first = arena.free_argument_ids(argument).expect("first free-argument query");
        let second = arena.free_argument_ids(argument).expect("cached free-argument query");
        assert_eq!(first, second);
        assert_eq!(arena.free_argument_counters(), (2, 1));
    }

    #[test]
    fn free_argument_cache_summarizes_shared_descendants_and_survives_append() {
        let mut arena = ExprArena::new();
        let left = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let right = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let shared = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [left, right].into())
            .unwrap();
        let root = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [shared, shared].into())
            .unwrap();
        assert_eq!(arena.free_argument_ids(root).unwrap(), BTreeSet::from([left, right]));
        let first = arena.free_argument_counter_details();
        assert!(first.newly_summarized >= 4);
        assert!(first.subtree_hits >= 1);

        let appended =
            arena.intern(ValueOperator::Scalar(ScalarOperation::Negate), [root].into()).unwrap();
        assert_eq!(arena.free_argument_ids(root).unwrap(), BTreeSet::from([left, right]));
        assert_eq!(arena.free_argument_ids(appended).unwrap(), BTreeSet::from([left, right]));
        let second = arena.free_argument_counter_details();
        assert!(second.root_hits >= first.root_hits + 1);
        assert!(second.newly_summarized >= first.newly_summarized + 1);
    }

    #[test]
    fn exact_substitution_preserves_distinct_fresh_binders() {
        let mut arena = ExprArena::new();
        let inner = arena.fresh_argument(0, ResolvedValueType::Int).unwrap();
        let outer = arena.fresh_argument(0, ResolvedValueType::Int).unwrap();
        assert_ne!(inner, outer);
        let body = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [inner, outer].into())
            .unwrap();
        let replacement = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let mut engine = ExactSubstitutionEngine::new();
        let result =
            engine.substitute(&mut arena, body, &BTreeMap::from([(inner, replacement)])).unwrap();
        let node = arena.node(result.root).unwrap();
        assert_eq!(node.inputs.as_ref(), &[replacement, outer]);
    }

    #[test]
    fn exact_substitution_cache_is_exact_and_does_not_reintern_on_hit() {
        let mut arena = ExprArena::new();
        let source = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let root =
            arena.intern(ValueOperator::Scalar(ScalarOperation::Negate), [source].into()).unwrap();
        let replacement = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let other_replacement = arena.intern_argument(2, ResolvedValueType::Int).unwrap();
        let mut engine = ExactSubstitutionEngine::new();
        let first =
            engine.substitute(&mut arena, root, &BTreeMap::from([(source, replacement)])).unwrap();
        let count_after_first = arena.node_count();
        let second =
            engine.substitute(&mut arena, root, &BTreeMap::from([(source, replacement)])).unwrap();
        assert!(second.cache_hit);
        assert_eq!(first.root, second.root);
        assert_eq!(arena.node_count(), count_after_first);

        let different = engine
            .substitute(&mut arena, root, &BTreeMap::from([(source, other_replacement)]))
            .unwrap();
        assert_ne!(different.root, first.root);
        let other_root =
            arena.intern(ValueOperator::Scalar(ScalarOperation::Negate), [root].into()).unwrap();
        let different_root = engine
            .substitute(&mut arena, other_root, &BTreeMap::from([(source, replacement)]))
            .unwrap();
        assert_ne!(different_root.root, first.root);
    }

    #[test]
    fn exact_substitution_uses_binder_shortcut_but_not_for_arbitrary_sources() {
        let mut arena = ExprArena::new();
        let used = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let unused = arena.intern_argument(1, ResolvedValueType::Int).unwrap();
        let replacement = arena.intern_argument(2, ResolvedValueType::Int).unwrap();
        let root =
            arena.intern(ValueOperator::Scalar(ScalarOperation::Negate), [used].into()).unwrap();
        let mut engine = ExactSubstitutionEngine::new();
        let disjoint =
            engine.substitute(&mut arena, root, &BTreeMap::from([(unused, replacement)])).unwrap();
        assert!(disjoint.binder_shortcut);
        assert_eq!(disjoint.root, root);

        let source =
            arena.intern(ValueOperator::Constant(TypedConstant::int(11)), Box::new([])).unwrap();
        let replacement =
            arena.intern(ValueOperator::Constant(TypedConstant::int(12)), Box::new([])).unwrap();
        let ordinary =
            engine.substitute(&mut arena, root, &BTreeMap::from([(source, replacement)])).unwrap();
        assert!(!ordinary.binder_shortcut);
        assert!(ordinary.nodes_visited > 0);
    }

    #[test]
    fn exact_substitution_prunes_disjoint_binder_subtrees_and_keeps_cold_order() {
        let mut arena = ExprArena::new();
        let used = arena.fresh_argument(0, ResolvedValueType::Int).unwrap();
        let disjoint = arena.fresh_argument(0, ResolvedValueType::Int).unwrap();
        let replacement = arena.fresh_argument(1, ResolvedValueType::Int).unwrap();
        assert_ne!(used, disjoint, "same-position fresh binders must remain exact identities");

        let mut large_disjoint = disjoint;
        for _ in 0..128 {
            large_disjoint = arena
                .intern(ValueOperator::Scalar(ScalarOperation::Negate), [large_disjoint].into())
                .unwrap();
        }
        let root = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [used, large_disjoint].into())
            .unwrap();
        let reachable = arena.reachable_node_count(root).unwrap();
        let replacements = BTreeMap::from([(used, replacement)]);
        let mut engine = ExactSubstitutionEngine::new();
        let first = engine.substitute(&mut arena, root, &replacements).unwrap();
        assert!(first.skipped_subtrees > 0);
        assert!(first.nodes_visited < reachable as u64);
        assert_eq!(first.nodes_reinterned, 1);
        let rewritten = arena.node(first.root).unwrap();
        assert_eq!(rewritten.inputs.as_ref(), &[replacement, large_disjoint]);

        // A repeated shared disjoint branch is skipped as one subtree, not walked once per edge.
        let shared_parent = arena
            .intern(
                ValueOperator::Scalar(ScalarOperation::Add),
                [large_disjoint, large_disjoint].into(),
            )
            .unwrap();
        let shared_root = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [used, shared_parent].into())
            .unwrap();
        let shared = engine.substitute(&mut arena, shared_root, &replacements).unwrap();
        assert!(shared.skipped_subtrees > 0);

        // A second cold engine sees the same exact traversal and output identity.
        let mut cold_again = ExactSubstitutionEngine::new();
        let repeated = cold_again.substitute(&mut arena, root, &replacements).unwrap();
        assert_eq!(repeated.root, first.root);
        assert_eq!(repeated.nodes_visited, first.nodes_visited);
        assert_eq!(repeated.skipped_subtrees, first.skipped_subtrees);
    }

    #[test]
    fn exact_substitution_pruning_supports_multiple_binders_but_not_non_argument_sources() {
        let mut arena = ExprArena::new();
        let first = arena.fresh_argument(0, ResolvedValueType::Int).unwrap();
        let second = arena.fresh_argument(1, ResolvedValueType::Int).unwrap();
        let untouched = arena.fresh_argument(2, ResolvedValueType::Int).unwrap();
        let replacement_first = arena.fresh_argument(3, ResolvedValueType::Int).unwrap();
        let replacement_second = arena.fresh_argument(4, ResolvedValueType::Int).unwrap();
        let used = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [first, second].into())
            .unwrap();
        let disjoint = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Negate), [untouched].into())
            .unwrap();
        let root = arena
            .intern(ValueOperator::Scalar(ScalarOperation::Add), [used, disjoint].into())
            .unwrap();
        let replacements =
            BTreeMap::from([(first, replacement_first), (second, replacement_second)]);
        let mut engine = ExactSubstitutionEngine::new();
        let result = engine.substitute(&mut arena, root, &replacements).unwrap();
        assert!(result.skipped_subtrees > 0);
        assert!(result.nodes_visited < arena.reachable_node_count(root).unwrap() as u64);

        let source =
            arena.intern(ValueOperator::Constant(TypedConstant::int(21)), Box::new([])).unwrap();
        let non_argument_replacement =
            arena.intern(ValueOperator::Constant(TypedConstant::int(22)), Box::new([])).unwrap();
        let non_argument = engine
            .substitute(&mut arena, root, &BTreeMap::from([(source, non_argument_replacement)]))
            .unwrap();
        assert_eq!(non_argument.skipped_subtrees, 0);
        assert!(non_argument.nodes_visited > 0);
    }

    #[test]
    fn exact_substitution_rejects_foreign_and_mismatched_pairs_before_caching() {
        let mut arena = ExprArena::new();
        let source = arena.intern_argument(0, ResolvedValueType::Int).unwrap();
        let root =
            arena.intern(ValueOperator::Scalar(ScalarOperation::Negate), [source].into()).unwrap();
        let mut foreign = ExprArena::new();
        let foreign_value = foreign.intern_argument(0, ResolvedValueType::Int).unwrap();
        let mut engine = ExactSubstitutionEngine::new();
        let foreign_error = engine
            .substitute(&mut arena, root, &BTreeMap::from([(source, foreign_value)]))
            .unwrap_err();
        assert!(matches!(foreign_error.source, ArenaError::ForeignExpression { .. }));
        assert!(engine.completed.is_empty());

        let bytes = arena.intern_argument(0, ResolvedValueType::Bytes).unwrap();
        let type_error =
            engine.substitute(&mut arena, root, &BTreeMap::from([(source, bytes)])).unwrap_err();
        assert!(matches!(type_error.source, ArenaError::TypeMismatch { .. }));
        assert!(engine.completed.is_empty());
    }
}
