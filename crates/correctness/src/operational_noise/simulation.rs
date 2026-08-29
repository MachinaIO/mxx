//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The production stages own graph reachability, lowering, normalization, and bound
//! classification; this driver owns target validation, diagnostics, and progress cadence.

use super::{
    OperationalSimulationDiagnostics, OperationalSimulationReport,
    arena::{
        ArenaError, ClosedExprId, ExprId, FamilyDomain, MatrixLayout, MatrixOperation,
        ResolvedMatrixType, ResolvedValueType, ScalarOperation, TrapdoorOperation, ValueOperator,
        ValueTransformOperation,
    },
    bound::{BoundClass, MatrixBound, product_bound_with_facts, tensor_bound_with_facts},
    error::{OperationalSimulationError, TargetError},
    g0::{
        AppliedRelationRule, BoundAuthority, BoundProjection, BoundRule, BoundScale, BoundValueRef,
        CanonicalEventKind, CanonicalProjectionRole, CanonicalStatementRows, EventKind,
        FeasibilitySink, FeasibilityTrace, G0Error, MonomialFactorEvidence, NormalizerEvent,
        StableHashVariant, StableSamplerOperation,
    },
    lower::{ProductionAdapter, ProductionRoot},
    program::{FamilyValueId, ValueProgramId},
    protocol::ProtocolPlan,
    report::{ReportTarget, analyze_roots, analyze_roots_with_sink},
};
use crate::{OperationalDecoderKind, ProtocolDecl};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, IntExpr, ParamEnv, Port, WireRef, WireType,
    node::{IntBinaryOp, IntCompareOp, NodeKind},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Zero;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use tracing::{debug, error, info};

const PROGRESS_TIME_CADENCE: Duration = Duration::from_secs(1);
/// Logical checker phase reported by progress events and diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckerPhase {
    Target,
    Lower,
    Normalize,
    Bound,
    Acceptance,
}

/// A decoder target after the graph-specific validation stage has closed its moduli.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedAcceptanceTarget {
    pub target_id: String,
    pub ciphertext_modulus: BigUint,
    pub kind: ResolvedDecoderKind,
}

/// Concrete data required by one exact decoder acceptance formula.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResolvedDecoderKind {
    Threshold { plaintext_modulus: BigUint },
    BooleanInterval,
}

/// The residual root projected for later certificate recording.
///
/// Arena handles remain crate-local so a future serializer must replace them with statement rows
/// before data crosses the crate boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CertificateResidualRoot {
    Closed { root: ClosedExprId, matrix: ResolvedMatrixType },
    Family { family: FamilyValueId, domain: FamilyDomain, matrix: ResolvedMatrixType },
}

/// Typed statement data selected by the pinned checker for one opt-in certificate emission.
///
/// This projection contains only threshold `p`, resolved `q`, and `ProductionRoots.residual`.
/// It deliberately has no decoder root or decoder result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationalCertificateProjection {
    pub target_id: String,
    pub plaintext_modulus: BigUint,
    pub ciphertext_modulus: BigUint,
    pub residual: CertificateResidualRoot,
    pub closure: CertificateClosure,
}

/// Stable schema identity for the non-emitting G0 base summary.
pub const BASE_FEASIBILITY_SCHEMA_ID: &str = "mxx.operational-noise.base-feasibility";
pub const BASE_FEASIBILITY_SCHEMA_VERSION: u32 = 1;

/// A typed, deliberately incomplete feasibility summary.  This is not a certificate report and
/// has no API that can be accepted or emitted as the required G0 artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BaseFeasibilitySummary {
    pub schema_id: &'static str,
    pub schema_version: u32,
    pub target_id: String,
    pub plaintext_modulus: String,
    pub ciphertext_modulus: String,
    pub accepted: bool,
    pub noise_bound: String,
    pub threshold_left: String,
    pub margin: String,
    pub counters: BaseFeasibilityCounters,
    pub n: BaseNBreakdown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BaseFeasibilityCounters {
    /// Counters from the ordinary accepted report, which combines residual and decoder
    /// normalization.  These are not residual-only certificate trace counters.
    pub ordinary_baseline: OrdinaryBaselineCounters,
    /// Reserved as a distinct type for later residual recorder observations.
    pub residual_trace: ResidualTraceCounters,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OrdinaryBaselineCounters {
    pub occurrences: u64,
    pub samples: u64,
    pub normalization_nodes_processed: u64,
    pub normalization_nodes_total: u64,
    pub normalization_exact_term_count: u64,
    pub normalization_relation_candidates: u64,
    pub normalization_relation_applied: u64,
    pub normalization_relation_remaining: u64,
    pub normalization_bounded_fold_count: u64,
}

/// Residual-only recorder counters are intentionally absent in the base summary.  A distinct
/// empty type keeps that absence explicit without introducing a collection of optional fields.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ResidualTraceCounters {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BaseNBreakdown {
    pub expression_rows: usize,
    pub program_rows: usize,
    pub source_rows: usize,
    pub event_rows: usize,
    pub total_rows: usize,
}

const G0_CPU_EVIDENCE_SCHEMA_ID: &str = "mxx.operational-noise.g0-cpu-evidence";
const G0_CPU_EVIDENCE_SCHEMA_VERSION: u32 = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum G0CpuEvidenceStatus {
    CpuObservationOnlyNotAcceptanceEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct G0CpuLutObservation {
    exact_row_count: String,
    exact_payload_logical_items: u64,
    index_use_frontier_products: Vec<String>,
    slice_group_frontier_products: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct G0CpuMetrics {
    descriptor_inventory_canonical_encoded_bytes: u64,
    inventory_retained_logical_items: u64,
    proof_payload_logical_items: u64,
    proof_payload_canonical_encoded_bytes: u64,
    lut_canonical_encoded_bytes: u64,
    lut_retained_logical_items: u64,
    recorder_peak_retained_logical_items: u64,
    proof_projection_peak_retained_logical_items: u64,
    generator_peak_retained_logical_items: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct G0CpuEvidence {
    schema_id: &'static str,
    schema_version: u32,
    status: G0CpuEvidenceStatus,
    base_feasibility: BaseFeasibilitySummary,
    residual_coverage_matrix: ResidualCoverageMatrix,
    lut: G0CpuLutObservation,
    metrics: G0CpuMetrics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExactRetainedN {
    expression_rows: u64,
    program_rows: u64,
    source_rows: u64,
    event_rows: u64,
    total_rows: u64,
}

/// The owned job, certificate projection, and ordinary accepted report from one opt-in run.
/// Arena handles remain tied to this job; callers cannot accidentally pair a projection with a
/// report or job from another lowering.
pub(crate) struct OperationalCertificateRun {
    pub job: super::job::CheckerJob,
    pub projection: OperationalCertificateProjection,
    pub accepted_report: super::report::OperationalReport,
    pub trace: FeasibilityTrace,
    #[cfg(test)]
    pub roots: super::lower::ProductionRoots,
}

/// A stable owner reference used by the proof-payload boundary.  The numbers are rows in
/// `CanonicalStatementRows`; they are never arena slots or raw handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum ProofPayloadScope {
    Closed { root_expression_row: u64 },
    Program { program_row: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ProofPayloadOwner {
    pub scope: ProofPayloadScope,
    pub expression_row: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ProofPayloadMonomial {
    pub central_factors: Vec<ProofPayloadOwner>,
    pub ordered_factors: Vec<ProofPayloadOwner>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ProofPayloadTerm {
    pub monomial: ProofPayloadMonomial,
    pub coefficient: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadValue {
    Exact {
        terms: Vec<ProofPayloadTerm>,
        coefficient_bound: super::facts::NumericContract<super::facts::CoefficientBound>,
        coefficient_producer: u64,
        summary: super::normal_form::BoundedSummary,
        summary_producer: Option<u64>,
    },
    Coefficient {
        bound: super::facts::NumericContract<super::facts::CoefficientBound>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadUniversalDispatch {
    pub preimage_family: u64,
    pub preimage_source: u64,
    pub trapdoor_source: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ProofPayloadRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadRelationRule {
    Universal {
        computed: u64,
        lhs: ProofPayloadMonomial,
        lhs_layout: Option<MatrixLayout>,
        rhs_result: u64,
    },
    Gadget {
        gadget: ProofPayloadOwner,
        decomposition: ProofPayloadOwner,
        input: u64,
        input_result: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadTermRef {
    pub value_event: u64,
    pub term_ordinal: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadCoefficientMergeSource {
    Operator { inputs: [ProofPayloadTermRef; 2] },
    Relation { application: u64, source_term_ordinal: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadCoefficientMerge {
    pub owner: ProofPayloadOwner,
    pub source: ProofPayloadCoefficientMergeSource,
    pub output: ProofPayloadMonomial,
    pub signed_contribution: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadSurvivorFold {
    pub coefficient: BigInt,
    pub bound: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadPreFoldPolynomial {
    pub result_event: u64,
    pub terms: Vec<ProofPayloadTerm>,
    pub summary: super::normal_form::BoundedSummary,
    pub summary_evidence: Option<ProofPayloadValueRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadFactorEvidence {
    pub bound: ProofPayloadValueRef,
    pub is_constant_polynomial: bool,
    pub support_upper: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadValueRef {
    Predecessor { binding_event: u64, input_position: u32, projection: BoundProjection },
    Result { event: u64, projection: BoundProjection },
    Transfer(u64),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadScale {
    Value(ProofPayloadValueRef),
    Magnitude(BigUint),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadAuthority {
    FactStore,
    ProgramFamilyFact,
    Operator,
    RelationPreimageSource { source: u64 },
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadRule {
    Authority(ProofPayloadAuthority),
    Identity {
        input: ProofPayloadValueRef,
    },
    RingAutomorphism {
        input: ProofPayloadValueRef,
        index: u64,
        ring_dimension: usize,
    },
    Sum {
        inputs: Vec<ProofPayloadValueRef>,
    },
    Maximum {
        inputs: Vec<ProofPayloadValueRef>,
    },
    Scale {
        value: ProofPayloadValueRef,
        scale: ProofPayloadScale,
    },
    MonomialProduct {
        monomial: ProofPayloadMonomial,
        factors: Vec<ProofPayloadFactorEvidence>,
    },
    WeightedSum {
        inputs: Vec<ProofPayloadValueRef>,
    },
    Product {
        left: ProofPayloadValueRef,
        right: ProofPayloadValueRef,
        facts: super::bound::MatrixProductFacts,
    },
    Tensor {
        left: ProofPayloadValueRef,
        right: ProofPayloadValueRef,
        left_is_constant_polynomial: bool,
        right_is_constant_polynomial: bool,
    },
}

/// One stable chronological observation.  Event positions remain the only local references;
/// all arena-local expression, monomial, and canonical-RHS handles are projected into values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadEvent {
    InvocationStart {
        root: ProofPayloadOwner,
    },
    Predecessor {
        consumer: ProofPayloadOwner,
        input_position: u32,
        predecessor: u64,
        source_result: u64,
    },
    Result {
        owner: ProofPayloadOwner,
        value: ProofPayloadValue,
    },
    InvocationEnd {
        root: ProofPayloadOwner,
        result: ProofPayloadValue,
        pre_fold_event: u64,
    },
    SpecializationComputed {
        owner: ProofPayloadOwner,
        dispatch: ProofPayloadUniversalDispatch,
        source: ProofPayloadRange,
    },
    SpecializationCacheHit {
        owner: ProofPayloadOwner,
        source: ProofPayloadRange,
    },
    AppliedRelation {
        owner: ProofPayloadOwner,
        source_monomial: ProofPayloadMonomial,
        outer_coefficient: BigInt,
        ordered_start: u32,
        ordered_end_exclusive: u32,
        rule: ProofPayloadRelationRule,
    },
    BoundTransfer {
        owner: ProofPayloadOwner,
        rule: ProofPayloadRule,
    },
    CoefficientMerge(ProofPayloadCoefficientMerge),
    PreFoldPolynomial(ProofPayloadPreFoldPolynomial),
    SurvivorFold(ProofPayloadSurvivorFold),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationalProofPayload {
    pub events: Vec<ProofPayloadEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedScalarKind {
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
    ThresholdDecode,
    Bit,
    Slice,
    Hash,
    ExtractCoefficient,
    LiftConstantPolynomial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedMatrixKind {
    Add,
    Subtract,
    Multiply,
    Negate,
    Scale,
    RingAutomorphism,
    Transpose,
    Slice,
    IndexedSlice,
    View,
    Concat,
    Tensor,
    CrtRecompose,
    ExtractCoefficient,
    LiftConstantPolynomial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedTrapdoorKind {
    Generate,
    Transform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedOperatorKind {
    Argument,
    Constant,
    Source,
    DeterministicHash,
    OpaqueFamilyElement,
    IndexMap,
    ExplicitElement,
    ProgramCall,
    ExtractCoefficient,
    Scalar(ObservedScalarKind),
    Matrix(ObservedMatrixKind),
    Trapdoor(ObservedTrapdoorKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedTransformKind {
    GadgetDecompose(ObservedGadgetKind),
    PackPolynomialCoefficients,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedGadgetKind {
    Regular,
    Small,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedHashKind {
    Plain,
    Decomposed,
    SmallDecomposed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedSamplerKind {
    Sample,
    UniformResidue,
    UniformInterval,
    Gaussian,
    Hash(ObservedHashKind),
    Trapdoor,
    Preimage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedRelationKind {
    Universal,
    Gadget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedBoundKind {
    Authority(ObservedAuthorityKind),
    Identity,
    RingAutomorphism,
    Sum,
    Maximum,
    Scale,
    MonomialProduct,
    WeightedSum,
    Product,
    Tensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservedAuthorityKind {
    FactStore,
    ProgramFamilyFact,
    Operator,
    RelationPreimageSource,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "domain", content = "kind")]
pub(crate) enum ObservedCoverageKind {
    Operator(ObservedOperatorKind),
    Transform(ObservedTransformKind),
    Sampler(ObservedSamplerKind),
    Relation(ObservedRelationKind),
    Bound(ObservedBoundKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "site")]
pub(crate) enum ObservedCoverageSite {
    ExpressionRow { row: u64 },
    SamplerEventRow { row: u64 },
    TraceEvent { index: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ObservedCoverageRow {
    pub kind: ObservedCoverageKind,
    pub count: u64,
    pub sites: Vec<ObservedCoverageSite>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ObservedCoverage {
    pub rows: Vec<ObservedCoverageRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ResidualCoverageMatrix {
    rows: Vec<ResidualCoverageRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ResidualCoverageRow {
    kind: ObservedCoverageKind,
    count: u64,
    sites: Vec<ObservedCoverageSite>,
    rust_item: &'static str,
    disposition: ResidualCoverageDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "disposition")]
enum ResidualCoverageDisposition {
    CheckedLean { semantics: &'static str, transfer: &'static str },
    G2LeanObligation { semantics: &'static str, transfer: &'static str },
    RejectBeforeGeneration { reason: &'static str },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidualCoverageClassification {
    rust_item: &'static str,
    disposition: ResidualCoverageDisposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawEventCoverageDisposition {
    CanonicalProjection,
    RejectBeforeCanonicalProjection { reason: &'static str },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadProjection {
    pub(crate) payload: OperationalProofPayload,
    pub(crate) generator_peak_retained_logical_items: u64,
    pub(crate) observed_coverage: ObservedCoverage,
}

pub(crate) struct ProjectedCertificateDocuments {
    pub(crate) cert: super::certificate_schema::CertificateDocumentV1,
    pub(crate) proof: ProofPayloadProjection,
}

/// Errors from the canonical proof-payload boundary.  The payload itself is already projected
/// into owned values; this error only protects length/count arithmetic while encoding it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CanonicalPayloadError {
    LengthOverflow,
}

pub(crate) trait LogicalItems {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError>;
}

impl<T: LogicalItems + ?Sized> LogicalItems for &T {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        (*self).logical_items()
    }
}

impl LogicalItems for ObservedScalarKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedMatrixKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedTrapdoorKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedOperatorKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_add(
            1,
            match self {
                Self::Scalar(kind) => kind.logical_items()?,
                Self::Matrix(kind) => kind.logical_items()?,
                Self::Trapdoor(kind) => kind.logical_items()?,
                Self::Argument |
                Self::Constant |
                Self::Source |
                Self::DeterministicHash |
                Self::OpaqueFamilyElement |
                Self::IndexMap |
                Self::ExplicitElement |
                Self::ProgramCall |
                Self::ExtractCoefficient => 0,
            },
        )
    }
}

impl LogicalItems for ObservedTransformKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_add(
            1,
            match self {
                Self::GadgetDecompose(kind) => kind.logical_items()?,
                Self::PackPolynomialCoefficients => 0,
            },
        )
    }
}

impl LogicalItems for ObservedGadgetKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedHashKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedSamplerKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_add(
            1,
            match self {
                Self::Hash(kind) => kind.logical_items()?,
                Self::Sample |
                Self::UniformResidue |
                Self::UniformInterval |
                Self::Gaussian |
                Self::Trapdoor |
                Self::Preimage => 0,
            },
        )
    }
}

impl LogicalItems for ObservedRelationKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedBoundKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_add(
            1,
            match self {
                Self::Authority(kind) => kind.logical_items()?,
                Self::Identity |
                Self::RingAutomorphism |
                Self::Sum |
                Self::Maximum |
                Self::Scale |
                Self::MonomialProduct |
                Self::WeightedSum |
                Self::Product |
                Self::Tensor => 0,
            },
        )
    }
}

impl LogicalItems for ObservedAuthorityKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for ObservedCoverageKind {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_add(
            1,
            match self {
                Self::Operator(kind) => kind.logical_items()?,
                Self::Transform(kind) => kind.logical_items()?,
                Self::Sampler(kind) => kind.logical_items()?,
                Self::Relation(kind) => kind.logical_items()?,
                Self::Bound(kind) => kind.logical_items()?,
            },
        )
    }
}

impl LogicalItems for ObservedCoverageSite {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(2)
    }
}

impl LogicalItems for ObservedCoverageRow {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.kind.logical_items(), Ok(1), logical_vec(&self.sites)])
    }
}

impl LogicalItems for ObservedCoverage {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        logical_vec(&self.rows)
    }
}

pub(crate) fn checked_add(left: u64, right: u64) -> Result<u64, CanonicalPayloadError> {
    left.checked_add(right).ok_or(CanonicalPayloadError::LengthOverflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, CanonicalPayloadError> {
    left.checked_mul(right).ok_or(CanonicalPayloadError::LengthOverflow)
}

fn logical_uniform_collection(
    len: usize,
    items_per_entry: u64,
) -> Result<u64, CanonicalPayloadError> {
    checked_add(
        1,
        checked_mul(
            u64::try_from(len).map_err(|_| CanonicalPayloadError::LengthOverflow)?,
            checked_add(1, items_per_entry)?,
        )?,
    )
}

pub(crate) fn checked_sum<I>(items: I) -> Result<u64, CanonicalPayloadError>
where
    I: IntoIterator,
    I::Item: LogicalItems,
{
    items.into_iter().try_fold(0_u64, |total, item| checked_add(total, item.logical_items()?))
}

pub(crate) fn logical_vec<T: LogicalItems>(items: &[T]) -> Result<u64, CanonicalPayloadError> {
    checked_add(
        checked_add(
            1,
            u64::try_from(items.len()).map_err(|_| CanonicalPayloadError::LengthOverflow)?,
        )?,
        checked_sum(items)?,
    )
}

impl LogicalItems for bool {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for u8 {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for u32 {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for u64 {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for usize {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for BigInt {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for BigUint {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for String {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for Result<u64, CanonicalPayloadError> {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        *self
    }
}

impl<T: LogicalItems> LogicalItems for Option<T> {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Some(value) => checked_add(1, value.logical_items()?),
            None => Ok(1),
        }
    }
}

impl<T: LogicalItems> LogicalItems for Vec<T> {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        logical_vec(self)
    }
}

impl<T: LogicalItems> LogicalItems for Box<[T]> {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        logical_vec(self)
    }
}

impl<T: LogicalItems, const N: usize> LogicalItems for [T; N] {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum(self)
    }
}

impl LogicalItems for MatrixLayout {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.name.logical_items(),
            self.row_stride.logical_items(),
            self.column_stride.logical_items(),
        ])
    }
}

impl LogicalItems for BoundProjection {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        Ok(1)
    }
}

impl LogicalItems for super::facts::CoefficientBound {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::ExactZero | Self::Large => Ok(1),
            Self::Finite(bound) => {
                checked_add(1, bound.maximum_absolute_coefficient.logical_items()?)
            }
        }
    }
}

impl LogicalItems for super::facts::NumericContract<super::facts::CoefficientBound> {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Missing => Ok(1),
            Self::Known(value) => checked_add(1, value.logical_items()?),
        }
    }
}

impl LogicalItems for super::normal_form::BoundedSummary {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        self.coefficient_bound().logical_items()
    }
}

impl LogicalItems for super::bound::MatrixProductFacts {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.left_is_constant_polynomial.logical_items(),
            self.right_is_constant_polynomial.logical_items(),
            self.right_known_zero_rows.logical_items(),
            self.left_support_upper.logical_items(),
            self.right_support_upper.logical_items(),
        ])
    }
}

impl LogicalItems for ProofPayloadScope {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Closed { root_expression_row } => {
                checked_add(1, root_expression_row.logical_items()?)
            }
            Self::Program { program_row } => checked_add(1, program_row.logical_items()?),
        }
    }
}

impl LogicalItems for ProofPayloadOwner {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.scope.logical_items(), self.expression_row.logical_items()])
    }
}

impl LogicalItems for ProofPayloadMonomial {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.central_factors.logical_items(), self.ordered_factors.logical_items()])
    }
}

impl LogicalItems for ProofPayloadTerm {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.monomial.logical_items(), self.coefficient.logical_items()])
    }
}

impl LogicalItems for ProofPayloadValue {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Exact {
                terms,
                coefficient_bound,
                coefficient_producer,
                summary,
                summary_producer,
            } => checked_add(
                1,
                checked_sum([
                    terms.logical_items(),
                    coefficient_bound.logical_items(),
                    coefficient_producer.logical_items(),
                    summary.logical_items(),
                    summary_producer.logical_items(),
                ])?,
            ),
            Self::Coefficient { bound } => checked_add(1, bound.logical_items()?),
        }
    }
}

impl LogicalItems for ProofPayloadUniversalDispatch {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.preimage_family.logical_items(),
            self.preimage_source.logical_items(),
            self.trapdoor_source.logical_items(),
        ])
    }
}

impl LogicalItems for ProofPayloadRange {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.start.logical_items(), self.end.logical_items()])
    }
}

impl LogicalItems for ProofPayloadRelationRule {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Universal { computed, lhs, lhs_layout, rhs_result } => checked_add(
                1,
                checked_sum([
                    computed.logical_items(),
                    lhs.logical_items(),
                    lhs_layout.logical_items(),
                    rhs_result.logical_items(),
                ])?,
            ),
            Self::Gadget { gadget, decomposition, input, input_result } => checked_add(
                1,
                checked_sum([
                    gadget.logical_items(),
                    decomposition.logical_items(),
                    input.logical_items(),
                    input_result.logical_items(),
                ])?,
            ),
        }
    }
}

impl LogicalItems for ProofPayloadTermRef {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.value_event.logical_items(), self.term_ordinal.logical_items()])
    }
}

impl LogicalItems for ProofPayloadCoefficientMergeSource {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Operator { inputs } => checked_add(1, inputs.logical_items()?),
            Self::Relation { application, source_term_ordinal } => checked_add(
                1,
                checked_sum([application.logical_items(), source_term_ordinal.logical_items()])?,
            ),
        }
    }
}

impl LogicalItems for ProofPayloadCoefficientMerge {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.owner.logical_items(),
            self.source.logical_items(),
            self.output.logical_items(),
            self.signed_contribution.logical_items(),
        ])
    }
}

impl LogicalItems for ProofPayloadSurvivorFold {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([self.coefficient.logical_items(), self.bound.logical_items()])
    }
}

impl LogicalItems for ProofPayloadValueRef {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Predecessor { binding_event, input_position, projection } => checked_add(
                1,
                checked_sum([
                    binding_event.logical_items(),
                    input_position.logical_items(),
                    projection.logical_items(),
                ])?,
            ),
            Self::Result { event, projection } => {
                checked_add(1, checked_sum([event.logical_items(), projection.logical_items()])?)
            }
            Self::Transfer(event) => checked_add(1, event.logical_items()?),
        }
    }
}

impl LogicalItems for ProofPayloadPreFoldPolynomial {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.result_event.logical_items(),
            self.terms.logical_items(),
            self.summary.logical_items(),
            self.summary_evidence.logical_items(),
        ])
    }
}

impl LogicalItems for ProofPayloadFactorEvidence {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        checked_sum([
            self.bound.logical_items(),
            self.is_constant_polynomial.logical_items(),
            self.support_upper.logical_items(),
        ])
    }
}

impl LogicalItems for ProofPayloadScale {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Value(value) => checked_add(1, value.logical_items()?),
            Self::Magnitude(value) => checked_add(1, value.logical_items()?),
        }
    }
}

impl LogicalItems for ProofPayloadAuthority {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::FactStore | Self::ProgramFamilyFact | Self::Operator | Self::Unavailable => Ok(1),
            Self::RelationPreimageSource { source } => checked_add(1, source.logical_items()?),
        }
    }
}

impl LogicalItems for ProofPayloadRule {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::Authority(authority) => checked_add(1, authority.logical_items()?),
            Self::Identity { input } => checked_add(1, input.logical_items()?),
            Self::RingAutomorphism { input, .. } => checked_add(3, input.logical_items()?),
            Self::Sum { inputs } | Self::Maximum { inputs } | Self::WeightedSum { inputs } => {
                checked_add(1, inputs.logical_items()?)
            }
            Self::Scale { value, scale } => {
                checked_add(1, checked_sum([value.logical_items(), scale.logical_items()])?)
            }
            Self::MonomialProduct { monomial, factors } => {
                checked_add(1, checked_sum([monomial.logical_items(), factors.logical_items()])?)
            }
            Self::Product { left, right, facts } => checked_add(
                1,
                checked_sum([left.logical_items(), right.logical_items(), facts.logical_items()])?,
            ),
            Self::Tensor {
                left,
                right,
                left_is_constant_polynomial,
                right_is_constant_polynomial,
            } => checked_add(
                1,
                checked_sum([
                    left.logical_items(),
                    right.logical_items(),
                    left_is_constant_polynomial.logical_items(),
                    right_is_constant_polynomial.logical_items(),
                ])?,
            ),
        }
    }
}

impl LogicalItems for ProofPayloadEvent {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        match self {
            Self::InvocationStart { root } => checked_add(1, root.logical_items()?),
            Self::Predecessor { consumer, input_position, predecessor, source_result } => {
                checked_add(
                    1,
                    checked_sum([
                        consumer.logical_items(),
                        input_position.logical_items(),
                        predecessor.logical_items(),
                        source_result.logical_items(),
                    ])?,
                )
            }
            Self::Result { owner, value } => {
                checked_add(1, checked_sum([owner.logical_items(), value.logical_items()])?)
            }
            Self::InvocationEnd { root, result, pre_fold_event } => checked_add(
                1,
                checked_sum([
                    root.logical_items(),
                    result.logical_items(),
                    pre_fold_event.logical_items(),
                ])?,
            ),
            Self::SpecializationComputed { owner, dispatch, source } => checked_add(
                1,
                checked_sum([
                    owner.logical_items(),
                    dispatch.logical_items(),
                    source.logical_items(),
                ])?,
            ),
            Self::SpecializationCacheHit { owner, source } => {
                checked_add(1, checked_sum([owner.logical_items(), source.logical_items()])?)
            }
            Self::AppliedRelation {
                owner,
                source_monomial,
                outer_coefficient,
                ordered_start,
                ordered_end_exclusive,
                rule,
            } => checked_add(
                1,
                checked_sum([
                    owner.logical_items(),
                    source_monomial.logical_items(),
                    outer_coefficient.logical_items(),
                    ordered_start.logical_items(),
                    ordered_end_exclusive.logical_items(),
                    rule.logical_items(),
                ])?,
            ),
            Self::BoundTransfer { owner, rule } => {
                checked_add(1, checked_sum([owner.logical_items(), rule.logical_items()])?)
            }
            Self::CoefficientMerge(merge) => checked_add(1, merge.logical_items()?),
            Self::PreFoldPolynomial(polynomial) => checked_add(1, polynomial.logical_items()?),
            Self::SurvivorFold(fold) => checked_add(1, fold.logical_items()?),
        }
    }
}

impl LogicalItems for OperationalProofPayload {
    fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        self.events.logical_items()
    }
}

struct CanonicalPayloadWriter {
    bytes: Vec<u8>,
}

impl CanonicalPayloadWriter {
    fn new() -> Self {
        let mut writer = Self { bytes: Vec::new() };
        writer.bytes.extend_from_slice(b"mxx-operational-proof-payload\0");
        writer.u8(1);
        writer
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<(), CanonicalPayloadError> {
        self.u64(u64::try_from(value).map_err(|_| CanonicalPayloadError::LengthOverflow)?);
        Ok(())
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), CanonicalPayloadError> {
        self.usize(value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), CanonicalPayloadError> {
        self.bytes(value.as_bytes())
    }

    fn bigint(&mut self, value: &BigInt) -> Result<(), CanonicalPayloadError> {
        let (sign, magnitude) = value.to_bytes_be();
        self.u8(match sign {
            Sign::NoSign => 0,
            Sign::Plus => 1,
            Sign::Minus => 2,
        });
        self.bytes(&magnitude)
    }

    fn biguint(&mut self, value: &BigUint) -> Result<(), CanonicalPayloadError> {
        self.bytes(&value.to_bytes_be())
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn option<T>(
        &mut self,
        value: &Option<T>,
        write: impl FnOnce(&mut Self, &T) -> Result<(), CanonicalPayloadError>,
    ) -> Result<(), CanonicalPayloadError> {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1);
                write(self, value)?;
            }
        }
        Ok(())
    }

    fn vec<T>(
        &mut self,
        values: &[T],
        mut write: impl FnMut(&mut Self, &T) -> Result<(), CanonicalPayloadError>,
    ) -> Result<(), CanonicalPayloadError> {
        self.usize(values.len())?;
        for value in values {
            write(self, value)?;
        }
        Ok(())
    }

    fn owner(&mut self, owner: &ProofPayloadOwner) -> Result<(), CanonicalPayloadError> {
        self.scope(&owner.scope)?;
        self.u64(owner.expression_row);
        Ok(())
    }

    fn scope(&mut self, scope: &ProofPayloadScope) -> Result<(), CanonicalPayloadError> {
        match scope {
            ProofPayloadScope::Closed { root_expression_row } => {
                self.u8(0);
                self.u64(*root_expression_row);
            }
            ProofPayloadScope::Program { program_row } => {
                self.u8(1);
                self.u64(*program_row);
            }
        }
        Ok(())
    }

    fn monomial(&mut self, monomial: &ProofPayloadMonomial) -> Result<(), CanonicalPayloadError> {
        self.vec(&monomial.central_factors, |writer, owner| writer.owner(owner))?;
        self.vec(&monomial.ordered_factors, |writer, owner| writer.owner(owner))
    }

    fn term(&mut self, term: &ProofPayloadTerm) -> Result<(), CanonicalPayloadError> {
        self.monomial(&term.monomial)?;
        self.bigint(&term.coefficient)
    }

    fn summary(
        &mut self,
        summary: &super::normal_form::BoundedSummary,
    ) -> Result<(), CanonicalPayloadError> {
        self.numeric_contract(&summary.coefficient_bound())
    }

    fn numeric_contract(
        &mut self,
        contract: &super::facts::NumericContract<super::facts::CoefficientBound>,
    ) -> Result<(), CanonicalPayloadError> {
        match contract {
            super::facts::NumericContract::Missing => self.u8(0),
            super::facts::NumericContract::Known(bound) => {
                self.u8(1);
                self.coefficient_bound(bound)?;
            }
        }
        Ok(())
    }

    fn coefficient_bound(
        &mut self,
        bound: &super::facts::CoefficientBound,
    ) -> Result<(), CanonicalPayloadError> {
        match bound {
            super::facts::CoefficientBound::ExactZero => self.u8(0),
            super::facts::CoefficientBound::Finite(value) => {
                self.u8(1);
                self.biguint(&value.maximum_absolute_coefficient)?;
            }
            super::facts::CoefficientBound::Large => self.u8(2),
        }
        Ok(())
    }

    fn value(&mut self, value: &ProofPayloadValue) -> Result<(), CanonicalPayloadError> {
        match value {
            ProofPayloadValue::Exact {
                terms,
                coefficient_bound,
                coefficient_producer,
                summary,
                summary_producer,
            } => {
                self.u8(0);
                self.vec(terms, |writer, term| writer.term(term))?;
                self.numeric_contract(coefficient_bound)?;
                self.u64(*coefficient_producer);
                self.summary(summary)?;
                self.option(summary_producer, |writer, event| {
                    writer.u64(*event);
                    Ok(())
                })
            }
            ProofPayloadValue::Coefficient { bound } => {
                self.u8(1);
                self.numeric_contract(bound)
            }
        }
    }

    fn dispatch(&mut self, dispatch: &ProofPayloadUniversalDispatch) {
        self.u64(dispatch.preimage_family);
        self.u64(dispatch.preimage_source);
        self.u64(dispatch.trapdoor_source);
    }

    fn range(&mut self, range: &ProofPayloadRange) {
        self.u64(range.start);
        self.u64(range.end);
    }

    fn layout(&mut self, layout: &Option<MatrixLayout>) -> Result<(), CanonicalPayloadError> {
        self.option(layout, |writer, layout| {
            writer.string(&layout.name)?;
            writer.usize(layout.row_stride)?;
            writer.usize(layout.column_stride)
        })
    }

    fn relation_rule(
        &mut self,
        rule: &ProofPayloadRelationRule,
    ) -> Result<(), CanonicalPayloadError> {
        match rule {
            ProofPayloadRelationRule::Universal { computed, lhs, lhs_layout, rhs_result } => {
                self.u8(0);
                self.u64(*computed);
                self.monomial(lhs)?;
                self.layout(lhs_layout)?;
                self.u64(*rhs_result);
            }
            ProofPayloadRelationRule::Gadget { gadget, decomposition, input, input_result } => {
                self.u8(1);
                self.owner(gadget)?;
                self.owner(decomposition)?;
                self.u64(*input);
                self.u64(*input_result);
            }
        }
        Ok(())
    }

    fn term_ref(&mut self, reference: &ProofPayloadTermRef) {
        self.u64(reference.value_event);
        self.u64(reference.term_ordinal);
    }

    fn merge_source(
        &mut self,
        source: &ProofPayloadCoefficientMergeSource,
    ) -> Result<(), CanonicalPayloadError> {
        match source {
            ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                self.u8(0);
                self.term_ref(&inputs[0]);
                self.term_ref(&inputs[1]);
            }
            ProofPayloadCoefficientMergeSource::Relation { application, source_term_ordinal } => {
                self.u8(1);
                self.u64(*application);
                self.u64(*source_term_ordinal);
            }
        }
        Ok(())
    }

    fn merge(&mut self, merge: &ProofPayloadCoefficientMerge) -> Result<(), CanonicalPayloadError> {
        self.owner(&merge.owner)?;
        self.merge_source(&merge.source)?;
        self.monomial(&merge.output)?;
        self.bigint(&merge.signed_contribution)
    }

    fn value_ref(&mut self, reference: &ProofPayloadValueRef) {
        match reference {
            ProofPayloadValueRef::Predecessor { binding_event, input_position, projection } => {
                self.u8(0);
                self.u64(*binding_event);
                self.u32(*input_position);
                self.projection(projection);
            }
            ProofPayloadValueRef::Result { event, projection } => {
                self.u8(1);
                self.u64(*event);
                self.projection(projection);
            }
            ProofPayloadValueRef::Transfer(event) => {
                self.u8(2);
                self.u64(*event);
            }
        }
    }

    fn projection(&mut self, projection: &BoundProjection) {
        self.u8(match projection {
            BoundProjection::Coefficient => 0,
            BoundProjection::Summary => 1,
        });
    }

    fn scale(&mut self, scale: &ProofPayloadScale) -> Result<(), CanonicalPayloadError> {
        match scale {
            ProofPayloadScale::Value(value) => {
                self.u8(0);
                self.value_ref(value);
            }
            ProofPayloadScale::Magnitude(value) => {
                self.u8(1);
                self.biguint(value)?;
            }
        }
        Ok(())
    }

    fn authority(&mut self, authority: &ProofPayloadAuthority) {
        match authority {
            ProofPayloadAuthority::FactStore => self.u8(0),
            ProofPayloadAuthority::ProgramFamilyFact => self.u8(1),
            ProofPayloadAuthority::Operator => self.u8(2),
            ProofPayloadAuthority::RelationPreimageSource { source } => {
                self.u8(3);
                self.u64(*source);
            }
            ProofPayloadAuthority::Unavailable => self.u8(4),
        }
    }

    fn factor_evidence(
        &mut self,
        factor: &ProofPayloadFactorEvidence,
    ) -> Result<(), CanonicalPayloadError> {
        self.value_ref(&factor.bound);
        self.bool(factor.is_constant_polynomial);
        self.option(&factor.support_upper, |writer, value| writer.usize(*value))
    }

    fn rule(&mut self, rule: &ProofPayloadRule) -> Result<(), CanonicalPayloadError> {
        match rule {
            ProofPayloadRule::Authority(authority) => {
                self.u8(0);
                self.authority(authority);
            }
            ProofPayloadRule::Identity { input } => {
                self.u8(1);
                self.value_ref(input);
            }
            ProofPayloadRule::RingAutomorphism { input, index, ring_dimension } => {
                self.u8(9);
                self.value_ref(input);
                self.u64(*index);
                self.usize(*ring_dimension)?;
            }
            ProofPayloadRule::Sum { inputs } => {
                self.u8(2);
                self.vec(inputs, |writer, input| {
                    writer.value_ref(input);
                    Ok(())
                })?;
            }
            ProofPayloadRule::Maximum { inputs } => {
                self.u8(3);
                self.vec(inputs, |writer, input| {
                    writer.value_ref(input);
                    Ok(())
                })?;
            }
            ProofPayloadRule::Scale { value, scale } => {
                self.u8(4);
                self.value_ref(value);
                self.scale(scale)?;
            }
            ProofPayloadRule::MonomialProduct { monomial, factors } => {
                self.u8(5);
                self.monomial(monomial)?;
                self.vec(factors, |writer, factor| writer.factor_evidence(factor))?;
            }
            ProofPayloadRule::WeightedSum { inputs } => {
                self.u8(6);
                self.vec(inputs, |writer, input| {
                    writer.value_ref(input);
                    Ok(())
                })?;
            }
            ProofPayloadRule::Product { left, right, facts } => {
                self.u8(7);
                self.value_ref(left);
                self.value_ref(right);
                self.product_facts(facts)?;
            }
            ProofPayloadRule::Tensor {
                left,
                right,
                left_is_constant_polynomial,
                right_is_constant_polynomial,
            } => {
                self.u8(8);
                self.value_ref(left);
                self.value_ref(right);
                self.bool(*left_is_constant_polynomial);
                self.bool(*right_is_constant_polynomial);
            }
        }
        Ok(())
    }

    fn product_facts(
        &mut self,
        facts: &super::bound::MatrixProductFacts,
    ) -> Result<(), CanonicalPayloadError> {
        self.bool(facts.left_is_constant_polynomial);
        self.bool(facts.right_is_constant_polynomial);
        self.option(&facts.right_known_zero_rows, |writer, value| writer.biguint(value))?;
        self.option(&facts.left_support_upper, |writer, value| writer.usize(*value))?;
        self.option(&facts.right_support_upper, |writer, value| writer.usize(*value))
    }

    fn event(&mut self, event: &ProofPayloadEvent) -> Result<(), CanonicalPayloadError> {
        match event {
            ProofPayloadEvent::InvocationStart { root } => {
                self.u8(0);
                self.owner(root)?;
            }
            ProofPayloadEvent::Predecessor {
                consumer,
                input_position,
                predecessor,
                source_result,
            } => {
                self.u8(1);
                self.owner(consumer)?;
                self.u32(*input_position);
                self.u64(*predecessor);
                self.u64(*source_result);
            }
            ProofPayloadEvent::Result { owner, value } => {
                self.u8(2);
                self.owner(owner)?;
                self.value(value)?;
            }
            ProofPayloadEvent::InvocationEnd { root, result, pre_fold_event } => {
                self.u8(3);
                self.owner(root)?;
                self.value(result)?;
                self.u64(*pre_fold_event);
            }
            ProofPayloadEvent::SpecializationComputed { owner, dispatch, source } => {
                self.u8(4);
                self.owner(owner)?;
                self.dispatch(dispatch);
                self.range(source);
            }
            ProofPayloadEvent::SpecializationCacheHit { owner, source } => {
                self.u8(5);
                self.owner(owner)?;
                self.range(source);
            }
            ProofPayloadEvent::AppliedRelation {
                owner,
                source_monomial,
                outer_coefficient,
                ordered_start,
                ordered_end_exclusive,
                rule,
            } => {
                self.u8(6);
                self.owner(owner)?;
                self.monomial(source_monomial)?;
                self.bigint(outer_coefficient)?;
                self.u32(*ordered_start);
                self.u32(*ordered_end_exclusive);
                self.relation_rule(rule)?;
            }
            ProofPayloadEvent::BoundTransfer { owner, rule } => {
                self.u8(7);
                self.owner(owner)?;
                self.rule(rule)?;
            }
            ProofPayloadEvent::CoefficientMerge(merge) => {
                self.u8(8);
                self.merge(merge)?;
            }
            ProofPayloadEvent::PreFoldPolynomial(polynomial) => {
                self.u8(9);
                self.u64(polynomial.result_event);
                self.vec(&polynomial.terms, |writer, term| writer.term(term))?;
                self.summary(&polynomial.summary)?;
                self.option(&polynomial.summary_evidence, |writer, value| {
                    writer.value_ref(value);
                    Ok(())
                })?;
            }
            ProofPayloadEvent::SurvivorFold(fold) => {
                self.u8(10);
                self.bigint(&fold.coefficient)?;
                self.u64(fold.bound);
            }
        }
        Ok(())
    }
}

impl OperationalProofPayload {
    /// Encode exactly this projected payload in a versioned, tagged deterministic format.
    pub(crate) fn encode_canonical(&self) -> Result<Vec<u8>, CanonicalPayloadError> {
        let mut writer = CanonicalPayloadWriter::new();
        writer.vec(&self.events, |writer, event| writer.event(event))?;
        Ok(writer.bytes)
    }

    pub(crate) fn logical_items(&self) -> Result<u64, CanonicalPayloadError> {
        LogicalItems::logical_items(self)
    }
}

/// The typed dependency inventory rooted at one residual production root.
///
/// Expression and program IDs remain job-local.  This inventory is an in-memory boundary only;
/// later serialization must replace these handles with canonical rows without adding decoder data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CertificateClosure {
    pub expressions: BTreeSet<ExprId>,
    pub programs: BTreeSet<ValueProgramId>,
    pub families: BTreeSet<FamilyValueId>,
    pub source_ids: BTreeSet<super::arena::SemanticSourceIdentity>,
    pub family_source_ids: BTreeSet<super::arena::SemanticFamilySourceIdentity>,
    pub event_ids: BTreeSet<super::arena::SampleEventId>,
    pub constant_expressions: BTreeSet<ExprId>,
}

enum CertificateWork {
    Expression(ExprId),
    Program(ValueProgramId),
    Family(FamilyValueId),
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum CertificateClosureError {
    #[error("certificate closure arena reference is invalid: {0}")]
    Arena(#[from] ArenaError),
    #[error(
        "certificate family {family:?} disagrees with its program {program:?}: family body {family_body:?}, program root {program_root:?}"
    )]
    FamilyProgramMismatch {
        family: FamilyValueId,
        program: ValueProgramId,
        family_body: ExprId,
        program_root: ExprId,
    },
    #[error("certificate closure has no monomial arena for retained root {monomial:?}")]
    MissingMonomialArena { monomial: super::monomial::MonomialId },
    #[error("certificate closure cannot read retained monomial {monomial:?}: {source}")]
    InvalidMonomial {
        monomial: super::monomial::MonomialId,
        #[source]
        source: super::monomial::MonomialError,
    },
}

/// Collect the transitive dependency closure of exactly one production root.
///
/// Callers pass the already projected residual root; this API cannot accept a decoder root and
/// never enumerates family lanes or selectors.
#[cfg(test)]
pub(crate) fn collect_residual_closure(
    job: &super::job::CheckerJob,
    root: &CertificateResidualRoot,
) -> Result<CertificateClosure, CertificateClosureError> {
    collect_residual_closure_with_plans(job, root, &[])
}

fn collect_residual_closure_with_plans(
    job: &super::job::CheckerJob,
    root: &CertificateResidualRoot,
    index_plans: &[&super::g0::IndexUsePlan],
) -> Result<CertificateClosure, CertificateClosureError> {
    let mut closure = CertificateClosure {
        expressions: BTreeSet::new(),
        programs: BTreeSet::new(),
        families: BTreeSet::new(),
        source_ids: BTreeSet::new(),
        family_source_ids: BTreeSet::new(),
        event_ids: BTreeSet::new(),
        constant_expressions: BTreeSet::new(),
    };
    let mut work = Vec::new();
    match root {
        CertificateResidualRoot::Closed { root, .. } => {
            work.push(CertificateWork::Expression(root.expression()))
        }
        CertificateResidualRoot::Family { family, .. } => {
            work.push(CertificateWork::Family(*family))
        }
    }
    walk_certificate_closure(job, &mut closure, work, index_plans)?;
    Ok(closure)
}

fn push_index_plan_dependencies(work: &mut Vec<CertificateWork>, plan: &super::g0::IndexUsePlan) {
    work.extend(plan.expression_roots().map(CertificateWork::Expression));
    work.extend(plan.result_family.into_iter().map(CertificateWork::Family));
    work.extend(plan.consumed_family.into_iter().map(CertificateWork::Family));
}

fn walk_certificate_closure(
    job: &super::job::CheckerJob,
    closure: &mut CertificateClosure,
    mut work: Vec<CertificateWork>,
    index_plans: &[&super::g0::IndexUsePlan],
) -> Result<(), CertificateClosureError> {
    let mut expression_plans = HashMap::<ExprId, Vec<usize>>::new();
    let mut family_plans = HashMap::<FamilyValueId, Vec<usize>>::new();
    for (plan_index, plan) in index_plans.iter().enumerate() {
        for expression in plan.result.into_iter().chain(plan.consumed) {
            expression_plans.entry(expression).or_default().push(plan_index);
        }
        for family in plan.result_family.into_iter().chain(plan.consumed_family) {
            family_plans.entry(family).or_default().push(plan_index);
        }
    }

    let mut seen_plans = HashSet::new();
    for (plan_index, plan) in index_plans.iter().enumerate() {
        if plan.is_residual_relevant(closure) && seen_plans.insert(plan_index) {
            push_index_plan_dependencies(&mut work, plan);
        }
    }

    while let Some(item) = work.pop() {
        match item {
            CertificateWork::Expression(expression) => {
                if !closure.expressions.insert(expression) {
                    continue;
                }
                if let Some(attached) = expression_plans.get(&expression) {
                    for &plan_index in attached {
                        let plan = index_plans[plan_index];
                        if plan.is_residual_relevant(closure) && seen_plans.insert(plan_index) {
                            push_index_plan_dependencies(&mut work, plan);
                        }
                    }
                }
                let node = job.expressions().node(expression)?;
                match &node.operator {
                    ValueOperator::Constant(_) => {
                        closure.constant_expressions.insert(expression);
                    }
                    ValueOperator::Source(source) => {
                        closure.source_ids.insert(source.clone());
                        if let Some(event) = source.sample_event {
                            closure.event_ids.insert(event);
                        }
                    }
                    ValueOperator::OpaqueFamilyElement { source } => {
                        closure.family_source_ids.insert(source.clone());
                    }
                    ValueOperator::Sample { event, .. } | ValueOperator::Sampler { event, .. } => {
                        closure.event_ids.insert(*event);
                    }
                    ValueOperator::Trapdoor(super::arena::TrapdoorOperation::Generate {
                        paired_public_event,
                        ..
                    }) => {
                        closure.event_ids.insert(*paired_public_event);
                    }
                    _ => {}
                }
                work.extend(node.inputs.iter().copied().map(CertificateWork::Expression));
                if let ValueOperator::ProgramCall { program } = node.operator {
                    work.push(CertificateWork::Program(program));
                }
            }
            CertificateWork::Program(program) => {
                if !closure.programs.insert(program) {
                    continue;
                }
                let record = job.programs().program(program)?;
                work.push(CertificateWork::Expression(record.root));
                if let Some(family) = job.programs().family_for_program(program) {
                    work.push(CertificateWork::Family(family));
                }
            }
            CertificateWork::Family(family) => {
                if !closure.families.insert(family) {
                    continue;
                }
                if let Some(attached) = family_plans.get(&family) {
                    for &plan_index in attached {
                        let plan = index_plans[plan_index];
                        if plan.is_residual_relevant(closure) && seen_plans.insert(plan_index) {
                            push_index_plan_dependencies(&mut work, plan);
                        }
                    }
                }
                let program = family.program();
                let family_body = job.programs().family_body(family)?;
                let program_root = job.programs().program(program)?.root;
                if family_body != program_root {
                    return Err(CertificateClosureError::FamilyProgramMismatch {
                        family,
                        program,
                        family_body,
                        program_root,
                    });
                }
                work.push(CertificateWork::Program(program));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub(crate) enum CertificateProjectionError {
    #[error(
        "certificate projection requires a threshold decoder for target {target_id:?}, got {kind:?}"
    )]
    UnsupportedDecoderKind { target_id: String, kind: OperationalDecoderKind },
    #[error("certificate residual root for target {target_id:?} is not a matrix: {actual:?}")]
    ResidualRootNotMatrix { target_id: String, actual: ResolvedValueType },
    #[error(
        "certificate residual modulus mismatch for target {target_id:?}: target q={target:?}, residual q={residual:?}"
    )]
    ResidualModulusMismatch { target_id: String, target: BigUint, residual: BigUint },
    #[error("certificate projection failed during lowering: {detail}")]
    Lowering { detail: String },
    #[error("certificate residual closure failed: {0}")]
    Closure(#[from] CertificateClosureError),
    #[error("operational checker rejected certificate projection: {0}")]
    Operational(#[from] OperationalSimulationError),
    #[error("operational checker report rejected certificate target {target_id:?}")]
    Rejected { target_id: String },
    #[error("operational checker report mismatched certificate target {target_id:?}: {detail}")]
    ReportMismatch { target_id: String, detail: String },
    #[error("proof payload projection failed: {detail}")]
    ProofPayload { detail: String },
    #[error("proof payload invariant failed: {context:?}")]
    ProofInvariant { context: Box<ProofInvariantContext> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProofEvidenceKind {
    Universal,
    Gadget,
    OperatorMerge,
    SurvivorFold,
    EventReference,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofInvariantMismatch {
    EventOrder {
        referenced: u64,
    },
    EventKind {
        referenced: u64,
        expected: &'static str,
    },
    MissingExactNormalForm {
        referenced: u64,
    },
    MissingTerm {
        referenced: u64,
        monomial: super::monomial::MonomialId,
    },
    Owner {
        expected: super::arena::ScopedExprId,
        actual: Option<super::arena::ScopedExprId>,
    },
    Coefficient {
        expected: BigInt,
        actual: BigInt,
    },
    SpliceRange {
        start: u64,
        end_exclusive: u64,
        ordered_len: u64,
    },
    SpliceOutput {
        expected_central: Vec<ProofPayloadOwner>,
        expected_ordered: Vec<ProofPayloadOwner>,
        actual_central: Vec<ProofPayloadOwner>,
        actual_ordered: Vec<ProofPayloadOwner>,
    },
    RhsReplay,
    MonomialRole {
        role: &'static str,
        expected: ProofPayloadMonomial,
        actual: ProofPayloadMonomial,
    },
    SurvivorMagnitude {
        expected: BigUint,
        actual: BigUint,
    },
    Operator {
        actual: Box<ValueOperator>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofInvariantContext {
    pub event: u64,
    pub owner: Option<super::arena::ScopedExprId>,
    pub evidence: ProofEvidenceKind,
    pub mismatch: ProofInvariantMismatch,
}

fn proof_invariant(
    event: usize,
    owner: Option<super::arena::ScopedExprId>,
    evidence: ProofEvidenceKind,
    mismatch: ProofInvariantMismatch,
) -> CertificateProjectionError {
    CertificateProjectionError::ProofInvariant {
        context: Box::new(ProofInvariantContext { event: event as u64, owner, evidence, mismatch }),
    }
}

fn payload_projection<T>(result: Result<T, G0Error>) -> Result<T, CertificateProjectionError> {
    result.map_err(proof_payload_error)
}

fn exact_scalar_tensor_contract(
    operation: &MatrixOperation,
    left: &ResolvedMatrixType,
    right: &ResolvedMatrixType,
) -> bool {
    let MatrixOperation::Tensor { output, left_layout, right_layout, output_layout } = operation
    else {
        return false;
    };
    let left_scalar = left.rows == 1 && left.columns == 1;
    let right_scalar = right.rows == 1 && right.columns == 1;
    let expected = if left_scalar && right_scalar {
        left
    } else if left_scalar {
        right
    } else if right_scalar {
        left
    } else {
        return false;
    };
    *left_layout == MatrixLayout::row_major(left.rows, left.columns) &&
        *right_layout == MatrixLayout::row_major(right.rows, right.columns) &&
        *output_layout == MatrixLayout::row_major(output.rows, output.columns) &&
        output.modulus == left.modulus &&
        output.modulus == right.modulus &&
        output.ring_dimension == left.ring_dimension &&
        output.ring_dimension == right.ring_dimension &&
        left.rows.checked_mul(right.rows) == Some(output.rows) &&
        left.columns.checked_mul(right.columns) == Some(output.columns) &&
        output == expected
}

/// Re-runs target resolution and production lowering only when certificate emission is explicitly
/// requested. Normal simulation entry points never call this function.
pub(crate) fn project_operational_certificate(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<OperationalCertificateProjection, CertificateProjectionError> {
    Ok(prepare_operational_certificate(protocol, request)?.projection)
}

/// Project the retained Rust trace into one canonical, in-memory proof payload.  This is a
/// crate-local boundary: it does not serialize, emit Lean, or retain a second dependency graph.
pub(crate) fn derive_proof_payload(
    run: &OperationalCertificateRun,
) -> Result<OperationalProofPayload, CertificateProjectionError> {
    Ok(derive_proof_payload_projection(run)?.payload)
}

fn closed_residual_expression(run: &OperationalCertificateRun) -> Option<ExprId> {
    match &run.projection.residual {
        CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
        CertificateResidualRoot::Family { .. } => None,
    }
}

pub(crate) fn derive_proof_payload_projection(
    run: &OperationalCertificateRun,
) -> Result<ProofPayloadProjection, CertificateProjectionError> {
    let refs = super::g0::derive_certificate_statement_rows(
        &run.job,
        &run.projection.closure,
        &run.trace,
        closed_residual_expression(run),
    )
    .map_err(proof_payload_error)?;
    derive_proof_payload_projection_with_refs(run, &refs)
}

pub(crate) fn derive_certificate_documents(
    run: &OperationalCertificateRun,
) -> Result<ProjectedCertificateDocuments, CertificateProjectionError> {
    let refs = super::g0::derive_certificate_statement_rows(
        &run.job,
        &run.projection.closure,
        &run.trace,
        closed_residual_expression(run),
    )
    .map_err(proof_payload_error)?;
    let proof = derive_proof_payload_projection_with_refs(run, &refs)?;
    let cert = super::certificate_schema::project_certificate_document(run, &refs)
        .map_err(|error| CertificateProjectionError::ProofPayload { detail: error.to_string() })?;
    Ok(ProjectedCertificateDocuments { cert, proof })
}

fn derive_proof_payload_projection_with_refs(
    run: &OperationalCertificateRun,
    refs: &CanonicalStatementRows,
) -> Result<ProofPayloadProjection, CertificateProjectionError> {
    validate_raw_event_coverage(&run.trace)?;
    let closure = &run.projection.closure;
    let closed_root_expression = match &run.projection.residual {
        CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
        CertificateResidualRoot::Family { .. } => None,
    };
    let closed_program = closed_wrapper_program(&run.trace, closed_root_expression);
    let mut monomial_arenas = HashMap::new();
    for scope in closure.programs.iter().copied() {
        if let Some(arena) = run.job.monomials().get(scope) {
            monomial_arenas.insert(arena.token(), arena);
        }
    }
    for (event_index, event) in run.trace.events.iter().enumerate() {
        let scope = match event {
            NormalizerEvent::InvocationStart { root } |
            NormalizerEvent::Result { owner: root, .. } |
            NormalizerEvent::InvocationEnd { root, .. } |
            NormalizerEvent::SpecializationComputed { owner: root, .. } |
            NormalizerEvent::SpecializationCacheHit { owner: root, .. } |
            NormalizerEvent::BoundTransfer { owner: root, .. } => root.program(),
            NormalizerEvent::CoefficientMerge(observation) => observation.owner.program(),
            NormalizerEvent::Predecessor { consumer, .. } => consumer.program(),
            NormalizerEvent::AppliedRelation(observation) => observation.owner.program(),
            NormalizerEvent::SurvivorFold(observation) => {
                match run.trace.events.get(observation.bound.0 as usize) {
                    Some(NormalizerEvent::BoundTransfer { owner, .. }) => owner.program(),
                    _ => {
                        return Err(proof_invariant(
                            event_index,
                            None,
                            ProofEvidenceKind::SurvivorFold,
                            ProofInvariantMismatch::EventKind {
                                referenced: observation.bound.0,
                                expected: "BoundTransfer",
                            },
                        ));
                    }
                }
            }
            // PreFoldPolynomial has no owner field; its monomial arenas are already covered by
            // the invocation roots in the residual closure.
            NormalizerEvent::PreFoldPolynomial(_) => continue,
        };
        if let Some(arena) = run.job.monomials().get(scope) {
            monomial_arenas.insert(arena.token(), arena);
        }
    }
    let mut rhs_events = HashMap::new();
    for (index, event) in run.trace.events.iter().enumerate() {
        if let NormalizerEvent::SpecializationComputed { key, replay, .. } = event {
            for (rhs, result) in replay.rhs_results.iter() {
                rhs_events.insert((key.clone(), replay.range, *rhs), (index as u64, result.0));
            }
        }
    }
    let projector = ProofPayloadProjector {
        job: &run.job,
        refs,
        monomial_arenas,
        rhs_events,
        closed_program,
        closed_root_expression: closed_root_expression
            .map(|expression| refs.expression(expression))
            .transpose()
            .map_err(proof_payload_error)?,
        frames: Vec::new(),
    };
    let retained_support_items = generator_support_logical_items(
        refs.retained_logical_items().map_err(proof_payload_error)?,
        &projector.monomial_arenas,
        &projector.rhs_events,
        projector.closed_program,
        projector.closed_root_expression,
    )
    .map_err(generator_retention_error)?;
    projector.project(&run.trace, closure, retained_support_items)
}

fn raw_event_coverage_disposition(kind: &EventKind) -> RawEventCoverageDisposition {
    match kind {
        EventKind::Sample { .. } | EventKind::Sampler { .. } => {
            RawEventCoverageDisposition::CanonicalProjection
        }
        EventKind::Trapdoor { .. } => {
            RawEventCoverageDisposition::RejectBeforeCanonicalProjection {
                reason: "raw EventKind::Trapdoor has no canonical residual event projection",
            }
        }
    }
}

fn validate_raw_event_coverage(trace: &FeasibilityTrace) -> Result<(), CertificateProjectionError> {
    for observation in trace.event_observations().values() {
        match raw_event_coverage_disposition(&observation.kind) {
            RawEventCoverageDisposition::CanonicalProjection => {}
            RawEventCoverageDisposition::RejectBeforeCanonicalProjection { reason } => {
                return Err(CertificateProjectionError::ProofPayload { detail: reason.to_owned() });
            }
        }
    }
    Ok(())
}

fn observed_scalar_kind(operation: &ScalarOperation) -> ObservedScalarKind {
    match operation {
        ScalarOperation::Add => ObservedScalarKind::Add,
        ScalarOperation::Subtract => ObservedScalarKind::Subtract,
        ScalarOperation::Multiply => ObservedScalarKind::Multiply,
        ScalarOperation::Divide => ObservedScalarKind::Divide,
        ScalarOperation::Remainder => ObservedScalarKind::Remainder,
        ScalarOperation::Negate => ObservedScalarKind::Negate,
        ScalarOperation::Equal => ObservedScalarKind::Equal,
        ScalarOperation::Less => ObservedScalarKind::Less,
        ScalarOperation::LessEqual => ObservedScalarKind::LessEqual,
        ScalarOperation::BoolToInt => ObservedScalarKind::BoolToInt,
        ScalarOperation::IntToReal => ObservedScalarKind::IntToReal,
        ScalarOperation::RealAdd => ObservedScalarKind::RealAdd,
        ScalarOperation::RealSubtract => ObservedScalarKind::RealSubtract,
        ScalarOperation::RealMultiply => ObservedScalarKind::RealMultiply,
        ScalarOperation::RealDivide => ObservedScalarKind::RealDivide,
        ScalarOperation::RealSqrt => ObservedScalarKind::RealSqrt,
        ScalarOperation::ThresholdDecode { .. } => ObservedScalarKind::ThresholdDecode,
        ScalarOperation::Bit { .. } => ObservedScalarKind::Bit,
        ScalarOperation::Slice { .. } => ObservedScalarKind::Slice,
        ScalarOperation::Hash { .. } => ObservedScalarKind::Hash,
        ScalarOperation::ExtractCoefficient { .. } => ObservedScalarKind::ExtractCoefficient,
        ScalarOperation::LiftConstantPolynomial { .. } => {
            ObservedScalarKind::LiftConstantPolynomial
        }
    }
}

fn observed_matrix_kind(operation: &MatrixOperation) -> ObservedMatrixKind {
    match operation {
        MatrixOperation::Add => ObservedMatrixKind::Add,
        MatrixOperation::Subtract => ObservedMatrixKind::Subtract,
        MatrixOperation::Multiply => ObservedMatrixKind::Multiply,
        MatrixOperation::Negate => ObservedMatrixKind::Negate,
        MatrixOperation::Scale => ObservedMatrixKind::Scale,
        MatrixOperation::RingAutomorphism { .. } => ObservedMatrixKind::RingAutomorphism,
        MatrixOperation::Transpose => ObservedMatrixKind::Transpose,
        MatrixOperation::Slice { .. } => ObservedMatrixKind::Slice,
        MatrixOperation::IndexedSlice { .. } => ObservedMatrixKind::IndexedSlice,
        MatrixOperation::View { .. } => ObservedMatrixKind::View,
        MatrixOperation::Concat { .. } => ObservedMatrixKind::Concat,
        MatrixOperation::Tensor { .. } => ObservedMatrixKind::Tensor,
        MatrixOperation::CrtRecompose { .. } => ObservedMatrixKind::CrtRecompose,
        MatrixOperation::ExtractCoefficient { .. } => ObservedMatrixKind::ExtractCoefficient,
        MatrixOperation::LiftConstantPolynomial { .. } => {
            ObservedMatrixKind::LiftConstantPolynomial
        }
    }
}

fn observed_operator_kind(operator: &ValueOperator) -> Option<ObservedOperatorKind> {
    Some(match operator {
        ValueOperator::Argument { .. } => ObservedOperatorKind::Argument,
        ValueOperator::Constant(_) => ObservedOperatorKind::Constant,
        ValueOperator::Source(_) => ObservedOperatorKind::Source,
        ValueOperator::Sample { .. } |
        ValueOperator::Sampler { .. } |
        ValueOperator::Transform(_) => return None,
        ValueOperator::DeterministicHash(_) => ObservedOperatorKind::DeterministicHash,
        ValueOperator::OpaqueFamilyElement { .. } => ObservedOperatorKind::OpaqueFamilyElement,
        ValueOperator::IndexMap { .. } => ObservedOperatorKind::IndexMap,
        ValueOperator::ExplicitElement { .. } => ObservedOperatorKind::ExplicitElement,
        ValueOperator::ProgramCall { .. } => ObservedOperatorKind::ProgramCall,
        ValueOperator::ExtractCoefficient { .. } => ObservedOperatorKind::ExtractCoefficient,
        ValueOperator::Scalar(operation) => {
            ObservedOperatorKind::Scalar(observed_scalar_kind(operation))
        }
        ValueOperator::Matrix(operation) => {
            ObservedOperatorKind::Matrix(observed_matrix_kind(operation))
        }
        ValueOperator::Trapdoor(operation) => ObservedOperatorKind::Trapdoor(match operation {
            TrapdoorOperation::Generate { .. } => ObservedTrapdoorKind::Generate,
            TrapdoorOperation::Transform { .. } => ObservedTrapdoorKind::Transform,
        }),
    })
}

fn observed_transform_kind(operator: &ValueOperator) -> Option<ObservedTransformKind> {
    match operator {
        ValueOperator::Transform(ValueTransformOperation::GadgetDecompose { small, .. }) => {
            Some(ObservedTransformKind::GadgetDecompose(if *small {
                ObservedGadgetKind::Small
            } else {
                ObservedGadgetKind::Regular
            }))
        }
        ValueOperator::Transform(ValueTransformOperation::PackPolynomialCoefficients {
            ..
        }) => Some(ObservedTransformKind::PackPolynomialCoefficients),
        ValueOperator::Argument { .. } |
        ValueOperator::Constant(_) |
        ValueOperator::Source(_) |
        ValueOperator::Sample { .. } |
        ValueOperator::Sampler { .. } |
        ValueOperator::DeterministicHash(_) |
        ValueOperator::OpaqueFamilyElement { .. } |
        ValueOperator::IndexMap { .. } |
        ValueOperator::ExplicitElement { .. } |
        ValueOperator::ProgramCall { .. } |
        ValueOperator::ExtractCoefficient { .. } |
        ValueOperator::Scalar(_) |
        ValueOperator::Matrix(_) |
        ValueOperator::Trapdoor(_) => None,
    }
}

fn observed_sampler_kind(kind: &CanonicalEventKind) -> ObservedSamplerKind {
    match kind {
        CanonicalEventKind::Sample { .. } => ObservedSamplerKind::Sample,
        CanonicalEventKind::Sampler { operation } => match operation {
            StableSamplerOperation::UniformResidue { .. } => ObservedSamplerKind::UniformResidue,
            StableSamplerOperation::UniformInterval { .. } => ObservedSamplerKind::UniformInterval,
            StableSamplerOperation::Gaussian { .. } => ObservedSamplerKind::Gaussian,
            StableSamplerOperation::Hash { variant, .. } => {
                ObservedSamplerKind::Hash(match variant {
                    StableHashVariant::Plain => ObservedHashKind::Plain,
                    StableHashVariant::Decomposed => ObservedHashKind::Decomposed,
                    StableHashVariant::SmallDecomposed => ObservedHashKind::SmallDecomposed,
                })
            }
            StableSamplerOperation::Trapdoor { .. } => ObservedSamplerKind::Trapdoor,
            StableSamplerOperation::Preimage { .. } => ObservedSamplerKind::Preimage,
        },
    }
}

fn observed_relation_kind(rule: &ProofPayloadRelationRule) -> ObservedRelationKind {
    match rule {
        ProofPayloadRelationRule::Universal { .. } => ObservedRelationKind::Universal,
        ProofPayloadRelationRule::Gadget { .. } => ObservedRelationKind::Gadget,
    }
}

fn observed_bound_kind(rule: &ProofPayloadRule) -> ObservedBoundKind {
    match rule {
        ProofPayloadRule::Authority(authority) => ObservedBoundKind::Authority(match authority {
            ProofPayloadAuthority::FactStore => ObservedAuthorityKind::FactStore,
            ProofPayloadAuthority::ProgramFamilyFact => ObservedAuthorityKind::ProgramFamilyFact,
            ProofPayloadAuthority::Operator => ObservedAuthorityKind::Operator,
            ProofPayloadAuthority::RelationPreimageSource { .. } => {
                ObservedAuthorityKind::RelationPreimageSource
            }
            ProofPayloadAuthority::Unavailable => ObservedAuthorityKind::Unavailable,
        }),
        ProofPayloadRule::Identity { .. } => ObservedBoundKind::Identity,
        ProofPayloadRule::RingAutomorphism { .. } => ObservedBoundKind::RingAutomorphism,
        ProofPayloadRule::Sum { .. } => ObservedBoundKind::Sum,
        ProofPayloadRule::Maximum { .. } => ObservedBoundKind::Maximum,
        ProofPayloadRule::Scale { .. } => ObservedBoundKind::Scale,
        ProofPayloadRule::MonomialProduct { .. } => ObservedBoundKind::MonomialProduct,
        ProofPayloadRule::WeightedSum { .. } => ObservedBoundKind::WeightedSum,
        ProofPayloadRule::Product { .. } => ObservedBoundKind::Product,
        ProofPayloadRule::Tensor { .. } => ObservedBoundKind::Tensor,
    }
}

fn derive_observed_coverage(
    job: &super::job::CheckerJob,
    closure: &CertificateClosure,
    refs: &CanonicalStatementRows,
    events: &[ProofPayloadEvent],
) -> Result<ObservedCoverage, CertificateProjectionError> {
    let mut sites = BTreeMap::<ObservedCoverageKind, BTreeSet<ObservedCoverageSite>>::new();
    let mut observe = |kind, site| {
        sites.entry(kind).or_default().insert(site);
    };
    for &expression in &closure.expressions {
        let node = job
            .expressions()
            .node(expression)
            .map_err(|error| proof_payload_error(G0Error::Arena(error)))?;
        let site = ObservedCoverageSite::ExpressionRow {
            row: refs.expression(expression).map_err(proof_payload_error)?,
        };
        if let Some(kind) = observed_operator_kind(&node.operator) {
            observe(ObservedCoverageKind::Operator(kind), site);
        }
        if let Some(kind) = observed_transform_kind(&node.operator) {
            observe(ObservedCoverageKind::Transform(kind), site);
        }
    }
    for (row, event) in refs.event_rows().rows().iter().enumerate() {
        observe(
            ObservedCoverageKind::Sampler(observed_sampler_kind(&event.kind)),
            ObservedCoverageSite::SamplerEventRow {
                row: u64::try_from(row).map_err(|_| {
                    generator_retention_error(CanonicalPayloadError::LengthOverflow)
                })?,
            },
        );
    }
    for (index, event) in events.iter().enumerate() {
        let site = ObservedCoverageSite::TraceEvent {
            index: u64::try_from(index)
                .map_err(|_| generator_retention_error(CanonicalPayloadError::LengthOverflow))?,
        };
        match event {
            ProofPayloadEvent::AppliedRelation { rule, .. } => {
                observe(ObservedCoverageKind::Relation(observed_relation_kind(rule)), site)
            }
            ProofPayloadEvent::BoundTransfer { rule, .. } => {
                observe(ObservedCoverageKind::Bound(observed_bound_kind(rule)), site)
            }
            _ => {}
        }
    }
    finalize_observed_coverage(sites)
}

fn finalize_observed_coverage(
    sites: BTreeMap<ObservedCoverageKind, BTreeSet<ObservedCoverageSite>>,
) -> Result<ObservedCoverage, CertificateProjectionError> {
    let rows = sites
        .into_iter()
        .map(|(kind, sites)| {
            let sites = sites.into_iter().collect::<Vec<_>>();
            let count = u64::try_from(sites.len())
                .map_err(|_| generator_retention_error(CanonicalPayloadError::LengthOverflow))?;
            Ok::<_, CertificateProjectionError>(ObservedCoverageRow { kind, count, sites })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ObservedCoverage { rows })
}

const NORMALIZE_RUST_ITEM: &str =
    "crates/correctness/src/operational_noise/normal_form.rs::Normalizer::normalize_inner";
const SAMPLER_BOUND_RUST_ITEM: &str =
    "crates/correctness/src/operational_noise/normal_form.rs::sampler_bound";
const RELATION_RUST_ITEM: &str = "crates/correctness/src/operational_noise/normal_form.rs::\
Normalizer::append_contextual_relation_evidence";
const BOUND_RUST_ITEM: &str =
    "crates/correctness/src/operational_noise/normal_form.rs::Normalizer::bound_normal_form";

fn checked_lean(
    rust_item: &'static str,
    semantics: &'static str,
    transfer: &'static str,
) -> ResidualCoverageClassification {
    ResidualCoverageClassification {
        rust_item,
        disposition: ResidualCoverageDisposition::CheckedLean { semantics, transfer },
    }
}

fn g2_obligation(
    rust_item: &'static str,
    semantics: &'static str,
    transfer: &'static str,
) -> ResidualCoverageClassification {
    ResidualCoverageClassification {
        rust_item,
        disposition: ResidualCoverageDisposition::G2LeanObligation { semantics, transfer },
    }
}

fn reject_before_generation(
    rust_item: &'static str,
    reason: &'static str,
) -> ResidualCoverageClassification {
    ResidualCoverageClassification {
        rust_item,
        disposition: ResidualCoverageDisposition::RejectBeforeGeneration { reason },
    }
}

fn scalar_coverage_classification(kind: ObservedScalarKind) -> ResidualCoverageClassification {
    match kind {
        ObservedScalarKind::Add => checked_lean(
            NORMALIZE_RUST_ITEM,
            "Mxx.Certificate.OperationalNoise.coefficient_add",
            "Mxx.Certificate.OperationalNoise.EventReplay.boundTransfer_sum",
        ),
        ObservedScalarKind::Subtract => checked_lean(
            NORMALIZE_RUST_ITEM,
            "Mxx.Certificate.OperationalNoise.coefficient_subtract",
            "Mxx.Certificate.OperationalNoise.EventReplay.boundTransfer_sum",
        ),
        ObservedScalarKind::Multiply => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "Mxx.Certificate.OperationalNoise.EventReplay.\
productMerge_contribution_coefficient",
            "Mxx.Certificate.OperationalNoise.EventReplay.boundTransfer_product",
        ),
        ObservedScalarKind::ThresholdDecode => reject_before_generation(
            NORMALIZE_RUST_ITEM,
            "threshold decoding is outside residual certificate generation",
        ),
        ObservedScalarKind::Divide |
        ObservedScalarKind::Remainder |
        ObservedScalarKind::Negate |
        ObservedScalarKind::Equal |
        ObservedScalarKind::Less |
        ObservedScalarKind::LessEqual |
        ObservedScalarKind::BoolToInt |
        ObservedScalarKind::IntToReal |
        ObservedScalarKind::RealAdd |
        ObservedScalarKind::RealSubtract |
        ObservedScalarKind::RealMultiply |
        ObservedScalarKind::RealDivide |
        ObservedScalarKind::RealSqrt |
        ObservedScalarKind::Bit |
        ObservedScalarKind::Slice |
        ObservedScalarKind::Hash |
        ObservedScalarKind::ExtractCoefficient |
        ObservedScalarKind::LiftConstantPolynomial => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "exact scalar normalization semantics",
            "coefficient-bound transfer",
        ),
    }
}

fn matrix_coverage_classification(kind: ObservedMatrixKind) -> ResidualCoverageClassification {
    match kind {
        ObservedMatrixKind::Add |
        ObservedMatrixKind::Subtract |
        ObservedMatrixKind::Multiply |
        ObservedMatrixKind::Negate |
        ObservedMatrixKind::Scale |
        ObservedMatrixKind::RingAutomorphism |
        ObservedMatrixKind::Transpose |
        ObservedMatrixKind::Slice |
        ObservedMatrixKind::IndexedSlice |
        ObservedMatrixKind::View |
        ObservedMatrixKind::Concat |
        ObservedMatrixKind::Tensor |
        ObservedMatrixKind::CrtRecompose |
        ObservedMatrixKind::ExtractCoefficient |
        ObservedMatrixKind::LiftConstantPolynomial => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "exact matrix normalization semantics",
            "matrix coefficient-bound transfer",
        ),
    }
}

fn operator_coverage_classification(kind: ObservedOperatorKind) -> ResidualCoverageClassification {
    match kind {
        ObservedOperatorKind::Scalar(kind) => scalar_coverage_classification(kind),
        ObservedOperatorKind::Matrix(kind) => matrix_coverage_classification(kind),
        ObservedOperatorKind::Trapdoor(
            ObservedTrapdoorKind::Generate | ObservedTrapdoorKind::Transform,
        ) => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "trapdoor operator normalization semantics",
            "trapdoor operator coefficient-bound transfer",
        ),
        ObservedOperatorKind::Argument |
        ObservedOperatorKind::Constant |
        ObservedOperatorKind::Source |
        ObservedOperatorKind::DeterministicHash |
        ObservedOperatorKind::OpaqueFamilyElement |
        ObservedOperatorKind::IndexMap |
        ObservedOperatorKind::ExplicitElement |
        ObservedOperatorKind::ProgramCall |
        ObservedOperatorKind::ExtractCoefficient => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "exact value normalization semantics",
            "value coefficient-bound transfer",
        ),
    }
}

fn transform_coverage_classification(
    kind: ObservedTransformKind,
) -> ResidualCoverageClassification {
    match kind {
        ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Regular) => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "regular gadget decomposition semantics",
            "regular gadget decomposition bound transfer",
        ),
        ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Small) => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "small gadget decomposition semantics",
            "small gadget decomposition bound transfer",
        ),
        ObservedTransformKind::PackPolynomialCoefficients => g2_obligation(
            NORMALIZE_RUST_ITEM,
            "polynomial coefficient packing semantics",
            "packing coefficient-bound transfer",
        ),
    }
}

fn sampler_coverage_classification(kind: ObservedSamplerKind) -> ResidualCoverageClassification {
    match kind {
        ObservedSamplerKind::Sample => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "declared sample semantics",
            "declared sample bound transfer",
        ),
        ObservedSamplerKind::UniformResidue => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "uniform residue sampler semantics",
            "uniform residue sampler bound transfer",
        ),
        ObservedSamplerKind::UniformInterval => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "uniform interval sampler semantics",
            "uniform interval sampler bound transfer",
        ),
        ObservedSamplerKind::Gaussian => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "Gaussian sampler semantics",
            "Gaussian sampler bound transfer",
        ),
        ObservedSamplerKind::Hash(ObservedHashKind::Plain) => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "plain hash sampler semantics",
            "plain hash sampler bound transfer",
        ),
        ObservedSamplerKind::Hash(ObservedHashKind::Decomposed) => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "decomposed hash sampler semantics",
            "decomposed hash sampler bound transfer",
        ),
        ObservedSamplerKind::Hash(ObservedHashKind::SmallDecomposed) => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "small decomposed hash sampler semantics",
            "small decomposed hash sampler bound transfer",
        ),
        ObservedSamplerKind::Trapdoor => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "canonical trapdoor sampler semantics",
            "canonical trapdoor sampler bound transfer",
        ),
        ObservedSamplerKind::Preimage => g2_obligation(
            SAMPLER_BOUND_RUST_ITEM,
            "preimage sampler semantics",
            "preimage sampler bound transfer",
        ),
    }
}

fn relation_coverage_classification(kind: ObservedRelationKind) -> ResidualCoverageClassification {
    match kind {
        ObservedRelationKind::Universal => g2_obligation(
            RELATION_RUST_ITEM,
            "universal relation replacement association",
            "universal relation replacement bound transfer",
        ),
        ObservedRelationKind::Gadget => g2_obligation(
            RELATION_RUST_ITEM,
            "gadget relation replacement association",
            "gadget relation replacement bound transfer",
        ),
    }
}

fn bound_coverage_classification(kind: ObservedBoundKind) -> ResidualCoverageClassification {
    match kind {
        ObservedBoundKind::Authority(ObservedAuthorityKind::Unavailable) => {
            reject_before_generation(
                BOUND_RUST_ITEM,
                "unavailable bound authority cannot justify a residual certificate",
            )
        }
        ObservedBoundKind::Authority(
            ObservedAuthorityKind::FactStore |
            ObservedAuthorityKind::ProgramFamilyFact |
            ObservedAuthorityKind::Operator |
            ObservedAuthorityKind::RelationPreimageSource,
        ) => g2_obligation(
            BOUND_RUST_ITEM,
            "bound authority association",
            "authority-provided coefficient bound",
        ),
        ObservedBoundKind::Identity |
        ObservedBoundKind::RingAutomorphism |
        ObservedBoundKind::Sum |
        ObservedBoundKind::Maximum |
        ObservedBoundKind::Scale |
        ObservedBoundKind::MonomialProduct |
        ObservedBoundKind::WeightedSum |
        ObservedBoundKind::Product |
        ObservedBoundKind::Tensor => g2_obligation(
            BOUND_RUST_ITEM,
            "bound-rule operand association",
            "bound-rule arithmetic transfer",
        ),
    }
}

fn residual_coverage_classification(kind: ObservedCoverageKind) -> ResidualCoverageClassification {
    match kind {
        ObservedCoverageKind::Operator(kind) => operator_coverage_classification(kind),
        ObservedCoverageKind::Transform(kind) => transform_coverage_classification(kind),
        ObservedCoverageKind::Sampler(kind) => sampler_coverage_classification(kind),
        ObservedCoverageKind::Relation(kind) => relation_coverage_classification(kind),
        ObservedCoverageKind::Bound(kind) => bound_coverage_classification(kind),
    }
}

fn derive_residual_coverage_matrix(
    coverage: &ObservedCoverage,
) -> Result<ResidualCoverageMatrix, CertificateProjectionError> {
    let mut previous_kind = None;
    let mut rows = Vec::with_capacity(coverage.rows.len());
    for row in &coverage.rows {
        if previous_kind.is_some_and(|previous| previous >= row.kind) {
            return Err(CertificateProjectionError::ProofPayload {
                detail: "observed coverage rows must be sorted and unique".to_owned(),
            });
        }
        if row.count !=
            u64::try_from(row.sites.len())
                .map_err(|_| generator_retention_error(CanonicalPayloadError::LengthOverflow))? ||
            !row.sites.windows(2).all(|sites| sites[0] < sites[1])
        {
            return Err(CertificateProjectionError::ProofPayload {
                detail: "observed coverage sites must be sorted, unique, and counted exactly"
                    .to_owned(),
            });
        }
        let classification = residual_coverage_classification(row.kind);
        if let ResidualCoverageDisposition::RejectBeforeGeneration { reason } =
            classification.disposition
        {
            return Err(CertificateProjectionError::ProofPayload { detail: reason.to_owned() });
        }
        rows.push(ResidualCoverageRow {
            kind: row.kind,
            count: row.count,
            sites: row.sites.clone(),
            rust_item: classification.rust_item,
            disposition: classification.disposition,
        });
        previous_kind = Some(row.kind);
    }
    Ok(ResidualCoverageMatrix { rows })
}

fn generator_support_logical_items(
    canonical_refs_items: u64,
    monomial_arenas: &HashMap<super::arena::ArenaToken, &super::monomial::MonomialArena>,
    rhs_events: &HashMap<
        (
            super::relation::RuntimeSpecializationKey,
            super::g0::EventRange,
            super::relation::CanonicalRhsId,
        ),
        (u64, u64),
    >,
    closed_program: Option<ValueProgramId>,
    closed_root_expression: Option<u64>,
) -> Result<u64, CanonicalPayloadError> {
    checked_sum([
        Ok(canonical_refs_items),
        logical_uniform_collection(monomial_arenas.len(), 2),
        // One RHS lookup key contains three typed references (specialization, range, RHS); its
        // value contains the Computed and Result event references.
        logical_uniform_collection(rhs_events.len(), 5),
        checked_add(1, u64::from(closed_program.is_some())),
        checked_add(1, u64::from(closed_root_expression.is_some())),
        OperationalProofPayload { events: Vec::new() }.logical_items(),
    ])
}

/// Add trace-owned scopes and relation dependencies to the same residual closure before any
/// canonical rows are assigned.  This keeps the closure, trace filtering, and canonical refs on
/// one authority instead of creating a second scope inventory for payload projection.
fn extend_certificate_closure(
    job: &super::job::CheckerJob,
    closure: &mut CertificateClosure,
    trace: &FeasibilityTrace,
) -> Result<(), CertificateClosureError> {
    let index_plans = trace.index_use_plans().collect::<Vec<_>>();
    let mut work = Vec::new();
    for event in &trace.events {
        match event {
            NormalizerEvent::InvocationStart { root } |
            NormalizerEvent::Result { owner: root, .. } |
            NormalizerEvent::InvocationEnd { root, .. } => {
                work.push(CertificateWork::Program(root.program()));
                work.push(CertificateWork::Expression(root.expression()));
            }
            NormalizerEvent::BoundTransfer { owner, rule } => {
                work.push(CertificateWork::Program(owner.program()));
                work.push(CertificateWork::Expression(owner.expression()));
                if let BoundRule::Authority(BoundAuthority::RelationPreimageSource { source }) =
                    rule
                {
                    work.push(CertificateWork::Program(owner.program()));
                    work.push(CertificateWork::Expression(*source));
                }
            }
            NormalizerEvent::SpecializationComputed { owner, key, .. } |
            NormalizerEvent::SpecializationCacheHit { owner, key, .. } => {
                work.push(CertificateWork::Program(owner.program()));
                work.push(CertificateWork::Expression(owner.expression()));
                push_universal_dependencies(&mut work, key);
            }
            NormalizerEvent::Predecessor { consumer, predecessor, .. } => {
                work.push(CertificateWork::Program(consumer.program()));
                work.push(CertificateWork::Expression(consumer.expression()));
                work.push(CertificateWork::Expression(*predecessor));
            }
            NormalizerEvent::AppliedRelation(observation) => {
                work.push(CertificateWork::Program(observation.owner.program()));
                work.push(CertificateWork::Expression(observation.owner.expression()));
                match &observation.rule {
                    AppliedRelationRule::Universal { key, .. } => {
                        push_universal_dependencies(&mut work, key);
                    }
                    AppliedRelationRule::Gadget { gadget, decomposition, input, .. } => {
                        work.push(CertificateWork::Program(gadget.program()));
                        work.push(CertificateWork::Program(decomposition.program()));
                        work.push(CertificateWork::Expression(gadget.expression()));
                        work.push(CertificateWork::Expression(decomposition.expression()));
                        work.push(CertificateWork::Expression(*input));
                    }
                }
            }
            NormalizerEvent::SurvivorFold(_) => {}
            NormalizerEvent::PreFoldPolynomial(_) => {}
            NormalizerEvent::CoefficientMerge(observation) => {
                work.push(CertificateWork::Program(observation.owner.program()));
                work.push(CertificateWork::Expression(observation.owner.expression()));
            }
        }
    }
    walk_certificate_closure(job, closure, work, &index_plans)?;

    // Normalization retains the monomial IDs used by the proof trace. Their descriptors can
    // refer to scoped factors detached from the residual expression DAG, so feed those exact
    // program/expression identities into the same closure walker before canonicalization.
    let monomial_arenas = closure
        .programs
        .iter()
        .filter_map(|&program| job.monomials().get(program))
        .map(|arena| (arena.token(), arena))
        .collect::<HashMap<_, _>>();
    let mut factor_work = Vec::new();
    if let Some(monomials) = trace.retained_monomial_roots() {
        for &monomial in monomials {
            let arena = monomial_arenas
                .get(&monomial.arena())
                .ok_or(CertificateClosureError::MissingMonomialArena { monomial })?;
            let descriptor = arena
                .descriptor(monomial)
                .map_err(|source| CertificateClosureError::InvalidMonomial { monomial, source })?;
            for factor in descriptor.central_factors.iter().chain(descriptor.ordered_factors.iter())
            {
                factor_work.push(CertificateWork::Program(factor.program()));
                factor_work.push(CertificateWork::Expression(factor.expression()));
            }
        }
    }
    walk_certificate_closure(job, closure, factor_work, &index_plans)
}

fn push_universal_dependencies(
    work: &mut Vec<CertificateWork>,
    key: &super::relation::RuntimeSpecializationKey,
) {
    work.push(CertificateWork::Family(key.dispatch.preimage_family));
    work.push(CertificateWork::Expression(key.dispatch.preimage_source.expression));
    work.push(CertificateWork::Expression(key.dispatch.trapdoor_source.expression));
    work.push(CertificateWork::Expression(key.index.expression()));
    work.push(CertificateWork::Program(key.index.program()));
}

fn closed_wrapper_program(
    trace: &FeasibilityTrace,
    root: Option<ExprId>,
) -> Option<ValueProgramId> {
    let root = root?;
    trace.events.iter().find_map(|event| match event {
        NormalizerEvent::InvocationStart { root: owner } if owner.expression() == root => {
            Some(owner.program())
        }
        _ => None,
    })
}

fn proof_payload_error(error: G0Error) -> CertificateProjectionError {
    CertificateProjectionError::ProofPayload { detail: error.to_string() }
}

fn generator_retention_error(_: CanonicalPayloadError) -> CertificateProjectionError {
    CertificateProjectionError::ProofPayload {
        detail: "proof payload generator retention overflow".to_owned(),
    }
}

struct ProofPayloadProjector<'a> {
    job: &'a super::job::CheckerJob,
    refs: &'a CanonicalStatementRows,
    monomial_arenas: HashMap<super::arena::ArenaToken, &'a super::monomial::MonomialArena>,
    rhs_events: HashMap<
        (
            super::relation::RuntimeSpecializationKey,
            super::g0::EventRange,
            super::relation::CanonicalRhsId,
        ),
        (u64, u64),
    >,
    closed_program: Option<ValueProgramId>,
    closed_root_expression: Option<u64>,
    frames: Vec<ProofProjectionFrame>,
}

struct ProofProjectionFrame {
    root: super::arena::ScopedExprId,
    start: usize,
    predecessor_bindings: HashMap<(super::arena::ScopedExprId, u32), u64>,
    last_exact_result: Option<u64>,
    last_pre_fold: Option<u64>,
}

fn active_predecessor_binding(
    frames: &[ProofProjectionFrame],
    owner: super::arena::ScopedExprId,
    input_position: u32,
) -> Option<u64> {
    frames
        .last()
        .and_then(|frame| frame.predecessor_bindings.get(&(owner, input_position)).copied())
}

fn reached_exact_coefficient_root(rule: &BoundRule) -> bool {
    match rule {
        BoundRule::Sum { inputs } => {
            !inputs.is_empty() &&
                inputs.iter().all(|input| {
                    matches!(
                        input,
                        BoundValueRef::Predecessor { projection: BoundProjection::Coefficient, .. }
                    )
                })
        }
        BoundRule::Product { left, right, .. } => [left, right].into_iter().all(|input| {
            matches!(
                input,
                BoundValueRef::Predecessor { projection: BoundProjection::Coefficient, .. }
            )
        }),
        _ => false,
    }
}

fn select_reached_exact_producers(
    candidates: &[(usize, &BoundRule)],
    summary: &super::facts::NumericContract<super::facts::CoefficientBound>,
) -> Result<(usize, Option<usize>), G0Error> {
    match summary {
        super::facts::NumericContract::Known(super::facts::CoefficientBound::ExactZero) => {
            let [(producer, _)] = candidates else {
                return Err(G0Error::UnsupportedBoundTransfer);
            };
            Ok((*producer, None))
        }
        super::facts::NumericContract::Known(super::facts::CoefficientBound::Finite(_)) => {
            if let [(producer, rule)] = candidates {
                if reached_exact_coefficient_root(rule) {
                    return Ok((*producer, None));
                }
            }
            let roots = candidates
                .iter()
                .filter(|(_, rule)| reached_exact_coefficient_root(rule))
                .collect::<Vec<_>>();
            let [root] = roots.as_slice() else {
                return Err(G0Error::UnsupportedBoundTransfer);
            };
            let Some((first, _)) = candidates.first() else {
                return Err(G0Error::UnsupportedBoundTransfer);
            };
            let Some((last, _)) = candidates.last() else {
                return Err(G0Error::UnsupportedBoundTransfer);
            };
            if root.0 != *first {
                return Err(G0Error::UnsupportedBoundTransfer);
            }
            Ok((*first, Some(*last)))
        }
        _ => Err(G0Error::UnsupportedBoundTransfer),
    }
}

fn coefficient_bound_is_within(
    recorded: &super::facts::NumericContract<super::facts::CoefficientBound>,
    replayed: &super::facts::NumericContract<super::facts::CoefficientBound>,
) -> bool {
    use super::facts::{CoefficientBound, NumericContract};
    match (recorded, replayed) {
        (NumericContract::Known(CoefficientBound::ExactZero), NumericContract::Known(_)) |
        (
            NumericContract::Known(CoefficientBound::Finite(_)),
            NumericContract::Known(CoefficientBound::Large),
        ) |
        (
            NumericContract::Known(CoefficientBound::Large),
            NumericContract::Known(CoefficientBound::Large),
        ) => true,
        (
            NumericContract::Known(CoefficientBound::Finite(recorded)),
            NumericContract::Known(CoefficientBound::Finite(replayed)),
        ) => recorded.maximum_absolute_coefficient <= replayed.maximum_absolute_coefficient,
        _ => false,
    }
}

fn reached_deferred_finite_summary_allowed(
    replayed_coefficient: &super::facts::NumericContract<super::facts::CoefficientBound>,
    has_owner_merge: bool,
) -> bool {
    matches!(
        replayed_coefficient,
        super::facts::NumericContract::Known(super::facts::CoefficientBound::Large)
    ) && has_owner_merge
}

impl<'a> ProofPayloadProjector<'a> {
    fn scalar_product_monomial(
        mut left: ProofPayloadMonomial,
        mut right: ProofPayloadMonomial,
        left_scalar: bool,
        right_scalar: bool,
    ) -> ProofPayloadMonomial {
        if left_scalar && !right_scalar {
            left.central_factors.extend(left.ordered_factors);
            left.central_factors.extend(right.central_factors);
            left.central_factors.sort();
            ProofPayloadMonomial {
                central_factors: left.central_factors,
                ordered_factors: right.ordered_factors,
            }
        } else if right_scalar && !left_scalar {
            right.central_factors.extend(right.ordered_factors);
            right.central_factors.extend(left.central_factors);
            right.central_factors.sort();
            ProofPayloadMonomial {
                central_factors: right.central_factors,
                ordered_factors: left.ordered_factors,
            }
        } else {
            left.central_factors.extend(right.central_factors);
            left.central_factors.sort();
            left.ordered_factors.extend(right.ordered_factors);
            ProofPayloadMonomial {
                central_factors: left.central_factors,
                ordered_factors: left.ordered_factors,
            }
        }
    }

    fn scalar_tensor_types(
        &self,
        node: &super::arena::ExprNode,
        operation: &MatrixOperation,
    ) -> Result<Option<(ResolvedMatrixType, ResolvedMatrixType)>, G0Error> {
        if !matches!(operation, MatrixOperation::Tensor { .. }) {
            return Ok(None);
        }
        let [left_expression, right_expression] = node.inputs.as_ref() else {
            return Ok(None);
        };
        let ResolvedValueType::Matrix(left) =
            self.job.expressions().value_type(*left_expression)?
        else {
            return Ok(None);
        };
        let ResolvedValueType::Matrix(right) =
            self.job.expressions().value_type(*right_expression)?
        else {
            return Ok(None);
        };
        if !exact_scalar_tensor_contract(operation, left, right) {
            return Ok(None);
        }
        Ok(Some((left.clone(), right.clone())))
    }

    fn project(
        mut self,
        trace: &FeasibilityTrace,
        closure: &CertificateClosure,
        retained_support_items: u64,
    ) -> Result<ProofPayloadProjection, CertificateProjectionError> {
        let event_count = trace.events.len();
        let mut events = Vec::with_capacity(event_count);
        let mut current_retained_logical_items = retained_support_items;
        let mut generator_peak_retained_logical_items = retained_support_items;
        for (index, event) in trace.events.iter().enumerate() {
            let projected = self.event(trace, index, event)?;
            current_retained_logical_items = checked_add(
                current_retained_logical_items,
                checked_add(1, projected.logical_items().map_err(generator_retention_error)?)
                    .map_err(generator_retention_error)?,
            )
            .map_err(generator_retention_error)?;
            generator_peak_retained_logical_items =
                generator_peak_retained_logical_items.max(current_retained_logical_items);
            events.push(projected);
            match event {
                NormalizerEvent::InvocationStart { root } => {
                    self.frames.push(ProofProjectionFrame {
                        root: *root,
                        start: index,
                        predecessor_bindings: HashMap::new(),
                        last_exact_result: None,
                        last_pre_fold: None,
                    });
                }
                NormalizerEvent::Predecessor { consumer, input_position, .. } => {
                    let frame = self.frames.last_mut().ok_or_else(|| {
                        proof_invariant(
                            index,
                            Some(*consumer),
                            ProofEvidenceKind::EventReference,
                            ProofInvariantMismatch::EventKind {
                                referenced: index as u64,
                                expected: "active invocation frame",
                            },
                        )
                    })?;
                    frame.predecessor_bindings.insert((*consumer, *input_position), index as u64);
                }
                NormalizerEvent::Result { owner, value } if value.exact_nf.is_some() => {
                    let frame = self
                        .frames
                        .last_mut()
                        .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
                    if frame.root == *owner {
                        frame.last_exact_result = Some(index as u64);
                    }
                }
                NormalizerEvent::PreFoldPolynomial(_) => {
                    self.frames
                        .last_mut()
                        .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?
                        .last_pre_fold = Some(index as u64);
                }
                NormalizerEvent::InvocationEnd { root, .. } => {
                    let frame = self
                        .frames
                        .pop()
                        .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
                    if frame.root != *root {
                        return Err(proof_payload_error(G0Error::RelationTraceInvariant));
                    }
                }
                _ => {}
            }
        }
        if !self.frames.is_empty() {
            return Err(proof_payload_error(G0Error::RelationTraceInvariant));
        }
        let observed_coverage = derive_observed_coverage(self.job, closure, self.refs, &events)?;
        current_retained_logical_items = checked_add(
            current_retained_logical_items,
            observed_coverage.logical_items().map_err(generator_retention_error)?,
        )
        .map_err(generator_retention_error)?;
        generator_peak_retained_logical_items =
            generator_peak_retained_logical_items.max(current_retained_logical_items);
        Ok(ProofPayloadProjection {
            payload: OperationalProofPayload { events },
            generator_peak_retained_logical_items,
            observed_coverage,
        })
    }

    fn owner(&self, owner: super::arena::ScopedExprId) -> Result<ProofPayloadOwner, G0Error> {
        let expression_row = self.refs.expression(owner.expression())?;
        let scope = if self.closed_program == Some(owner.program()) {
            ProofPayloadScope::Closed {
                root_expression_row: self
                    .closed_root_expression
                    .ok_or(G0Error::UnsupportedBoundTransfer)?,
            }
        } else {
            // Validate that user-owned program scopes are still real finalized programs.  The
            // synthetic closed wrapper is handled above and deliberately has no Program row.
            self.job
                .programs()
                .project_program(owner.program())
                .map_err(|_| G0Error::UnsupportedBoundTransfer)?;
            ProofPayloadScope::Program { program_row: self.refs.program(owner.program())? }
        };
        Ok(ProofPayloadOwner { scope, expression_row })
    }

    fn expression(&self, expression: super::arena::ExprId) -> Result<u64, G0Error> {
        self.refs.expression(expression)
    }

    fn authority(&self, authority: &BoundAuthority) -> Result<ProofPayloadAuthority, G0Error> {
        Ok(match authority {
            BoundAuthority::FactStore => ProofPayloadAuthority::FactStore,
            BoundAuthority::ProgramFamilyFact => ProofPayloadAuthority::ProgramFamilyFact,
            BoundAuthority::Operator => ProofPayloadAuthority::Operator,
            BoundAuthority::Unavailable => ProofPayloadAuthority::Unavailable,
            BoundAuthority::RelationPreimageSource { source } => {
                ProofPayloadAuthority::RelationPreimageSource {
                    source: self.expression(*source).map_err(|source| {
                        G0Error::CanonicalProjectionReference {
                            role: CanonicalProjectionRole::RelationPreimageSource,
                            source: Box::new(source),
                        }
                    })?,
                }
            }
        })
    }

    fn monomial(
        &self,
        monomial: super::monomial::MonomialId,
    ) -> Result<ProofPayloadMonomial, G0Error> {
        let arena =
            self.monomial_arenas.get(&monomial.arena()).ok_or(G0Error::UnsupportedBoundTransfer)?;
        let descriptor =
            arena.descriptor(monomial).map_err(|_| G0Error::UnsupportedBoundTransfer)?;
        let mut central_factors = descriptor
            .central_factors
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, factor)| {
                self.owner(factor).map_err(|source| G0Error::CanonicalProjectionReference {
                    role: CanonicalProjectionRole::MonomialCentralFactor {
                        monomial,
                        ordinal: ordinal as u64,
                    },
                    source: Box::new(source),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        central_factors.sort();
        Ok(ProofPayloadMonomial {
            central_factors,
            ordered_factors: descriptor
                .ordered_factors
                .iter()
                .copied()
                .enumerate()
                .map(|(ordinal, factor)| {
                    self.owner(factor).map_err(|source| G0Error::CanonicalProjectionReference {
                        role: CanonicalProjectionRole::MonomialOrderedFactor {
                            monomial,
                            ordinal: ordinal as u64,
                        },
                        source: Box::new(source),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn recorded_value_bound(
        value: &super::g0::RecordedValue,
        projection: &BoundProjection,
    ) -> Result<super::facts::NumericContract<super::facts::CoefficientBound>, G0Error> {
        match projection {
            BoundProjection::Coefficient => Ok(value.coefficient_bound.clone()),
            BoundProjection::Summary => value
                .exact_nf
                .as_ref()
                .map(|normal_form| normal_form.bounded_summary.coefficient_bound())
                .ok_or(G0Error::UnsupportedBoundTransfer),
        }
    }

    fn recorded_reference_bound(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        owner: super::arena::ScopedExprId,
        reference: &BoundValueRef,
        visiting: &mut BTreeSet<u64>,
    ) -> Result<super::facts::NumericContract<super::facts::CoefficientBound>, G0Error> {
        let value_at = |event: super::g0::EventIndex, projection: &BoundProjection| {
            let value = match trace.events.get(event.0 as usize) {
                Some(NormalizerEvent::Result { value, .. }) |
                Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                _ => return Err(G0Error::UnsupportedBoundTransfer),
            };
            Self::recorded_value_bound(value, projection)
        };
        match reference {
            BoundValueRef::Predecessor { input_position, projection } => {
                let frame = self.frames.last().ok_or(G0Error::UnsupportedBoundTransfer)?;
                let (binding_event, source_result) = trace.events[frame.start..current]
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(offset, event)| match event {
                        NormalizerEvent::Predecessor {
                            consumer,
                            input_position: position,
                            source_result,
                            ..
                        } if *consumer == owner && position == input_position => {
                            Some((frame.start + offset, *source_result))
                        }
                        _ => None,
                    })
                    .ok_or(G0Error::UnsupportedBoundTransfer)?;
                if source_result.0 as usize >= binding_event {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                value_at(source_result, projection)
            }
            BoundValueRef::Result { event, projection } => {
                if event.0 as usize >= current {
                    return Err(G0Error::UnsupportedBoundTransfer);
                }
                value_at(*event, projection)
            }
            BoundValueRef::Transfer(event) => {
                self.replay_transfer_bound(trace, *event, owner, visiting)
            }
        }
    }

    fn replay_matrix_transfer(
        &self,
        owner: super::arena::ScopedExprId,
        left: super::facts::NumericContract<super::facts::CoefficientBound>,
        right: super::facts::NumericContract<super::facts::CoefficientBound>,
        facts: &super::bound::MatrixProductFacts,
        tensor: bool,
    ) -> Result<super::facts::NumericContract<super::facts::CoefficientBound>, G0Error> {
        let (
            super::facts::NumericContract::Known(left_bound),
            super::facts::NumericContract::Known(right_bound),
        ) = (left, right)
        else {
            return Ok(super::facts::NumericContract::Missing);
        };
        let node = self.job.expressions().node(owner.expression())?;
        let [left_expression, right_expression] = node.inputs.as_ref() else {
            return Err(G0Error::UnsupportedBoundTransfer);
        };
        let ResolvedValueType::Matrix(left_type) =
            self.job.expressions().value_type(*left_expression)?
        else {
            return Err(G0Error::UnsupportedBoundTransfer);
        };
        let ResolvedValueType::Matrix(right_type) =
            self.job.expressions().value_type(*right_expression)?
        else {
            return Err(G0Error::UnsupportedBoundTransfer);
        };
        let matrix_bound =
            |matrix: &ResolvedMatrixType, bound: &super::facts::CoefficientBound| MatrixBound {
                matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                    modulus: matrix.modulus.clone().into(),
                    ring_dimension: matrix.ring_dimension,
                    rows: matrix.rows,
                    columns: matrix.columns,
                },
                coefficient_class: match bound {
                    super::facts::CoefficientBound::ExactZero => BoundClass::ExactZero,
                    super::facts::CoefficientBound::Finite(value) => {
                        BoundClass::bounded(value.maximum_absolute_coefficient.clone())
                    }
                    super::facts::CoefficientBound::Large => BoundClass::Large,
                },
            };
        let left = matrix_bound(left_type, &left_bound);
        let right = matrix_bound(right_type, &right_bound);
        let output = if tensor {
            tensor_bound_with_facts(&left, &right, facts)
        } else {
            product_bound_with_facts(&left, &right, facts)
        }
        .map_err(|_| G0Error::UnsupportedBoundTransfer)?;
        Ok(super::facts::NumericContract::Known(match output.coefficient_class {
            BoundClass::ExactZero => super::facts::CoefficientBound::ExactZero,
            BoundClass::Bounded { maximum_absolute_coefficient } => {
                super::facts::CoefficientBound::finite(maximum_absolute_coefficient)
            }
            BoundClass::Large => super::facts::CoefficientBound::Large,
        }))
    }

    fn replay_transfer_bound(
        &self,
        trace: &FeasibilityTrace,
        event: super::g0::EventIndex,
        owner: super::arena::ScopedExprId,
        visiting: &mut BTreeSet<u64>,
    ) -> Result<super::facts::NumericContract<super::facts::CoefficientBound>, G0Error> {
        if !visiting.insert(event.0) {
            return Err(G0Error::UnsupportedBoundTransfer);
        }
        let Some(NormalizerEvent::BoundTransfer { owner: actual_owner, rule }) =
            trace.events.get(event.0 as usize)
        else {
            return Err(G0Error::UnsupportedBoundTransfer);
        };
        if *actual_owner != owner {
            return Err(G0Error::UnsupportedBoundTransfer);
        }
        let reference = |value: &BoundValueRef, visiting: &mut BTreeSet<u64>| {
            self.recorded_reference_bound(trace, event.0 as usize, owner, value, visiting)
        };
        let output = match rule {
            BoundRule::Authority(_) => return Err(G0Error::UnsupportedBoundTransfer),
            BoundRule::Identity { input } | BoundRule::RingAutomorphism { input, .. } => {
                reference(input, visiting)?
            }
            BoundRule::Sum { inputs } => super::facts::add_bounds(
                &inputs
                    .iter()
                    .map(|input| reference(input, visiting))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            BoundRule::Scale { value, scale } => {
                let value = reference(value, visiting)?;
                match scale {
                    BoundScale::Magnitude(magnitude) => {
                        super::facts::product_bounds_with_factor(&[value], magnitude)
                    }
                    BoundScale::Value(scale) => {
                        super::facts::product_bounds(&[value, reference(scale, visiting)?])
                    }
                }
            }
            BoundRule::MonomialProduct { factors, .. } => super::facts::product_bounds(
                &factors.iter().map(|factor| reference(&factor.bound, visiting)).collect::<Result<
                    Vec<_>,
                    _,
                >>(
                )?,
            ),
            BoundRule::Product { left, right, facts } => self.replay_matrix_transfer(
                owner,
                reference(left, visiting)?,
                reference(right, visiting)?,
                facts,
                false,
            )?,
            BoundRule::Tensor {
                left,
                right,
                left_is_constant_polynomial,
                right_is_constant_polynomial,
            } => self.replay_matrix_transfer(
                owner,
                reference(left, visiting)?,
                reference(right, visiting)?,
                &super::bound::MatrixProductFacts {
                    left_is_constant_polynomial: *left_is_constant_polynomial,
                    right_is_constant_polynomial: *right_is_constant_polynomial,
                    ..Default::default()
                },
                true,
            )?,
            BoundRule::Maximum { .. } | BoundRule::WeightedSum { .. } => {
                return Err(G0Error::UnsupportedBoundTransfer)
            }
        };
        visiting.remove(&event.0);
        Ok(output)
    }

    fn exact_value_producers(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        owner: super::arena::ScopedExprId,
        value: &super::g0::RecordedValue,
    ) -> Result<(u64, Option<u64>), CertificateProjectionError> {
        let frame = self
            .frames
            .last()
            .filter(|frame| frame.root.program() == owner.program())
            .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
        let boundary = trace.events[frame.start..current]
            .iter()
            .enumerate()
            .rev()
            .find_map(|(offset, event)| match event {
                NormalizerEvent::Result { owner: prior_owner, .. } if *prior_owner == owner => {
                    Some(frame.start + offset + 1)
                }
                NormalizerEvent::Predecessor { consumer, .. } if *consumer == owner => {
                    Some(frame.start + offset + 1)
                }
                _ => None,
            })
            .unwrap_or(frame.start);
        let candidates = trace.events[boundary..current]
            .iter()
            .enumerate()
            .filter_map(|(offset, event)| match event {
                NormalizerEvent::BoundTransfer { owner: transfer_owner, rule }
                    if *transfer_owner == owner =>
                {
                    Some((boundary + offset, rule))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let summary = value
            .exact_nf
            .as_ref()
            .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?
            .bounded_summary
            .coefficient_bound();
        let (coefficient_producer, summary_producer) =
            select_reached_exact_producers(&candidates, &summary).map_err(proof_payload_error)?;
        let coefficient_rule = match trace.events.get(coefficient_producer) {
            Some(NormalizerEvent::BoundTransfer { rule, .. }) => rule,
            _ => return Err(proof_payload_error(G0Error::UnsupportedBoundTransfer)),
        };
        if !matches!(coefficient_rule, BoundRule::Authority(_)) {
            let replayed_coefficient = self
                .replay_transfer_bound(
                    trace,
                    super::g0::EventIndex(coefficient_producer as u64),
                    owner,
                    &mut BTreeSet::new(),
                )
                .map_err(proof_payload_error)?;
            if !coefficient_bound_is_within(&value.coefficient_bound, &replayed_coefficient) {
                return Err(proof_payload_error(G0Error::UnsupportedBoundTransfer));
            }
            if summary_producer.is_none() &&
                matches!(
                    summary,
                    super::facts::NumericContract::Known(super::facts::CoefficientBound::Finite(_))
                )
            {
                let has_owner_merge =
                    trace.events[coefficient_producer + 1..current].iter().any(|event| {
                        matches!(
                            event,
                            NormalizerEvent::CoefficientMerge(merge) if merge.owner == owner
                        )
                    });
                if !reached_deferred_finite_summary_allowed(&replayed_coefficient, has_owner_merge)
                {
                    return Err(proof_payload_error(G0Error::UnsupportedBoundTransfer));
                }
            }
        }
        if let Some(summary_producer) = summary_producer {
            let replayed_summary = self
                .replay_transfer_bound(
                    trace,
                    super::g0::EventIndex(summary_producer as u64),
                    owner,
                    &mut BTreeSet::new(),
                )
                .map_err(proof_payload_error)?;
            if replayed_summary != summary {
                return Err(proof_payload_error(G0Error::UnsupportedBoundTransfer));
            }
        }
        Ok((coefficient_producer as u64, summary_producer.map(|event| event as u64)))
    }

    fn exact_value(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        owner: super::arena::ScopedExprId,
        value: &super::g0::RecordedValue,
    ) -> Result<ProofPayloadValue, CertificateProjectionError> {
        let Some(normal_form) = &value.exact_nf else {
            return Err(proof_payload_error(G0Error::RelationTraceInvariant));
        };
        let (coefficient_producer, summary_producer) =
            self.exact_value_producers(trace, current, owner, value)?;
        Ok(ProofPayloadValue::Exact {
            terms: payload_projection(self.terms(normal_form))?,
            coefficient_bound: value.coefficient_bound.clone(),
            coefficient_producer,
            summary: normal_form.bounded_summary.clone(),
            summary_producer,
        })
    }

    fn value(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        owner: super::arena::ScopedExprId,
        value: &super::g0::RecordedValue,
    ) -> Result<ProofPayloadValue, CertificateProjectionError> {
        if value.exact_nf.is_some() {
            self.exact_value(trace, current, owner, value)
        } else {
            Ok(ProofPayloadValue::Coefficient { bound: value.coefficient_bound.clone() })
        }
    }

    fn terms(
        &self,
        normal_form: &super::normal_form::PolynomialNF,
    ) -> Result<Vec<ProofPayloadTerm>, G0Error> {
        let mut terms = normal_form
            .exact_terms
            .iter()
            .map(|(monomial, coefficient)| {
                Ok(ProofPayloadTerm {
                    monomial: self.monomial(*monomial)?,
                    coefficient: coefficient.clone(),
                })
            })
            .collect::<Result<Vec<_>, G0Error>>()?;
        terms.sort_by(|left, right| left.monomial.cmp(&right.monomial));
        Ok(terms)
    }

    fn pre_fold(
        &self,
        observation: &super::g0::PreFoldPolynomial,
        current: usize,
    ) -> Result<ProofPayloadPreFoldPolynomial, CertificateProjectionError> {
        let frame = self
            .frames
            .last()
            .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
        let result_event = frame
            .last_exact_result
            .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
        Ok(ProofPayloadPreFoldPolynomial {
            result_event,
            terms: payload_projection(self.terms(&observation.polynomial))?,
            summary: observation.polynomial.bounded_summary.clone(),
            summary_evidence: observation
                .summary_evidence
                .as_ref()
                .map(|evidence| self.value_ref(evidence, current, frame.root))
                .transpose()?,
        })
    }

    fn range(
        &self,
        range: super::g0::EventRange,
        current: usize,
    ) -> Result<ProofPayloadRange, G0Error> {
        if range.start.0 > range.end.0 || range.end.0 as usize > current {
            return Err(G0Error::MalformedSpecializationRange);
        }
        Ok(ProofPayloadRange { start: range.start.0, end: range.end.0 })
    }

    fn value_ref(
        &self,
        value: &BoundValueRef,
        current: usize,
        owner: super::arena::ScopedExprId,
    ) -> Result<ProofPayloadValueRef, CertificateProjectionError> {
        let event = match value {
            BoundValueRef::Predecessor { input_position, projection } => {
                let binding_event =
                    active_predecessor_binding(&self.frames, owner, *input_position).ok_or_else(
                        || {
                            proof_invariant(
                                current,
                                Some(owner),
                                ProofEvidenceKind::EventReference,
                                ProofInvariantMismatch::EventKind {
                                    referenced: current as u64,
                                    expected: "prior predecessor binding",
                                },
                            )
                        },
                    )?;
                return Ok(ProofPayloadValueRef::Predecessor {
                    binding_event,
                    input_position: *input_position,
                    projection: projection.clone(),
                });
            }
            BoundValueRef::Result { event, projection } => {
                self.prior_event(*event, current, None, ProofEvidenceKind::EventReference)?;
                return Ok(ProofPayloadValueRef::Result {
                    event: event.0,
                    projection: projection.clone(),
                });
            }
            BoundValueRef::Transfer(event) => {
                self.prior_event(*event, current, None, ProofEvidenceKind::EventReference)?;
                *event
            }
        };
        Ok(ProofPayloadValueRef::Transfer(event.0))
    }

    fn term_ref(
        &self,
        trace: &FeasibilityTrace,
        reference: super::g0::RecordedTermRef,
        current: usize,
        owner: super::arena::ScopedExprId,
        evidence: ProofEvidenceKind,
    ) -> Result<ProofPayloadTermRef, CertificateProjectionError> {
        self.prior_event(reference.value_event, current, Some(owner), evidence)?;
        let value = match trace.events.get(reference.value_event.0 as usize) {
            Some(NormalizerEvent::Result { value, .. }) |
            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
            _ => {
                return Err(proof_invariant(
                    current,
                    Some(owner),
                    evidence,
                    ProofInvariantMismatch::EventKind {
                        referenced: reference.value_event.0,
                        expected: "Result or InvocationEnd",
                    },
                ));
            }
        };
        let normal_form = value.exact_nf.as_ref().ok_or_else(|| {
            proof_invariant(
                current,
                Some(owner),
                evidence,
                ProofInvariantMismatch::MissingExactNormalForm {
                    referenced: reference.value_event.0,
                },
            )
        })?;
        let mut terms = normal_form
            .exact_terms
            .keys()
            .map(|monomial| {
                payload_projection(self.monomial(*monomial)).map(|term| (*monomial, term))
            })
            .collect::<Result<Vec<_>, _>>()?;
        terms.sort_by(|left, right| left.1.cmp(&right.1));
        let term_ordinal = terms
            .iter()
            .position(|(monomial, _)| *monomial == reference.monomial)
            .ok_or_else(|| {
                proof_invariant(
                    current,
                    Some(owner),
                    evidence,
                    ProofInvariantMismatch::MissingTerm {
                        referenced: reference.value_event.0,
                        monomial: reference.monomial,
                    },
                )
            })? as u64;
        Ok(ProofPayloadTermRef { value_event: reference.value_event.0, term_ordinal })
    }

    fn validate_relation_output(
        &self,
        applied: &super::g0::AppliedRelation,
        source_term: super::monomial::MonomialId,
        output: &ProofPayloadMonomial,
        current: usize,
        evidence: ProofEvidenceKind,
    ) -> Result<(), CertificateProjectionError> {
        let source = payload_projection(self.monomial(applied.source_monomial))?;
        let replacement = payload_projection(self.monomial(source_term))?;
        let start = usize::try_from(applied.ordered_start).map_err(|_| {
            proof_invariant(
                current,
                Some(applied.owner),
                evidence,
                ProofInvariantMismatch::SpliceRange {
                    start: applied.ordered_start as u64,
                    end_exclusive: applied.ordered_end_exclusive as u64,
                    ordered_len: source.ordered_factors.len() as u64,
                },
            )
        })?;
        let end = usize::try_from(applied.ordered_end_exclusive).map_err(|_| {
            proof_invariant(
                current,
                Some(applied.owner),
                evidence,
                ProofInvariantMismatch::SpliceRange {
                    start: applied.ordered_start as u64,
                    end_exclusive: applied.ordered_end_exclusive as u64,
                    ordered_len: source.ordered_factors.len() as u64,
                },
            )
        })?;
        if start > end || end > source.ordered_factors.len() {
            return Err(proof_invariant(
                current,
                Some(applied.owner),
                evidence,
                ProofInvariantMismatch::SpliceRange {
                    start: applied.ordered_start as u64,
                    end_exclusive: applied.ordered_end_exclusive as u64,
                    ordered_len: source.ordered_factors.len() as u64,
                },
            ));
        }
        let mut central_factors = source.central_factors;
        central_factors.extend(replacement.central_factors);
        central_factors.sort();
        let mut ordered_factors = Vec::with_capacity(
            source.ordered_factors.len() - (end - start) + replacement.ordered_factors.len(),
        );
        ordered_factors.extend_from_slice(&source.ordered_factors[..start]);
        ordered_factors.extend(replacement.ordered_factors);
        ordered_factors.extend_from_slice(&source.ordered_factors[end..]);
        if output.central_factors != central_factors || output.ordered_factors != ordered_factors {
            return Err(proof_invariant(
                current,
                Some(applied.owner),
                evidence,
                ProofInvariantMismatch::SpliceOutput {
                    expected_central: central_factors,
                    expected_ordered: ordered_factors,
                    actual_central: output.central_factors.clone(),
                    actual_ordered: output.ordered_factors.clone(),
                },
            ));
        }
        Ok(())
    }

    fn coefficient_merge(
        &self,
        trace: &FeasibilityTrace,
        observation: &super::g0::CoefficientMerge,
        current: usize,
    ) -> Result<ProofPayloadCoefficientMerge, CertificateProjectionError> {
        let output = payload_projection(self.monomial(observation.output))?;
        let source = match &observation.source {
            super::g0::CoefficientMergeSource::Operator { inputs } => {
                let evidence = ProofEvidenceKind::OperatorMerge;
                let node = payload_projection(
                    self.job
                        .expressions()
                        .node(observation.owner.expression())
                        .map_err(G0Error::from),
                )?;
                let operation = match &node.operator {
                    ValueOperator::Matrix(
                        operation @ (MatrixOperation::Add | MatrixOperation::Subtract),
                    ) |
                    ValueOperator::Matrix(operation @ MatrixOperation::Multiply) => {
                        operation.clone()
                    }
                    ValueOperator::Matrix(operation @ MatrixOperation::Tensor { .. })
                        if payload_projection(self.scalar_tensor_types(node, operation))?
                            .is_some() =>
                    {
                        operation.clone()
                    }
                    _ => {
                        return Err(proof_invariant(
                            current,
                            Some(observation.owner),
                            evidence,
                            ProofInvariantMismatch::Operator {
                                actual: Box::new(node.operator.clone()),
                            },
                        ));
                    }
                };
                let right = match trace.events.get(inputs[1].value_event.0 as usize) {
                    Some(NormalizerEvent::Result { value, .. }) |
                    Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value
                        .exact_nf
                        .as_ref()
                        .and_then(|normal_form| normal_form.exact_terms.get(&inputs[1].monomial)),
                    _ => None,
                }
                .ok_or_else(|| {
                    proof_invariant(
                        current,
                        Some(observation.owner),
                        evidence,
                        ProofInvariantMismatch::MissingTerm {
                            referenced: inputs[1].value_event.0,
                            monomial: inputs[1].monomial,
                        },
                    )
                })?;
                let expected = match operation {
                    MatrixOperation::Add => right.clone(),
                    MatrixOperation::Subtract => -right.clone(),
                    MatrixOperation::Multiply | MatrixOperation::Tensor { .. } => {
                        let left = match trace.events.get(inputs[0].value_event.0 as usize) {
                            Some(NormalizerEvent::Result { value, .. }) |
                            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => {
                                value.exact_nf.as_ref().and_then(|normal_form| {
                                    normal_form.exact_terms.get(&inputs[0].monomial)
                                })
                            }
                            _ => None,
                        }
                        .ok_or_else(|| {
                            proof_invariant(
                                current,
                                Some(observation.owner),
                                evidence,
                                ProofInvariantMismatch::MissingTerm {
                                    referenced: inputs[0].value_event.0,
                                    monomial: inputs[0].monomial,
                                },
                            )
                        })?;
                        left * right
                    }
                    _ => unreachable!("operator classification is exhaustive"),
                };
                if observation.signed_contribution != expected {
                    return Err(proof_invariant(
                        current,
                        Some(observation.owner),
                        evidence,
                        ProofInvariantMismatch::Coefficient {
                            expected,
                            actual: observation.signed_contribution.clone(),
                        },
                    ));
                }
                match operation {
                    MatrixOperation::Add | MatrixOperation::Subtract => {
                        for (role, input) in [("left", inputs[0]), ("right", inputs[1])] {
                            let actual = payload_projection(self.monomial(input.monomial))?;
                            if actual != output {
                                return Err(proof_invariant(
                                    current,
                                    Some(observation.owner),
                                    evidence,
                                    ProofInvariantMismatch::MonomialRole {
                                        role,
                                        expected: output.clone(),
                                        actual,
                                    },
                                ));
                            }
                        }
                    }
                    MatrixOperation::Multiply | MatrixOperation::Tensor { .. } => {
                        let left = payload_projection(self.monomial(inputs[0].monomial))?;
                        let right = payload_projection(self.monomial(inputs[1].monomial))?;
                        let left_type = payload_projection(
                            self.job
                                .expressions()
                                .value_type(node.inputs[0])
                                .map_err(G0Error::from),
                        )?;
                        let right_type = payload_projection(
                            self.job
                                .expressions()
                                .value_type(node.inputs[1])
                                .map_err(G0Error::from),
                        )?;
                        let left_scalar = matches!(
                            left_type,
                            ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1
                        );
                        let right_scalar = matches!(
                            right_type,
                            ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1
                        );
                        let expected =
                            Self::scalar_product_monomial(left, right, left_scalar, right_scalar);
                        if output != expected {
                            return Err(proof_invariant(
                                current,
                                Some(observation.owner),
                                evidence,
                                ProofInvariantMismatch::SpliceOutput {
                                    expected_central: expected.central_factors,
                                    expected_ordered: expected.ordered_factors,
                                    actual_central: output.central_factors.clone(),
                                    actual_ordered: output.ordered_factors.clone(),
                                },
                            ));
                        }
                    }
                    _ => unreachable!("operator classification is exhaustive"),
                }
                ProofPayloadCoefficientMergeSource::Operator {
                    inputs: [
                        self.term_ref(trace, inputs[0], current, observation.owner, evidence)?,
                        self.term_ref(trace, inputs[1], current, observation.owner, evidence)?,
                    ],
                }
            }
            super::g0::CoefficientMergeSource::Relation { application, source_term } => {
                let evidence = match trace.events.get(application.0 as usize) {
                    Some(NormalizerEvent::AppliedRelation(super::g0::AppliedRelation {
                        rule: AppliedRelationRule::Universal { .. },
                        ..
                    })) => ProofEvidenceKind::Universal,
                    Some(NormalizerEvent::AppliedRelation(super::g0::AppliedRelation {
                        rule: AppliedRelationRule::Gadget { .. },
                        ..
                    })) => ProofEvidenceKind::Gadget,
                    _ => ProofEvidenceKind::Universal,
                };
                self.prior_event(*application, current, Some(observation.owner), evidence)?;
                let applied = match trace.events.get(application.0 as usize) {
                    Some(NormalizerEvent::AppliedRelation(applied))
                        if applied.owner == observation.owner =>
                    {
                        applied
                    }
                    Some(NormalizerEvent::AppliedRelation(applied)) => {
                        return Err(proof_invariant(
                            current,
                            Some(observation.owner),
                            evidence,
                            ProofInvariantMismatch::Owner {
                                expected: observation.owner,
                                actual: Some(applied.owner),
                            },
                        ));
                    }
                    _ => {
                        return Err(proof_invariant(
                            current,
                            Some(observation.owner),
                            evidence,
                            ProofInvariantMismatch::EventKind {
                                referenced: application.0,
                                expected: "AppliedRelation",
                            },
                        ));
                    }
                };
                let source_event = match &applied.rule {
                    super::g0::AppliedRelationRule::Universal { key, source, rhs, .. } => {
                        self.rhs_event(
                            key,
                            *source,
                            *rhs,
                            current,
                            applied.owner,
                            ProofEvidenceKind::Universal,
                        )?
                        .1
                    }
                    super::g0::AppliedRelationRule::Gadget { input_result, .. } => {
                        self.prior_event(
                            *input_result,
                            current,
                            Some(applied.owner),
                            ProofEvidenceKind::Gadget,
                        )?;
                        input_result.0
                    }
                };
                let source_event_index = super::g0::EventIndex(source_event);
                let source_value = match trace.events.get(source_event as usize) {
                    Some(NormalizerEvent::Result { value, .. }) |
                    Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                    _ => {
                        return Err(proof_invariant(
                            current,
                            Some(observation.owner),
                            evidence,
                            ProofInvariantMismatch::EventKind {
                                referenced: source_event,
                                expected: "Result or InvocationEnd",
                            },
                        ));
                    }
                };
                let source_nf = source_value.exact_nf.as_ref().ok_or_else(|| {
                    proof_invariant(
                        current,
                        Some(observation.owner),
                        evidence,
                        ProofInvariantMismatch::MissingExactNormalForm { referenced: source_event },
                    )
                })?;
                let source_coefficient =
                    source_nf.exact_terms.get(source_term).ok_or_else(|| {
                        proof_invariant(
                            current,
                            Some(observation.owner),
                            evidence,
                            ProofInvariantMismatch::MissingTerm {
                                referenced: source_event,
                                monomial: *source_term,
                            },
                        )
                    })?;
                let expected_coefficient = &applied.outer_coefficient * source_coefficient;
                if expected_coefficient != observation.signed_contribution {
                    return Err(proof_invariant(
                        current,
                        Some(observation.owner),
                        evidence,
                        ProofInvariantMismatch::Coefficient {
                            expected: expected_coefficient,
                            actual: observation.signed_contribution.clone(),
                        },
                    ));
                }
                let source_ordinal = self
                    .term_ref(
                        trace,
                        super::g0::RecordedTermRef {
                            value_event: source_event_index,
                            monomial: *source_term,
                        },
                        current,
                        observation.owner,
                        evidence,
                    )?
                    .term_ordinal;
                self.validate_relation_output(applied, *source_term, &output, current, evidence)?;
                ProofPayloadCoefficientMergeSource::Relation {
                    application: application.0,
                    source_term_ordinal: source_ordinal,
                }
            }
        };
        Ok(ProofPayloadCoefficientMerge {
            owner: payload_projection(self.owner(observation.owner))?,
            source,
            output,
            signed_contribution: observation.signed_contribution.clone(),
        })
    }

    fn prior_event(
        &self,
        event: super::g0::EventIndex,
        current: usize,
        owner: Option<super::arena::ScopedExprId>,
        evidence: ProofEvidenceKind,
    ) -> Result<(), CertificateProjectionError> {
        if event.0 as usize >= current {
            return Err(proof_invariant(
                current,
                owner,
                evidence,
                ProofInvariantMismatch::EventOrder { referenced: event.0 },
            ));
        }
        Ok(())
    }

    fn rule(
        &self,
        rule: &BoundRule,
        current: usize,
        owner: super::arena::ScopedExprId,
    ) -> Result<ProofPayloadRule, CertificateProjectionError> {
        let value = |value: &BoundValueRef| self.value_ref(value, current, owner);
        Ok(match rule {
            BoundRule::Authority(authority) => {
                ProofPayloadRule::Authority(payload_projection(self.authority(authority))?)
            }
            BoundRule::Identity { input } => ProofPayloadRule::Identity { input: value(input)? },
            BoundRule::RingAutomorphism { input, index, ring_dimension } => {
                ProofPayloadRule::RingAutomorphism {
                    input: value(input)?,
                    index: *index,
                    ring_dimension: *ring_dimension,
                }
            }
            BoundRule::Sum { inputs } => ProofPayloadRule::Sum {
                inputs: inputs.iter().map(value).collect::<Result<Vec<_>, _>>()?,
            },
            BoundRule::Maximum { inputs } => ProofPayloadRule::Maximum {
                inputs: inputs.iter().map(value).collect::<Result<Vec<_>, _>>()?,
            },
            BoundRule::Scale { value: input, scale } => ProofPayloadRule::Scale {
                value: value(input)?,
                scale: match scale {
                    BoundScale::Value(input) => ProofPayloadScale::Value(value(input)?),
                    BoundScale::Magnitude(magnitude) => {
                        ProofPayloadScale::Magnitude(magnitude.clone())
                    }
                },
            },
            BoundRule::MonomialProduct { monomial, factors } => ProofPayloadRule::MonomialProduct {
                monomial: payload_projection(self.monomial(*monomial))?,
                factors: factors
                    .iter()
                    .map(|factor| self.factor(factor, current, owner))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            BoundRule::WeightedSum { inputs } => ProofPayloadRule::WeightedSum {
                inputs: inputs.iter().map(value).collect::<Result<Vec<_>, _>>()?,
            },
            BoundRule::Product { left, right, facts } => ProofPayloadRule::Product {
                left: value(left)?,
                right: value(right)?,
                facts: facts.clone(),
            },
            BoundRule::Tensor {
                left,
                right,
                left_is_constant_polynomial,
                right_is_constant_polynomial,
            } => ProofPayloadRule::Tensor {
                left: value(left)?,
                right: value(right)?,
                left_is_constant_polynomial: *left_is_constant_polynomial,
                right_is_constant_polynomial: *right_is_constant_polynomial,
            },
        })
    }

    fn factor(
        &self,
        factor: &MonomialFactorEvidence,
        current: usize,
        owner: super::arena::ScopedExprId,
    ) -> Result<ProofPayloadFactorEvidence, CertificateProjectionError> {
        Ok(ProofPayloadFactorEvidence {
            bound: self.value_ref(&factor.bound, current, owner)?,
            is_constant_polynomial: factor.is_constant_polynomial,
            support_upper: factor.support_upper,
        })
    }

    fn relation_rule(
        &self,
        rule: &AppliedRelationRule,
        current: usize,
        owner: super::arena::ScopedExprId,
    ) -> Result<ProofPayloadRelationRule, CertificateProjectionError> {
        Ok(match rule {
            AppliedRelationRule::Universal { key, source, lhs, rhs } => {
                let (computed, rhs_result) = self.rhs_event(
                    key,
                    *source,
                    *rhs,
                    current,
                    owner,
                    ProofEvidenceKind::Universal,
                )?;
                ProofPayloadRelationRule::Universal {
                    computed,
                    lhs: payload_projection(self.monomial(lhs.monomial))?,
                    lhs_layout: lhs.layout.clone(),
                    rhs_result,
                }
            }
            AppliedRelationRule::Gadget { gadget, decomposition, input, input_result } => {
                self.prior_event(*input_result, current, Some(*gadget), ProofEvidenceKind::Gadget)?;
                ProofPayloadRelationRule::Gadget {
                    gadget: payload_projection(self.owner(*gadget))?,
                    decomposition: payload_projection(self.owner(*decomposition))?,
                    input: payload_projection(self.expression(*input))?,
                    input_result: input_result.0,
                }
            }
        })
    }

    fn specialization_dispatch(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
    ) -> Result<ProofPayloadUniversalDispatch, G0Error> {
        Ok(ProofPayloadUniversalDispatch {
            preimage_family: self.refs.family(key.dispatch.preimage_family)?,
            preimage_source: self.refs.expression(key.dispatch.preimage_source.expression)?,
            trapdoor_source: self.refs.expression(key.dispatch.trapdoor_source.expression)?,
        })
    }

    fn rhs_event(
        &self,
        key: &super::relation::RuntimeSpecializationKey,
        source: super::g0::EventRange,
        rhs: super::relation::CanonicalRhsId,
        current: usize,
        owner: super::arena::ScopedExprId,
        evidence: ProofEvidenceKind,
    ) -> Result<(u64, u64), CertificateProjectionError> {
        let (computed, result) =
            self.rhs_events.get(&(key.clone(), source, rhs)).copied().ok_or_else(|| {
                proof_invariant(current, Some(owner), evidence, ProofInvariantMismatch::RhsReplay)
            })?;
        if computed >= current as u64 || result >= current as u64 {
            return Err(proof_invariant(
                current,
                Some(owner),
                evidence,
                ProofInvariantMismatch::EventOrder { referenced: computed.max(result) },
            ));
        }
        Ok((computed, result))
    }

    fn event(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        event: &NormalizerEvent,
    ) -> Result<ProofPayloadEvent, CertificateProjectionError> {
        Ok(match event {
            NormalizerEvent::InvocationStart { root } => {
                ProofPayloadEvent::InvocationStart { root: payload_projection(self.owner(*root))? }
            }
            NormalizerEvent::Predecessor {
                consumer,
                input_position,
                predecessor,
                source_result,
            } => {
                self.prior_event(
                    *source_result,
                    current,
                    Some(*consumer),
                    ProofEvidenceKind::EventReference,
                )?;
                ProofPayloadEvent::Predecessor {
                    consumer: payload_projection(self.owner(*consumer))?,
                    input_position: *input_position,
                    predecessor: payload_projection(self.expression(*predecessor))?,
                    source_result: source_result.0,
                }
            }
            NormalizerEvent::Result { owner, value } => ProofPayloadEvent::Result {
                owner: payload_projection(self.owner(*owner))?,
                value: self.value(trace, current, *owner, value)?,
            },
            NormalizerEvent::InvocationEnd { root, result, .. } => {
                let pre_fold_event = self
                    .frames
                    .last()
                    .filter(|frame| frame.root == *root)
                    .and_then(|frame| frame.last_pre_fold)
                    .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
                let result_event = self
                    .frames
                    .last()
                    .filter(|frame| frame.root == *root)
                    .and_then(|frame| frame.last_exact_result)
                    .ok_or_else(|| proof_payload_error(G0Error::RelationTraceInvariant))?;
                let result_value = match trace.events.get(result_event as usize) {
                    Some(NormalizerEvent::Result { owner, value }) if *owner == *root => value,
                    _ => return Err(proof_payload_error(G0Error::RelationTraceInvariant)),
                };
                if result_value.exact_nf.is_none() || result_value.exact_nf != result.exact_nf {
                    return Err(proof_payload_error(G0Error::RelationTraceInvariant));
                }
                ProofPayloadEvent::InvocationEnd {
                    root: payload_projection(self.owner(*root))?,
                    result: self.value(trace, result_event as usize, *root, result_value)?,
                    pre_fold_event,
                }
            }
            NormalizerEvent::SpecializationComputed { owner, key, replay } => {
                ProofPayloadEvent::SpecializationComputed {
                    owner: payload_projection(self.owner(*owner))?,
                    dispatch: payload_projection(self.specialization_dispatch(key))?,
                    source: payload_projection(self.range(replay.range, current))?,
                }
            }
            NormalizerEvent::SpecializationCacheHit { owner, source, .. } => {
                ProofPayloadEvent::SpecializationCacheHit {
                    owner: payload_projection(self.owner(*owner))?,
                    source: payload_projection(self.range(*source, current))?,
                }
            }
            NormalizerEvent::AppliedRelation(observation) => ProofPayloadEvent::AppliedRelation {
                owner: payload_projection(self.owner(observation.owner))?,
                source_monomial: payload_projection(self.monomial(observation.source_monomial))?,
                outer_coefficient: observation.outer_coefficient.clone(),
                ordered_start: observation.ordered_start,
                ordered_end_exclusive: observation.ordered_end_exclusive,
                rule: self.relation_rule(&observation.rule, current, observation.owner)?,
            },
            NormalizerEvent::BoundTransfer { owner, rule } => ProofPayloadEvent::BoundTransfer {
                owner: payload_projection(self.owner(*owner))?,
                rule: self.rule(rule, current, *owner)?,
            },
            NormalizerEvent::SurvivorFold(observation) => {
                self.prior_event(
                    observation.bound,
                    current,
                    None,
                    ProofEvidenceKind::SurvivorFold,
                )?;
                let (_, _, magnitude) =
                    payload_projection(trace.resolve_survivor_bound(observation.bound))?;
                if *observation.coefficient.magnitude() != magnitude {
                    return Err(proof_invariant(
                        current,
                        None,
                        ProofEvidenceKind::SurvivorFold,
                        ProofInvariantMismatch::SurvivorMagnitude {
                            expected: magnitude,
                            actual: observation.coefficient.magnitude().clone(),
                        },
                    ));
                }
                ProofPayloadEvent::SurvivorFold(ProofPayloadSurvivorFold {
                    coefficient: observation.coefficient.clone(),
                    bound: observation.bound.0,
                })
            }
            NormalizerEvent::PreFoldPolynomial(observation) => {
                ProofPayloadEvent::PreFoldPolynomial(self.pre_fold(observation, current)?)
            }
            NormalizerEvent::CoefficientMerge(observation) => ProofPayloadEvent::CoefficientMerge(
                self.coefficient_merge(trace, observation, current)?,
            ),
        })
    }
}

/// Resolve, lower, report, and retain one accepted opt-in certificate run.
///
/// The report is deliberately produced through the ordinary decoder-inclusive authority.  The
/// certificate projection still contains only the residual closure, while this run keeps the
/// owned job alive for later serialization and handle resolution.
pub(crate) fn prepare_operational_certificate(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<OperationalCertificateRun, CertificateProjectionError> {
    let mut emit = |_| {};
    let mut control = SimulationControl::new(&mut emit);
    request.validate(protocol.params.iter().map(|parameter| parameter.name.clone())).map_err(
        |error| CertificateProjectionError::Operational(OperationalSimulationError::Request(error)),
    )?;
    let target = resolve_target(protocol, request, &mut control)?;
    let target_id = target.target_id.clone();
    let plaintext_modulus = match &target.kind {
        ResolvedDecoderKind::Threshold { plaintext_modulus } => plaintext_modulus.clone(),
        ResolvedDecoderKind::BooleanInterval => {
            return Err(CertificateProjectionError::UnsupportedDecoderKind {
                target_id,
                kind: OperationalDecoderKind::BooleanInterval,
            });
        }
    };
    let parameters = request
        .environment
        .iter()
        .filter_map(|(name, value)| match value {
            super::OperationalParameterValue::Integer(value) => Some((name.clone(), value.clone())),
            super::OperationalParameterValue::Rational { .. } => None,
        })
        .collect::<BTreeMap<_, _>>();
    let plan = ProtocolPlan::build(protocol, &request.target_id)
        .map_err(|error| CertificateProjectionError::Lowering { detail: error.to_string() })?;
    let (job, roots, mut trace) =
        ProductionAdapter::new_with_feasibility(protocol, &plan, parameters)
            .map_err(|error| CertificateProjectionError::Lowering { detail: error.to_string() })?
            .lower_with_feasibility()
            .map_err(|error| CertificateProjectionError::Lowering { detail: error.to_string() })?;
    let residual = project_residual_root(&job, &roots.residual, &target)?;
    let available_index_plans = trace.index_use_plans().collect::<Vec<_>>();
    let mut closure = collect_residual_closure_with_plans(&job, &residual, &available_index_plans)?;
    // Freeze the genuine LUT fixed point before proof-only expressions expand this same closure.
    trace.retain_residual_index_use_plans(&closure);
    let mut job = job;
    let accepted_report = analyze_roots_with_sink(
        &mut job,
        &roots,
        &ReportTarget {
            target_id: target.target_id.clone(),
            plaintext_modulus: plaintext_modulus.clone(),
            ciphertext_modulus: target.ciphertext_modulus.clone(),
            boolean_interval: false,
        },
        &mut trace,
    )
    .map_err(|error| {
        CertificateProjectionError::Operational(OperationalSimulationError::from(
            super::error::ProductionError::from(error),
        ))
    })?;
    let report_plaintext_modulus = match &accepted_report.acceptance {
        super::OperationalAcceptanceReport::Threshold { plaintext_modulus, .. } => {
            plaintext_modulus
        }
        _ => {
            return Err(CertificateProjectionError::ReportMismatch {
                target_id: target.target_id,
                detail: "ordinary report did not use threshold acceptance".to_owned(),
            });
        }
    };
    if accepted_report.target_id != request.target_id ||
        accepted_report.ciphertext_modulus != target.ciphertext_modulus ||
        report_plaintext_modulus != &plaintext_modulus
    {
        return Err(CertificateProjectionError::ReportMismatch {
            target_id: request.target_id.clone(),
            detail: format!(
                "report target={:?}, report p={:?}, report q={:?}; resolved p={:?}, resolved q={:?}",
                accepted_report.target_id,
                report_plaintext_modulus,
                accepted_report.ciphertext_modulus,
                plaintext_modulus,
                target.ciphertext_modulus,
            ),
        });
    }
    extend_certificate_closure(&job, &mut closure, &trace)
        .map_err(CertificateProjectionError::Closure)?;
    let closed_root_expression = match &residual {
        CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
        CertificateResidualRoot::Family { .. } => None,
    };
    if let Some(wrapper) = closed_wrapper_program(&trace, closed_root_expression) {
        // The zero-argument wrapper exists only to authorize a closed root.  Its expressions
        // remain in the closure, but it is not a user-visible Program row.
        closure.programs.remove(&wrapper);
    }
    trace.retain_residual(&closure);
    let projection = OperationalCertificateProjection {
        target_id,
        plaintext_modulus,
        ciphertext_modulus: target.ciphertext_modulus.clone(),
        residual,
        closure,
    };
    if !accepted_report.accepted {
        return Err(CertificateProjectionError::Rejected { target_id: request.target_id.clone() });
    }
    Ok(OperationalCertificateRun {
        job,
        projection,
        accepted_report,
        trace,
        #[cfg(test)]
        roots,
    })
}

/// Prepare the typed, non-emitting base summary used by the later G0 feasibility evidence stage.
/// Coverage, frontier/L, proof-payload T, artifact-byte, and retained-memory observations are
/// intentionally absent; this type cannot represent a final certificate artifact.
pub fn prepare_base_feasibility_summary(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<BaseFeasibilitySummary, String> {
    let run =
        prepare_operational_certificate(protocol, request).map_err(|error| error.to_string())?;
    let closure = &run.projection.closure;
    // Build the residual-only Stage-1 inventory from the same owned job and closure.  The base
    // summary does not expose it or claim final artifact completeness, but descriptor conflicts
    // must still fail closed on this opt-in path.
    let rows = super::g0::derive_certificate_statement_rows(
        &run.job,
        closure,
        &run.trace,
        closed_residual_expression(&run),
    )
    .map_err(|error| error.to_string())?;
    base_feasibility_summary_from_run(&run, &rows)
}

fn base_feasibility_summary_from_run(
    run: &OperationalCertificateRun,
    rows: &CanonicalStatementRows,
) -> Result<BaseFeasibilitySummary, String> {
    let (plaintext_modulus, threshold_left, margin) = match &run.accepted_report.acceptance {
        super::OperationalAcceptanceReport::Threshold {
            plaintext_modulus,
            threshold_left,
            margin,
        } => (plaintext_modulus, threshold_left, margin),
        super::OperationalAcceptanceReport::BooleanInterval { .. } => {
            return Err("base feasibility summary requires threshold acceptance".to_owned());
        }
    };
    let normalization = run.accepted_report.counters.normalization;
    let source_rows = rows.sources().len();
    let total_rows = rows
        .expressions()
        .len()
        .checked_add(rows.programs().len())
        .and_then(|total| total.checked_add(source_rows))
        .and_then(|total| total.checked_add(rows.events().len()))
        .ok_or_else(|| "base feasibility N count overflow".to_owned())?;
    Ok(BaseFeasibilitySummary {
        schema_id: BASE_FEASIBILITY_SCHEMA_ID,
        schema_version: BASE_FEASIBILITY_SCHEMA_VERSION,
        target_id: run.accepted_report.target_id.clone(),
        plaintext_modulus: plaintext_modulus.to_string(),
        ciphertext_modulus: run.accepted_report.ciphertext_modulus.to_string(),
        accepted: run.accepted_report.accepted,
        noise_bound: run.accepted_report.noise_bound.to_string(),
        threshold_left: threshold_left.to_string(),
        margin: margin.to_string(),
        counters: BaseFeasibilityCounters {
            ordinary_baseline: OrdinaryBaselineCounters {
                occurrences: run.accepted_report.counters.occurrences,
                samples: run.accepted_report.counters.samples,
                normalization_nodes_processed: normalization.nodes_processed,
                normalization_nodes_total: normalization.nodes_total,
                normalization_exact_term_count: normalization.final_exact_term_count,
                normalization_relation_candidates: normalization.relation_candidates,
                normalization_relation_applied: normalization.relation_applied,
                normalization_relation_remaining: normalization.relation_remaining,
                normalization_bounded_fold_count: normalization.bounded_fold_count,
            },
            residual_trace: ResidualTraceCounters::default(),
        },
        n: BaseNBreakdown {
            expression_rows: rows.expressions().len(),
            program_rows: rows.programs().len(),
            source_rows,
            event_rows: rows.events().len(),
            total_rows,
        },
    })
}

/// Serialize a base summary deterministically.  It remains a review/input summary, not an
/// acceptance or certificate-emission API.
pub fn serialize_base_feasibility_summary(
    summary: &BaseFeasibilitySummary,
) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(summary)
}

fn checked_cardinality(len: usize, label: &str) -> Result<u64, String> {
    u64::try_from(len).map_err(|_| format!("G0 CPU evidence {label} cardinality overflow"))
}

fn exact_retained_n(rows: &CanonicalStatementRows) -> Result<ExactRetainedN, String> {
    let expression_rows = checked_cardinality(rows.expressions().len(), "expression")?;
    let program_rows = checked_cardinality(rows.programs().len(), "program")?;
    let source_rows = checked_cardinality(rows.sources().len(), "source")?;
    let event_rows = checked_cardinality(rows.events().len(), "event")?;
    let total_rows = [expression_rows, program_rows, source_rows, event_rows]
        .into_iter()
        .try_fold(0_u64, |total, count| {
            total.checked_add(count).ok_or_else(|| "G0 CPU evidence N overflow".to_owned())
        })?;
    Ok(ExactRetainedN { expression_rows, program_rows, source_rows, event_rows, total_rows })
}

fn g0_cpu_lut_observation(lut: &super::g0::G0LutEvidence) -> Result<G0CpuLutObservation, String> {
    let mut observed_rows = BigUint::ZERO;
    for unit in &lut.index_uses {
        let rows = BigUint::from(unit.rows.len());
        if rows.to_string() != unit.frontier_product {
            return Err("G0 CPU evidence index LUT frontier product mismatch".to_owned());
        }
        observed_rows += rows;
    }
    for unit in &lut.slice_groups {
        let rows = BigUint::from(unit.rows.len());
        if rows.to_string() != unit.frontier_product {
            return Err("G0 CPU evidence slice LUT frontier product mismatch".to_owned());
        }
        observed_rows += rows;
    }
    if observed_rows != lut.l_rows {
        return Err("G0 CPU evidence LUT row total mismatch".to_owned());
    }
    Ok(G0CpuLutObservation {
        exact_row_count: lut.l_rows.to_string(),
        exact_payload_logical_items: lut
            .logical_items()
            .map_err(|_| "G0 CPU evidence LUT logical-item overflow".to_owned())?,
        index_use_frontier_products: lut
            .index_uses
            .iter()
            .map(|unit| unit.frontier_product.clone())
            .collect(),
        slice_group_frontier_products: lut
            .slice_groups
            .iter()
            .map(|unit| unit.frontier_product.clone())
            .collect(),
    })
}

fn aggregate_generator_peak_retained_logical_items<I>(
    phases: I,
) -> Result<u64, CanonicalPayloadError>
where
    I: IntoIterator<Item = Result<u64, CanonicalPayloadError>>,
{
    phases.into_iter().try_fold(0_u64, |peak, phase| Ok(peak.max(phase?)))
}

/// Produce deterministic CPU-observation evidence bytes without emitting a file or claiming a
/// G0 hard gate, protocol execution, or GPU evidence.
pub fn prepare_g0_cpu_evidence_bytes(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<Vec<u8>, String> {
    let run =
        prepare_operational_certificate(protocol, request).map_err(|error| error.to_string())?;
    let closure = &run.projection.closure;
    let statement_rows = super::g0::derive_certificate_statement_rows(
        &run.job,
        closure,
        &run.trace,
        closed_residual_expression(&run),
    )
    .map_err(|error| error.to_string())?;
    let base_feasibility = base_feasibility_summary_from_run(&run, &statement_rows)?;
    let n = exact_retained_n(&statement_rows)?;

    let inventory = super::g0::inventory_from_statement_rows(&statement_rows);
    if checked_cardinality(inventory.sources.len(), "canonical source row")? != n.source_rows {
        return Err("G0 CPU evidence source-row authority mismatch".to_owned());
    }
    if checked_cardinality(inventory.events.len(), "canonical event row")? != n.event_rows {
        return Err("G0 CPU evidence event-row authority mismatch".to_owned());
    }
    let inventory_retained_logical_items =
        inventory.retained_logical_items().map_err(|error| error.to_string())?;
    let descriptor_inventory_canonical_encoded_bytes = checked_cardinality(
        inventory.encode_canonical().map_err(|error| error.to_string())?.len(),
        "descriptor inventory byte",
    )?;
    drop(inventory);

    let lut =
        super::g0::derive_lut_evidence_with_refs(&run.job, closure, &run.trace, &statement_rows)
            .map_err(|error| error.to_string())?;
    let lut_observation = g0_cpu_lut_observation(&lut)?;
    let lut_retained_logical_items =
        lut.retained_logical_items().map_err(|error| error.to_string())?;
    let lut_canonical_encoded_bytes = checked_cardinality(
        lut.encode_canonical().map_err(|error| error.to_string())?.len(),
        "LUT byte",
    )?;
    drop(lut);

    let ProofPayloadProjection {
        payload,
        generator_peak_retained_logical_items: proof_projection_peak_retained_logical_items,
        observed_coverage,
    } = derive_proof_payload_projection_with_refs(&run, &statement_rows)
        .map_err(|error| error.to_string())?;
    let residual_coverage_matrix =
        derive_residual_coverage_matrix(&observed_coverage).map_err(|error| error.to_string())?;
    let proof_payload_logical_items = payload
        .logical_items()
        .map_err(|_| "G0 CPU evidence proof payload logical-item overflow".to_owned())?;
    let proof_payload_canonical_encoded_bytes = checked_cardinality(
        payload
            .encode_canonical()
            .map_err(|_| "G0 CPU evidence proof payload encoding overflow".to_owned())?
            .len(),
        "proof payload byte",
    )?;
    let retention = run.trace.recorder_retention();
    let generator_peak_retained_logical_items = aggregate_generator_peak_retained_logical_items([
        Ok(inventory_retained_logical_items),
        Ok(lut_retained_logical_items),
        Ok(proof_projection_peak_retained_logical_items),
    ])
    .map_err(|_| "G0 CPU evidence generator retention overflow".to_owned())?;

    if checked_cardinality(base_feasibility.n.expression_rows, "base expression")? !=
        n.expression_rows ||
        checked_cardinality(base_feasibility.n.program_rows, "base program")? != n.program_rows ||
        checked_cardinality(base_feasibility.n.source_rows, "base source")? != n.source_rows ||
        checked_cardinality(base_feasibility.n.event_rows, "base event")? != n.event_rows ||
        checked_cardinality(base_feasibility.n.total_rows, "base total")? != n.total_rows
    {
        return Err("G0 CPU evidence N authority mismatch".to_owned());
    }

    let evidence = G0CpuEvidence {
        schema_id: G0_CPU_EVIDENCE_SCHEMA_ID,
        schema_version: G0_CPU_EVIDENCE_SCHEMA_VERSION,
        status: G0CpuEvidenceStatus::CpuObservationOnlyNotAcceptanceEvidence,
        base_feasibility,
        residual_coverage_matrix,
        lut: lut_observation,
        metrics: G0CpuMetrics {
            descriptor_inventory_canonical_encoded_bytes,
            inventory_retained_logical_items,
            proof_payload_logical_items,
            proof_payload_canonical_encoded_bytes,
            lut_canonical_encoded_bytes,
            lut_retained_logical_items,
            recorder_peak_retained_logical_items: retention.peak_logical_items,
            proof_projection_peak_retained_logical_items,
            generator_peak_retained_logical_items,
        },
    };
    serde_json::to_vec(&evidence).map_err(|error| error.to_string())
}

fn project_residual_root(
    job: &super::job::CheckerJob,
    root: &ProductionRoot,
    target: &ResolvedAcceptanceTarget,
) -> Result<CertificateResidualRoot, CertificateProjectionError> {
    let projected = match root {
        ProductionRoot::Closed(root) => {
            let actual = job.expressions().value_type(root.expression()).map_err(|error| {
                CertificateProjectionError::Lowering { detail: error.to_string() }
            })?;
            let ResolvedValueType::Matrix(matrix) = actual else {
                return Err(CertificateProjectionError::ResidualRootNotMatrix {
                    target_id: target.target_id.clone(),
                    actual: actual.clone(),
                });
            };
            CertificateResidualRoot::Closed { root: *root, matrix: matrix.clone() }
        }
        ProductionRoot::Family(family) => {
            let actual = job.programs().family_element_type(*family).map_err(|error| {
                CertificateProjectionError::Lowering { detail: error.to_string() }
            })?;
            let ResolvedValueType::Matrix(matrix) = actual else {
                return Err(CertificateProjectionError::ResidualRootNotMatrix {
                    target_id: target.target_id.clone(),
                    actual,
                });
            };
            let domain = job.programs().family_domain(*family).map_err(|error| {
                CertificateProjectionError::Lowering { detail: error.to_string() }
            })?;
            CertificateResidualRoot::Family { family: *family, domain, matrix: matrix.clone() }
        }
    };
    let modulus = match &projected {
        CertificateResidualRoot::Closed { matrix, .. } |
        CertificateResidualRoot::Family { matrix, .. } => &matrix.modulus,
    };
    if modulus != &target.ciphertext_modulus {
        return Err(CertificateProjectionError::ResidualModulusMismatch {
            target_id: target.target_id.clone(),
            target: target.ciphertext_modulus.clone(),
            residual: modulus.clone(),
        });
    }
    Ok(projected)
}

/// One structured, allocation-free progress record supplied at instrumented checker boundaries.
/// Progress is best-effort: opaque lowering or normalization operations may run between events,
/// so this is not a wall-clock heartbeat guarantee.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressEvent {
    pub phase: CheckerPhase,
    pub event: ProgressEventKind,
    pub processed: u64,
    pub total_or_discovered: Option<u64>,
    pub elapsed_ms: u64,
    /// Number of expression-DAG nodes normalized so far.
    pub normalization_nodes_processed: u64,
    /// Number of expression-DAG nodes reachable from the current root set.
    pub normalization_nodes_total: u64,
    /// Number of exact terms retained by the current normal form.
    pub normalization_exact_term_count: u64,
    /// Number of bounded-only aggregations performed by normalization.
    pub normalization_bounded_fold_count: u64,
    /// Number of relation candidates inspected/applied/remaining.
    pub normalization_relation_candidates: u64,
    pub normalization_relations_applied: u64,
    pub normalization_relations_remaining: u64,
    pub owned_elements: u64,
    pub program: Option<String>,
    pub scope: Option<String>,
    pub node: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgressEventKind {
    Start,
    Progress,
    Complete,
}

#[derive(Clone, Debug)]
struct ProgressState {
    processed: u64,
    last_emitted: Instant,
}

impl ProgressState {
    fn new(started: Instant) -> Self {
        Self { processed: 0, last_emitted: started }
    }
}

/// One job's mutable progress and diagnostics owner.
pub(crate) struct SimulationControl<'a> {
    started: Instant,
    phase: CheckerPhase,
    owned_elements: Arc<AtomicUsize>,
    diagnostics: OperationalSimulationDiagnostics,
    progress: ProgressState,
    progress_site: Option<(String, String, u64)>,
    emit_progress: &'a mut dyn FnMut(ProgressEvent),
}

impl<'a> SimulationControl<'a> {
    fn new(emit_progress: &'a mut dyn FnMut(ProgressEvent)) -> Self {
        let started = Instant::now();
        Self {
            started,
            phase: CheckerPhase::Target,
            owned_elements: Arc::new(AtomicUsize::new(0)),
            diagnostics: OperationalSimulationDiagnostics::default(),
            progress: ProgressState::new(started),
            progress_site: None,
            emit_progress,
        }
    }

    /// Attaches the currently processed graph occurrence to cadence events.
    /// Callers update it at node/scope boundaries; it never requires a scan.
    pub(crate) fn set_progress_site(&mut self, program: String, scope: String, node: u64) {
        self.progress_site = Some((program, scope, node));
    }

    pub(crate) fn reserve_owned_elements(
        &mut self,
        requested: usize,
    ) -> Result<(), OperationalSimulationError> {
        let _ = self.owned_elements.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(requested))
        });
        Ok(())
    }

    /// Counts one node, edge, branch, iteration, or bound-work boundary and emits at the
    /// required work/time cadence without scanning any checker-owned collection.
    pub(crate) fn work(
        &mut self,
        units: u64,
        total_or_discovered: Option<u64>,
        _unused_graph_size: Option<u64>,
    ) -> Result<(), OperationalSimulationError> {
        self.progress.processed = self.progress.processed.saturating_add(units);
        let now = Instant::now();
        if now.duration_since(self.progress.last_emitted) >= PROGRESS_TIME_CADENCE {
            self.emit(ProgressEventKind::Progress, total_or_discovered, now);
            self.progress.last_emitted = now;
        }
        Ok(())
    }

    fn begin_phase(&mut self, phase: CheckerPhase) -> Result<Instant, OperationalSimulationError> {
        self.phase = phase;
        self.progress = ProgressState::new(Instant::now());
        self.emit(ProgressEventKind::Start, None, Instant::now());
        Ok(Instant::now())
    }

    fn complete_phase(
        &mut self,
        phase_started: Instant,
        total_or_discovered: Option<u64>,
        _unused_graph_size: Option<u64>,
    ) -> Result<Duration, OperationalSimulationError> {
        let now = Instant::now();
        self.emit(ProgressEventKind::Complete, total_or_discovered, now);
        Ok(now.duration_since(phase_started))
    }

    fn emit(&mut self, event: ProgressEventKind, total_or_discovered: Option<u64>, now: Instant) {
        (self.emit_progress)(ProgressEvent {
            phase: self.phase,
            event,
            processed: self.progress.processed,
            total_or_discovered,
            elapsed_ms: now.duration_since(self.started).as_millis() as u64,
            normalization_nodes_processed: self.diagnostics.normalization_node_count,
            normalization_nodes_total: self.diagnostics.normalization_node_total,
            normalization_exact_term_count: self.diagnostics.normalization_exact_term_count,
            normalization_bounded_fold_count: self.diagnostics.normalization_bounded_fold_count,
            normalization_relation_candidates: self.diagnostics.normalization_relation_count,
            normalization_relations_applied: self.diagnostics.normalization_relation_applied,
            normalization_relations_remaining: self.diagnostics.normalization_relation_remaining,
            owned_elements: self.owned_elements.load(Ordering::Relaxed) as u64,
            program: self.progress_site.as_ref().map(|(program, _, _)| program.clone()),
            scope: self.progress_site.as_ref().map(|(_, scope, _)| scope.clone()),
            node: self.progress_site.as_ref().map(|(_, _, node)| *node),
        });
    }
}

/// Checks one closed protocol candidate using production lowering, normal-form
/// normalization, and bound validation. Unsupported graph facts remain typed
/// errors; this entry point has no compatibility or heuristic fallback.
pub fn check_operational_noise_candidate(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let target = request.target_id.clone();
    let started = Instant::now();
    let mut last_progress = None;
    info!(target, "operational noise simulation begin");
    let result = check_operational_noise_candidate_with_progress(protocol, request, |event| {
        last_progress = Some(event.clone());
        match event.event {
            ProgressEventKind::Progress => debug!(
                target,
                phase = ?event.phase,
                elapsed_ms = event.elapsed_ms,
                processed = event.processed,
                total_or_discovered = ?event.total_or_discovered,
                owned_elements = event.owned_elements,
                normalization_nodes_processed = event.normalization_nodes_processed,
                normalization_nodes_total = event.normalization_nodes_total,
                normalization_exact_term_count = event.normalization_exact_term_count,
                normalization_bounded_fold_count = event.normalization_bounded_fold_count,
                normalization_relation_candidates = event.normalization_relation_candidates,
                normalization_relations_applied = event.normalization_relations_applied,
                normalization_relations_remaining = event.normalization_relations_remaining,
                program = ?event.program,
                scope = ?event.scope,
                node = ?event.node,
                "operational noise simulation progress"
            ),
            ProgressEventKind::Start => info!(
                target,
                phase = ?event.phase,
                elapsed_ms = event.elapsed_ms,
                owned_elements = event.owned_elements,
                "operational noise simulation phase begin"
            ),
            ProgressEventKind::Complete => info!(
                target,
                phase = ?event.phase,
                elapsed_ms = event.elapsed_ms,
                processed = event.processed,
                owned_elements = event.owned_elements,
                normalization_nodes_processed = event.normalization_nodes_processed,
                normalization_nodes_total = event.normalization_nodes_total,
                normalization_exact_term_count = event.normalization_exact_term_count,
                normalization_bounded_fold_count = event.normalization_bounded_fold_count,
                normalization_relation_candidates = event.normalization_relation_candidates,
                normalization_relations_applied = event.normalization_relations_applied,
                normalization_relations_remaining = event.normalization_relations_remaining,
                "operational noise simulation phase complete"
            ),
        }
    });
    match &result {
        Ok(report) => info!(
            target,
            elapsed = ?started.elapsed(),
            accepted = report.accepted,
            noise_bound = %report.noise_bound,
            diagnostics = ?report.diagnostics,
            "operational noise simulation complete"
        ),
        Err(simulation_error) => error!(
            target,
            elapsed = ?started.elapsed(),
            error = %simulation_error,
            partial_progress = ?last_progress,
            "operational noise simulation failed with partial diagnostics"
        ),
    }
    result
}

/// Checks one closed protocol candidate and exposes best-effort progress at instrumented work
/// boundaries without coupling the checker to a logging implementation. The callback is not a
/// wall-clock heartbeat while control is inside an opaque third-party operation.
pub fn check_operational_noise_candidate_with_progress(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
    mut emit: impl FnMut(ProgressEvent),
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let mut control = SimulationControl::new(&mut emit);
    let target_started = control.begin_phase(CheckerPhase::Target)?;
    control.work(
        protocol.params.len().saturating_add(request.environment.len()) as u64,
        None,
        None,
    )?;
    request
        .validate(protocol.params.iter().map(|parameter| parameter.name.clone()))
        .map_err(OperationalSimulationError::Request)?;
    let target = resolve_target(protocol, request, &mut control)?;
    control.work(1, None, None)?;
    control.complete_phase(target_started, None, None)?;
    // Production authority: once the request and decoder target have been validated,
    // all graph reachability, lowering, relation registration, normalization, and reporting go
    // through the job-local arenas.
    let parameters = request
        .environment
        .iter()
        .filter_map(|(name, value)| match value {
            super::OperationalParameterValue::Integer(value) => Some((name.clone(), value.clone())),
            super::OperationalParameterValue::Rational { .. } => None,
        })
        .collect::<BTreeMap<_, _>>();
    let plan = ProtocolPlan::build(protocol, &request.target_id).map_err(|error| {
        OperationalSimulationError::from(super::error::ProductionError::internal(
            super::error::ProductionPhase::Adapter,
            error.to_string(),
        ))
    })?;
    let (mut job, roots) = ProductionAdapter::new(protocol, &plan, parameters)
        .map_err(|error| {
            OperationalSimulationError::from(super::error::ProductionError::from(error))
        })?
        .lower()
        .map_err(|error| {
            OperationalSimulationError::from(super::error::ProductionError::from(error))
        })?;
    let report = analyze_roots(
        &mut job,
        &roots,
        &ReportTarget {
            target_id: target.target_id.clone(),
            plaintext_modulus: match target.kind {
                ResolvedDecoderKind::Threshold { ref plaintext_modulus } => {
                    plaintext_modulus.clone()
                }
                ResolvedDecoderKind::BooleanInterval => BigUint::from(2_u8),
            },
            ciphertext_modulus: target.ciphertext_modulus,
            boolean_interval: matches!(target.kind, ResolvedDecoderKind::BooleanInterval),
        },
    )
    .map_err(|error| {
        OperationalSimulationError::from(super::error::ProductionError::from(error))
    })?;
    Ok(report.into_simulation_report())
}

fn resolve_target(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
    control: &mut SimulationControl<'_>,
) -> Result<ResolvedAcceptanceTarget, OperationalSimulationError> {
    let mut target = None;
    let mut declarations = Vec::new();
    for candidate in &protocol.bundle.operational_decoder_targets {
        control.work(1, Some(protocol.bundle.operational_decoder_targets.len() as u64), None)?;
        if candidate.target_id != request.target_id {
            continue;
        }
        control.reserve_owned_elements(1)?;
        declarations.push(super::error::TargetDeclarationSite {
            target_id: candidate.target_id.clone(),
            residual: stage_output(candidate),
            decoder: super::error::DecoderWireRef {
                stage: candidate.decoder_stage.clone(),
                node: candidate.decoder_node,
                port: 0,
            },
        });
        target.get_or_insert(candidate);
    }
    let Some(target) = target else {
        return Err(OperationalSimulationError::Target(TargetError::MissingTargetId {
            target_id: request.target_id.clone(),
        }));
    };
    if declarations.len() != 1 {
        return Err(OperationalSimulationError::Target(TargetError::DuplicateTargetId {
            target_id: request.target_id.clone(),
            declarations: declarations.into_boxed_slice(),
        }));
    };
    let stage =
        protocol.stages().iter().find(|stage| stage.id == target.residual_stage).ok_or_else(
            || {
                OperationalSimulationError::Target(TargetError::MissingStage {
                    target_id: target.target_id.clone(),
                    role: super::error::TargetStageRole::Residual,
                    stage: target.residual_stage.clone(),
                })
            },
        )?;
    let output = stage.graph.outputs().get(&target.residual_output).ok_or_else(|| {
        OperationalSimulationError::Target(TargetError::MissingResidualOutput {
            target_id: target.target_id.clone(),
            residual: super::error::StageOutputRef {
                stage: target.residual_stage.clone(),
                output: target.residual_output.clone(),
            },
        })
    })?;
    let wire = output.value;
    control.set_progress_site(stage.id.0.clone(), "root".to_owned(), wire.node.0 as u64);
    control.work(1, None, None)?;
    let wire_type = stage
        .graph
        .root_scope()
        .node(wire.node)
        .and_then(|node| node.output_types().get(wire.port.0 as usize))
        .ok_or_else(|| {
            OperationalSimulationError::Target(TargetError::MissingResidualOutput {
                target_id: target.target_id.clone(),
                residual: super::error::StageOutputRef {
                    stage: target.residual_stage.clone(),
                    output: target.residual_output.clone(),
                },
            })
        })?;
    let matrix = match wire_type {
        WireType::Matrix(matrix) => matrix,
        WireType::IndexedFamily { element, .. } => match element.as_ref() {
            WireType::Matrix(matrix) => matrix,
            actual => {
                return Err(OperationalSimulationError::Target(TargetError::InvalidResidualSort {
                    target_id: target.target_id.clone(),
                    residual: stage_output(target),
                    actual: actual.clone(),
                }));
            }
        },
        actual => {
            return Err(OperationalSimulationError::Target(TargetError::InvalidResidualSort {
                target_id: target.target_id.clone(),
                residual: stage_output(target),
                actual: actual.clone(),
            }))
        }
    };
    let decoder_stage =
        protocol.stages().iter().find(|stage| stage.id == target.decoder_stage).ok_or_else(
            || {
                OperationalSimulationError::Target(TargetError::MissingStage {
                    target_id: target.target_id.clone(),
                    role: super::error::TargetStageRole::Decoder,
                    stage: target.decoder_stage.clone(),
                })
            },
        )?;
    let decoder = super::error::DecoderWireRef {
        stage: target.decoder_stage.clone(),
        node: target.decoder_node,
        port: 0,
    };
    let node = decoder_stage.graph.root_scope().node(target.decoder_node).ok_or_else(|| {
        OperationalSimulationError::Target(TargetError::MissingDecoderWire {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
        })
    })?;
    let endpoint = protocol.bundle.endpoints.entries.iter().find(|endpoint| {
        endpoint.stage == target.decoder_stage &&
            decoder_stage
                .graph
                .outputs()
                .get(&endpoint.workflow_output.output)
                .is_some_and(|output| output.value.node == target.decoder_node)
    });
    let endpoint_output = endpoint.and_then(|endpoint| {
        decoder_stage
            .graph
            .outputs()
            .get(&endpoint.workflow_output.output)
            .map(|output| output.value)
    });
    let endpoint_ref = endpoint_output.map(|wire| super::error::DecoderWireRef {
        stage: target.decoder_stage.clone(),
        node: wire.node,
        port: wire.port.0,
    });
    if endpoint_ref.as_ref() != Some(&decoder) {
        return Err(OperationalSimulationError::Target(
            TargetError::DecoderWorkflowOutputMismatch {
                target_id: target.target_id.clone(),
                expected: decoder.clone(),
                actual: endpoint_ref,
            },
        ));
    }
    if let Some(scope) = endpoint
        .and_then(|endpoint| decoder_stage.semantic_anchors.get(&endpoint.semantic_anchor))
        .and_then(|wires| wires.iter().find(|wire| wire.wire.node == target.decoder_node))
        .map(|wire| wire.scope.clone())
        .filter(|scope| *scope != FrozenGraphScopeId::Root)
    {
        return Err(OperationalSimulationError::Target(TargetError::DecoderWireNotRoot {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            actual_scope: scope,
        }));
    }
    let semantic_ref = endpoint
        .and_then(|endpoint| {
            decoder_stage.semantic_anchors.get(&endpoint.semantic_anchor).and_then(|wires| {
                (wires.len() == 1 && wires[0].scope == FrozenGraphScopeId::Root)
                    .then_some(&wires[0])
            })
        })
        .map(|wire| super::error::DecoderWireRef {
            stage: target.decoder_stage.clone(),
            node: wire.wire.node,
            port: wire.wire.port.0,
        });
    if semantic_ref.as_ref() != Some(&decoder) {
        return Err(OperationalSimulationError::Target(
            TargetError::DecoderSemanticAnchorMismatch {
                target_id: target.target_id.clone(),
                expected: decoder.clone(),
                actual: semantic_ref,
            },
        ));
    }
    let actual_kind = node.kind().clone();
    let kind_matches = matches!(
        (&target.kind, node.kind()),
        (OperationalDecoderKind::ThresholdDecode { .. }, NodeKind::ThresholdDecode { .. }) |
            (OperationalDecoderKind::BooleanInterval, NodeKind::IntCompare(IntCompareOp::Equal))
    );
    if !kind_matches {
        return Err(OperationalSimulationError::Target(TargetError::DecoderKindMismatch {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            expected: target.kind.clone(),
            actual: actual_kind,
        }));
    }
    if matches!(target.kind, OperationalDecoderKind::BooleanInterval) &&
        !boolean_interval_decoder_matches(&decoder_stage.graph, target.decoder_node, wire)
    {
        return Err(OperationalSimulationError::Target(
            TargetError::DecoderInputDoesNotConsumeResidual {
                target_id: target.target_id.clone(),
                decoder: decoder.clone(),
                residual: stage_output(target),
                actual_input: decoder_stage
                    .graph
                    .root_scope()
                    .arguments(node)
                    .and_then(|arguments| arguments.first().copied()),
            },
        ));
    }
    let arguments = decoder_stage.graph.root_scope().arguments(node).unwrap_or_default();
    let expected_arity = match &target.kind {
        OperationalDecoderKind::ThresholdDecode { .. } => 1,
        OperationalDecoderKind::BooleanInterval => 2,
    };
    if arguments.len() != expected_arity {
        return Err(OperationalSimulationError::Target(TargetError::DecoderArityMismatch {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            expected: expected_arity,
            actual: arguments.len(),
        }));
    }
    let output_types = node.output_types();
    if output_types.len() != 1 {
        return Err(OperationalSimulationError::Target(TargetError::DecoderOutputCountMismatch {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            expected: 1,
            actual: output_types.len(),
        }));
    }
    if decoder.port as usize >= output_types.len() {
        return Err(OperationalSimulationError::Target(TargetError::DecoderOutputPortOutOfRange {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            output_count: output_types.len(),
        }));
    }
    if output_types[decoder.port as usize] != WireType::Bool {
        return Err(OperationalSimulationError::Target(TargetError::DecoderOutputTypeMismatch {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            expected: WireType::Bool,
            actual: output_types[decoder.port as usize].clone(),
        }));
    }
    let expected_snapshot = target_decoder_snapshot(&target.kind, node.kind(), output_types);
    let actual_snapshot = node_decoder_snapshot(node.kind(), Some(output_types));
    if expected_snapshot != actual_snapshot {
        return Err(OperationalSimulationError::Target(TargetError::DecoderAttributeMismatch {
            target_id: target.target_id.clone(),
            decoder: decoder.clone(),
            expected: expected_snapshot,
            actual: actual_snapshot,
        }));
    }
    let mut consumes_residual = false;
    if target.decoder_stage == target.residual_stage {
        for input in arguments.iter().copied() {
            if wire_consumes(&decoder_stage.graph, input, wire, control)? {
                consumes_residual = true;
                break;
            }
        }
    } else {
        // Cross-stage decoder inputs are represented by an explicit artifact
        // binding rather than by a same-graph wire edge.  Treat that binding
        // as the generic provenance edge; do not infer it from names or
        // matrix shape alone.
        consumes_residual = decoder_stage.bindings.iter().any(|binding| {
            binding.producer_stage == target.residual_stage &&
                binding.producer_output == crate::ArtifactName(target.residual_output.clone())
        });
    }
    if !consumes_residual {
        return Err(OperationalSimulationError::Target(
            TargetError::DecoderInputDoesNotConsumeResidual {
                target_id: target.target_id.clone(),
                decoder: decoder.clone(),
                residual: stage_output(target),
                actual_input: arguments.first().copied(),
            },
        ));
    }
    if let Some(WireType::Matrix(decoder_input)) = decoder_stage
        .graph
        .root_scope()
        .node(arguments[0].node)
        .and_then(|input| input.output_types().get(arguments[0].port.0 as usize))
    {
        if decoder_input.modulus != matrix.modulus {
            return Err(OperationalSimulationError::Target(TargetError::DecoderModulusMismatch {
                target_id: target.target_id.clone(),
                decoder: decoder.clone(),
                residual_modulus: matrix.modulus.clone(),
                decoder_modulus: decoder_input.modulus.clone(),
            }));
        }
    }
    control.reserve_owned_elements(request.environment.len())?;
    for _ in &request.environment {
        control.work(1, Some(request.environment.len() as u64), None)?;
    }
    let environment = ParamEnv {
        integers: request
            .environment
            .iter()
            .filter_map(|(name, value)| match value {
                super::OperationalParameterValue::Integer(value) => {
                    Some((name.clone(), value.clone()))
                }
                super::OperationalParameterValue::Rational { .. } => None,
            })
            .collect(),
        ..ParamEnv::default()
    };
    let modulus = matrix.modulus.evaluate(&environment).map_err(|_| {
        OperationalSimulationError::Target(TargetError::NonClosedCiphertextModulus {
            target_id: target.target_id.clone(),
            expression: matrix.modulus.clone(),
        })
    })?;
    let ciphertext_modulus =
        modulus.to_biguint().filter(|value| !value.is_zero()).ok_or_else(|| {
            OperationalSimulationError::Target(TargetError::NonPositiveCiphertextModulus {
                target_id: target.target_id.clone(),
                actual: modulus,
            })
        })?;
    let kind = match &target.kind {
        OperationalDecoderKind::BooleanInterval => ResolvedDecoderKind::BooleanInterval,
        OperationalDecoderKind::ThresholdDecode { plaintext_modulus } => {
            let value = plaintext_modulus.evaluate(&environment).map_err(|_| {
                OperationalSimulationError::Target(TargetError::NonClosedPlaintextModulus {
                    target_id: target.target_id.clone(),
                    expression: plaintext_modulus.clone(),
                })
            })?;
            let plaintext_modulus =
                value.to_biguint().filter(|value| !value.is_zero()).ok_or_else(|| {
                    OperationalSimulationError::Target(TargetError::NonPositivePlaintextModulus {
                        target_id: target.target_id.clone(),
                        actual: value,
                    })
                })?;
            ResolvedDecoderKind::Threshold { plaintext_modulus }
        }
    };
    Ok(ResolvedAcceptanceTarget { target_id: target.target_id.clone(), ciphertext_modulus, kind })
}

fn node_kind_and_arguments<const N: usize>(
    graph: &Graph,
    wire: WireRef,
) -> Option<(&NodeKind, [WireRef; N])> {
    if wire.port != Port(0) {
        return None;
    }
    let node = graph.root_scope().node(wire.node)?;
    Some((node.kind(), graph.root_scope().arguments(node)?.try_into().ok()?))
}

// Keep this structurally identical to the bundle validator's executable
// interval-chain check. The operational checker must not accept a merely
// connected equality whose interior arithmetic differs from the endpoint.
fn boolean_interval_decoder_matches(
    graph: &Graph,
    decoder_node: mxx_ir_core::NodeId,
    residual: WireRef,
) -> bool {
    let Some(WireType::Matrix(residual_type)) = graph
        .root_scope()
        .node(residual.node)
        .and_then(|node| node.output_types().get(residual.port.0 as usize))
    else {
        return false
    };
    let Some((NodeKind::IntCompare(IntCompareOp::Equal), [sum, two])) =
        node_kind_and_arguments(graph, WireRef { node: decoder_node, port: Port(0) })
    else {
        return false
    };
    let Some((NodeKind::IntBinary(IntBinaryOp::Add), [lower_int, upper_int])) =
        node_kind_and_arguments(graph, sum)
    else {
        return false
    };
    let Some((NodeKind::BoolToInt, [lower_ok])) = node_kind_and_arguments(graph, lower_int) else {
        return false
    };
    let Some((NodeKind::BoolToInt, [upper_ok])) = node_kind_and_arguments(graph, upper_int) else {
        return false
    };
    let Some((NodeKind::IntCompare(IntCompareOp::LessEqual), [quarter, coefficient])) =
        node_kind_and_arguments(graph, lower_ok)
    else {
        return false
    };
    let Some((NodeKind::IntCompare(IntCompareOp::LessEqual), [upper_coefficient, upper])) =
        node_kind_and_arguments(graph, upper_ok)
    else {
        return false
    };
    let Some((NodeKind::IntBinary(IntBinaryOp::Multiply), [upper_quarter, three])) =
        node_kind_and_arguments(graph, upper)
    else {
        return false
    };
    let Some((NodeKind::EvaluateInt(quarter_expression), [])) =
        node_kind_and_arguments(graph, quarter)
    else {
        return false
    };
    let Some((NodeKind::ExtractCoefficient { position, .. }, [coefficient_input])) =
        node_kind_and_arguments(graph, coefficient)
    else {
        return false
    };
    let expected_quarter = IntExpr::RoundDiv(
        Box::new(IntExpr::Sub(
            Box::new(residual_type.modulus.clone()),
            Box::new(IntExpr::constant(2)),
        )),
        Box::new(IntExpr::constant(4)),
    );
    coefficient_input == residual &&
        upper_coefficient == coefficient &&
        upper_quarter == quarter &&
        position == &IntExpr::constant(0) &&
        quarter_expression == &expected_quarter &&
        matches!(node_kind_and_arguments::<0>(graph, two), Some((NodeKind::ConstantInt(value), [])) if value == &BigInt::from(2)) &&
        matches!(node_kind_and_arguments::<0>(graph, three), Some((NodeKind::ConstantInt(value), [])) if value == &BigInt::from(3))
}

fn stage_output(target: &crate::OperationalDecoderTarget) -> super::error::StageOutputRef {
    super::error::StageOutputRef {
        stage: target.residual_stage.clone(),
        output: target.residual_output.clone(),
    }
}

fn target_decoder_snapshot(
    kind: &OperationalDecoderKind,
    node_kind: &NodeKind,
    output_types: &[WireType],
) -> super::error::DecoderSnapshot {
    match kind {
        OperationalDecoderKind::ThresholdDecode { plaintext_modulus } => {
            super::error::DecoderSnapshot {
                kind: kind.clone(),
                operand_count: 1,
                output_types: output_types.into(),
                plaintext_modulus: plaintext_modulus.clone(),
                length: match node_kind {
                    NodeKind::ThresholdDecode { length, .. } => Some(length.clone()),
                    _ => None,
                },
                output_bool: Some(true),
            }
        }
        OperationalDecoderKind::BooleanInterval => super::error::DecoderSnapshot {
            kind: kind.clone(),
            operand_count: 2,
            output_types: output_types.into(),
            plaintext_modulus: IntExpr::constant(0),
            length: None,
            output_bool: None,
        },
    }
}

fn node_decoder_snapshot(
    kind: &NodeKind,
    output_types: Option<&[WireType]>,
) -> super::error::DecoderSnapshot {
    match kind {
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            super::error::DecoderSnapshot {
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: plaintext_modulus.clone(),
                },
                operand_count: 1,
                output_types: output_types.unwrap_or(&[]).into(),
                plaintext_modulus: plaintext_modulus.clone(),
                length: Some(length.clone()),
                output_bool: Some(*output_bool),
            }
        }
        NodeKind::IntCompare(IntCompareOp::Equal) => super::error::DecoderSnapshot {
            kind: OperationalDecoderKind::BooleanInterval,
            operand_count: 2,
            output_types: output_types.unwrap_or(&[]).into(),
            plaintext_modulus: IntExpr::constant(0),
            length: None,
            output_bool: None,
        },
        _ => unreachable!("decoder kind was checked before snapshot construction"),
    }
}

fn wire_consumes(
    graph: &mxx_ir_core::Graph,
    current: WireRef,
    target: WireRef,
    control: &mut SimulationControl<'_>,
) -> Result<bool, OperationalSimulationError> {
    let mut pending = vec![current];
    control.reserve_owned_elements(1)?;
    let mut visited = std::collections::BTreeSet::new();
    while let Some(wire) = pending.pop() {
        control.set_progress_site("target".to_owned(), "root".to_owned(), wire.node.0 as u64);
        control.work(1, Some(visited.len() as u64), None)?;
        if wire == target {
            return Ok(true);
        }
        if !visited.insert(wire) {
            continue;
        }
        control.reserve_owned_elements(1)?;
        let Some(node) = graph.root_scope().node(wire.node) else {
            continue;
        };
        if let Some(arguments) = graph.root_scope().arguments(node) {
            control.reserve_owned_elements(arguments.len())?;
            pending.extend(arguments.iter().copied());
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::{super::g0::FeasibilitySink, *};
    use crate::{
        OutputRef, ProtocolDecl, ProtocolStage, StageId,
        bundle::{
            ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor,
            EndpointAnchors, EndpointSemanticBinding, EndpointSpecId, InputContract,
            InputContractEntry, InputValueContract, OperationalDecoderTarget, ProtocolInputBinding,
            ProtocolInputDestination, Workflow,
        },
        operational_noise::facts::{BoundExpression, CoefficientBound, NumericContract},
    };
    use mxx_dsl::{Bool, DslContext, IdealSpec, Int, Parallel, Ring, SemanticAnchor};
    use mxx_ir_core::{IntExpr, node::ConstantMatrix};

    fn predecessor_coefficient(input_position: u32) -> BoundValueRef {
        BoundValueRef::Predecessor { input_position, projection: BoundProjection::Coefficient }
    }

    fn finite_coefficient_bound(value: u64) -> NumericContract<CoefficientBound> {
        NumericContract::Known(CoefficientBound::finite(value))
    }

    #[test]
    fn reached_exact_producer_resolver_leaves_shared_finite_summary_for_semantic_fold() {
        let rule = BoundRule::Sum {
            inputs: vec![predecessor_coefficient(0), predecessor_coefficient(1)].into_boxed_slice(),
        };
        let candidates = [(107_405, &rule)];

        assert_eq!(
            select_reached_exact_producers(&candidates, &finite_coefficient_bound(4)),
            Ok((107_405, None))
        );
    }

    #[test]
    fn reached_exact_producer_resolver_uses_endpoints_not_repeated_rule_classes() {
        let coefficient = BoundRule::Product {
            left: predecessor_coefficient(0),
            right: predecessor_coefficient(1),
            facts: Default::default(),
        };
        let internal_one = BoundRule::Sum {
            inputs: vec![BoundValueRef::Transfer(super::super::g0::EventIndex(11))]
                .into_boxed_slice(),
        };
        let internal_two = BoundRule::Sum {
            inputs: vec![BoundValueRef::Transfer(super::super::g0::EventIndex(12))]
                .into_boxed_slice(),
        };
        let summary = BoundRule::Sum {
            inputs: vec![BoundValueRef::Transfer(super::super::g0::EventIndex(13))]
                .into_boxed_slice(),
        };
        let candidates =
            [(10, &coefficient), (11, &internal_one), (12, &internal_two), (13, &summary)];

        assert_eq!(
            select_reached_exact_producers(&candidates, &finite_coefficient_bound(8)),
            Ok((10, Some(13)))
        );
    }

    #[test]
    fn reached_exact_producer_resolver_rejects_multiple_coefficient_roots() {
        let first = BoundRule::Sum {
            inputs: vec![predecessor_coefficient(0), predecessor_coefficient(1)].into_boxed_slice(),
        };
        let second = BoundRule::Product {
            left: predecessor_coefficient(0),
            right: predecessor_coefficient(1),
            facts: Default::default(),
        };
        let candidates = [(20, &first), (21, &second)];

        assert!(matches!(
            select_reached_exact_producers(&candidates, &finite_coefficient_bound(16)),
            Err(G0Error::UnsupportedBoundTransfer)
        ));
    }

    #[test]
    fn reached_exact_coefficient_allows_post_merge_cancellation() {
        assert!(coefficient_bound_is_within(
            &NumericContract::Known(CoefficientBound::ExactZero),
            &finite_coefficient_bound(2),
        ));
        assert!(!coefficient_bound_is_within(
            &finite_coefficient_bound(3),
            &finite_coefficient_bound(2),
        ));
    }

    #[test]
    fn reached_exact_deferred_finite_summary_requires_large_transfer_and_merge() {
        let large = NumericContract::Known(CoefficientBound::Large);
        assert!(reached_deferred_finite_summary_allowed(&large, true));
        assert!(!reached_deferred_finite_summary_allowed(&large, false));
        assert!(!reached_deferred_finite_summary_allowed(&finite_coefficient_bound(4), true));
    }

    #[test]
    fn observed_coverage_classifiers_cover_all_typed_discriminants() {
        let matrix = ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).expect("matrix type");
        let layout = MatrixLayout::row_major(1, 1);
        let scalar_cases = vec![
            (ScalarOperation::Add, ObservedScalarKind::Add),
            (ScalarOperation::Subtract, ObservedScalarKind::Subtract),
            (ScalarOperation::Multiply, ObservedScalarKind::Multiply),
            (ScalarOperation::Divide, ObservedScalarKind::Divide),
            (ScalarOperation::Remainder, ObservedScalarKind::Remainder),
            (ScalarOperation::Negate, ObservedScalarKind::Negate),
            (ScalarOperation::Equal, ObservedScalarKind::Equal),
            (ScalarOperation::Less, ObservedScalarKind::Less),
            (ScalarOperation::LessEqual, ObservedScalarKind::LessEqual),
            (ScalarOperation::BoolToInt, ObservedScalarKind::BoolToInt),
            (ScalarOperation::IntToReal, ObservedScalarKind::IntToReal),
            (ScalarOperation::RealAdd, ObservedScalarKind::RealAdd),
            (ScalarOperation::RealSubtract, ObservedScalarKind::RealSubtract),
            (ScalarOperation::RealMultiply, ObservedScalarKind::RealMultiply),
            (ScalarOperation::RealDivide, ObservedScalarKind::RealDivide),
            (ScalarOperation::RealSqrt, ObservedScalarKind::RealSqrt),
            (
                ScalarOperation::ThresholdDecode {
                    plaintext_modulus: 2_u8.into(),
                    length: 1,
                    output_bool: false,
                },
                ObservedScalarKind::ThresholdDecode,
            ),
            (ScalarOperation::Bit { position: 3 }, ObservedScalarKind::Bit),
            (ScalarOperation::Slice { start: 2, end_exclusive: 5 }, ObservedScalarKind::Slice),
            (
                ScalarOperation::Hash { tag: "tag".to_owned(), dynamic_tags: Box::new([7]) },
                ObservedScalarKind::Hash,
            ),
            (
                ScalarOperation::ExtractCoefficient { row: 1, column: 2 },
                ObservedScalarKind::ExtractCoefficient,
            ),
            (
                ScalarOperation::LiftConstantPolynomial {
                    output: matrix.clone(),
                    coefficient_bits: 8,
                },
                ObservedScalarKind::LiftConstantPolynomial,
            ),
        ];
        for (operation, expected) in scalar_cases {
            assert_eq!(observed_scalar_kind(&operation), expected);
        }

        let matrix_cases = vec![
            (MatrixOperation::Add, ObservedMatrixKind::Add),
            (MatrixOperation::Subtract, ObservedMatrixKind::Subtract),
            (MatrixOperation::Multiply, ObservedMatrixKind::Multiply),
            (MatrixOperation::Negate, ObservedMatrixKind::Negate),
            (MatrixOperation::Scale, ObservedMatrixKind::Scale),
            (MatrixOperation::Transpose, ObservedMatrixKind::Transpose),
            (
                MatrixOperation::Slice {
                    row_start: 0,
                    row_end_exclusive: 1,
                    column_start: 0,
                    column_end_exclusive: 1,
                    layout: layout.clone(),
                },
                ObservedMatrixKind::Slice,
            ),
            (
                MatrixOperation::IndexedSlice { output: matrix.clone(), layout: layout.clone() },
                ObservedMatrixKind::IndexedSlice,
            ),
            (
                MatrixOperation::View { output: matrix.clone(), layout: layout.clone() },
                ObservedMatrixKind::View,
            ),
            (
                MatrixOperation::Concat { axis: 0, output: matrix.clone(), layout: layout.clone() },
                ObservedMatrixKind::Concat,
            ),
            (
                MatrixOperation::Tensor {
                    output: matrix.clone(),
                    left_layout: layout.clone(),
                    right_layout: layout.clone(),
                    output_layout: layout.clone(),
                },
                ObservedMatrixKind::Tensor,
            ),
            (
                MatrixOperation::CrtRecompose {
                    plaintext_moduli: Box::new([2_u8.into()]),
                    reconstruction_coefficients: Box::new([1_u8.into()]),
                    output: matrix.clone(),
                },
                ObservedMatrixKind::CrtRecompose,
            ),
            (
                MatrixOperation::ExtractCoefficient { row: 0, column: 0 },
                ObservedMatrixKind::ExtractCoefficient,
            ),
            (
                MatrixOperation::LiftConstantPolynomial {
                    output: matrix.clone(),
                    coefficient_bits: 8,
                },
                ObservedMatrixKind::LiftConstantPolynomial,
            ),
        ];
        for (operation, expected) in matrix_cases {
            assert_eq!(observed_matrix_kind(&operation), expected);
        }

        let output = super::super::g0::StableValueType::Matrix {
            modulus: "17".to_owned(),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let sampler_cases = vec![
            (
                CanonicalEventKind::Sample {
                    descriptor: super::super::g0::StableSampleDescriptor {
                        definition: "sample".to_owned(),
                        parameters: vec![],
                        output_type: output.clone(),
                        gadget_base: None,
                        digit_count: None,
                        decomposition: None,
                    },
                },
                ObservedSamplerKind::Sample,
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::UniformResidue { output: output.clone() },
                },
                ObservedSamplerKind::UniformResidue,
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::UniformInterval {
                        output: output.clone(),
                        minimum: "0".to_owned(),
                        maximum: "1".to_owned(),
                    },
                },
                ObservedSamplerKind::UniformInterval,
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::Gaussian {
                        output: output.clone(),
                        sigma: "1".to_owned(),
                        max_coefficient_bound: "2".to_owned(),
                    },
                },
                ObservedSamplerKind::Gaussian,
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::Hash {
                        output: output.clone(),
                        variant: super::super::g0::StableHashVariant::Plain,
                        tag_prefix: vec![],
                        tag_expressions: vec![],
                        tag_decimal_expressions: vec![],
                        tag_u64_le_expressions: vec![],
                        base: None,
                        digit_count: None,
                    },
                },
                ObservedSamplerKind::Hash(ObservedHashKind::Plain),
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::Trapdoor {
                        output: output.clone(),
                        sigma: "1".to_owned(),
                        gadget_base: 2,
                        digit_count: 1,
                        preimage_max_coefficient_bound: "2".to_owned(),
                    },
                },
                ObservedSamplerKind::Trapdoor,
            ),
            (
                CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::Preimage {
                        output,
                        max_coefficient_bound: "2".to_owned(),
                    },
                },
                ObservedSamplerKind::Preimage,
            ),
        ];
        for (kind, expected) in sampler_cases {
            assert_eq!(observed_sampler_kind(&kind), expected);
        }

        let value = ProofPayloadValueRef::Transfer(0);
        let monomial =
            ProofPayloadMonomial { central_factors: Vec::new(), ordered_factors: Vec::new() };
        let bound_cases = vec![
            (
                ProofPayloadRule::Authority(ProofPayloadAuthority::Operator),
                ObservedBoundKind::Authority(ObservedAuthorityKind::Operator),
            ),
            (ProofPayloadRule::Identity { input: value.clone() }, ObservedBoundKind::Identity),
            (ProofPayloadRule::Sum { inputs: vec![value.clone()] }, ObservedBoundKind::Sum),
            (ProofPayloadRule::Maximum { inputs: vec![value.clone()] }, ObservedBoundKind::Maximum),
            (
                ProofPayloadRule::Scale {
                    value: value.clone(),
                    scale: ProofPayloadScale::Magnitude(2_u8.into()),
                },
                ObservedBoundKind::Scale,
            ),
            (
                ProofPayloadRule::MonomialProduct {
                    monomial: monomial.clone(),
                    factors: Vec::new(),
                },
                ObservedBoundKind::MonomialProduct,
            ),
            (
                ProofPayloadRule::WeightedSum { inputs: vec![value.clone()] },
                ObservedBoundKind::WeightedSum,
            ),
            (
                ProofPayloadRule::Product {
                    left: value.clone(),
                    right: value.clone(),
                    facts: super::super::bound::MatrixProductFacts::default(),
                },
                ObservedBoundKind::Product,
            ),
            (
                ProofPayloadRule::Tensor {
                    left: value.clone(),
                    right: value,
                    left_is_constant_polynomial: false,
                    right_is_constant_polynomial: true,
                },
                ObservedBoundKind::Tensor,
            ),
        ];
        for (rule, expected) in bound_cases {
            assert_eq!(observed_bound_kind(&rule), expected);
        }
        assert_eq!(
            observed_relation_kind(&ProofPayloadRelationRule::Universal {
                computed: 0,
                lhs: monomial.clone(),
                lhs_layout: None,
                rhs_result: 0,
            }),
            ObservedRelationKind::Universal,
        );
        assert_eq!(
            observed_relation_kind(&ProofPayloadRelationRule::Gadget {
                gadget: ProofPayloadOwner {
                    scope: ProofPayloadScope::Closed { root_expression_row: 0 },
                    expression_row: 0,
                },
                decomposition: ProofPayloadOwner {
                    scope: ProofPayloadScope::Closed { root_expression_row: 0 },
                    expression_row: 1,
                },
                input: 0,
                input_result: 0,
            }),
            ObservedRelationKind::Gadget,
        );

        assert_eq!(
            observed_transform_kind(&ValueOperator::Transform(
                ValueTransformOperation::GadgetDecompose {
                    output: matrix.clone(),
                    base: 2,
                    small: true,
                    digit_count: 1,
                }
            )),
            Some(ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Small)),
        );
        assert_eq!(
            observed_transform_kind(&ValueOperator::Transform(
                ValueTransformOperation::GadgetDecompose {
                    output: matrix.clone(),
                    base: 2,
                    small: false,
                    digit_count: 1,
                }
            )),
            Some(ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Regular)),
        );
        assert_eq!(
            observed_transform_kind(&ValueOperator::Transform(
                ValueTransformOperation::PackPolynomialCoefficients {
                    output: matrix,
                    coefficient_bits: 8,
                }
            )),
            Some(ObservedTransformKind::PackPolynomialCoefficients),
        );
        assert_eq!(
            observed_operator_kind(&ValueOperator::Transform(
                ValueTransformOperation::PackPolynomialCoefficients {
                    output: ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap(),
                    coefficient_bits: 8,
                }
            )),
            None,
        );

        for variant in [
            StableHashVariant::Plain,
            StableHashVariant::Decomposed,
            StableHashVariant::SmallDecomposed,
        ] {
            let expected = match variant {
                StableHashVariant::Plain => ObservedHashKind::Plain,
                StableHashVariant::Decomposed => ObservedHashKind::Decomposed,
                StableHashVariant::SmallDecomposed => ObservedHashKind::SmallDecomposed,
            };
            assert_eq!(
                observed_sampler_kind(&CanonicalEventKind::Sampler {
                    operation: StableSamplerOperation::Hash {
                        output: super::super::g0::StableValueType::Matrix {
                            modulus: "17".to_owned(),
                            ring_dimension: 1,
                            rows: 1,
                            columns: 1,
                        },
                        variant,
                        tag_prefix: vec![],
                        tag_expressions: vec![],
                        tag_decimal_expressions: vec![],
                        tag_u64_le_expressions: vec![],
                        base: None,
                        digit_count: None,
                    },
                }),
                ObservedSamplerKind::Hash(expected),
            );
        }

        for (authority, expected) in [
            (ProofPayloadAuthority::FactStore, ObservedAuthorityKind::FactStore),
            (ProofPayloadAuthority::ProgramFamilyFact, ObservedAuthorityKind::ProgramFamilyFact),
            (ProofPayloadAuthority::Operator, ObservedAuthorityKind::Operator),
            (
                ProofPayloadAuthority::RelationPreimageSource { source: 0 },
                ObservedAuthorityKind::RelationPreimageSource,
            ),
            (ProofPayloadAuthority::Unavailable, ObservedAuthorityKind::Unavailable),
        ] {
            assert_eq!(
                observed_bound_kind(&ProofPayloadRule::Authority(authority)),
                ObservedBoundKind::Authority(expected),
            );
        }

        let source = super::super::arena::SemanticSourceIdentity {
            stable_definition: "source".to_owned(),
            invocation: "invocation".to_owned(),
            sample_event: None,
            output_role: "output".to_owned(),
            sampler: None,
            artifact: None,
            value_type: ResolvedValueType::Int,
            coordinates: Box::new([]),
            matrix_constant: None,
        };
        let family_source = super::super::arena::SemanticFamilySourceIdentity {
            stable_definition: "family".to_owned(),
            invocation: "invocation".to_owned(),
            element_type: ResolvedValueType::Int,
            domain: FamilyDomain::new(0, 1).unwrap(),
            artifact: None,
        };
        let operator_cases = vec![
            (
                ValueOperator::Argument { position: 0, value_type: ResolvedValueType::Int },
                ObservedOperatorKind::Argument,
            ),
            (
                ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                ObservedOperatorKind::Constant,
            ),
            (ValueOperator::Source(source), ObservedOperatorKind::Source),
            (
                ValueOperator::DeterministicHash(
                    super::super::arena::DeterministicHashDescriptor {
                        definition:
                            super::super::arena::DeterministicHashDefinition::MxxPolynomialHash,
                        version: 1,
                        key_byte_length: 32,
                        output: ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap(),
                        tag_prefix: Box::new([]),
                        binary_tag_count: 0,
                        decimal_tag_count: 0,
                        u64_le_tag_count: 0,
                        dynamic_tag_count: 0,
                    },
                ),
                ObservedOperatorKind::DeterministicHash,
            ),
            (
                ValueOperator::OpaqueFamilyElement { source: family_source },
                ObservedOperatorKind::OpaqueFamilyElement,
            ),
            (
                ValueOperator::IndexMap {
                    definition: super::super::arena::IndexFunctionDefinitionId(3),
                    parameters: Box::new([5]),
                },
                ObservedOperatorKind::IndexMap,
            ),
            (
                ValueOperator::ExplicitElement {
                    domain: FamilyDomain::new(0, 1).unwrap(),
                    element_type: ResolvedValueType::Int,
                },
                ObservedOperatorKind::ExplicitElement,
            ),
            (
                ValueOperator::ProgramCall {
                    program: ValueProgramId::new(super::super::arena::ArenaToken::fresh(), 0),
                },
                ObservedOperatorKind::ProgramCall,
            ),
            (
                ValueOperator::ExtractCoefficient {
                    position: 0,
                    canonical_input_exclusive_upper: Some(17_u8.into()),
                },
                ObservedOperatorKind::ExtractCoefficient,
            ),
            (
                ValueOperator::Scalar(ScalarOperation::Bit { position: 1 }),
                ObservedOperatorKind::Scalar(ObservedScalarKind::Bit),
            ),
            (
                ValueOperator::Matrix(MatrixOperation::Scale),
                ObservedOperatorKind::Matrix(ObservedMatrixKind::Scale),
            ),
            (
                ValueOperator::Trapdoor(TrapdoorOperation::Generate {
                    descriptor: "generate".to_owned(),
                    parameters: Box::new([]),
                    paired_public_event: super::super::arena::SampleEventId(1),
                    paired_public_output_role: "public".to_owned(),
                }),
                ObservedOperatorKind::Trapdoor(ObservedTrapdoorKind::Generate),
            ),
            (
                ValueOperator::Trapdoor(TrapdoorOperation::Transform {
                    descriptor: "transform".to_owned(),
                    output: ResolvedValueType::Trapdoor,
                    parameters: Box::new([]),
                }),
                ObservedOperatorKind::Trapdoor(ObservedTrapdoorKind::Transform),
            ),
        ];
        for (operator, expected) in operator_cases {
            assert_eq!(observed_operator_kind(&operator), Some(expected));
        }

        let mut repeated_parameter_sites = BTreeMap::new();
        for (row, position) in [(4, 1), (9, 7)] {
            let kind =
                observed_operator_kind(&ValueOperator::Scalar(ScalarOperation::Bit { position }))
                    .unwrap();
            repeated_parameter_sites
                .entry(ObservedCoverageKind::Operator(kind))
                .or_insert_with(BTreeSet::new)
                .insert(ObservedCoverageSite::ExpressionRow { row });
        }
        let coverage = finalize_observed_coverage(repeated_parameter_sites).unwrap();
        assert_eq!(coverage.rows.len(), 1);
        assert_eq!(coverage.rows[0].count, 2);
        assert_eq!(
            coverage.rows[0].sites,
            vec![
                ObservedCoverageSite::ExpressionRow { row: 4 },
                ObservedCoverageSite::ExpressionRow { row: 9 },
            ]
        );
        // Scalar 1; Operator tag + Scalar 2; domain tag + Operator 3; each site 2;
        // sites Vec 1 + len 2 + sites 4 = 7; row 3 + count 1 + sites 7 = 11;
        // rows Vec 1 + len 1 + row 11 = 13.
        assert_eq!(coverage.logical_items(), Ok(13));
    }

    #[test]
    fn residual_coverage_classification_splits_variants_and_rejects_unsupported_rows() {
        for operation in [ObservedScalarKind::Add, ObservedScalarKind::Subtract] {
            let classification = residual_coverage_classification(ObservedCoverageKind::Operator(
                ObservedOperatorKind::Scalar(operation),
            ));
            assert!(matches!(
                classification.disposition,
                ResidualCoverageDisposition::CheckedLean { .. }
            ));
        }
        let multiply = residual_coverage_classification(ObservedCoverageKind::Operator(
            ObservedOperatorKind::Scalar(ObservedScalarKind::Multiply),
        ));
        assert!(matches!(
            multiply.disposition,
            ResidualCoverageDisposition::G2LeanObligation { .. }
        ));
        let regular = residual_coverage_classification(ObservedCoverageKind::Transform(
            ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Regular),
        ));
        let small = residual_coverage_classification(ObservedCoverageKind::Transform(
            ObservedTransformKind::GadgetDecompose(ObservedGadgetKind::Small),
        ));
        assert_ne!(regular.disposition, small.disposition);
        let hash_classifications = [
            ObservedHashKind::Plain,
            ObservedHashKind::Decomposed,
            ObservedHashKind::SmallDecomposed,
        ]
        .map(|hash| {
            residual_coverage_classification(ObservedCoverageKind::Sampler(
                ObservedSamplerKind::Hash(hash),
            ))
        });
        for classification in hash_classifications {
            assert!(matches!(
                classification.disposition,
                ResidualCoverageDisposition::G2LeanObligation { .. }
            ));
        }
        assert!(hash_classifications.windows(2).all(|pair| pair[0] != pair[1]));
        for rejected in [
            ObservedCoverageKind::Operator(ObservedOperatorKind::Scalar(
                ObservedScalarKind::ThresholdDecode,
            )),
            ObservedCoverageKind::Bound(ObservedBoundKind::Authority(
                ObservedAuthorityKind::Unavailable,
            )),
        ] {
            let coverage = ObservedCoverage {
                rows: vec![ObservedCoverageRow {
                    kind: rejected,
                    count: 1,
                    sites: vec![ObservedCoverageSite::ExpressionRow { row: 0 }],
                }],
            };
            assert!(derive_residual_coverage_matrix(&coverage).is_err());
        }
    }

    #[test]
    fn raw_event_coverage_classifier_is_exhaustive_and_rejects_trapdoors() {
        let matrix = ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap();
        let cases = [
            EventKind::Sample {
                descriptor: super::super::arena::SampleDescriptor::new(
                    "sample",
                    ResolvedValueType::Int,
                ),
            },
            EventKind::Sampler {
                operation: super::super::arena::SamplerOperation::UniformResidue { output: matrix },
            },
            EventKind::Trapdoor {
                operation: TrapdoorOperation::Generate {
                    descriptor: "trapdoor".to_owned(),
                    parameters: Box::new([]),
                    paired_public_event: super::super::arena::SampleEventId(0),
                    paired_public_output_role: "public".to_owned(),
                },
            },
        ];
        assert!(matches!(
            raw_event_coverage_disposition(&cases[0]),
            RawEventCoverageDisposition::CanonicalProjection
        ));
        assert!(matches!(
            raw_event_coverage_disposition(&cases[1]),
            RawEventCoverageDisposition::CanonicalProjection
        ));
        assert!(matches!(
            raw_event_coverage_disposition(&cases[2]),
            RawEventCoverageDisposition::RejectBeforeCanonicalProjection { .. }
        ));
    }

    fn payload_rule_mentions_transfer(rule: &ProofPayloadRule, event: usize) -> bool {
        let value = |value: &ProofPayloadValueRef| matches!(value, ProofPayloadValueRef::Transfer(candidate) if *candidate as usize == event);
        match rule {
            ProofPayloadRule::Authority(_) => false,
            ProofPayloadRule::Identity { input } |
            ProofPayloadRule::RingAutomorphism { input, .. } => value(input),
            ProofPayloadRule::Sum { inputs } |
            ProofPayloadRule::Maximum { inputs } |
            ProofPayloadRule::WeightedSum { inputs } => inputs.iter().any(value),
            ProofPayloadRule::Scale { value: input, scale } => {
                value(input) || matches!(scale, ProofPayloadScale::Value(input) if value(input))
            }
            ProofPayloadRule::MonomialProduct { factors, .. } => {
                factors.iter().any(|factor| value(&factor.bound))
            }
            ProofPayloadRule::Product { left, right, .. } |
            ProofPayloadRule::Tensor { left, right, .. } => value(left) || value(right),
        }
    }

    fn assert_payload_event_refs_are_local(payload: &OperationalProofPayload) {
        fn value_ref_before(reference: &ProofPayloadValueRef, before: &impl Fn(u64)) {
            match reference {
                ProofPayloadValueRef::Predecessor { binding_event, .. } => before(*binding_event),
                ProofPayloadValueRef::Result { event, .. } |
                ProofPayloadValueRef::Transfer(event) => before(*event),
            }
        }

        fn rule_refs_before(rule: &ProofPayloadRule, before: &impl Fn(u64)) {
            let value = |reference: &ProofPayloadValueRef| value_ref_before(reference, before);
            match rule {
                ProofPayloadRule::Authority(_) => {}
                ProofPayloadRule::Identity { input } |
                ProofPayloadRule::RingAutomorphism { input, .. } => value(input),
                ProofPayloadRule::Sum { inputs } |
                ProofPayloadRule::Maximum { inputs } |
                ProofPayloadRule::WeightedSum { inputs } => inputs.iter().for_each(value),
                ProofPayloadRule::Scale { value: input, scale } => {
                    value(input);
                    if let ProofPayloadScale::Value(reference) = scale {
                        value(reference);
                    }
                }
                ProofPayloadRule::MonomialProduct { factors, .. } => {
                    factors.iter().for_each(|factor| value(&factor.bound));
                }
                ProofPayloadRule::Product { left, right, .. } |
                ProofPayloadRule::Tensor { left, right, .. } => {
                    value(left);
                    value(right);
                }
            }
        }

        for (index, event) in payload.events.iter().enumerate() {
            let before = |reference: u64| assert!((reference as usize) < index);
            match event {
                ProofPayloadEvent::Predecessor { source_result, .. } => before(*source_result),
                ProofPayloadEvent::SpecializationComputed { source, .. } |
                ProofPayloadEvent::SpecializationCacheHit { source, .. } => {
                    assert!((source.end as usize) <= index);
                    assert!(source.start <= source.end);
                }
                ProofPayloadEvent::AppliedRelation { rule, .. } => match rule {
                    ProofPayloadRelationRule::Universal { computed, rhs_result, .. } => {
                        before(*computed);
                        before(*rhs_result);
                    }
                    ProofPayloadRelationRule::Gadget { input_result, .. } => before(*input_result),
                },
                ProofPayloadEvent::CoefficientMerge(observation) => match &observation.source {
                    ProofPayloadCoefficientMergeSource::Operator { inputs } => {
                        for input in inputs {
                            before(input.value_event);
                        }
                    }
                    ProofPayloadCoefficientMergeSource::Relation { application, .. } => {
                        before(*application)
                    }
                },
                ProofPayloadEvent::PreFoldPolynomial(snapshot) => {
                    before(snapshot.result_event);
                    if let Some(reference) = &snapshot.summary_evidence {
                        value_ref_before(reference, &before);
                    }
                }
                ProofPayloadEvent::BoundTransfer { rule, .. } => rule_refs_before(rule, &before),
                ProofPayloadEvent::SurvivorFold(observation) => before(observation.bound),
                ProofPayloadEvent::InvocationEnd { pre_fold_event, .. } => before(*pre_fold_event),
                _ => {}
            }
        }
    }

    fn payload_frame_data(
        payload: &OperationalProofPayload,
    ) -> (Vec<(usize, usize, ProofPayloadOwner)>, Vec<Option<usize>>) {
        let mut stack = Vec::new();
        let mut outer = Vec::new();
        let mut immediate = vec![None; payload.events.len()];
        for (index, event) in payload.events.iter().enumerate() {
            if let ProofPayloadEvent::InvocationStart { root } = event {
                immediate[index] = stack.last().map(|(start, _)| *start);
                stack.push((index, *root));
            } else {
                immediate[index] = stack.last().map(|(start, _)| *start);
            }
            if let ProofPayloadEvent::InvocationEnd { root, .. } = event {
                let (start, started_root) = stack.pop().expect("balanced invocation frames");
                assert_eq!(&started_root, root);
                if stack.is_empty() {
                    outer.push((start, index, *root));
                }
            }
        }
        assert!(stack.is_empty(), "payload invocation frames must be balanced");
        (outer, immediate)
    }

    fn manual_projection_support_items(
        run: &OperationalCertificateRun,
    ) -> (u64, usize, usize, u64, u64) {
        let closure = &run.projection.closure;
        let refs = super::super::g0::derive_certificate_statement_rows(
            &run.job,
            closure,
            &run.trace,
            closed_residual_expression(run),
        )
        .expect("canonical refs for manual pipeline count");
        let canonical_refs_items =
            refs.retained_logical_items().expect("canonical refs retained count");
        let shallow_refs_items = (1 + 3 * (closure.expressions.len() + closure.programs.len()))
            as u64 +
            (1 + 3 * closure.event_ids.len()) as u64;
        let mut arena_tokens = BTreeSet::new();
        for program in &closure.programs {
            if let Some(arena) = run.job.monomials().get(*program) {
                arena_tokens.insert(arena.token());
            }
        }
        for event in &run.trace.events {
            let program = match event {
                NormalizerEvent::InvocationStart { root } |
                NormalizerEvent::Result { owner: root, .. } |
                NormalizerEvent::InvocationEnd { root, .. } |
                NormalizerEvent::SpecializationComputed { owner: root, .. } |
                NormalizerEvent::SpecializationCacheHit { owner: root, .. } |
                NormalizerEvent::BoundTransfer { owner: root, .. } => Some(root.program()),
                NormalizerEvent::CoefficientMerge(observation) => Some(observation.owner.program()),
                NormalizerEvent::Predecessor { consumer, .. } => Some(consumer.program()),
                NormalizerEvent::AppliedRelation(observation) => Some(observation.owner.program()),
                NormalizerEvent::SurvivorFold(observation) => {
                    match run.trace.events.get(observation.bound.0 as usize) {
                        Some(NormalizerEvent::BoundTransfer { owner, .. }) => Some(owner.program()),
                        _ => None,
                    }
                }
                NormalizerEvent::PreFoldPolynomial(_) => None,
            };
            if let Some(arena) = program.and_then(|program| run.job.monomials().get(program)) {
                arena_tokens.insert(arena.token());
            }
        }
        let rhs_lookups = run
            .trace
            .events
            .iter()
            .filter_map(|event| match event {
                NormalizerEvent::SpecializationComputed { replay, .. } => {
                    Some(replay.rhs_results.len())
                }
                _ => None,
            })
            .sum::<usize>();
        let closed_root = match &run.projection.residual {
            CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
            CertificateResidualRoot::Family { .. } => None,
        };
        let closed_program = closed_root.is_some_and(|root| {
            run.trace.events.iter().any(|event| {
                matches!(event, NormalizerEvent::InvocationStart { root: owner } if owner.expression() == root)
            })
        });
        let arenas = arena_tokens.len() as u64;
        let rhs = rhs_lookups as u64;
        let support = canonical_refs_items +
            (1 + 3 * arenas) +
            (1 + 6 * rhs) +
            (1 + u64::from(closed_program)) +
            (1 + u64::from(closed_root.is_some()));
        (support, arena_tokens.len(), rhs_lookups, canonical_refs_items, shallow_refs_items)
    }

    fn repeat_certificate_normalization(run: &mut OperationalCertificateRun) {
        let plaintext_modulus = match &run.accepted_report.acceptance {
            super::super::OperationalAcceptanceReport::Threshold { plaintext_modulus, .. } => {
                plaintext_modulus.clone()
            }
            super::super::OperationalAcceptanceReport::BooleanInterval { .. } => {
                panic!("repeated fixture requires threshold acceptance")
            }
        };
        analyze_roots_with_sink(
            &mut run.job,
            &run.roots,
            &ReportTarget {
                target_id: run.accepted_report.target_id.clone(),
                plaintext_modulus,
                ciphertext_modulus: run.accepted_report.ciphertext_modulus.clone(),
                boolean_interval: false,
            },
            &mut run.trace,
        )
        .expect("repeat accepted trace normalization");
        let closed_root_expression = match &run.projection.residual {
            CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
            CertificateResidualRoot::Family { .. } => None,
        };
        extend_certificate_closure(&run.job, &mut run.projection.closure, &run.trace)
            .expect("repeat trace closure");
        if let Some(wrapper) = closed_wrapper_program(&run.trace, closed_root_expression) {
            run.projection.closure.programs.remove(&wrapper);
        }
        run.trace.retain_residual(&run.projection.closure);
    }

    fn threshold_certificate_protocol() -> ProtocolDecl {
        let stage_id = StageId("certificate-stage".to_owned());
        let ring = Ring::new(256, 1);
        let left_factor = ring.gaussian((1, 1), 1, 2);
        let right_factor = ring.gaussian((1, 1), 1, 3);
        let left_scalar = left_factor.clone() + left_factor;
        let right_scalar = right_factor.clone() + right_factor.clone() + right_factor;
        let matrix = ring.uniform_residue((2, 2));
        let left_outer = left_scalar * matrix.clone();
        let right_outer = right_scalar * matrix;
        let product = left_outer * right_outer;
        let finite = ring.constant(
            (2, 2),
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(3)] },
        );
        let residual = (product.clone() - product + finite)
            .semantic_anchor("certificate.residual")
            .expect("residual anchor");
        let decoded = residual
            .clone()
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("decoded output")
            .semantic_anchor("certificate.decoded")
            .expect("decoded anchor");
        let stage = DslContext::new("certificate-stage")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoded output")
            .build()
            .expect("certificate graph");
        let decoder_node = stage.graph.outputs()["decoded"].value.node;
        let ideal = IdealSpec::new(
            DslContext::new("certificate-ideal")
                .bool_output("result", Bool::constant(false))
                .expect("ideal output")
                .build()
                .expect("ideal graph"),
        )
        .expect("pure ideal");
        let endpoint = EndpointSpecId::ThresholdDecode;
        ProtocolDecl::new(crate::ProtocolDecl {
            params: Vec::new(),
            bundle: ClosedProtocolBundle {
                workflow: Workflow {
                    stages: vec![ProtocolStage {
                        id: stage_id.clone(),
                        graph: stage.graph,
                        semantic_anchors: stage.anchors,
                        derivation_attachments: stage.derivation_attachments,
                        bindings: Vec::new(),
                    }],
                    entrypoint: stage_id.clone(),
                },
                ideal,
                requirements: Vec::new(),
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "result".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: stage_id.clone(),
                        semantic_anchor: "certificate.decoded".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: stage_id.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "result".to_owned(),
                    }],
                },
                operational_decoder_targets: vec![OperationalDecoderTarget {
                    target_id: "certificate-threshold".to_owned(),
                    residual_stage: stage_id.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage_id,
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }],
                endpoint_specs: vec![endpoint],
                input_contract: InputContract::default(),
                input_bindings: Vec::new(),
                precondition_spec: crate::ProtocolPreconditionSpec::default(),
            },
        })
        .expect("threshold certificate protocol")
    }

    fn indexed_threshold_certificate_protocol() -> ProtocolDecl {
        let stage_id = StageId("indexed-certificate-stage".to_owned());
        let ring = Ring::new(256, 1);
        let source_value = ring.constant(
            (1, 1),
            ConstantMatrix::Polynomial { coefficients: vec![IntExpr::constant(3)] },
        );
        let source =
            Parallel::range(7).map_values(|_| source_value.clone()).expect("indexed source family");
        let indices = Parallel::range(4)
            .map_values(|index| index.as_int().mul(Int::constant(2)))
            .expect("indexed selector family");
        let gathered = source.parallel_gather(indices).expect("indexed gathered family");
        let residual = gathered
            .get_static(0)
            .semantic_anchor("indexed.certificate.residual")
            .expect("indexed residual anchor");
        let decoded = residual
            .clone()
            .threshold_decode_bools(IntExpr::constant(2), 1)
            .into_iter()
            .next()
            .expect("indexed decoded output")
            .semantic_anchor("indexed.certificate.decoded")
            .expect("indexed decoded anchor");
        let stage = DslContext::new("indexed-certificate-stage")
            .private_output("operational-residual", residual)
            .expect("indexed residual output")
            .bool_output("decoded", decoded)
            .expect("indexed decoded output")
            .build()
            .expect("indexed certificate graph");
        let decoder_node = stage.graph.outputs()["decoded"].value.node;
        let ideal = IdealSpec::new(
            DslContext::new("indexed-certificate-ideal")
                .bool_output("result", Bool::constant(false))
                .expect("ideal output")
                .build()
                .expect("ideal graph"),
        )
        .expect("pure ideal");
        let endpoint = EndpointSpecId::ThresholdDecode;
        ProtocolDecl::new(crate::ProtocolDecl {
            params: Vec::new(),
            bundle: ClosedProtocolBundle {
                workflow: Workflow {
                    stages: vec![ProtocolStage {
                        id: stage_id.clone(),
                        graph: stage.graph,
                        semantic_anchors: stage.anchors,
                        derivation_attachments: stage.derivation_attachments,
                        bindings: Vec::new(),
                    }],
                    entrypoint: stage_id.clone(),
                },
                ideal,
                requirements: Vec::new(),
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "result".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: stage_id.clone(),
                        semantic_anchor: "indexed.certificate.decoded".to_owned(),
                        semantics: EndpointSemanticBinding::ThresholdDecode,
                        workflow_output: OutputRef {
                            stage: stage_id.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "result".to_owned(),
                    }],
                },
                operational_decoder_targets: vec![OperationalDecoderTarget {
                    target_id: "indexed-certificate-threshold".to_owned(),
                    residual_stage: stage_id.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage_id,
                    decoder_node,
                    kind: OperationalDecoderKind::ThresholdDecode {
                        plaintext_modulus: IntExpr::constant(2),
                    },
                }],
                endpoint_specs: vec![endpoint],
                input_contract: InputContract::default(),
                input_bindings: Vec::new(),
                precondition_spec: crate::ProtocolPreconditionSpec::default(),
            },
        })
        .expect("indexed threshold certificate protocol")
    }

    fn boolean_interval_certificate_protocol() -> ProtocolDecl {
        let stage_id = StageId("boolean-certificate-stage".to_owned());
        let ring = Ring::new(17, 1);
        let residual = ring
            .zero((1, 1))
            .semantic_anchor("boolean.residual")
            .expect("residual anchor")
            .semantic_anchor("boolean.carrier")
            .expect("carrier anchor");
        let coefficient = residual.clone().extract_coefficient(0);
        let quarter = Int::evaluate(IntExpr::RoundDiv(
            Box::new(IntExpr::Sub(Box::new(IntExpr::constant(17)), Box::new(IntExpr::constant(2)))),
            Box::new(IntExpr::constant(4)),
        ));
        let decoded = quarter
            .clone()
            .less_equal(coefficient.clone())
            .to_int()
            .add(coefficient.less_equal(quarter.mul(Int::constant(3))).to_int())
            .equal(Int::constant(2))
            .semantic_anchor("boolean.decoded")
            .expect("decoded anchor");
        let stage = DslContext::new("boolean-certificate-stage")
            .private_output("operational-residual", residual)
            .expect("residual output")
            .bool_output("decoded", decoded)
            .expect("decoded output")
            .build()
            .expect("boolean certificate graph");
        let decoder_node = stage.graph.outputs()["decoded"].value.node;
        let message = crate::ProtocolInputId::from("message");
        let ideal = IdealSpec::new(
            DslContext::new("boolean-certificate-ideal")
                .bool_output("result", ring.bool_input("message"))
                .expect("ideal output")
                .build()
                .expect("ideal graph"),
        )
        .expect("pure ideal");
        let endpoint = EndpointSpecId::BooleanInterval;
        ProtocolDecl::new(crate::ProtocolDecl {
            params: Vec::new(),
            bundle: ClosedProtocolBundle {
                workflow: Workflow {
                    stages: vec![ProtocolStage {
                        id: stage_id.clone(),
                        graph: stage.graph,
                        semantic_anchors: stage.anchors,
                        derivation_attachments: stage.derivation_attachments,
                        bindings: Vec::new(),
                    }],
                    entrypoint: stage_id.clone(),
                },
                ideal,
                requirements: Vec::new(),
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: "decoded".to_owned(),
                        ideal_input: "result".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointAnchors {
                    entries: vec![EndpointAnchor {
                        spec: endpoint,
                        stage: stage_id.clone(),
                        semantic_anchor: "boolean.decoded".to_owned(),
                        semantics: EndpointSemanticBinding::BooleanInterval {
                            residual_stage: stage_id.clone(),
                            residual_anchor: "boolean.residual".to_owned(),
                            carrier_stage: stage_id.clone(),
                            carrier_anchor: "boolean.carrier".to_owned(),
                            message: message.clone(),
                        },
                        workflow_output: OutputRef {
                            stage: stage_id.clone(),
                            output: "decoded".to_owned(),
                        },
                        ideal_output: "result".to_owned(),
                    }],
                },
                operational_decoder_targets: vec![OperationalDecoderTarget {
                    target_id: "certificate-boolean".to_owned(),
                    residual_stage: stage_id.clone(),
                    residual_output: "operational-residual".to_owned(),
                    decoder_stage: stage_id,
                    decoder_node,
                    kind: OperationalDecoderKind::BooleanInterval,
                }],
                endpoint_specs: vec![endpoint],
                input_contract: InputContract {
                    inputs: vec![InputContractEntry {
                        id: message.clone(),
                        name: "message".to_owned(),
                        value: InputValueContract::Boolean,
                    }],
                },
                input_bindings: vec![ProtocolInputBinding {
                    input: message,
                    destinations: vec![ProtocolInputDestination::Ideal {
                        input: "message".to_owned(),
                    }],
                }],
                precondition_spec: crate::ProtocolPreconditionSpec::default(),
            },
        })
        .expect("boolean certificate protocol")
    }

    #[test]
    fn closed_target_rejects_a_same_modulus_decoder_that_does_not_consume_the_residual() {
        let protocol = crate::protocol_example::protocol();
        let request = super::super::OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(1.into()),
            )],
            layouts: Vec::new(),
            target_id: "example-threshold".to_owned(),
        };
        let mut emit = |_| {};
        let mut control = SimulationControl::new(&mut emit);
        let error = resolve_target(&protocol, &request, &mut control)
            .expect_err("the decoder input must be derived from the declared residual");
        assert!(matches!(
            error,
            OperationalSimulationError::Target(TargetError::DecoderInputDoesNotConsumeResidual {
                target_id,
                ..
            }) if target_id == "example-threshold"
        ));
    }

    #[test]
    fn owned_element_reservations_are_observed_without_rejecting() {
        let mut emit = |_| {};
        let mut control = SimulationControl::new(&mut emit);
        assert!(control.reserve_owned_elements(usize::MAX).is_ok());
        assert!(control.reserve_owned_elements(3).is_ok());
        assert_eq!(control.owned_elements.load(Ordering::Relaxed), usize::MAX);
    }

    #[test]
    fn certificate_projection_uses_threshold_target_and_residual_root() {
        let protocol = threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("valid threshold certificate run");
        assert!(run.accepted_report.accepted);
        assert!(run.trace.lowering_complete > 0);
        let starts = run
            .trace
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    crate::operational_noise::g0::NormalizerEvent::InvocationStart { .. }
                )
            })
            .count();
        let ends = run
            .trace
            .events
            .iter()
            .filter(|event| {
                matches!(event, crate::operational_noise::g0::NormalizerEvent::InvocationEnd { .. })
            })
            .count();
        let results = run
            .trace
            .events
            .iter()
            .filter(|event| {
                matches!(event, crate::operational_noise::g0::NormalizerEvent::Result { .. })
            })
            .count();
        assert_eq!(starts, 1);
        assert_eq!(ends, 1);
        assert!(results > 0);
        assert_eq!(run.accepted_report.target_id, request.target_id);
        assert_eq!(run.accepted_report.ciphertext_modulus, 256_u16.into());
        assert!(matches!(
            run.accepted_report.acceptance,
            super::super::OperationalAcceptanceReport::Threshold { plaintext_modulus, .. }
                if plaintext_modulus == 2_u8.into()
        ));
        let projection = run.projection;
        assert_eq!(projection.target_id, "certificate-threshold");
        assert_eq!(projection.plaintext_modulus, 2_u8.into());
        assert_eq!(projection.ciphertext_modulus, 256_u16.into());
        let CertificateResidualRoot::Closed { matrix, .. } = projection.residual else {
            panic!("certificate residual should be the closed production residual root")
        };
        assert_eq!(matrix.modulus, 256_u16.into());
        assert_eq!(matrix.ring_dimension, 1);
        assert!(!projection.closure.expressions.is_empty());
        // The synthetic zero-argument wrapper is a Closed scope, never a user-visible Program
        // row.  There are no user programs in this closed residual fixture.
        assert!(projection.closure.programs.is_empty());
        assert!(projection.closure.families.is_empty());
    }

    #[test]
    fn certificate_report_matches_ordinary_analysis() {
        let protocol = threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("valid threshold certificate run");
        let plan = ProtocolPlan::build(&protocol, &request.target_id).expect("protocol plan");
        let (mut ordinary_job, ordinary_roots) =
            ProductionAdapter::new(&protocol, &plan, BTreeMap::new())
                .expect("ordinary production adapter")
                .lower()
                .expect("ordinary production lowering");
        let plaintext_modulus = match &run.accepted_report.acceptance {
            super::super::OperationalAcceptanceReport::Threshold { plaintext_modulus, .. } => {
                plaintext_modulus.clone()
            }
            super::super::OperationalAcceptanceReport::BooleanInterval { .. } => {
                panic!("threshold fixture must use threshold acceptance")
            }
        };
        let ordinary_report = analyze_roots(
            &mut ordinary_job,
            &ordinary_roots,
            &ReportTarget {
                target_id: request.target_id,
                plaintext_modulus,
                ciphertext_modulus: run.accepted_report.ciphertext_modulus.clone(),
                boolean_interval: false,
            },
        )
        .expect("ordinary report");
        assert_eq!(ordinary_report, run.accepted_report);
    }

    #[test]
    fn proof_payload_projects_the_accepted_trace_without_arena_handles() {
        let protocol = threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("valid threshold certificate run");
        let validation_program = run
            .trace
            .events
            .iter()
            .find_map(|event| match event {
                NormalizerEvent::InvocationStart { root } => Some(root.program()),
                _ => None,
            })
            .expect("threshold invocation program");
        let monomials =
            run.job.monomials().get(validation_program).expect("threshold monomial arena");
        run.trace
            .validate_normalization_observations_with_monomials(monomials)
            .expect("threshold product trace validates");
        run.trace
            .validate_normalization_observations_with_state(
                monomials,
                &super::super::relation::NormalizationCache::new(),
            )
            .expect("threshold product state validates");
        let payload = derive_proof_payload(&run).expect("canonical proof payload");
        assert_payload_event_refs_are_local(&payload);
        let survivor_folds = payload
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ProofPayloadEvent::SurvivorFold(observation) => Some((index, observation)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!survivor_folds.is_empty(), "accepted payload must contain a survivor fold");
        assert!(survivor_folds.iter().all(|(index, fold)| fold.bound < *index as u64));
        for (_, fold) in &survivor_folds {
            let Some(ProofPayloadEvent::BoundTransfer { owner, rule }) =
                payload.events.get(fold.bound as usize)
            else {
                panic!("survivor fold must point to a bound transfer")
            };
            match rule {
                ProofPayloadRule::MonomialProduct { .. } => {
                    assert_eq!(fold.coefficient.magnitude(), &BigUint::from(1_u8));
                }
                ProofPayloadRule::Scale {
                    value: ProofPayloadValueRef::Transfer(previous),
                    scale: ProofPayloadScale::Magnitude(magnitude),
                } => {
                    assert_eq!(fold.coefficient.magnitude(), magnitude);
                    let Some(ProofPayloadEvent::BoundTransfer {
                        owner: previous_owner,
                        rule: ProofPayloadRule::MonomialProduct { .. },
                    }) = payload.events.get(*previous as usize)
                    else {
                        panic!("scale must consume its monomial product")
                    };
                    assert_eq!(owner, previous_owner);
                }
                _ => panic!("survivor fold must resolve to monomial product or scale"),
            }
        }
        let second = prepare_operational_certificate(&protocol, &request)
            .expect("equivalent threshold certificate run");
        let second_payload = derive_proof_payload(&second).expect("canonical second payload");
        assert_eq!(payload, second_payload);
        assert_eq!(payload.events.len(), run.trace.events.len());
        let snapshots = payload
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ProofPayloadEvent::PreFoldPolynomial(snapshot) => Some((index, snapshot)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!snapshots.is_empty(), "accepted payload must retain pre-fold snapshots");
        for (index, snapshot) in &snapshots {
            if let Some(evidence) = &snapshot.summary_evidence {
                match evidence {
                    ProofPayloadValueRef::Result { event, .. } |
                    ProofPayloadValueRef::Transfer(event) => {
                        assert!((*event as usize) < *index)
                    }
                    ProofPayloadValueRef::Predecessor { .. } => {}
                }
            }
        }
        assert!(snapshots.iter().any(|(index, snapshot)| {
            matches!(
                snapshot.summary.coefficient_bound(),
                NumericContract::Known(CoefficientBound::Finite(_))
            ) && matches!(
                snapshot.summary_evidence,
                Some(ProofPayloadValueRef::Result {
                    event,
                    projection: BoundProjection::Summary,
                }) if (event as usize) < *index &&
                    matches!(payload.events.get(event as usize), Some(ProofPayloadEvent::Result { .. }))
            )
        }));
        assert!(
            payload
                .events
                .iter()
                .any(|event| { matches!(event, ProofPayloadEvent::InvocationEnd { .. }) })
        );
        assert!(
            payload
                .events
                .iter()
                .any(|event| { matches!(event, ProofPayloadEvent::Result { .. }) })
        );
        let merges = payload
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ProofPayloadEvent::CoefficientMerge(observation) => Some((index, observation)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let product_merge = merges
            .iter()
            .find(|(_, merge)| merge.signed_contribution == BigInt::from(6_u8))
            .expect("nonunit production product merge");
        let ProofPayloadCoefficientMergeSource::Operator { inputs } = &product_merge.1.source
        else {
            panic!("production product merge must use operator inputs")
        };
        assert_eq!(inputs.iter().map(|source| source.term_ordinal).collect::<Vec<_>>(), vec![0, 0]);
        assert_eq!(
            product_merge
                .1
                .output
                .central_factors
                .iter()
                .map(|owner| owner.expression_row)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(
            product_merge
                .1
                .output
                .ordered_factors
                .iter()
                .map(|owner| owner.expression_row)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        for (merge_index, merge) in &merges {
            let ProofPayloadCoefficientMergeSource::Operator { inputs } = &merge.source else {
                continue;
            };
            assert_eq!(inputs.len(), 2);
            assert_eq!(inputs[0].term_ordinal, inputs[1].term_ordinal);
            for (input_position, source) in inputs.iter().enumerate() {
                let predecessor_result = payload
                    .events
                    .iter()
                    .find_map(|event| match event {
                        ProofPayloadEvent::Predecessor {
                            consumer,
                            input_position: position,
                            source_result,
                            ..
                        } if consumer == &merge.owner && *position == input_position as u32 => {
                            Some(*source_result)
                        }
                        _ => None,
                    })
                    .expect("merge source predecessor");
                assert_eq!(source.value_event, predecessor_result);
            }
            let owner_result = payload
                .events
                .iter()
                .enumerate()
                .find(|(_, event)| {
                    matches!(event, ProofPayloadEvent::Result { owner, .. } if owner == &merge.owner)
                })
                .map(|(index, _)| index)
                .expect("merge owner result");
            assert!(*merge_index < owner_result, "merge is recorded before its owner Result");
        }
        let final_result = payload
            .events
            .iter()
            .rev()
            .find_map(|event| match event {
                ProofPayloadEvent::InvocationEnd { result, .. } => Some(result),
                _ => None,
            })
            .expect("final invocation result");
        let ProofPayloadValue::Exact { terms, .. } = final_result else {
            panic!("threshold residual should retain exact polynomial output")
        };
        assert!(terms.is_empty(), "the subtract merge cancels the final monomial");
        let owner_scope = |event: &ProofPayloadEvent| match event {
            ProofPayloadEvent::InvocationStart { root } |
            ProofPayloadEvent::InvocationEnd { root, .. } => Some(root.scope),
            ProofPayloadEvent::Result { owner, .. } |
            ProofPayloadEvent::SpecializationComputed { owner, .. } |
            ProofPayloadEvent::SpecializationCacheHit { owner, .. } |
            ProofPayloadEvent::AppliedRelation { owner, .. } |
            ProofPayloadEvent::BoundTransfer { owner, .. } => Some(owner.scope),
            ProofPayloadEvent::CoefficientMerge(observation) => Some(observation.owner.scope),
            ProofPayloadEvent::SurvivorFold(_) => None,
            ProofPayloadEvent::Predecessor { consumer, .. } => Some(consumer.scope),
            ProofPayloadEvent::PreFoldPolynomial(_) => None,
        };
        let scopes = payload.events.iter().filter_map(owner_scope).collect::<Vec<_>>();
        let expected_scope = scopes.first().copied().expect("payload owner scope");
        assert!(matches!(expected_scope, ProofPayloadScope::Closed { .. }));
        assert!(scopes.iter().all(|scope| *scope == expected_scope));
    }

    #[test]
    fn proof_payload_projects_singleton_universal_relation_trace() {
        let protocol = super::super::lower::tests::singleton_preimage_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let mut run = prepare_operational_certificate(&protocol, &request)
            .expect("singleton universal certificate run");
        repeat_certificate_normalization(&mut run);
        let payload = derive_proof_payload(&run).expect("singleton universal payload");
        assert_payload_event_refs_are_local(&payload);
        assert_eq!(run.projection.closure.families.len(), 1);
        let relation_sources = run
            .trace
            .events
            .iter()
            .filter_map(|event| match event {
                NormalizerEvent::BoundTransfer {
                    rule: BoundRule::Authority(BoundAuthority::RelationPreimageSource { source }),
                    ..
                } => Some(*source),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert!(!relation_sources.is_empty(), "fixture must exercise relation source authority");
        assert!(
            relation_sources.iter().all(|source| run
                .projection
                .closure
                .expressions
                .contains(source))
        );
        let relation_source_rows = payload
            .events
            .iter()
            .filter_map(|event| match event {
                ProofPayloadEvent::BoundTransfer {
                    rule:
                        ProofPayloadRule::Authority(ProofPayloadAuthority::RelationPreimageSource {
                            source,
                        }),
                    ..
                } => Some(*source),
                _ => None,
            })
            .collect::<Vec<_>>();
        let refs = super::super::g0::derive_certificate_statement_rows(
            &run.job,
            &run.projection.closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("canonical relation source rows");
        assert!(refs.expressions().iter().enumerate().all(|(row, descriptor)| {
            descriptor.inputs.iter().all(|dependency| *dependency < row as u64)
        }));
        let expected_relation_source_rows = relation_sources
            .iter()
            .map(|source| refs.expression(*source).expect("retained relation source"))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            relation_source_rows.iter().copied().collect::<BTreeSet<_>>(),
            expected_relation_source_rows
        );

        let retained_monomials =
            run.trace.retained_monomial_roots().expect("enabled trace retains monomial roots");
        let closed_root = match &run.projection.residual {
            CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
            CertificateResidualRoot::Family { .. } => None,
        };
        let wrapper = closed_wrapper_program(&run.trace, closed_root);
        let mut retained_factor_count = 0_usize;
        for &monomial in retained_monomials {
            let Some(arena) =
                run.projection.closure.programs.iter().copied().chain(wrapper).find_map(
                    |program| {
                        run.job
                            .monomials()
                            .get(program)
                            .filter(|arena| arena.token() == monomial.arena())
                    },
                )
            else {
                continue;
            };
            let descriptor = arena.descriptor(monomial).expect("retained descriptor");
            for factor in descriptor.central_factors.iter().chain(descriptor.ordered_factors.iter())
            {
                retained_factor_count += 1;
                assert!(
                    run.projection.closure.programs.contains(&factor.program()) ||
                        wrapper == Some(factor.program())
                );
                assert!(run.projection.closure.expressions.contains(&factor.expression()));
                let node = run.job.expressions().node(factor.expression()).expect("factor node");
                assert!(
                    node.inputs.iter().all(|input| run
                        .projection
                        .closure
                        .expressions
                        .contains(input))
                );
            }
        }
        assert!(retained_factor_count > 0, "fixture must retain relation monomial factors");
        let computed = payload
            .events
            .iter()
            .enumerate()
            .find_map(|(index, event)| match event {
                ProofPayloadEvent::SpecializationComputed { owner, dispatch, source } => {
                    Some((index, *owner, dispatch, *source))
                }
                _ => None,
            })
            .expect("universal specialization computation");
        assert!(computed.3.start < computed.3.end);
        assert!(computed.3.end <= payload.events.len() as u64);
        let applied = payload
            .events
            .iter()
            .enumerate()
            .find_map(|(index, event)| match event {
                ProofPayloadEvent::AppliedRelation { owner, rule, .. }
                    if matches!(rule, ProofPayloadRelationRule::Universal { .. }) =>
                {
                    Some((index, *owner, rule))
                }
                _ => None,
            })
            .expect("universal relation application");
        assert!(applied.0 > computed.0);
        let ProofPayloadRelationRule::Universal {
            computed: applied_computed,
            lhs,
            lhs_layout,
            rhs_result,
        } = applied.2
        else {
            unreachable!("filtered universal relation")
        };
        assert!((*applied_computed as usize) < payload.events.len());
        assert!(*rhs_result as usize >= computed.3.start as usize);
        assert!((*rhs_result as usize) < payload.events.len());
        assert!(lhs.central_factors.is_empty());
        assert!(lhs.ordered_factors.len() >= 2);
        assert!(lhs_layout.is_none());
        assert!(matches!(
            payload.events.get(*rhs_result as usize),
            Some(ProofPayloadEvent::InvocationEnd { .. })
        ));
        assert_eq!(applied.1.scope, computed.1.scope);
        let relation_merges = payload
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ProofPayloadEvent::CoefficientMerge(observation) => {
                    if let ProofPayloadCoefficientMergeSource::Relation {
                        application,
                        source_term_ordinal,
                    } = &observation.source
                    {
                        Some((index, *application, *source_term_ordinal, observation))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!relation_merges.is_empty(), "universal RHS terms carry relation provenance");
        for (index, application, source_term_ordinal, merge) in &relation_merges {
            assert!(*application < payload.events.len() as u64);
            assert!(*index < payload.events.len());
            assert!(*application < *index as u64);
            assert_eq!(merge.owner, applied.1);
            let _ = source_term_ordinal;
        }
        let mut frame_stack = Vec::new();
        let mut frame_at = vec![None; payload.events.len()];
        for (index, event) in payload.events.iter().enumerate() {
            if matches!(event, ProofPayloadEvent::InvocationStart { .. }) {
                frame_stack.push(index);
            }
            frame_at[index] = frame_stack.last().copied();
            if matches!(event, ProofPayloadEvent::InvocationEnd { .. }) {
                frame_stack.pop();
            }
        }
        assert_eq!(frame_at[applied.0], frame_at[computed.0]);
        assert!(payload.events.iter().all(|event| match event {
            ProofPayloadEvent::BoundTransfer { owner, rule } => {
                !payload_rule_mentions_transfer(rule, applied.0) || owner.scope == applied.1.scope
            }
            _ => true,
        }));
        let hit = payload
            .events
            .iter()
            .enumerate()
            .find_map(|(index, event)| match event {
                ProofPayloadEvent::SpecializationCacheHit { owner, source } => {
                    Some((index, *owner, *source))
                }
                _ => None,
            })
            .expect("universal specialization cache hit");
        assert!(hit.0 > applied.0);
        assert_eq!(hit.1.scope, computed.1.scope);
        assert_eq!(hit.2, computed.3);
        assert!(hit.2.end <= hit.0 as u64);
        assert!(matches!(
            payload.events.get(hit.2.end as usize - 1),
            Some(ProofPayloadEvent::InvocationEnd { .. })
        ));
        let computed_snapshots = payload.events[computed.3.start as usize..computed.3.end as usize]
            .iter()
            .enumerate()
            .filter_map(|(offset, event)| {
                matches!(event, ProofPayloadEvent::PreFoldPolynomial(_))
                    .then_some(computed.3.start as usize + offset)
            })
            .collect::<Vec<_>>();
        let computed_nested_ends = payload.events
            [computed.3.start as usize..computed.3.end as usize]
            .iter()
            .filter(|event| matches!(event, ProofPayloadEvent::InvocationEnd { .. }))
            .count();
        assert!(!computed_snapshots.is_empty());
        assert_eq!(computed_snapshots.len(), computed_nested_ends);
        for snapshot in computed_snapshots {
            assert!(snapshot < computed.3.end as usize);
            assert!(
                payload.events[snapshot + 1..computed.3.end as usize]
                    .iter()
                    .any(|event| matches!(event, ProofPayloadEvent::InvocationEnd { .. }))
            );
        }
        assert!(!matches!(payload.events[hit.0], ProofPayloadEvent::InvocationStart { .. }));

        let (outer_frames, immediate_frames) = payload_frame_data(&payload);
        assert_eq!(outer_frames.len(), 2, "the repeated run must have two outer invocations");
        let first_outer = outer_frames[0];
        let second_outer = outer_frames[1];
        assert!(first_outer.0 <= computed.0 && computed.0 <= first_outer.1);
        assert!(second_outer.0 <= hit.0 && hit.0 <= second_outer.1);
        let direct_child_starts =
            |(outer_start, outer_end, _): (usize, usize, ProofPayloadOwner)| {
                payload.events[outer_start..=outer_end]
                    .iter()
                    .enumerate()
                    .filter(|(offset, event)| {
                        matches!(event, ProofPayloadEvent::InvocationStart { .. }) &&
                            immediate_frames[outer_start + offset] == Some(outer_start)
                    })
                    .count()
            };
        assert!(direct_child_starts(first_outer) > 0);
        assert_eq!(direct_child_starts(second_outer), 0);
        let mut outer_snapshots = Vec::new();
        for (outer_start, outer_end, outer_root) in &outer_frames {
            let snapshots = payload.events[*outer_start..=*outer_end]
                .iter()
                .enumerate()
                .filter_map(|(offset, event)| match event {
                    ProofPayloadEvent::PreFoldPolynomial(snapshot) => {
                        let index = *outer_start + offset;
                        (immediate_frames[index] == Some(*outer_start)).then_some((index, snapshot))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(snapshots.len(), 1, "each outer invocation has one fresh snapshot");
            let (snapshot_index, snapshot) = snapshots[0];
            assert!(*outer_start < snapshot_index && snapshot_index < *outer_end);
            if let Some(evidence) = &snapshot.summary_evidence {
                let event = match evidence {
                    ProofPayloadValueRef::Result { event, .. } |
                    ProofPayloadValueRef::Transfer(event) => *event as usize,
                    ProofPayloadValueRef::Predecessor { .. } => {
                        unreachable!("pre-fold summaries cannot use predecessor evidence")
                    }
                };
                assert!(*outer_start <= event && event < snapshot_index);
            }
            for (index, event) in payload.events[*outer_start..=*outer_end].iter().enumerate() {
                let index = *outer_start + index;
                if matches!(event, ProofPayloadEvent::AppliedRelation { .. }) ||
                    matches!(
                        event,
                        ProofPayloadEvent::CoefficientMerge(ProofPayloadCoefficientMerge {
                            source: ProofPayloadCoefficientMergeSource::Relation { .. },
                            ..
                        })
                    )
                {
                    assert!(index < snapshot_index);
                }
                if let ProofPayloadEvent::SurvivorFold(fold) = event {
                    assert!(snapshot_index < index && index < *outer_end);
                    assert!(*outer_start <= fold.bound as usize && (fold.bound as usize) < index);
                    assert_eq!(immediate_frames[index], immediate_frames[fold.bound as usize]);
                }
            }
            let root_result = payload.events[*outer_start..snapshot_index]
                .iter()
                .enumerate()
                .filter_map(|(offset, event)| {
                    matches!(event, ProofPayloadEvent::Result { owner, .. } if owner == outer_root)
                        .then_some(outer_start + offset)
                })
                .last()
                .expect("outer root result before its snapshot");
            assert!(root_result < snapshot_index);
            assert!(matches!(
                payload.events[*outer_end],
                ProofPayloadEvent::InvocationEnd { root, .. } if root == *outer_root
            ));
            assert!(snapshot_index < *outer_end);
            outer_snapshots.push(snapshot_index);
        }
        assert_ne!(outer_snapshots[0], outer_snapshots[1]);

        for (index, event) in payload.events.iter().enumerate() {
            let ProofPayloadEvent::AppliedRelation { rule, .. } = event else { continue };
            let ProofPayloadRelationRule::Universal { computed, rhs_result, .. } = rule else {
                continue
            };
            let Some(ProofPayloadEvent::SpecializationComputed { source, .. }) =
                payload.events.get(*computed as usize)
            else {
                panic!("universal application must cite its computed specialization")
            };
            assert!(*rhs_result as usize >= source.start as usize);
            assert!((*rhs_result as usize) < source.end as usize);
            assert_ne!(immediate_frames[index], immediate_frames[*rhs_result as usize]);
        }
        let second = prepare_operational_certificate(&protocol, &request)
            .and_then(|run| derive_proof_payload(&run))
            .expect("stable singleton universal payload");
        assert!(
            second.events.iter().all(|event| {
                !matches!(event, ProofPayloadEvent::SpecializationCacheHit { .. })
            })
        );
        assert!(
            second
                .events
                .iter()
                .any(|event| { matches!(event, ProofPayloadEvent::SpecializationComputed { .. }) })
        );
    }

    #[test]
    fn certificate_schema_v1_shares_canonical_refs_with_chronological_proof() {
        let protocol = super::super::lower::tests::singleton_preimage_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("singleton certificate run");
        let ordinary_report = run.accepted_report.clone();
        let trace_events = run.trace.events.clone();
        let prior_proof =
            derive_proof_payload_projection(&run).expect("prior proof projection").payload;
        let documents = derive_certificate_documents(&run).expect("typed certificate documents");
        assert_eq!(run.accepted_report, ordinary_report);
        assert_eq!(run.trace.events, trace_events);
        assert_eq!(
            prior_proof.encode_canonical().expect("prior proof bytes"),
            documents.proof.payload.encode_canonical().expect("shared proof bytes")
        );
        assert_eq!(
            documents.cert.schema_id,
            super::super::certificate_schema::CERTIFICATE_SCHEMA_ID
        );
        assert_eq!(documents.cert.schema_version, 1);
        assert!(!documents.cert.expressions.is_empty());
        assert_certificate_references_are_dense(&documents);
        assert_payload_event_refs_are_local(&documents.proof.payload);
        let encoded = documents.cert.encode_canonical().expect("certificate bytes");
        let text = std::str::from_utf8(&encoded).expect("certificate JSON");
        let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            value.as_object().unwrap().keys().map(String::as_str).collect::<BTreeSet<_>>(),
            [
                "schemaId",
                "schemaVersion",
                "plaintextModulus",
                "ciphertextModulus",
                "ringDimension",
                "expressions",
                "programs",
                "sources",
                "events",
                "indexUses",
                "sliceGroups",
                "residualRoot",
            ]
            .into_iter()
            .collect()
        );
        for forbidden in [
            "decoder",
            "target_id",
            "accepted_report",
            "proof_payload",
            "coverage",
            "metrics",
            "descriptor_bytes",
            "descriptor_hash",
            "opaque_descriptor",
        ] {
            assert!(!text.contains(forbidden), "unexpected certificate field {forbidden}");
        }
    }

    fn assert_certificate_references_are_dense(documents: &ProjectedCertificateDocuments) {
        let expressions = &documents.cert.expressions;
        let programs = &documents.cert.programs;
        for (row, expression) in expressions.iter().enumerate() {
            assert!(expression.inputs.iter().all(|dependency| (*dependency as usize) < row));
            assert!(expression.program.is_none_or(|program| (program as usize) < programs.len()));
            match &expression.descriptor {
                super::super::g0::CanonicalExpressionDescriptor::Source {
                    source: super::super::g0::CanonicalExpressionSource::Direct { source },
                } => assert!((*source as usize) < documents.cert.sources.len()),
                super::super::g0::CanonicalExpressionDescriptor::Source {
                    source: super::super::g0::CanonicalExpressionSource::Family { source, selector },
                } => {
                    assert!((*source as usize) < documents.cert.sources.len());
                    assert!((*selector as usize) < row);
                }
                super::super::g0::CanonicalExpressionDescriptor::Event { .. } |
                super::super::g0::CanonicalExpressionDescriptor::Operation { .. } => {}
            }
        }
        let assert_owner = |owner: ProofPayloadOwner| {
            assert!((owner.expression_row as usize) < expressions.len());
            match owner.scope {
                ProofPayloadScope::Closed { root_expression_row } => {
                    assert!((root_expression_row as usize) < expressions.len());
                    assert!(matches!(
                        documents.cert.residual_root,
                        super::super::certificate_schema::CertificateResidualRootV1::Closed {
                            expression,
                            ..
                        } if expression == root_expression_row
                    ));
                }
                ProofPayloadScope::Program { program_row } => {
                    assert!((program_row as usize) < programs.len());
                }
            }
        };
        let assert_monomial = |monomial: &ProofPayloadMonomial| {
            for owner in monomial.central_factors.iter().chain(&monomial.ordered_factors) {
                assert_owner(*owner);
            }
        };
        for event in &documents.proof.payload.events {
            match event {
                ProofPayloadEvent::InvocationStart { root } |
                ProofPayloadEvent::InvocationEnd { root, .. } => {
                    assert_owner(*root);
                }
                ProofPayloadEvent::Predecessor { consumer, .. } |
                ProofPayloadEvent::Result { owner: consumer, .. } |
                ProofPayloadEvent::SpecializationComputed { owner: consumer, .. } |
                ProofPayloadEvent::SpecializationCacheHit { owner: consumer, .. } |
                ProofPayloadEvent::BoundTransfer { owner: consumer, .. } => {
                    assert_owner(*consumer);
                }
                ProofPayloadEvent::AppliedRelation { owner, source_monomial, rule, .. } => {
                    assert_owner(*owner);
                    assert_monomial(source_monomial);
                    match rule {
                        ProofPayloadRelationRule::Universal { lhs, .. } => assert_monomial(lhs),
                        ProofPayloadRelationRule::Gadget { gadget, decomposition, .. } => {
                            assert_owner(*gadget);
                            assert_owner(*decomposition);
                        }
                    }
                }
                ProofPayloadEvent::CoefficientMerge(merge) => {
                    assert_owner(merge.owner);
                    assert_monomial(&merge.output);
                }
                ProofPayloadEvent::PreFoldPolynomial(polynomial) => {
                    for term in &polynomial.terms {
                        assert_monomial(&term.monomial);
                    }
                }
                ProofPayloadEvent::SurvivorFold(_) => {}
            }
        }
    }

    #[test]
    fn certificate_schema_v1_is_deterministic_for_independent_runs() {
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let first = prepare_operational_certificate(
            &super::super::lower::tests::singleton_preimage_protocol(),
            &request,
        )
        .expect("first certificate run");
        let second = prepare_operational_certificate(
            &super::super::lower::tests::singleton_preimage_protocol(),
            &request,
        )
        .expect("second certificate run");
        let first = derive_certificate_documents(&first).expect("first certificate documents");
        let second = derive_certificate_documents(&second).expect("second certificate documents");
        assert_eq!(
            first.cert.encode_canonical().expect("first certificate bytes"),
            second.cert.encode_canonical().expect("second certificate bytes")
        );
        assert_eq!(
            first.proof.payload.encode_canonical().expect("first proof bytes"),
            second.proof.payload.encode_canonical().expect("second proof bytes")
        );
    }

    #[test]
    fn certificate_schema_v1_matches_test_golden() {
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let run = prepare_operational_certificate(
            &super::super::lower::tests::singleton_preimage_protocol(),
            &request,
        )
        .expect("golden certificate run");
        let bytes = derive_certificate_documents(&run)
            .expect("golden certificate documents")
            .cert
            .encode_canonical()
            .expect("golden certificate bytes");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/operational-noise-certificate-v1.json");
        if std::env::var_os("MXX_REGENERATE_CORRECTNESS").as_deref() ==
            Some(std::ffi::OsStr::new("1"))
        {
            std::fs::create_dir_all(path.parent().expect("golden parent"))
                .expect("create certificate testdata");
            std::fs::write(&path, &bytes).expect("write certificate test golden");
        }
        let golden = std::fs::read(&path).expect("read certificate test golden");
        assert_eq!(bytes, golden);
    }

    #[test]
    fn certificate_schema_v1_projects_indexed_residual_tables() {
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "indexed-certificate-threshold".to_owned(),
        };
        let run =
            prepare_operational_certificate(&indexed_threshold_certificate_protocol(), &request)
                .expect("indexed certificate run");
        let documents = derive_certificate_documents(&run).expect("indexed certificate documents");
        assert!(!documents.cert.index_uses.is_empty() || !documents.cert.slice_groups.is_empty());
        assert_certificate_references_are_dense(&documents);
        assert_payload_event_refs_are_local(&documents.proof.payload);
    }

    #[test]
    fn certificate_projection_rejects_boolean_interval_targets() {
        let protocol = boolean_interval_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-boolean".to_owned(),
        };
        assert!(matches!(
            project_operational_certificate(&protocol, &request),
            Err(CertificateProjectionError::UnsupportedDecoderKind {
                target_id,
                kind: OperationalDecoderKind::BooleanInterval,
            }) if target_id == "certificate-boolean"
        ));
    }

    #[test]
    fn certificate_projection_rejects_residual_matrix_modulus_mismatch() {
        let mut job = super::super::job::CheckerJob::new();
        let scalar = job
            .expressions_mut()
            .intern(
                super::super::arena::ValueOperator::Constant(
                    super::super::arena::TypedConstant::int(0),
                ),
                Box::new([]),
            )
            .expect("zero scalar");
        let matrix_type = super::super::arena::ResolvedMatrixType::new(256_u16.into(), 1, 1, 1)
            .expect("matrix type");
        let matrix = job
            .expressions_mut()
            .intern(
                super::super::arena::ValueOperator::Matrix(
                    super::super::arena::MatrixOperation::LiftConstantPolynomial {
                        output: matrix_type,
                        coefficient_bits: 8,
                    },
                ),
                Box::new([scalar]),
            )
            .expect("zero matrix");
        let root = super::super::lower::ProductionRoot::Closed(
            job.expressions().close(matrix).expect("closed matrix"),
        );
        let target = ResolvedAcceptanceTarget {
            target_id: "certificate-threshold".to_owned(),
            ciphertext_modulus: 255_u16.into(),
            kind: ResolvedDecoderKind::Threshold { plaintext_modulus: 2_u8.into() },
        };
        assert!(matches!(
            project_residual_root(&job, &root, &target),
            Err(CertificateProjectionError::ResidualModulusMismatch {
                target_id,
                target,
                residual,
            }) if target_id == "certificate-threshold" &&
                target == 255_u16.into() &&
                residual == 256_u16.into()
        ));
    }

    #[test]
    fn residual_closure_traverses_closed_expression_inputs() {
        let mut job = super::super::job::CheckerJob::new();
        let (root, left, right) = job
            .with_arena_stores(|expressions, _, _| {
                let left = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                    Box::new([]),
                )?;
                let right = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                    Box::new([]),
                )?;
                let root = expressions.intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[left, right],
                )?;
                Ok::<_, super::super::arena::ArenaError>((expressions.close(root)?, left, right))
            })
            .expect("closed expression");
        let root_expression = root.expression();
        let projected = CertificateResidualRoot::Closed {
            root,
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        let closure = collect_residual_closure(&job, &projected).expect("closed residual closure");
        assert_eq!(closure.expressions.len(), 3);
        assert!(closure.expressions.contains(&root_expression));
        assert!(closure.expressions.contains(&left));
        assert!(closure.expressions.contains(&right));
        assert_eq!(closure.constant_expressions.len(), 2);
        assert!(closure.programs.is_empty());
        assert!(closure.families.is_empty());
    }

    #[test]
    fn lut_plans_freeze_before_proof_only_closure_expansion() {
        use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef};

        let matrix = ResolvedMatrixType::new(17_u8.into(), 1, 1, 1).unwrap();
        let raw_event = super::super::arena::SampleEventId(91);
        let mut job = super::super::job::CheckerJob::new();
        let (
            residual_root,
            residual,
            first_index,
            second_index,
            proof_program,
            proof_gadget,
            raw_gadget,
            raw_source,
            raw_identity,
        ) = job
            .with_arena_stores(|expressions, programs, _| {
                let zero = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                    Box::new([]),
                )?;
                let residual = expressions.intern(
                    ValueOperator::Matrix(MatrixOperation::LiftConstantPolynomial {
                        output: matrix.clone(),
                        coefficient_bits: 1,
                    }),
                    Box::new([zero]),
                )?;
                let first_index = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                    Box::new([]),
                )?;
                let second_index = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                    Box::new([]),
                )?;
                let proof_gadget = expressions.intern(
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                        output: ResolvedMatrixType::new(17_u8.into(), 1, 2, 1)?,
                        base: 4,
                        small: false,
                        digit_count: 2,
                    }),
                    Box::new([residual]),
                )?;
                let proof_program = programs.finalize(
                    expressions,
                    super::super::arena::ProgramSignature {
                        inputs: Box::new([]),
                        output: ResolvedValueType::Matrix(ResolvedMatrixType::new(
                            17_u8.into(),
                            1,
                            2,
                            1,
                        )?),
                    },
                    proof_gadget,
                )?;
                let raw_identity = super::super::arena::SemanticSourceIdentity {
                    stable_definition: "proof-only-plan-source".to_owned(),
                    invocation: "proof-only-plan-source".to_owned(),
                    sample_event: Some(raw_event),
                    output_role: "matrix".to_owned(),
                    sampler: Some(super::super::arena::SampleDescriptor::new(
                        "proof-only-plan-event",
                        ResolvedValueType::Matrix(matrix.clone()),
                    )),
                    artifact: None,
                    value_type: ResolvedValueType::Matrix(matrix.clone()),
                    coordinates: Box::new([]),
                    matrix_constant: None,
                };
                let raw_source = expressions
                    .intern(ValueOperator::Source(raw_identity.clone()), Box::new([]))?;
                let raw_gadget = expressions.intern(
                    ValueOperator::Transform(ValueTransformOperation::GadgetDecompose {
                        output: ResolvedMatrixType::new(17_u8.into(), 1, 2, 1)?,
                        base: 8,
                        small: false,
                        digit_count: 2,
                    }),
                    Box::new([raw_source]),
                )?;
                Ok::<_, super::super::arena::ArenaError>((
                    expressions.close(residual)?,
                    residual,
                    first_index,
                    second_index,
                    proof_program,
                    proof_gadget,
                    raw_gadget,
                    raw_source,
                    raw_identity,
                ))
            })
            .unwrap();
        let owner = |path| super::super::protocol::PlannedWire {
            stage: StageId("closure-phase".to_owned()),
            occurrence: super::super::protocol::ProgramOccurrence {
                definition: FrozenGraphScopeId::Root,
                path,
            },
            wire: WireRef { node: NodeId(path), port: Port(0) },
        };
        let plan = |path, result, consumed, index| super::super::g0::IndexUsePlan {
            kind: super::super::g0::IndexUseKind::IntegerExpression,
            owner: owner(path),
            result: Some(result),
            result_family: None,
            consumed,
            consumed_family: None,
            index,
            frontier: Box::new([]),
            output_type: ResolvedValueType::Int,
            output_range: None,
            slice_group: None,
        };
        let mut trace = FeasibilityTrace::default();
        trace.record_index_use(plan(1, residual, None, first_index)).unwrap();
        trace.record_index_use(plan(2, first_index, None, second_index)).unwrap();
        trace.record_index_use(plan(3, proof_gadget, Some(raw_gadget), second_index)).unwrap();
        trace
            .record_source(
                super::super::g0::SourceHandle::Expression(raw_source),
                super::super::g0::SourceClass::UnboundOccurrenceInput {
                    owner: owner(4),
                    identity: super::super::g0::InputSourceIdentity::Expression(raw_identity),
                },
            )
            .unwrap();
        trace
            .record_event(super::super::g0::EventObservation {
                event: raw_event,
                owner: owner(5),
                kind: super::super::g0::EventKind::Sample {
                    descriptor: super::super::arena::SampleDescriptor::new(
                        "proof-only-plan-event",
                        ResolvedValueType::Matrix(matrix.clone()),
                    ),
                },
            })
            .unwrap();

        let projected =
            CertificateResidualRoot::Closed { root: residual_root, matrix: matrix.clone() };
        let available = trace.index_use_plans().collect::<Vec<_>>();
        let mut closure =
            collect_residual_closure_with_plans(&job, &projected, &available).unwrap();
        assert!(closure.expressions.contains(&first_index));
        assert!(closure.expressions.contains(&second_index));
        assert!(!closure.expressions.contains(&proof_gadget));
        assert!(!closure.expressions.contains(&raw_gadget));
        trace.retain_residual_index_use_plans(&closure);
        assert_eq!(trace.index_use_plans().count(), 2);
        assert_eq!(trace.source_observations().len(), 1);
        assert_eq!(trace.event_observations().len(), 1);

        let frozen = trace.index_use_plans().collect::<Vec<_>>();
        walk_certificate_closure(
            &job,
            &mut closure,
            vec![CertificateWork::Program(proof_program)],
            &frozen,
        )
        .unwrap();
        assert!(closure.programs.contains(&proof_program));
        assert!(closure.expressions.contains(&proof_gadget));
        assert!(!closure.expressions.contains(&raw_gadget));
        assert!(!closure.expressions.contains(&raw_source));
        assert!(!closure.event_ids.contains(&raw_event));
        trace.retain_residual(&closure);
        assert_eq!(trace.index_use_plans().count(), 2);
        assert!(trace.source_observations().is_empty());
        assert!(trace.event_observations().is_empty());

        let rows = super::super::g0::derive_certificate_statement_rows(
            &job,
            &closure,
            &trace,
            Some(residual),
        )
        .unwrap();
        let proof_row = rows.expression(proof_gadget).unwrap() as usize;
        assert!(matches!(
            rows.expressions()[proof_row].descriptor,
            super::super::g0::CanonicalExpressionDescriptor::Event {
                operator: super::super::g0::CanonicalEventOperator::GadgetDecompose { .. }
            }
        ));
        assert_eq!(rows.events().len(), 1);
        let n = exact_retained_n(&rows).unwrap();
        assert_eq!(n.total_rows, n.expression_rows + n.program_rows + n.source_rows + n.event_rows);
    }

    #[test]
    fn residual_closure_collects_typed_sources_and_events_only_from_residual() {
        use super::super::g0::FeasibilitySink;
        use mxx_ir_core::NodeId;
        let mut job = super::super::job::CheckerJob::new();
        let (root, source_identity, source_event, sample_event, decoder_event) = job
            .with_arena_stores(|expressions, _, _| {
                let source_identity = super::super::arena::SemanticSourceIdentity {
                    stable_definition: "source-definition".to_owned(),
                    invocation: "source-invocation".to_owned(),
                    sample_event: Some(super::super::arena::SampleEventId(41)),
                    output_role: "source-output".to_owned(),
                    sampler: Some(super::super::arena::SampleDescriptor::new(
                        "source-sampler",
                        super::super::arena::ResolvedValueType::Int,
                    )),
                    artifact: None,
                    value_type: super::super::arena::ResolvedValueType::Int,
                    coordinates: Box::new([3]),
                    matrix_constant: None,
                };
                let source = expressions
                    .intern(ValueOperator::Source(source_identity.clone()), Box::new([]))?;
                let sample_event = super::super::arena::SampleEventId(7);
                let sample = expressions.intern(
                    ValueOperator::Sample {
                        event: sample_event,
                        descriptor: super::super::arena::SampleDescriptor::new(
                            "sample-definition",
                            super::super::arena::ResolvedValueType::Int,
                        ),
                    },
                    Box::new([]),
                )?;
                let root = expressions.intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[source, sample],
                )?;
                let decoder_event = super::super::arena::SampleEventId(99);
                let decoder = expressions.intern(
                    ValueOperator::Sample {
                        event: decoder_event,
                        descriptor: super::super::arena::SampleDescriptor::new(
                            "decoder-only",
                            super::super::arena::ResolvedValueType::Int,
                        ),
                    },
                    Box::new([]),
                )?;
                let _decoder = expressions.close(decoder)?;
                Ok::<_, super::super::arena::ArenaError>((
                    expressions.close(root)?,
                    source_identity,
                    super::super::arena::SampleEventId(41),
                    sample_event,
                    decoder_event,
                ))
            })
            .expect("source and event expressions");
        let projected = CertificateResidualRoot::Closed {
            root,
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        let closure = collect_residual_closure(&job, &projected).expect("residual closure");
        assert_eq!(closure.source_ids, [source_identity].into_iter().collect());
        assert_eq!(closure.event_ids.len(), 2);
        assert!(closure.event_ids.contains(&source_event));
        assert!(closure.event_ids.contains(&sample_event));
        assert!(!closure.event_ids.contains(&decoder_event));
        let mut trace = super::super::g0::FeasibilityTrace::default();
        let owner = |path| super::super::protocol::PlannedWire {
            stage: crate::StageId("inventory-events".to_owned()),
            occurrence: super::super::protocol::ProgramOccurrence {
                definition: FrozenGraphScopeId::Root,
                path,
            },
            wire: WireRef { node: NodeId(path), port: Port(0) },
        };
        trace
            .record_event(super::super::g0::EventObservation {
                event: source_event,
                owner: owner(1),
                kind: super::super::g0::EventKind::Sample {
                    descriptor: super::super::arena::SampleDescriptor::new(
                        "source-sampler",
                        super::super::arena::ResolvedValueType::Int,
                    ),
                },
            })
            .unwrap();
        trace
            .record_event(super::super::g0::EventObservation {
                event: sample_event,
                owner: owner(2),
                kind: super::super::g0::EventKind::Sample {
                    descriptor: super::super::arena::SampleDescriptor::new(
                        "sample-definition",
                        super::super::arena::ResolvedValueType::Int,
                    ),
                },
            })
            .unwrap();
        let inventory = super::super::g0::derive_inventory(&job, &closure, &trace, None)
            .expect("residual descriptor inventory");
        assert_eq!(inventory.events.len(), 2);
        assert_eq!(inventory.sources.len(), 1);
        let first = inventory.encode_canonical().expect("canonical inventory");
        let second = inventory.encode_canonical().expect("canonical inventory");
        assert_eq!(first, second);
        assert_eq!(inventory.canonical_encoded_size().expect("encoded size"), first.len());
        assert!(!String::from_utf8(first).unwrap().contains("decoder-only"));
    }

    #[test]
    fn residual_closure_collects_sampler_and_trapdoor_pair_events() {
        let matrix = super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
            .expect("matrix type");
        let mut job = super::super::job::CheckerJob::new();
        let (sampler_root, trapdoor_root) = job
            .with_arena_stores(|expressions, _, _| {
                let sampler_left = expressions.intern(
                    ValueOperator::Sampler {
                        event: super::super::arena::SampleEventId(51),
                        operation: super::super::arena::SamplerOperation::UniformResidue {
                            output: matrix.clone(),
                        },
                    },
                    Box::new([]),
                )?;
                let sampler_right = expressions.intern(
                    ValueOperator::Sampler {
                        event: super::super::arena::SampleEventId(52),
                        operation: super::super::arena::SamplerOperation::UniformResidue {
                            output: matrix.clone(),
                        },
                    },
                    Box::new([]),
                )?;
                let sampler_root = expressions.intern_matrix_transform(
                    super::super::arena::MatrixOperation::Add,
                    &[sampler_left, sampler_right],
                )?;
                let trapdoor_root = expressions.intern(
                    ValueOperator::Trapdoor(super::super::arena::TrapdoorOperation::Generate {
                        descriptor: "closure-paired-trapdoor".to_owned(),
                        parameters: Box::new([]),
                        paired_public_event: super::super::arena::SampleEventId(53),
                        paired_public_output_role: "value".to_owned(),
                    }),
                    Box::new([]),
                )?;
                Ok::<_, super::super::arena::ArenaError>((
                    expressions.close(sampler_root)?,
                    expressions.close(trapdoor_root)?,
                ))
            })
            .expect("sampler and trapdoor roots");
        let sampler_closure = collect_residual_closure(
            &job,
            &CertificateResidualRoot::Closed { root: sampler_root, matrix: matrix.clone() },
        )
        .expect("sampler closure");
        assert_eq!(
            sampler_closure.event_ids,
            [super::super::arena::SampleEventId(51), super::super::arena::SampleEventId(52)]
                .into_iter()
                .collect()
        );
        let trapdoor_closure = collect_residual_closure(
            &job,
            &CertificateResidualRoot::Closed { root: trapdoor_root, matrix },
        )
        .expect("trapdoor closure");
        assert_eq!(
            trapdoor_closure.event_ids,
            [super::super::arena::SampleEventId(53)].into_iter().collect()
        );
    }

    #[test]
    fn residual_closure_keeps_family_body_typed_without_lane_enumeration() {
        let mut job = super::super::job::CheckerJob::new();
        let (family, body) = job
            .with_arena_stores(|expressions, programs, _| {
                let body =
                    expressions.intern_argument(0, super::super::arena::ResolvedValueType::Int)?;
                let family = programs.generated_family_from_body(
                    expressions,
                    super::super::arena::FamilyDomain::new(4, 8)?,
                    body,
                )?;
                Ok::<_, super::super::arena::ArenaError>((family, body))
            })
            .expect("indexed family");
        let projected = CertificateResidualRoot::Family {
            family,
            domain: super::super::arena::FamilyDomain::new(4, 8).expect("family domain"),
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        let closure = collect_residual_closure(&job, &projected).expect("family residual closure");
        assert_eq!(closure.expressions.len(), 1);
        assert!(closure.expressions.contains(&body));
        assert_eq!(closure.programs.len(), 1);
        assert!(closure.programs.contains(&family.program()));
        assert_eq!(closure.families.len(), 1);
        assert!(closure.families.contains(&family));
    }

    #[test]
    fn residual_closure_includes_transitive_program_call_bodies() {
        let mut job = super::super::job::CheckerJob::new();
        let (root, inner, outer, inner_body, outer_body, one) = job
            .with_arena_stores(|expressions, programs, _| {
                let inner_body = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(2)),
                    Box::new([]),
                )?;
                let signature = super::super::arena::ProgramSignature {
                    inputs: Box::new([]),
                    output: super::super::arena::ResolvedValueType::Int,
                };
                let inner = programs.finalize(expressions, signature.clone(), inner_body)?;
                let inner_call =
                    expressions.intern_slice(ValueOperator::ProgramCall { program: inner }, &[])?;
                let one = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                    Box::new([]),
                )?;
                let outer_body = expressions.intern_slice(
                    ValueOperator::Scalar(super::super::arena::ScalarOperation::Add),
                    &[inner_call, one],
                )?;
                let outer = programs.finalize(expressions, signature, outer_body)?;
                let root_expression =
                    expressions.intern_slice(ValueOperator::ProgramCall { program: outer }, &[])?;
                Ok::<_, super::super::arena::ArenaError>((
                    expressions.close(root_expression)?,
                    inner,
                    outer,
                    inner_body,
                    outer_body,
                    one,
                ))
            })
            .expect("nested program calls");
        let projected = CertificateResidualRoot::Closed {
            root,
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        let closure =
            collect_residual_closure(&job, &projected).expect("program-call residual closure");
        assert_eq!(closure.programs, [inner, outer].into_iter().collect());
        assert!(closure.expressions.contains(&root.expression()));
        assert!(closure.expressions.contains(&inner_body));
        assert!(closure.expressions.contains(&outer_body));
        assert!(closure.expressions.contains(&one));
        assert!(closure.families.is_empty());
    }

    #[test]
    fn residual_closure_excludes_decoder_root() {
        let mut job = super::super::job::CheckerJob::new();
        let (residual, decoder) = job
            .with_arena_stores(|expressions, _, _| {
                let residual = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                    Box::new([]),
                )?;
                let decoder = expressions.intern(
                    ValueOperator::Constant(super::super::arena::TypedConstant::int(1)),
                    Box::new([]),
                )?;
                Ok::<_, super::super::arena::ArenaError>((
                    expressions.close(residual)?,
                    expressions.close(decoder)?,
                ))
            })
            .expect("two roots");
        let projected = CertificateResidualRoot::Closed {
            root: residual,
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        let closure = collect_residual_closure(&job, &projected).expect("residual closure");
        assert!(closure.expressions.contains(&residual.expression()));
        assert_eq!(closure.constant_expressions, [residual.expression()].into_iter().collect());
        assert!(!closure.expressions.contains(&decoder.expression()));
    }

    #[test]
    fn residual_closure_rejects_a_foreign_closed_root() {
        let mut source = super::super::job::CheckerJob::new();
        let root = source
            .expressions_mut()
            .intern(
                ValueOperator::Constant(super::super::arena::TypedConstant::int(0)),
                Box::new([]),
            )
            .and_then(|expression| source.expressions().close(expression))
            .expect("source root");
        let target = super::super::job::CheckerJob::new();
        let projected = CertificateResidualRoot::Closed {
            root,
            matrix: super::super::arena::ResolvedMatrixType::new(17_u8.into(), 1, 1, 1)
                .expect("matrix type"),
        };
        assert!(matches!(
            collect_residual_closure(&target, &projected),
            Err(CertificateClosureError::Arena(
                super::super::arena::ArenaError::ForeignExpression { .. }
            ))
        ));
    }

    #[test]
    fn proof_payload_encoding_and_logical_items_are_allocation_independent() {
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let first_run =
            prepare_operational_certificate(&threshold_certificate_protocol(), &request)
                .expect("first accepted run");
        let second_run =
            prepare_operational_certificate(&threshold_certificate_protocol(), &request)
                .expect("independently allocated accepted run");
        let first_projection =
            derive_proof_payload_projection(&first_run).expect("first retained projection");
        let second_projection =
            derive_proof_payload_projection(&second_run).expect("second retained projection");
        let first_payload = derive_proof_payload(&first_run).expect("first proof payload");
        let second_payload = derive_proof_payload(&second_run).expect("second proof payload");
        assert_eq!(first_projection.payload, first_payload);
        assert_eq!(second_projection.payload, second_payload);
        let first_canonical_payload_bytes =
            first_payload.encode_canonical().expect("first canonical payload");
        let second_canonical_payload_bytes =
            second_payload.encode_canonical().expect("second canonical payload");
        assert_eq!(first_canonical_payload_bytes, second_canonical_payload_bytes);
        assert_eq!(first_payload.logical_items(), second_payload.logical_items());
        assert_eq!(
            first_projection.generator_peak_retained_logical_items,
            second_projection.generator_peak_retained_logical_items,
        );
        assert_eq!(first_projection.observed_coverage, second_projection.observed_coverage);
        for row in &first_projection.observed_coverage.rows {
            assert_eq!(row.count, u64::try_from(row.sites.len()).unwrap());
            assert!(row.sites.windows(2).all(|sites| sites[0] < sites[1]));
        }
        let (manual_support, _, _, _, _) = manual_projection_support_items(&first_run);
        let payload_items = first_payload.logical_items().expect("logical item count");
        let coverage_items =
            first_projection.observed_coverage.logical_items().expect("coverage logical items");
        assert_eq!(
            first_projection.generator_peak_retained_logical_items,
            manual_support + payload_items + coverage_items,
        );
        assert!(coverage_items > 0);
        assert!(first_projection.generator_peak_retained_logical_items >= payload_items);
        assert!(first_payload.logical_items().expect("logical item count") > 0);

        // The empty vector contributes its length field and nothing else; this is a small
        // independent audit of the recursive structural count.
        let empty = OperationalProofPayload { events: Vec::new() };
        assert_eq!(empty.logical_items(), Ok(1));
        assert!(empty.encode_canonical().expect("empty canonical payload").len() > 0);
    }

    #[test]
    fn proof_payload_generator_peak_counts_nested_rhs_lifecycle_once() {
        let protocol = super::super::lower::tests::singleton_preimage_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let mut run = prepare_operational_certificate(&protocol, &request)
            .expect("singleton universal certificate run");
        repeat_certificate_normalization(&mut run);
        let projection =
            derive_proof_payload_projection(&run).expect("retained universal projection");
        let (frames, immediate) = payload_frame_data(&projection.payload);
        assert!(frames.iter().any(|(start, end, _)| {
            (*start..=*end).any(|index| immediate[index].is_some_and(|parent| parent != *start))
        }));
        let (manual_support, arena_count, rhs_count, canonical_refs_items, shallow_refs_items) =
            manual_projection_support_items(&run);
        assert!(arena_count > 0);
        assert!(rhs_count > 0);
        assert!(canonical_refs_items > shallow_refs_items);
        let payload_items = projection.payload.logical_items().expect("payload logical items");
        let coverage_items =
            projection.observed_coverage.logical_items().expect("coverage logical items");
        assert_eq!(
            projection.generator_peak_retained_logical_items,
            manual_support + payload_items + coverage_items,
        );
        assert!(coverage_items > 0);
        let mut expected_trace_sites = BTreeMap::new();
        for (index, event) in projection.payload.events.iter().enumerate() {
            let kind = match event {
                ProofPayloadEvent::AppliedRelation { rule, .. } => {
                    Some(ObservedCoverageKind::Relation(observed_relation_kind(rule)))
                }
                ProofPayloadEvent::BoundTransfer { rule, .. } => {
                    Some(ObservedCoverageKind::Bound(observed_bound_kind(rule)))
                }
                _ => None,
            };
            if let Some(kind) = kind {
                expected_trace_sites
                    .entry(kind)
                    .or_insert_with(BTreeSet::new)
                    .insert(ObservedCoverageSite::TraceEvent { index: index as u64 });
            }
        }
        let actual_trace_sites = projection
            .observed_coverage
            .rows
            .iter()
            .filter(|row| {
                matches!(
                    row.kind,
                    ObservedCoverageKind::Relation(_) | ObservedCoverageKind::Bound(_)
                )
            })
            .map(|row| (row.kind, row.sites.iter().copied().collect::<BTreeSet<_>>()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(actual_trace_sites, expected_trace_sites);
        assert!(
            projection
                .payload
                .events
                .iter()
                .any(|event| { matches!(event, ProofPayloadEvent::SpecializationCacheHit { .. }) })
        );
        assert!(projection.observed_coverage.rows.iter().any(|row| {
            row.kind == ObservedCoverageKind::Relation(ObservedRelationKind::Universal) &&
                row.count > 1
        }));
        let canonical_refs = super::super::g0::derive_certificate_statement_rows(
            &run.job,
            &run.projection.closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("canonical refs");
        let sampler_sites = projection
            .observed_coverage
            .rows
            .iter()
            .filter(|row| matches!(row.kind, ObservedCoverageKind::Sampler(_)))
            .flat_map(|row| row.sites.iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(sampler_sites.len(), canonical_refs.event_rows().rows().len());
        assert!(
            projection.observed_coverage.rows.iter().any(|row| {
                matches!(row.kind, ObservedCoverageKind::Operator(_)) && row.count > 1
            })
        );
        assert_payload_event_refs_are_local(&projection.payload);
    }

    #[test]
    fn honest_protocol_difference_changes_payload_encoding_and_logical_items() {
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let threshold_request = super::super::OperationalCheckRequest {
            target_id: "certificate-threshold".to_owned(),
            ..request.clone()
        };
        let singleton_run = prepare_operational_certificate(
            &super::super::lower::tests::singleton_preimage_protocol(),
            &request,
        )
        .expect("honest singleton run");
        let threshold_run =
            prepare_operational_certificate(&threshold_certificate_protocol(), &threshold_request)
                .expect("honest threshold run");
        let singleton_payload = derive_proof_payload(&singleton_run).expect("singleton payload");
        let threshold_payload = derive_proof_payload(&threshold_run).expect("threshold payload");
        assert_ne!(
            singleton_payload.encode_canonical().expect("singleton canonical payload"),
            threshold_payload.encode_canonical().expect("threshold canonical payload")
        );
        assert_ne!(singleton_payload.logical_items(), threshold_payload.logical_items());
    }

    #[test]
    fn representative_payload_audits_recursive_encoding_and_logical_items() {
        let owner = ProofPayloadOwner {
            scope: ProofPayloadScope::Closed { root_expression_row: 7 },
            expression_row: 11,
        };
        // scope = 1 tag + 1 row = 2; owner = scope 2 + expression row 1 = 3.
        let monomial = ProofPayloadMonomial {
            central_factors: vec![owner],
            ordered_factors: vec![owner, owner],
        };
        // central Vec = 1 length + 1 element + owner 3 = 5;
        // ordered Vec = 1 length + 2 elements + 2 * owner 3 = 9;
        // monomial = 5 + 9 = 14; term = monomial 14 + coefficient 1 = 15.
        let term =
            ProofPayloadTerm { monomial: monomial.clone(), coefficient: BigInt::from(-7_i32) };
        let exact_summary = super::super::normal_form::BoundedSummary::finite(
            BoundExpression::new(BigUint::from(2_u8)),
        );
        let exact = ProofPayloadValue::Exact {
            terms: vec![term.clone()],
            coefficient_bound: exact_summary.coefficient_bound(),
            coefficient_producer: 1,
            summary: exact_summary,
            summary_producer: Some(1),
        };
        // finite summary = Known tag 1 + (Finite tag 1 + BigUint 1) = 3;
        // Exact value = enum tag 1 + terms Vec 17 + coefficient bound 3 + producer 1 +
        // summary 3 + optional summary producer 2 = 27.
        let product_facts = super::super::bound::MatrixProductFacts {
            left_is_constant_polynomial: true,
            right_is_constant_polynomial: false,
            right_known_zero_rows: Some(BigUint::from(3_u8)),
            left_support_upper: Some(2),
            right_support_upper: None,
        };
        // Product refs: Result = 1 tag + event 1 + projection 1 = 3;
        // Transfer = 1 tag + event 1 = 2; facts = 1 + 1 + 2 + 2 + 1 = 7;
        // Product rule = enum tag 1 + 3 + 2 + 7 = 13; BoundTransfer = 1 + owner 3 + rule 13 = 17.
        let product_rule = ProofPayloadRule::Product {
            left: ProofPayloadValueRef::Result { event: 1, projection: BoundProjection::Summary },
            right: ProofPayloadValueRef::Transfer(0),
            facts: product_facts,
        };
        let operator_merge = ProofPayloadCoefficientMerge {
            owner,
            source: ProofPayloadCoefficientMergeSource::Operator {
                inputs: [
                    ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                    ProofPayloadTermRef { value_event: 1, term_ordinal: 0 },
                ],
            },
            output: monomial.clone(),
            signed_contribution: BigInt::from(-7_i32),
        };
        // TermRef = 1 + 1 = 2; Operator source = tag 1 + 2 * 2 = 5;
        // merge = owner 3 + source 5 + monomial 14 + integer 1 = 23;
        // CoefficientMerge event = tag 1 + merge 23 = 24.
        let relation_merge = ProofPayloadCoefficientMerge {
            owner,
            source: ProofPayloadCoefficientMergeSource::Relation {
                application: 2,
                source_term_ordinal: 0,
            },
            output: monomial.clone(),
            signed_contribution: BigInt::from(5_i32),
        };
        // Relation source = tag 1 + application 1 + ordinal 1 = 3;
        // relation merge = 3 + 3 + 14 + 1 = 21; event = 1 + 21 = 22.
        let pre_fold = ProofPayloadPreFoldPolynomial {
            result_event: 1,
            terms: vec![term],
            summary: super::super::normal_form::BoundedSummary::finite(BoundExpression::new(
                BigUint::from(2_u8),
            )),
            summary_evidence: Some(ProofPayloadValueRef::Result {
                event: 1,
                projection: BoundProjection::Coefficient,
            }),
        };
        // PreFold = result event 1 + terms Vec 17 + summary 3
        //   + Some(ref (1 + 1 + 1) = 4) = 25;
        // event = tag 1 + 25 = 26. SurvivorFold = event tag 1 + (integer 1 + bound 1) = 3.
        let payload = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root: owner }, // 1 + 3 = 4
                ProofPayloadEvent::Result { owner, value: exact },  // 1 + 3 + 27 = 31
                ProofPayloadEvent::BoundTransfer { owner, rule: product_rule }, // 17
                ProofPayloadEvent::CoefficientMerge(operator_merge), // 24
                ProofPayloadEvent::CoefficientMerge(relation_merge), // 22
                ProofPayloadEvent::PreFoldPolynomial(pre_fold),     // 26
                ProofPayloadEvent::SurvivorFold(ProofPayloadSurvivorFold {
                    coefficient: BigInt::from(-2_i32),
                    bound: 5,
                }), // 3
            ],
        };
        // OperationalProofPayload is the events Vec: 1 + length 7 +
        // (4 + 31 + 17 + 24 + 22 + 26 + 3) = 8 + 127 = 135.
        assert_eq!(payload.logical_items(), Ok(135));
        let canonical_payload_bytes = payload.encode_canonical().expect("representative payload");

        // Scalar changes are semantically represented bytes but remain one logical item.
        let mut scalar_changed = payload.clone();
        if let ProofPayloadEvent::Result { value: ProofPayloadValue::Exact { terms, .. }, .. } =
            &mut scalar_changed.events[1]
        {
            terms[0].coefficient = BigInt::from(42_i32);
        } else {
            panic!("representative result value");
        }
        assert_ne!(
            canonical_payload_bytes,
            scalar_changed.encode_canonical().expect("scalar-changed payload")
        );
        assert_eq!(scalar_changed.logical_items(), Ok(135));

        // Removing one Some option changes the structural count by exactly one.
        let mut product_changed = payload.clone();
        if let ProofPayloadEvent::BoundTransfer {
            rule: ProofPayloadRule::Product { facts, .. },
            ..
        } = &mut product_changed.events[2]
        {
            facts.right_support_upper = Some(4);
        } else {
            panic!("representative product rule");
        }
        assert_ne!(
            canonical_payload_bytes,
            product_changed.encode_canonical().expect("product-changed payload")
        );
        assert_eq!(product_changed.logical_items(), Ok(136));

        let mut evidence_removed = payload.clone();
        if let ProofPayloadEvent::PreFoldPolynomial(polynomial) = &mut evidence_removed.events[5] {
            polynomial.summary_evidence = None;
        } else {
            panic!("representative PreFold");
        }
        assert_ne!(
            canonical_payload_bytes,
            evidence_removed.encode_canonical().expect("evidence-removed payload")
        );
        assert_eq!(evidence_removed.logical_items(), Ok(132));

        let mut survivor_changed = payload.clone();
        if let ProofPayloadEvent::SurvivorFold(fold) = &mut survivor_changed.events[6] {
            fold.bound = 9;
        } else {
            panic!("representative SurvivorFold");
        }
        assert_ne!(
            canonical_payload_bytes,
            survivor_changed.encode_canonical().expect("survivor-changed payload")
        );
        assert_eq!(survivor_changed.logical_items(), Ok(135));
    }

    #[test]
    fn base_summary_is_typed_incomplete_and_serializes_deterministically() {
        let protocol = threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let first = prepare_base_feasibility_summary(&protocol, &request)
            .expect("base feasibility summary");
        let second = prepare_base_feasibility_summary(&protocol, &request)
            .expect("repeat base feasibility summary");
        assert_eq!(first.schema_id, BASE_FEASIBILITY_SCHEMA_ID);
        assert_eq!(first.schema_version, BASE_FEASIBILITY_SCHEMA_VERSION);
        assert!(first.accepted);
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("ordinary accepted baseline");
        let expected_source_rows = run
            .projection
            .closure
            .source_ids
            .len()
            .checked_add(run.projection.closure.family_source_ids.len())
            .and_then(|count| count.checked_add(run.projection.closure.constant_expressions.len()))
            .expect("source rows");
        assert_eq!(first.n.source_rows, expected_source_rows);
        assert_eq!(first.counters.residual_trace, ResidualTraceCounters::default());
        assert_eq!(
            first.counters.ordinary_baseline.normalization_nodes_processed,
            run.accepted_report.counters.normalization.nodes_processed
        );
        assert_eq!(
            first.n.total_rows,
            first.n.expression_rows +
                first.n.program_rows +
                first.n.source_rows +
                first.n.event_rows
        );
        assert_eq!(first, second);
        let first_bytes = serialize_base_feasibility_summary(&first).expect("serialize summary");
        let second_bytes = serialize_base_feasibility_summary(&second).expect("serialize repeat");
        assert_eq!(first_bytes, second_bytes);
        let json = String::from_utf8(first_bytes).expect("UTF-8 JSON");
        assert!(!json.contains("milliseconds"));
        assert!(!json.contains("artifact_emission"));
    }

    #[test]
    fn aggregate_generator_retention_is_each_phase_max_and_propagates_overflow() {
        assert_eq!(aggregate_generator_peak_retained_logical_items([Ok(7), Ok(11), Ok(5)]), Ok(11));
        assert_eq!(
            aggregate_generator_peak_retained_logical_items([
                Ok(7),
                Err(CanonicalPayloadError::LengthOverflow),
                Ok(11),
            ]),
            Err(CanonicalPayloadError::LengthOverflow)
        );
    }

    #[test]
    fn g0_cpu_evidence_is_deterministic_exact_and_explicitly_pre_gate() {
        fn object_keys(value: &serde_json::Value) -> BTreeSet<&str> {
            value.as_object().expect("JSON object").keys().map(String::as_str).collect()
        }
        fn collect_keys<'a>(value: &'a serde_json::Value, keys: &mut BTreeSet<&'a str>) {
            match value {
                serde_json::Value::Object(object) => {
                    for (key, value) in object {
                        keys.insert(key);
                        collect_keys(value, keys);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        collect_keys(value, keys);
                    }
                }
                _ => {}
            }
        }

        let protocol = super::super::lower::tests::singleton_preimage_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "singleton-preimage".to_owned(),
        };
        let first = prepare_g0_cpu_evidence_bytes(&protocol, &request).expect("CPU evidence");
        let second =
            prepare_g0_cpu_evidence_bytes(&protocol, &request).expect("repeat CPU evidence");
        assert_eq!(first, second);
        let document: serde_json::Value = serde_json::from_slice(&first).expect("evidence JSON");
        assert_eq!(
            object_keys(&document),
            [
                "schema_id",
                "schema_version",
                "status",
                "base_feasibility",
                "residual_coverage_matrix",
                "lut",
                "metrics",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(document["schema_id"], G0_CPU_EVIDENCE_SCHEMA_ID);
        assert_eq!(document["schema_version"], G0_CPU_EVIDENCE_SCHEMA_VERSION);
        assert_eq!(document["schema_version"], 6);
        assert_eq!(document["status"], "CpuObservationOnlyNotAcceptanceEvidence");
        let base = &document["base_feasibility"];
        assert_eq!(
            object_keys(base),
            [
                "schema_id",
                "schema_version",
                "target_id",
                "plaintext_modulus",
                "ciphertext_modulus",
                "accepted",
                "noise_bound",
                "threshold_left",
                "margin",
                "counters",
                "n",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            object_keys(&base["counters"]),
            ["ordinary_baseline", "residual_trace"].into_iter().collect()
        );
        assert_eq!(
            object_keys(&base["counters"]["ordinary_baseline"]),
            [
                "occurrences",
                "samples",
                "normalization_nodes_processed",
                "normalization_nodes_total",
                "normalization_exact_term_count",
                "normalization_relation_candidates",
                "normalization_relation_applied",
                "normalization_relation_remaining",
                "normalization_bounded_fold_count",
            ]
            .into_iter()
            .collect()
        );
        assert!(object_keys(&base["counters"]["residual_trace"]).is_empty());
        assert_eq!(
            object_keys(&base["n"]),
            ["expression_rows", "program_rows", "source_rows", "event_rows", "total_rows"]
                .into_iter()
                .collect()
        );
        assert_eq!(
            object_keys(&document["residual_coverage_matrix"]),
            ["rows"].into_iter().collect()
        );
        for row in document["residual_coverage_matrix"]["rows"].as_array().expect("coverage rows") {
            assert_eq!(
                object_keys(row),
                ["kind", "count", "sites", "rust_item", "disposition"].into_iter().collect()
            );
            assert_eq!(object_keys(&row["kind"]), ["domain", "kind"].into_iter().collect());
            if let Some(nested_kind) = row["kind"]["kind"].as_object() {
                assert_eq!(nested_kind.len(), 1);
                assert!(nested_kind.keys().all(|key| matches!(
                    key.as_str(),
                    "scalar" | "matrix" | "trapdoor" | "gadget_decompose" | "hash" | "authority"
                )));
            }
            assert!(
                row["rust_item"]
                    .as_str()
                    .expect("Rust item")
                    .starts_with("crates/correctness/src/operational_noise/")
            );
            let disposition = &row["disposition"];
            assert_eq!(
                object_keys(disposition),
                ["disposition", "semantics", "transfer"].into_iter().collect()
            );
            assert!(matches!(
                disposition["disposition"].as_str(),
                Some("checked_lean" | "g2_lean_obligation")
            ));
            for site in row["sites"].as_array().expect("coverage sites") {
                let expected =
                    if site["site"] == "trace_event" { ["site", "index"] } else { ["site", "row"] };
                assert_eq!(object_keys(site), expected.into_iter().collect());
            }
        }
        assert_eq!(
            object_keys(&document["lut"]),
            [
                "exact_row_count",
                "exact_payload_logical_items",
                "index_use_frontier_products",
                "slice_group_frontier_products",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            object_keys(&document["metrics"]),
            [
                "descriptor_inventory_canonical_encoded_bytes",
                "inventory_retained_logical_items",
                "proof_payload_logical_items",
                "proof_payload_canonical_encoded_bytes",
                "lut_canonical_encoded_bytes",
                "lut_retained_logical_items",
                "recorder_peak_retained_logical_items",
                "proof_projection_peak_retained_logical_items",
                "generator_peak_retained_logical_items",
            ]
            .into_iter()
            .collect()
        );

        let run = prepare_operational_certificate(&protocol, &request).expect("comparison run");
        let closure = &run.projection.closure;
        let statement_rows = super::super::g0::derive_certificate_statement_rows(
            &run.job,
            closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("statement rows");
        let n = exact_retained_n(&statement_rows).expect("exact N");
        assert_eq!(base["n"]["expression_rows"], n.expression_rows);
        assert_eq!(base["n"]["program_rows"], n.program_rows);
        assert_eq!(base["n"]["source_rows"], n.source_rows);
        assert_eq!(base["n"]["event_rows"], n.event_rows);
        assert_eq!(base["n"]["total_rows"], n.total_rows);

        let inventory = super::super::g0::derive_inventory(
            &run.job,
            closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("comparison inventory");
        assert_eq!(inventory.sources.len() as u64, n.source_rows);
        assert_eq!(inventory.events.len() as u64, n.event_rows);
        let inventory_bytes = inventory.encode_canonical().unwrap().len() as u64;
        let inventory_retained = inventory.retained_logical_items().unwrap();
        let lut = super::super::g0::derive_lut_evidence(
            &run.job,
            closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("comparison LUT");
        let lut_bytes = lut.encode_canonical().unwrap().len() as u64;
        let lut_retained = lut.retained_logical_items().unwrap();
        let frontier_sum = document["lut"]["index_use_frontier_products"]
            .as_array()
            .unwrap()
            .iter()
            .chain(document["lut"]["slice_group_frontier_products"].as_array().unwrap())
            .map(|value| value.as_str().unwrap().parse::<BigUint>().unwrap())
            .fold(BigUint::ZERO, |sum, value| sum + value);
        assert_eq!(document["lut"]["exact_row_count"], frontier_sum.to_string());
        assert_eq!(frontier_sum, lut.l_rows);

        let projection = derive_proof_payload_projection(&run).expect("comparison projection");
        let payload_t = projection.payload.logical_items().unwrap();
        let payload_bytes = projection.payload.encode_canonical().unwrap().len() as u64;
        let retention = run.trace.recorder_retention();
        let metrics = &document["metrics"];
        assert_eq!(metrics["descriptor_inventory_canonical_encoded_bytes"], inventory_bytes);
        assert_eq!(metrics["inventory_retained_logical_items"], inventory_retained);
        assert_eq!(metrics["proof_payload_logical_items"], payload_t);
        assert_eq!(metrics["proof_payload_canonical_encoded_bytes"], payload_bytes);
        assert_eq!(metrics["lut_canonical_encoded_bytes"], lut_bytes);
        assert_eq!(metrics["lut_retained_logical_items"], lut_retained);
        assert_eq!(metrics["recorder_peak_retained_logical_items"], retention.peak_logical_items);
        assert_eq!(
            metrics["proof_projection_peak_retained_logical_items"],
            projection.generator_peak_retained_logical_items
        );
        assert_eq!(
            metrics["generator_peak_retained_logical_items"],
            inventory_retained
                .max(lut_retained)
                .max(projection.generator_peak_retained_logical_items)
        );
        let expected_matrix =
            derive_residual_coverage_matrix(&projection.observed_coverage).unwrap();
        assert_eq!(expected_matrix.rows.len(), projection.observed_coverage.rows.len());
        for (matrix_row, observed_row) in
            expected_matrix.rows.iter().zip(&projection.observed_coverage.rows)
        {
            assert_eq!(matrix_row.kind, observed_row.kind);
            assert_eq!(matrix_row.count, observed_row.count);
            assert_eq!(matrix_row.sites, observed_row.sites);
            assert_eq!(matrix_row.count, matrix_row.sites.len() as u64);
            assert!(matrix_row.sites.windows(2).all(|sites| sites[0] < sites[1]));
            assert!(!matches!(
                matrix_row.disposition,
                ResidualCoverageDisposition::RejectBeforeGeneration { .. }
            ));
        }
        assert!(expected_matrix.rows.windows(2).all(|rows| rows[0].kind < rows[1].kind));
        assert_eq!(
            document["residual_coverage_matrix"],
            serde_json::to_value(expected_matrix).unwrap()
        );

        let mut all_keys = BTreeSet::new();
        collect_keys(&document, &mut all_keys);
        for forbidden in [
            "l",
            "l_logical_items",
            "artifact",
            "current",
            "size",
            "rss",
            "time",
            "runtime",
            "gpu",
            "benchmark",
            "estimate",
            "dispositions",
            "generation",
            "acceptance",
        ] {
            assert!(!all_keys.contains(forbidden), "forbidden evidence key {forbidden}");
        }
        let encoded = std::str::from_utf8(&first).unwrap();
        for forbidden in ["l_logical_items", "artifact_bytes", "runtime", "benchmark", "estimate"] {
            assert!(!encoded.contains(forbidden), "forbidden evidence substring {forbidden}");
        }
    }

    fn serialized_lut_logical_items(value: &serde_json::Value) -> u64 {
        match value {
            serde_json::Value::Array(values) => {
                1 + values.len() as u64 +
                    values.iter().map(serialized_lut_logical_items).sum::<u64>()
            }
            serde_json::Value::Object(fields) => {
                fields.values().map(serialized_lut_logical_items).sum()
            }
            _ => 1,
        }
    }

    fn independent_lut_logical_items(lut: &serde_json::Value) -> u64 {
        let mut optional_payloads = 0_u64;
        let mut semantic_ranges = 0_u64;
        for unit in lut["indexUses"].as_array().unwrap() {
            optional_payloads += ["result", "consumed", "output_range"]
                .into_iter()
                .filter(|field| !unit[*field].is_null())
                .count() as u64;
            semantic_ranges += u64::from(!unit["output_range"].is_null());
            semantic_ranges += unit["frontier"].as_array().unwrap().len() as u64;
        }
        for unit in lut["sliceGroups"].as_array().unwrap() {
            optional_payloads += ["result", "consumed", "row_span", "column_span"]
                .into_iter()
                .filter(|field| !unit[*field].is_null())
                .count() as u64;
            semantic_ranges += unit["frontier"].as_array().unwrap().len() as u64;
            semantic_ranges += unit["members"].as_array().unwrap().len() as u64;
        }
        serialized_lut_logical_items(lut) + optional_payloads - 3 * semantic_ranges
    }

    #[test]
    fn g0_cpu_evidence_observes_nonempty_honest_index_lut() {
        let protocol = indexed_threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "indexed-certificate-threshold".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request).expect("comparison run");
        for plan in run.trace.index_use_plans() {
            assert!(plan.is_residual_relevant(&run.projection.closure));
            assert!(run.projection.closure.expressions.contains(&plan.index));
            for expression in plan.result.into_iter().chain(plan.consumed) {
                assert!(run.projection.closure.expressions.contains(&expression));
            }
            for axis in &plan.frontier {
                assert!(run.projection.closure.expressions.contains(&axis.expression()));
            }
            if let Some(group) = &plan.slice_group {
                for member in &group.members {
                    assert!(run.projection.closure.expressions.contains(&member.expression));
                }
            }
            for family in plan.result_family.into_iter().chain(plan.consumed_family) {
                assert!(run.projection.closure.families.contains(&family));
            }
        }
        let first = prepare_g0_cpu_evidence_bytes(&protocol, &request)
            .expect("accepted indexed CPU evidence");
        let second = prepare_g0_cpu_evidence_bytes(&protocol, &request)
            .expect("repeat accepted indexed CPU evidence");
        assert_eq!(first, second);
        let document: serde_json::Value = serde_json::from_slice(&first).unwrap();
        let index_products = document["lut"]["index_use_frontier_products"].as_array().unwrap();
        let slice_products = document["lut"]["slice_group_frontier_products"].as_array().unwrap();
        assert!(!index_products.is_empty() || !slice_products.is_empty());

        let lut = super::super::g0::derive_lut_evidence(
            &run.job,
            &run.projection.closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("nonempty LUT");
        assert!(!lut.index_uses.is_empty() || !lut.slice_groups.is_empty());
        for unit in &lut.index_uses {
            assert_eq!(unit.rows.len().to_string(), unit.frontier_product);
        }
        for unit in &lut.slice_groups {
            assert_eq!(unit.rows.len().to_string(), unit.frontier_product);
        }
        let summed_rows = lut
            .index_uses
            .iter()
            .map(|unit| BigUint::from(unit.rows.len()))
            .chain(lut.slice_groups.iter().map(|unit| BigUint::from(unit.rows.len())))
            .fold(BigUint::ZERO, |sum, rows| sum + rows);
        assert_eq!(summed_rows, lut.l_rows);
        assert_eq!(document["lut"]["exact_row_count"], summed_rows.to_string());
        let canonical_lut: serde_json::Value =
            serde_json::from_slice(&lut.encode_canonical().unwrap()).unwrap();
        let independent_logical_items = independent_lut_logical_items(&canonical_lut);
        assert!(independent_logical_items > 0);
        assert_eq!(document["lut"]["exact_payload_logical_items"], independent_logical_items);
        assert_eq!(
            document["metrics"]["lut_canonical_encoded_bytes"],
            lut.encode_canonical().unwrap().len() as u64
        );
    }

    #[test]
    fn g0_cpu_evidence_counts_shared_four_role_slice_lut() {
        let protocol = super::super::lower::tests::generated_indexed_slice_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(8.into()),
            )],
            layouts: Vec::new(),
            target_id: "generated-indexed-slice-threshold".to_owned(),
        };
        let evidence = prepare_g0_cpu_evidence_bytes(&protocol, &request)
            .expect("shared indexed-slice CPU evidence");
        let document: serde_json::Value = serde_json::from_slice(&evidence).unwrap();

        let run = prepare_operational_certificate(&protocol, &request).expect("comparison run");
        let lut = super::super::g0::derive_lut_evidence(
            &run.job,
            &run.projection.closure,
            &run.trace,
            closed_residual_expression(&run),
        )
        .expect("shared indexed-slice LUT");
        assert!(!lut.slice_groups.is_empty());
        assert!(lut.slice_groups.iter().all(|group| {
            group.row_span.is_some() && group.column_span.is_some() && group.members.len() == 4
        }));

        let canonical_lut: serde_json::Value =
            serde_json::from_slice(&lut.encode_canonical().unwrap()).unwrap();
        let slice_groups = canonical_lut["sliceGroups"].as_array().unwrap();
        assert!(!slice_groups.is_empty());
        for group in slice_groups {
            assert!(!group["row_span"].is_null());
            assert!(!group["column_span"].is_null());
            let members = group["members"].as_array().unwrap();
            assert_eq!(members.len(), 4);
            assert!(members.iter().all(|member| member["range"].as_array().unwrap().len() == 2));
        }
        assert_eq!(
            document["lut"]["exact_payload_logical_items"],
            independent_lut_logical_items(&canonical_lut)
        );
    }

    #[test]
    fn proof_invariant_context_keeps_major_evidence_branches_typed() {
        let cases = [
            (ProofEvidenceKind::Universal, ProofInvariantMismatch::RhsReplay),
            (
                ProofEvidenceKind::Gadget,
                ProofInvariantMismatch::EventKind {
                    referenced: 3,
                    expected: "Result or InvocationEnd",
                },
            ),
            (
                ProofEvidenceKind::OperatorMerge,
                ProofInvariantMismatch::Coefficient { expected: 2.into(), actual: 3.into() },
            ),
            (
                ProofEvidenceKind::SurvivorFold,
                ProofInvariantMismatch::SurvivorMagnitude {
                    expected: 5_u8.into(),
                    actual: 7_u8.into(),
                },
            ),
        ];
        for (evidence, mismatch) in cases {
            let error = proof_invariant(11, None, evidence, mismatch.clone());
            assert_eq!(
                error,
                CertificateProjectionError::ProofInvariant {
                    context: Box::new(ProofInvariantContext {
                        event: 11,
                        owner: None,
                        evidence,
                        mismatch,
                    }),
                }
            );
        }
    }

    #[test]
    fn scalar_tensor_contract_and_monomial_reconstruction_are_exact() {
        let matrix = |rows, columns| ResolvedMatrixType {
            modulus: 257_u16.into(),
            ring_dimension: 8,
            rows,
            columns,
        };
        let tensor =
            |left: &ResolvedMatrixType, right: &ResolvedMatrixType, output: ResolvedMatrixType| {
                MatrixOperation::Tensor {
                    left_layout: MatrixLayout::row_major(left.rows, left.columns),
                    right_layout: MatrixLayout::row_major(right.rows, right.columns),
                    output_layout: MatrixLayout::row_major(output.rows, output.columns),
                    output,
                }
            };
        let scalar = matrix(1, 1);
        let vector = matrix(1, 14);
        assert!(exact_scalar_tensor_contract(
            &tensor(&scalar, &vector, vector.clone()),
            &scalar,
            &vector,
        ));
        assert!(exact_scalar_tensor_contract(
            &tensor(&vector, &scalar, vector.clone()),
            &vector,
            &scalar,
        ));
        assert!(exact_scalar_tensor_contract(
            &tensor(&scalar, &scalar, scalar.clone()),
            &scalar,
            &scalar,
        ));
        let left_non_scalar = matrix(1, 2);
        let right_non_scalar = matrix(1, 7);
        assert!(!exact_scalar_tensor_contract(
            &tensor(&left_non_scalar, &right_non_scalar, matrix(1, 14)),
            &left_non_scalar,
            &right_non_scalar,
        ));
        let mut non_row_major = tensor(&scalar, &vector, vector.clone());
        if let MatrixOperation::Tensor { left_layout, .. } = &mut non_row_major {
            left_layout.name = "opaque-layout".to_owned();
        }
        assert!(!exact_scalar_tensor_contract(&non_row_major, &scalar, &vector));

        let owner = |row| ProofPayloadOwner {
            scope: ProofPayloadScope::Closed { root_expression_row: 0 },
            expression_row: row,
        };
        let left = ProofPayloadMonomial {
            central_factors: vec![owner(3)],
            ordered_factors: vec![owner(2)],
        };
        let right = ProofPayloadMonomial {
            central_factors: vec![owner(4)],
            ordered_factors: vec![owner(5)],
        };
        assert_eq!(
            ProofPayloadProjector::scalar_product_monomial(
                left.clone(),
                right.clone(),
                true,
                false,
            ),
            ProofPayloadMonomial {
                central_factors: vec![owner(2), owner(3), owner(4)],
                ordered_factors: vec![owner(5)],
            }
        );
        assert_eq!(
            ProofPayloadProjector::scalar_product_monomial(
                left.clone(),
                right.clone(),
                false,
                true,
            ),
            ProofPayloadMonomial {
                central_factors: vec![owner(3), owner(4), owner(5)],
                ordered_factors: vec![owner(2)],
            }
        );
        assert_eq!(
            ProofPayloadProjector::scalar_product_monomial(left, right, true, true),
            ProofPayloadMonomial {
                central_factors: vec![owner(3), owner(4)],
                ordered_factors: vec![owner(2), owner(5)],
            }
        );
    }
}
