//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The production stages own graph reachability, lowering, normalization, and bound
//! classification; this driver owns target validation, diagnostics, and progress cadence.

use super::{
    OperationalSimulationDiagnostics, OperationalSimulationReport,
    arena::{
        ArenaError, ClosedExprId, ExprId, FamilyDomain, MatrixLayout, MatrixOperation,
        ResolvedMatrixType, ResolvedValueType, ValueOperator,
    },
    error::{OperationalSimulationError, TargetError},
    g0::{
        AppliedRelationRule, BoundAuthority, BoundProjection, BoundRule, BoundScale, BoundValueRef,
        CanonicalResidualRefs, FeasibilityTrace, G0Error, MonomialFactorEvidence, NormalizerEvent,
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
    collections::{BTreeMap, BTreeSet, HashMap},
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
/// `CanonicalResidualRefs`; they are never arena slots or raw handles.
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
    Exact { terms: Vec<ProofPayloadTerm>, summary: super::normal_form::BoundedSummary },
    Coefficient { bound: super::facts::NumericContract<super::facts::CoefficientBound> },
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
    Predecessor { input_position: u32, projection: BoundProjection },
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

fn checked_add(left: u64, right: u64) -> Result<u64, CanonicalPayloadError> {
    left.checked_add(right).ok_or(CanonicalPayloadError::LengthOverflow)
}

fn checked_sum<I>(items: I) -> Result<u64, CanonicalPayloadError>
where
    I: IntoIterator,
    I::Item: LogicalItems,
{
    items.into_iter().try_fold(0_u64, |total, item| checked_add(total, item.logical_items()?))
}

fn logical_vec<T: LogicalItems>(items: &[T]) -> Result<u64, CanonicalPayloadError> {
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
            Self::Exact { terms, summary } => {
                checked_add(1, checked_sum([terms.logical_items(), summary.logical_items()])?)
            }
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
            Self::Predecessor { input_position, projection } => checked_add(
                1,
                checked_sum([input_position.logical_items(), projection.logical_items()])?,
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
            Self::InvocationEnd { root, result } => {
                checked_add(1, checked_sum([root.logical_items(), result.logical_items()])?)
            }
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
            ProofPayloadValue::Exact { terms, summary } => {
                self.u8(0);
                self.vec(terms, |writer, term| writer.term(term))?;
                self.summary(summary)
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
            ProofPayloadValueRef::Predecessor { input_position, projection } => {
                self.u8(0);
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
            ProofPayloadEvent::InvocationEnd { root, result } => {
                self.u8(3);
                self.owner(root)?;
                self.value(result)?;
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
}

/// Collect the transitive dependency closure of exactly one production root.
///
/// Callers pass the already projected residual root; this API cannot accept a decoder root and
/// never enumerates family lanes or selectors.
pub(crate) fn collect_residual_closure(
    job: &super::job::CheckerJob,
    root: &CertificateResidualRoot,
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
    walk_certificate_closure(job, &mut closure, work)?;
    Ok(closure)
}

fn walk_certificate_closure(
    job: &super::job::CheckerJob,
    closure: &mut CertificateClosure,
    mut work: Vec<CertificateWork>,
) -> Result<(), CertificateClosureError> {
    while let Some(item) = work.pop() {
        match item {
            CertificateWork::Expression(expression) => {
                if !closure.expressions.insert(expression) {
                    continue;
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
    let closure = &run.projection.closure;
    let closed_root_expression = match &run.projection.residual {
        CertificateResidualRoot::Closed { root, .. } => Some(root.expression()),
        CertificateResidualRoot::Family { .. } => None,
    };
    let closed_program = closed_wrapper_program(&run.trace, closed_root_expression);
    let refs = super::g0::canonical_residual_refs(&run.job, closure, &run.trace)
        .map_err(proof_payload_error)?;
    let mut monomial_arenas = HashMap::new();
    for scope in closure.programs.iter().copied() {
        if let Some(arena) = run.job.monomials().get(scope) {
            monomial_arenas.insert(arena.token(), arena);
        }
    }
    for event in &run.trace.events {
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
                    _ => return Err(proof_payload_error(G0Error::RelationTraceInvariant)),
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
        refs: &refs,
        monomial_arenas,
        rhs_events,
        closed_program,
        closed_root_expression: closed_root_expression
            .map(|expression| refs.expression(expression))
            .transpose()
            .map_err(proof_payload_error)?,
    };
    projector.project(&run.trace)
}

/// Add trace-owned scopes and relation dependencies to the same residual closure before any
/// canonical rows are assigned.  This keeps the closure, trace filtering, and canonical refs on
/// one authority instead of creating a second scope inventory for payload projection.
fn extend_certificate_closure(
    job: &super::job::CheckerJob,
    closure: &mut CertificateClosure,
    trace: &FeasibilityTrace,
) -> Result<(), CertificateClosureError> {
    let mut work = Vec::new();
    for event in &trace.events {
        match event {
            NormalizerEvent::InvocationStart { root } |
            NormalizerEvent::Result { owner: root, .. } |
            NormalizerEvent::InvocationEnd { root, .. } |
            NormalizerEvent::BoundTransfer { owner: root, .. } => {
                work.push(CertificateWork::Program(root.program()));
                work.push(CertificateWork::Expression(root.expression()));
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
    walk_certificate_closure(job, closure, work)
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

struct ProofPayloadProjector<'a> {
    job: &'a super::job::CheckerJob,
    refs: &'a CanonicalResidualRefs,
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
}

impl<'a> ProofPayloadProjector<'a> {
    fn project(
        self,
        trace: &FeasibilityTrace,
    ) -> Result<OperationalProofPayload, CertificateProjectionError> {
        let event_count = trace.events.len();
        let mut events = Vec::with_capacity(event_count);
        for (index, event) in trace.events.iter().enumerate() {
            events.push(self.event(trace, index, event).map_err(proof_payload_error)?);
        }
        Ok(OperationalProofPayload { events })
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
                ProofPayloadAuthority::RelationPreimageSource { source: self.expression(*source)? }
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
            .map(|factor| self.owner(factor))
            .collect::<Result<Vec<_>, _>>()?;
        central_factors.sort();
        Ok(ProofPayloadMonomial {
            central_factors,
            ordered_factors: descriptor
                .ordered_factors
                .iter()
                .copied()
                .map(|factor| self.owner(factor))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn value(&self, value: &super::g0::RecordedValue) -> Result<ProofPayloadValue, G0Error> {
        let Some(normal_form) = &value.exact_nf else {
            return Ok(ProofPayloadValue::Coefficient { bound: value.coefficient_bound.clone() });
        };
        Ok(ProofPayloadValue::Exact {
            terms: self.terms(normal_form)?,
            summary: normal_form.bounded_summary.clone(),
        })
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
    ) -> Result<ProofPayloadPreFoldPolynomial, G0Error> {
        Ok(ProofPayloadPreFoldPolynomial {
            terms: self.terms(&observation.polynomial)?,
            summary: observation.polynomial.bounded_summary.clone(),
            summary_evidence: observation
                .summary_evidence
                .as_ref()
                .map(|evidence| self.value_ref(evidence, current))
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
    ) -> Result<ProofPayloadValueRef, G0Error> {
        let event = match value {
            BoundValueRef::Predecessor { input_position, projection } => {
                return Ok(ProofPayloadValueRef::Predecessor {
                    input_position: *input_position,
                    projection: projection.clone(),
                });
            }
            BoundValueRef::Result { event, projection } => {
                self.prior_event(*event, current)?;
                return Ok(ProofPayloadValueRef::Result {
                    event: event.0,
                    projection: projection.clone(),
                });
            }
            BoundValueRef::Transfer(event) => {
                self.prior_event(*event, current)?;
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
    ) -> Result<ProofPayloadTermRef, G0Error> {
        self.prior_event(reference.value_event, current)?;
        let value = match trace.events.get(reference.value_event.0 as usize) {
            Some(NormalizerEvent::Result { value, .. }) |
            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
            _ => return Err(G0Error::RelationTraceInvariant),
        };
        let normal_form = value.exact_nf.as_ref().ok_or(G0Error::RelationTraceInvariant)?;
        let mut terms = normal_form
            .exact_terms
            .keys()
            .filter_map(|monomial| self.monomial(*monomial).ok().map(|term| (*monomial, term)))
            .collect::<Vec<_>>();
        terms.sort_by(|left, right| left.1.cmp(&right.1));
        let term_ordinal = terms
            .iter()
            .position(|(monomial, _)| *monomial == reference.monomial)
            .ok_or(G0Error::RelationTraceInvariant)? as u64;
        Ok(ProofPayloadTermRef { value_event: reference.value_event.0, term_ordinal })
    }

    fn validate_relation_output(
        &self,
        applied: &super::g0::AppliedRelation,
        source_term: super::monomial::MonomialId,
        output: &ProofPayloadMonomial,
    ) -> Result<(), G0Error> {
        let source = self.monomial(applied.source_monomial)?;
        let replacement = self.monomial(source_term)?;
        let start =
            usize::try_from(applied.ordered_start).map_err(|_| G0Error::RelationTraceInvariant)?;
        let end = usize::try_from(applied.ordered_end_exclusive)
            .map_err(|_| G0Error::RelationTraceInvariant)?;
        if start > end || end > source.ordered_factors.len() {
            return Err(G0Error::RelationTraceInvariant);
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
            return Err(G0Error::RelationTraceInvariant);
        }
        Ok(())
    }

    fn coefficient_merge(
        &self,
        trace: &FeasibilityTrace,
        observation: &super::g0::CoefficientMerge,
        current: usize,
    ) -> Result<ProofPayloadCoefficientMerge, G0Error> {
        let output = self.monomial(observation.output)?;
        let source = match &observation.source {
            super::g0::CoefficientMergeSource::Operator { inputs } => {
                let node = self.job.expressions().node(observation.owner.expression())?;
                let operation = match &node.operator {
                    ValueOperator::Matrix(
                        operation @ (MatrixOperation::Add | MatrixOperation::Subtract),
                    ) |
                    ValueOperator::Matrix(operation @ MatrixOperation::Multiply) => {
                        operation.clone()
                    }
                    _ => return Err(G0Error::RelationTraceInvariant),
                };
                let right = match trace.events.get(inputs[1].value_event.0 as usize) {
                    Some(NormalizerEvent::Result { value, .. }) |
                    Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value
                        .exact_nf
                        .as_ref()
                        .and_then(|normal_form| normal_form.exact_terms.get(&inputs[1].monomial)),
                    _ => None,
                }
                .ok_or(G0Error::RelationTraceInvariant)?;
                let expected = match operation {
                    MatrixOperation::Add => right.clone(),
                    MatrixOperation::Subtract => -right.clone(),
                    MatrixOperation::Multiply => {
                        let left = match trace.events.get(inputs[0].value_event.0 as usize) {
                            Some(NormalizerEvent::Result { value, .. }) |
                            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => {
                                value.exact_nf.as_ref().and_then(|normal_form| {
                                    normal_form.exact_terms.get(&inputs[0].monomial)
                                })
                            }
                            _ => None,
                        }
                        .ok_or(G0Error::RelationTraceInvariant)?;
                        left * right
                    }
                    _ => unreachable!("operator classification is exhaustive"),
                };
                if observation.signed_contribution != expected {
                    return Err(G0Error::RelationTraceInvariant);
                }
                match operation {
                    MatrixOperation::Add | MatrixOperation::Subtract => {
                        if self.monomial(inputs[0].monomial)? != output ||
                            self.monomial(inputs[1].monomial)? != output
                        {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                    }
                    MatrixOperation::Multiply => {
                        let left = self.monomial(inputs[0].monomial)?;
                        let right = self.monomial(inputs[1].monomial)?;
                        let left_type = self.job.expressions().value_type(node.inputs[0])?;
                        let right_type = self.job.expressions().value_type(node.inputs[1])?;
                        let left_scalar = matches!(
                            left_type,
                            ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1
                        );
                        let right_scalar = matches!(
                            right_type,
                            ResolvedValueType::Matrix(matrix) if matrix.rows == 1 && matrix.columns == 1
                        );
                        let (central_factors, ordered_factors) = if left_scalar && !right_scalar {
                            let mut central_factors = left.central_factors;
                            central_factors.extend(left.ordered_factors);
                            central_factors.extend(right.central_factors);
                            central_factors.sort();
                            (central_factors, right.ordered_factors)
                        } else if right_scalar && !left_scalar {
                            let mut central_factors = right.central_factors;
                            central_factors.extend(right.ordered_factors);
                            central_factors.extend(left.central_factors);
                            central_factors.sort();
                            (central_factors, left.ordered_factors)
                        } else {
                            let mut central_factors = left.central_factors;
                            central_factors.extend(right.central_factors);
                            central_factors.sort();
                            let mut ordered_factors = left.ordered_factors;
                            ordered_factors.extend(right.ordered_factors);
                            (central_factors, ordered_factors)
                        };
                        if output.central_factors != central_factors ||
                            output.ordered_factors != ordered_factors
                        {
                            return Err(G0Error::RelationTraceInvariant);
                        }
                    }
                    _ => unreachable!("operator classification is exhaustive"),
                }
                ProofPayloadCoefficientMergeSource::Operator {
                    inputs: [
                        self.term_ref(trace, inputs[0], current)?,
                        self.term_ref(trace, inputs[1], current)?,
                    ],
                }
            }
            super::g0::CoefficientMergeSource::Relation { application, source_term } => {
                self.prior_event(*application, current)?;
                let applied = match trace.events.get(application.0 as usize) {
                    Some(NormalizerEvent::AppliedRelation(applied))
                        if applied.owner == observation.owner =>
                    {
                        applied
                    }
                    _ => return Err(G0Error::RelationTraceInvariant),
                };
                let source_event = match &applied.rule {
                    super::g0::AppliedRelationRule::Universal { key, source, rhs, .. } => {
                        self.rhs_event(key, *source, *rhs, current)?.1
                    }
                    super::g0::AppliedRelationRule::Gadget { input_result, .. } => {
                        self.prior_event(*input_result, current)?;
                        input_result.0
                    }
                };
                let source_event_index = super::g0::EventIndex(source_event);
                let source_value = match trace.events.get(source_event as usize) {
                    Some(NormalizerEvent::Result { value, .. }) |
                    Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
                    _ => return Err(G0Error::RelationTraceInvariant),
                };
                let source_nf =
                    source_value.exact_nf.as_ref().ok_or(G0Error::RelationTraceInvariant)?;
                let source_coefficient = source_nf
                    .exact_terms
                    .get(source_term)
                    .ok_or(G0Error::RelationTraceInvariant)?;
                if &applied.outer_coefficient * source_coefficient !=
                    observation.signed_contribution
                {
                    return Err(G0Error::RelationTraceInvariant);
                }
                let source_ordinal = self
                    .term_ref(
                        trace,
                        super::g0::RecordedTermRef {
                            value_event: source_event_index,
                            monomial: *source_term,
                        },
                        current,
                    )?
                    .term_ordinal;
                self.validate_relation_output(applied, *source_term, &output)?;
                ProofPayloadCoefficientMergeSource::Relation {
                    application: application.0,
                    source_term_ordinal: source_ordinal,
                }
            }
        };
        Ok(ProofPayloadCoefficientMerge {
            owner: self.owner(observation.owner)?,
            source,
            output,
            signed_contribution: observation.signed_contribution.clone(),
        })
    }

    fn prior_event(&self, event: super::g0::EventIndex, current: usize) -> Result<(), G0Error> {
        if event.0 as usize >= current {
            return Err(G0Error::RelationTraceInvariant);
        }
        Ok(())
    }

    fn rule(&self, rule: &BoundRule, current: usize) -> Result<ProofPayloadRule, G0Error> {
        let value = |value: &BoundValueRef| self.value_ref(value, current);
        Ok(match rule {
            BoundRule::Authority(authority) => {
                ProofPayloadRule::Authority(self.authority(authority)?)
            }
            BoundRule::Identity { input } => ProofPayloadRule::Identity { input: value(input)? },
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
                monomial: self.monomial(*monomial)?,
                factors: factors
                    .iter()
                    .map(|factor| self.factor(factor, current))
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
    ) -> Result<ProofPayloadFactorEvidence, G0Error> {
        Ok(ProofPayloadFactorEvidence {
            bound: self.value_ref(&factor.bound, current)?,
            is_constant_polynomial: factor.is_constant_polynomial,
            support_upper: factor.support_upper,
        })
    }

    fn relation_rule(
        &self,
        rule: &AppliedRelationRule,
        current: usize,
    ) -> Result<ProofPayloadRelationRule, G0Error> {
        Ok(match rule {
            AppliedRelationRule::Universal { key, source, lhs, rhs } => {
                let (computed, rhs_result) = self.rhs_event(key, *source, *rhs, current)?;
                ProofPayloadRelationRule::Universal {
                    computed,
                    lhs: self.monomial(lhs.monomial)?,
                    lhs_layout: lhs.layout.clone(),
                    rhs_result,
                }
            }
            AppliedRelationRule::Gadget { gadget, decomposition, input, input_result } => {
                self.prior_event(*input_result, current)?;
                ProofPayloadRelationRule::Gadget {
                    gadget: self.owner(*gadget)?,
                    decomposition: self.owner(*decomposition)?,
                    input: self.expression(*input)?,
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
    ) -> Result<(u64, u64), G0Error> {
        let (computed, result) = self
            .rhs_events
            .get(&(key.clone(), source, rhs))
            .copied()
            .ok_or(G0Error::RelationTraceInvariant)?;
        if computed >= current as u64 || result >= current as u64 {
            return Err(G0Error::RelationTraceInvariant);
        }
        Ok((computed, result))
    }

    fn event(
        &self,
        trace: &FeasibilityTrace,
        current: usize,
        event: &NormalizerEvent,
    ) -> Result<ProofPayloadEvent, G0Error> {
        Ok(match event {
            NormalizerEvent::InvocationStart { root } => {
                ProofPayloadEvent::InvocationStart { root: self.owner(*root)? }
            }
            NormalizerEvent::Predecessor {
                consumer,
                input_position,
                predecessor,
                source_result,
            } => {
                self.prior_event(*source_result, current)?;
                ProofPayloadEvent::Predecessor {
                    consumer: self.owner(*consumer)?,
                    input_position: *input_position,
                    predecessor: self.expression(*predecessor)?,
                    source_result: source_result.0,
                }
            }
            NormalizerEvent::Result { owner, value } => {
                ProofPayloadEvent::Result { owner: self.owner(*owner)?, value: self.value(value)? }
            }
            NormalizerEvent::InvocationEnd { root, result, .. } => {
                ProofPayloadEvent::InvocationEnd {
                    root: self.owner(*root)?,
                    result: self.value(result)?,
                }
            }
            NormalizerEvent::SpecializationComputed { owner, key, replay } => {
                ProofPayloadEvent::SpecializationComputed {
                    owner: self.owner(*owner)?,
                    dispatch: self.specialization_dispatch(key)?,
                    source: self.range(replay.range, current)?,
                }
            }
            NormalizerEvent::SpecializationCacheHit { owner, source, .. } => {
                ProofPayloadEvent::SpecializationCacheHit {
                    owner: self.owner(*owner)?,
                    source: self.range(*source, current)?,
                }
            }
            NormalizerEvent::AppliedRelation(observation) => ProofPayloadEvent::AppliedRelation {
                owner: self.owner(observation.owner)?,
                source_monomial: self.monomial(observation.source_monomial)?,
                outer_coefficient: observation.outer_coefficient.clone(),
                ordered_start: observation.ordered_start,
                ordered_end_exclusive: observation.ordered_end_exclusive,
                rule: self.relation_rule(&observation.rule, current)?,
            },
            NormalizerEvent::BoundTransfer { owner, rule } => ProofPayloadEvent::BoundTransfer {
                owner: self.owner(*owner)?,
                rule: self.rule(rule, current)?,
            },
            NormalizerEvent::SurvivorFold(observation) => {
                self.prior_event(observation.bound, current)?;
                let (_, _, magnitude) = trace.resolve_survivor_bound(observation.bound)?;
                if *observation.coefficient.magnitude() != magnitude {
                    return Err(G0Error::RelationTraceInvariant);
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
    let mut closure = collect_residual_closure(&job, &residual)?;
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
    let closure = &run.projection.closure;
    // Build the residual-only Stage-1 inventory from the same owned job and closure.  The base
    // summary does not expose it or claim final artifact completeness, but descriptor conflicts
    // must still fail closed on this opt-in path.
    super::g0::derive_inventory(&run.job, closure, &run.trace)
        .map_err(|error| error.to_string())?;
    let source_rows = closure
        .source_ids
        .len()
        .checked_add(closure.family_source_ids.len())
        .and_then(|count| count.checked_add(closure.constant_expressions.len()))
        .ok_or_else(|| "base feasibility source-row count overflow".to_owned())?;
    let total_rows = closure
        .expressions
        .len()
        .checked_add(closure.programs.len())
        .and_then(|total| total.checked_add(source_rows))
        .and_then(|total| total.checked_add(closure.event_ids.len()))
        .ok_or_else(|| "base feasibility N count overflow".to_owned())?;
    Ok(BaseFeasibilitySummary {
        schema_id: BASE_FEASIBILITY_SCHEMA_ID,
        schema_version: BASE_FEASIBILITY_SCHEMA_VERSION,
        target_id: run.accepted_report.target_id,
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
            expression_rows: closure.expressions.len(),
            program_rows: closure.programs.len(),
            source_rows,
            event_rows: closure.event_ids.len(),
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
    use mxx_dsl::{Bool, DslContext, IdealSpec, Int, Ring, SemanticAnchor};
    use mxx_ir_core::{IntExpr, node::ConstantMatrix};

    fn payload_rule_mentions_transfer(rule: &ProofPayloadRule, event: usize) -> bool {
        let value = |value: &ProofPayloadValueRef| matches!(value, ProofPayloadValueRef::Transfer(candidate) if *candidate as usize == event);
        match rule {
            ProofPayloadRule::Authority(_) => false,
            ProofPayloadRule::Identity { input } => value(input),
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
                    if let Some(
                        ProofPayloadValueRef::Result { event, .. } |
                        ProofPayloadValueRef::Transfer(event),
                    ) = &snapshot.summary_evidence
                    {
                        before(*event);
                    }
                }
                ProofPayloadEvent::SurvivorFold(observation) => before(observation.bound),
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
        let endpoint = EndpointSpecId::ToyThresholdDecode;
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
        let endpoint = EndpointSpecId::DiamondBooleanInterval;
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
                        semantics: EndpointSemanticBinding::DiamondBoolean {
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
        let protocol = crate::toy_example::protocol();
        let request = super::super::OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                super::super::OperationalParameterValue::Integer(1.into()),
            )],
            layouts: Vec::new(),
            target_id: "toy-threshold".to_owned(),
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
            }) if target_id == "toy-threshold"
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
        let inventory = super::super::g0::derive_inventory(&job, &closure, &trace)
            .expect("residual descriptor inventory");
        assert_eq!(inventory.events.len(), 2);
        assert_eq!(inventory.sources.len(), 1);
        let first = inventory.encode_canonical().expect("canonical inventory");
        let second = inventory.encode_canonical().expect("canonical inventory");
        assert_eq!(first, second);
        assert_eq!(inventory.canonical_encoded_size().expect("encoded size"), first.len());
        assert!(
            inventory
                .operators
                .iter()
                .all(|operator| !operator.to_string().contains("decoder-only"))
        );
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
        let first_payload = derive_proof_payload(&first_run).expect("first proof payload");
        let second_payload = derive_proof_payload(&second_run).expect("second proof payload");
        let first_canonical_payload_bytes =
            first_payload.encode_canonical().expect("first canonical payload");
        let second_canonical_payload_bytes =
            second_payload.encode_canonical().expect("second canonical payload");
        assert_eq!(first_canonical_payload_bytes, second_canonical_payload_bytes);
        assert_eq!(first_payload.logical_items(), second_payload.logical_items());
        assert!(first_payload.logical_items().expect("logical item count") > 0);

        // The empty vector contributes its length field and nothing else; this is a small
        // independent audit of the recursive structural count.
        let empty = OperationalProofPayload { events: Vec::new() };
        assert_eq!(empty.logical_items(), Ok(1));
        assert!(empty.encode_canonical().expect("empty canonical payload").len() > 0);
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
        let exact = ProofPayloadValue::Exact {
            terms: vec![term.clone()],
            summary: super::super::normal_form::BoundedSummary::finite(BoundExpression::new(
                BigUint::from(2_u8),
            )),
        };
        // finite summary = Known tag 1 + (Finite tag 1 + BigUint 1) = 3;
        // Exact value = enum tag 1 + terms Vec (1 + length 1 + term 15) + summary 3 = 21.
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
            terms: vec![term],
            summary: super::super::normal_form::BoundedSummary::finite(BoundExpression::new(
                BigUint::from(2_u8),
            )),
            summary_evidence: Some(ProofPayloadValueRef::Result {
                event: 1,
                projection: BoundProjection::Coefficient,
            }),
        };
        // PreFold = terms Vec 17 + summary 3 + Some(ref (1 + 1 + 1) = 4) = 24;
        // event = tag 1 + 24 = 25. SurvivorFold = event tag 1 + (integer 1 + bound 1) = 3.
        let payload = OperationalProofPayload {
            events: vec![
                ProofPayloadEvent::InvocationStart { root: owner }, // 1 + 3 = 4
                ProofPayloadEvent::Result { owner, value: exact },  // 1 + 3 + 21 = 25
                ProofPayloadEvent::BoundTransfer { owner, rule: product_rule }, // 17
                ProofPayloadEvent::CoefficientMerge(operator_merge), // 24
                ProofPayloadEvent::CoefficientMerge(relation_merge), // 22
                ProofPayloadEvent::PreFoldPolynomial(pre_fold),     // 25
                ProofPayloadEvent::SurvivorFold(ProofPayloadSurvivorFold {
                    coefficient: BigInt::from(-2_i32),
                    bound: 5,
                }), // 3
            ],
        };
        // OperationalProofPayload is the events Vec: 1 + length 7 +
        // (4 + 25 + 17 + 24 + 22 + 25 + 3) = 8 + 120 = 128.
        assert_eq!(payload.logical_items(), Ok(128));
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
        assert_eq!(scalar_changed.logical_items(), Ok(128));

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
        assert_eq!(product_changed.logical_items(), Ok(129));

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
        assert_eq!(evidence_removed.logical_items(), Ok(125));

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
        assert_eq!(survivor_changed.logical_items(), Ok(128));
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
}
