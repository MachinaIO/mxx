//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The production stages own graph reachability, lowering, normalization, and bound
//! classification; this driver owns target validation, diagnostics, and progress cadence.

use super::{
    OperationalSimulationDiagnostics, OperationalSimulationReport,
    arena::{
        ArenaError, ClosedExprId, ExprId, FamilyDomain, MatrixLayout, ResolvedMatrixType,
        ResolvedValueType, ValueOperator,
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
use num_bigint::{BigInt, BigUint};
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
    pub term_ordinal: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProofPayloadCoefficientMergeSource {
    Value(ProofPayloadTermRef),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProofPayloadCoefficientMerge {
    pub owner: ProofPayloadOwner,
    pub sources: Box<[ProofPayloadCoefficientMergeSource]>,
    pub output: ProofPayloadMonomial,
    pub signed_contribution: BigInt,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationalProofPayload {
    pub events: Vec<ProofPayloadEvent>,
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
        Ok(ProofPayloadValue::Exact { terms, summary: normal_form.bounded_summary.clone() })
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
        self.term_ref_with_missing(trace, reference, current, false)
    }

    fn term_ref_with_missing(
        &self,
        trace: &FeasibilityTrace,
        reference: super::g0::RecordedTermRef,
        current: usize,
        allow_missing: bool,
    ) -> Result<ProofPayloadTermRef, G0Error> {
        self.prior_event(reference.value_event, current)?;
        let value = match trace.events.get(reference.value_event.0 as usize) {
            Some(NormalizerEvent::Result { value, .. }) |
            Some(NormalizerEvent::InvocationEnd { result: value, .. }) => value,
            _ => return Err(G0Error::RelationTraceInvariant),
        };
        let term_ordinal = value.exact_nf.as_ref().and_then(|normal_form| {
            let mut terms = normal_form
                .exact_terms
                .keys()
                .filter_map(|monomial| self.monomial(*monomial).ok().map(|term| (*monomial, term)))
                .collect::<Vec<_>>();
            terms.sort_by(|left, right| left.1.cmp(&right.1));
            terms
                .iter()
                .position(|(monomial, _)| *monomial == reference.monomial)
                .map(|position| position as u64)
        });
        if !allow_missing &&
            term_ordinal.is_none() &&
            matches!(
                trace.events.get(reference.value_event.0 as usize),
                Some(NormalizerEvent::Result { .. }) | Some(NormalizerEvent::InvocationEnd { .. })
            ) &&
            value.exact_nf.is_some()
        {
            return Err(G0Error::RelationTraceInvariant);
        }
        Ok(ProofPayloadTermRef { value_event: reference.value_event.0, term_ordinal })
    }

    fn coefficient_merge(
        &self,
        trace: &FeasibilityTrace,
        observation: &super::g0::CoefficientMerge,
        current: usize,
    ) -> Result<ProofPayloadCoefficientMerge, G0Error> {
        let source = |source: &super::g0::CoefficientMergeSource| match source {
            super::g0::CoefficientMergeSource::Value(reference) => {
                Ok(ProofPayloadCoefficientMergeSource::Value(
                    self.term_ref(trace, *reference, current)?,
                ))
            }
        };
        Ok(ProofPayloadCoefficientMerge {
            owner: self.owner(observation.owner)?,
            sources: observation
                .sources
                .iter()
                .map(source)
                .collect::<Result<Vec<_>, G0Error>>()?
                .into_boxed_slice(),
            output: self.monomial(observation.output)?,
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
    use super::*;
    use crate::{
        OutputRef, ProtocolDecl, ProtocolStage, StageId,
        bundle::{
            ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor,
            EndpointAnchors, EndpointSemanticBinding, EndpointSpecId, InputContract,
            InputContractEntry, InputValueContract, OperationalDecoderTarget, ProtocolInputBinding,
            ProtocolInputDestination, Workflow,
        },
    };
    use mxx_dsl::{Bool, DslContext, IdealSpec, Int, Ring, SemanticAnchor};

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
        let residual =
            ring.zero((1, 1)).semantic_anchor("certificate.residual").expect("residual anchor");
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
    fn proof_payload_projects_the_accepted_trace_without_arena_handles() {
        let protocol = threshold_certificate_protocol();
        let request = super::super::OperationalCheckRequest {
            environment: Vec::new(),
            layouts: Vec::new(),
            target_id: "certificate-threshold".to_owned(),
        };
        let run = prepare_operational_certificate(&protocol, &request)
            .expect("valid threshold certificate run");
        let payload = derive_proof_payload(&run).expect("canonical proof payload");
        let second = prepare_operational_certificate(&protocol, &request)
            .expect("equivalent threshold certificate run");
        let second_payload = derive_proof_payload(&second).expect("canonical second payload");
        assert_eq!(payload, second_payload);
        assert_eq!(payload.events.len(), run.trace.events.len());
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
        let owner_scope = |event: &ProofPayloadEvent| match event {
            ProofPayloadEvent::InvocationStart { root } |
            ProofPayloadEvent::InvocationEnd { root, .. } => Some(root.scope),
            ProofPayloadEvent::Result { owner, .. } |
            ProofPayloadEvent::SpecializationComputed { owner, .. } |
            ProofPayloadEvent::SpecializationCacheHit { owner, .. } |
            ProofPayloadEvent::AppliedRelation { owner, .. } |
            ProofPayloadEvent::BoundTransfer { owner, .. } => Some(owner.scope),
            ProofPayloadEvent::CoefficientMerge(observation) => Some(observation.owner.scope),
            ProofPayloadEvent::Predecessor { consumer, .. } => Some(consumer.scope),
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
        assert_eq!(computed.3.end, computed.0 as u64);
        assert!(computed.3.start < computed.3.end);
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
        assert_eq!(*applied_computed as usize, computed.0);
        assert!(*rhs_result as usize >= computed.3.start as usize);
        assert!((*rhs_result as usize) < computed.0);
        assert!(lhs.central_factors.is_empty());
        assert!(lhs.ordered_factors.len() >= 2);
        assert!(lhs_layout.is_none());
        assert!(matches!(
            payload.events.get(*rhs_result as usize),
            Some(ProofPayloadEvent::InvocationEnd { .. })
        ));
        assert_eq!(applied.1.scope, computed.1.scope);
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
