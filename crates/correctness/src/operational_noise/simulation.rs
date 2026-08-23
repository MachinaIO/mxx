//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The production stages own graph reachability, lowering, normalization, and bound
//! classification; this driver owns target validation, diagnostics, and progress cadence.

use super::{
    OperationalSimulationDiagnostics, OperationalSimulationReport,
    arena::{
        ArenaError, ClosedExprId, ExprId, FamilyDomain, ResolvedMatrixType, ResolvedValueType,
        ValueOperator,
    },
    error::{OperationalSimulationError, TargetError},
    lower::{ProductionAdapter, ProductionRoot},
    program::{FamilyValueId, ValueProgramId},
    protocol::ProtocolPlan,
    report::{ReportTarget, analyze_roots},
};
use crate::{OperationalDecoderKind, ProtocolDecl};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, IntExpr, ParamEnv, Port, WireRef, WireType,
    node::{IntBinaryOp, IntCompareOp, NodeKind},
};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use std::{
    collections::{BTreeMap, BTreeSet},
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

/// The typed dependency inventory rooted at one residual production root.
///
/// Expression and program IDs remain job-local.  This inventory is an in-memory boundary only;
/// later serialization must replace these handles with canonical rows without adding decoder data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CertificateClosure {
    pub expressions: BTreeSet<ExprId>,
    pub programs: BTreeSet<ValueProgramId>,
    pub families: BTreeSet<FamilyValueId>,
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
/// Callers pass `ProductionRoots.residual`; this API intentionally accepts no decoder root and
/// never enumerates family lanes or selectors.
pub(crate) fn collect_residual_closure(
    job: &super::job::CheckerJob,
    root: &ProductionRoot,
) -> Result<CertificateClosure, CertificateClosureError> {
    enum Work {
        Expression(ExprId),
        Program(ValueProgramId),
        Family(FamilyValueId),
    }

    let mut expressions = BTreeSet::new();
    let mut programs = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut work = Vec::new();
    match root {
        ProductionRoot::Closed(root) => work.push(Work::Expression(root.expression())),
        ProductionRoot::Family(family) => work.push(Work::Family(*family)),
    }

    while let Some(item) = work.pop() {
        match item {
            Work::Expression(expression) => {
                if !expressions.insert(expression) {
                    continue;
                }
                let node = job.expressions().node(expression)?;
                let inputs = node.inputs.clone();
                let program = match &node.operator {
                    ValueOperator::ProgramCall { program } => Some(*program),
                    _ => None,
                };
                work.extend(inputs.into_iter().map(Work::Expression));
                if let Some(program) = program {
                    work.push(Work::Program(program));
                }
            }
            Work::Program(program) => {
                if !programs.insert(program) {
                    continue;
                }
                let record = job.programs().program(program)?;
                let root = record.root;
                if let Some(family) = job.programs().family_for_program(program) {
                    work.push(Work::Family(family));
                }
                work.push(Work::Expression(root));
            }
            Work::Family(family) => {
                if !families.insert(family) {
                    continue;
                }
                let program = family.program();
                let program_root = job.programs().program(program)?.root;
                let family_body = job.programs().family_body(family)?;
                if family_body != program_root {
                    return Err(CertificateClosureError::FamilyProgramMismatch {
                        family,
                        program,
                        family_body,
                        program_root,
                    });
                }
                work.push(Work::Program(program));
            }
        }
    }

    Ok(CertificateClosure { expressions, programs, families })
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
}

/// Re-runs target resolution and production lowering only when certificate emission is explicitly
/// requested. Normal simulation entry points never call this function.
pub(crate) fn project_operational_certificate(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<OperationalCertificateProjection, CertificateProjectionError> {
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
    let (job, roots) = ProductionAdapter::new(protocol, &plan, parameters)
        .map_err(|error| CertificateProjectionError::Lowering { detail: error.to_string() })?
        .lower()
        .map_err(|error| CertificateProjectionError::Lowering { detail: error.to_string() })?;
    let residual = project_residual_root(&job, &roots.residual, &target)?;
    let closure = collect_residual_closure(&job, &roots.residual)?;
    Ok(OperationalCertificateProjection {
        target_id,
        plaintext_modulus,
        ciphertext_modulus: target.ciphertext_modulus,
        residual,
        closure,
    })
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
        let projection = project_operational_certificate(&protocol, &request)
            .expect("valid threshold certificate projection");
        assert_eq!(projection.target_id, "certificate-threshold");
        assert_eq!(projection.plaintext_modulus, 2_u8.into());
        assert_eq!(projection.ciphertext_modulus, 256_u16.into());
        let CertificateResidualRoot::Closed { matrix, .. } = projection.residual else {
            panic!("certificate residual should be the closed production residual root")
        };
        assert_eq!(matrix.modulus, 256_u16.into());
        assert_eq!(matrix.ring_dimension, 1);
        assert!(!projection.closure.expressions.is_empty());
        assert!(projection.closure.programs.is_empty());
        assert!(projection.closure.families.is_empty());
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
        let closure =
            collect_residual_closure(&job, &super::super::lower::ProductionRoot::Closed(root))
                .expect("closed residual closure");
        assert_eq!(closure.expressions.len(), 3);
        assert!(closure.expressions.contains(&root_expression));
        assert!(closure.expressions.contains(&left));
        assert!(closure.expressions.contains(&right));
        assert!(closure.programs.is_empty());
        assert!(closure.families.is_empty());
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
        let closure =
            collect_residual_closure(&job, &super::super::lower::ProductionRoot::Family(family))
                .expect("family residual closure");
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
        let closure =
            collect_residual_closure(&job, &super::super::lower::ProductionRoot::Closed(root))
                .expect("program-call residual closure");
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
        let roots = super::super::lower::ProductionRoots {
            residual: super::super::lower::ProductionRoot::Closed(residual),
            decoder: super::super::lower::ProductionRoot::Closed(decoder),
            occurrences: 0,
            samples: 0,
        };
        let closure = collect_residual_closure(&job, &roots.residual).expect("residual closure");
        assert!(closure.expressions.contains(&residual.expression()));
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
        assert!(matches!(
            collect_residual_closure(&target, &super::super::lower::ProductionRoot::Closed(root),),
            Err(CertificateClosureError::Arena(
                super::super::arena::ArenaError::ForeignExpression { .. }
            ))
        ));
    }
}
