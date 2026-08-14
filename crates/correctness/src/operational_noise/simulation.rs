//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The graph-specific stages remain injected here: this driver owns the one deadline,
//! cumulative allocation budget, diagnostics, and progress cadence for a checker job.
//! It deliberately does not manufacture a bound when lowering or relation rewriting is
//! unavailable.

use super::{
    OperationalAcceptanceReport, OperationalSimulationDiagnostics, OperationalSimulationReport,
    analysis::MxxAnalysis,
    bound::BoundEvaluator,
    error::{
        CheckerPhase, OperationalSimulationError, ResourceLimitKind, ResourceObserved, TargetError,
    },
    extract::{ExtractionControl, ProposalNodeClassification, extract_best_proposal},
    lower::{GraphLowerer, LoweredValue},
    relation::{RelationApplier, RelationSearcher, RewriteContext, SharedRewriteBudget},
};
use crate::{OperationalDecoderKind, ProtocolDecl, StageId};
use egg::{EGraph, Rewrite, Runner};
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, ParamEnv, WireRef, WireType};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

const PROGRESS_WORK_CADENCE: u64 = 4_096;
const PROGRESS_TIME_CADENCE: Duration = Duration::from_secs(1);

/// Fixed private production ceilings for one complete checker job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckerLimits {
    pub iteration_limit: usize,
    pub node_limit: usize,
    pub total_owned_element_limit: usize,
    pub total_time_limit: Duration,
    pub relation_sources_per_eclass: usize,
    pub switch_case_limit: usize,
    pub recurrence_step_limit: BigUint,
    pub max_integer_bits: BigUint,
}

impl CheckerLimits {
    pub(crate) fn production() -> Self {
        Self {
            iteration_limit: 32,
            node_limit: 2_000_000,
            total_owned_element_limit: 2_000_000,
            total_time_limit: Duration::from_secs(120),
            relation_sources_per_eclass: 64,
            switch_case_limit: 65_536,
            recurrence_step_limit: BigUint::from(10_000_000_u32),
            max_integer_bits: BigUint::from(16_777_216_u32),
        }
    }
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

/// One structured, allocation-free progress record supplied to the caller's logging backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgressEvent {
    pub phase: CheckerPhase,
    pub event: ProgressEventKind,
    pub processed: u64,
    pub total_or_discovered: Option<u64>,
    pub elapsed_ms: u64,
    pub egraph_nodes: Option<u64>,
    pub program: Option<String>,
    pub scope: Option<String>,
    pub node: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressEventKind {
    Start,
    Progress,
    Complete,
}

#[derive(Clone, Debug)]
struct ProgressState {
    processed: u64,
    last_emitted: Instant,
    next_work_threshold: u64,
}

impl ProgressState {
    fn new(started: Instant) -> Self {
        Self { processed: 0, last_emitted: started, next_work_threshold: PROGRESS_WORK_CADENCE }
    }
}

/// One job's mutable controls.  Stage implementations receive this instead of constructing
/// phase-local deadlines or resource counters.
pub(crate) struct SimulationControl<'a> {
    limits: &'a CheckerLimits,
    started: Instant,
    deadline: Instant,
    phase: CheckerPhase,
    owned_elements: Arc<AtomicUsize>,
    diagnostics: OperationalSimulationDiagnostics,
    progress: ProgressState,
    progress_site: Option<(String, String, u64)>,
    emit_progress: &'a mut dyn FnMut(ProgressEvent),
}

impl<'a> SimulationControl<'a> {
    fn new(limits: &'a CheckerLimits, emit_progress: &'a mut dyn FnMut(ProgressEvent)) -> Self {
        let started = Instant::now();
        Self {
            limits,
            started,
            deadline: started + limits.total_time_limit,
            phase: CheckerPhase::Target,
            owned_elements: Arc::new(AtomicUsize::new(0)),
            diagnostics: OperationalSimulationDiagnostics::default(),
            progress: ProgressState::new(started),
            progress_site: None,
            emit_progress,
        }
    }

    pub(crate) fn diagnostics_mut(&mut self) -> &mut OperationalSimulationDiagnostics {
        &mut self.diagnostics
    }

    /// Attaches the currently processed graph occurrence to cadence events.
    /// Callers update it at node/scope boundaries; it never requires a scan.
    pub(crate) fn set_progress_site(&mut self, program: String, scope: String, node: u64) {
        self.progress_site = Some((program, scope, node));
    }

    fn rewrite_budget(&self) -> SharedRewriteBudget {
        SharedRewriteBudget::from_shared(
            self.deadline,
            self.limits.total_owned_element_limit,
            Arc::clone(&self.owned_elements),
        )
    }

    pub(crate) fn check_deadline(&mut self) -> Result<(), OperationalSimulationError> {
        let now = Instant::now();
        if now >= self.deadline {
            return Err(self.resource_error(
                ResourceLimitKind::TotalTime,
                ResourceObserved::Duration {
                    limit: self.limits.total_time_limit,
                    observed: now.duration_since(self.started),
                },
            ));
        }
        Ok(())
    }

    pub(crate) fn reserve_owned_elements(
        &mut self,
        requested: usize,
    ) -> Result<(), OperationalSimulationError> {
        self.check_deadline()?;
        let mut current = self.owned_elements.load(Ordering::Relaxed);
        loop {
            let observed = current.checked_add(requested).unwrap_or(usize::MAX);
            if observed > self.limits.total_owned_element_limit {
                return Err(self.resource_error(
                    ResourceLimitKind::TotalOwnedElements,
                    ResourceObserved::Counter {
                        limit: self.limits.total_owned_element_limit.min(u64::MAX as usize) as u64,
                        observed: observed.min(u64::MAX as usize) as u64,
                    },
                ));
            }
            match self.owned_elements.compare_exchange_weak(
                current,
                observed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(next) => current = next,
            }
        }
    }

    pub(crate) fn check_egraph_nodes(
        &mut self,
        observed: usize,
    ) -> Result<(), OperationalSimulationError> {
        self.check_counter(ResourceLimitKind::EGraphNodes, self.limits.node_limit, observed)
    }

    pub(crate) fn check_rewrite_iterations(
        &mut self,
        observed: usize,
    ) -> Result<(), OperationalSimulationError> {
        self.check_counter(
            ResourceLimitKind::RewriteIterations,
            self.limits.iteration_limit,
            observed,
        )
    }

    pub(crate) fn check_integer_bits(
        &mut self,
        value: &BigUint,
        operation: impl Into<String>,
    ) -> Result<(), OperationalSimulationError> {
        self.check_deadline()?;
        let observed = BigUint::from(value.bits());
        if observed > self.limits.max_integer_bits {
            return Err(self.resource_error(
                ResourceLimitKind::IntegerBits,
                ResourceObserved::IntegerBits {
                    limit: self.limits.max_integer_bits.clone(),
                    observed,
                    operation: operation.into(),
                },
            ));
        }
        Ok(())
    }

    /// Counts one node, edge, branch, iteration, or bound-work boundary and emits at the
    /// required work/time cadence without scanning any checker-owned collection.
    pub(crate) fn work(
        &mut self,
        units: u64,
        total_or_discovered: Option<u64>,
        egraph_nodes: Option<u64>,
    ) -> Result<(), OperationalSimulationError> {
        self.check_deadline()?;
        self.progress.processed = self.progress.processed.saturating_add(units);
        let now = Instant::now();
        if self.progress.processed >= self.progress.next_work_threshold ||
            now.duration_since(self.progress.last_emitted) >= PROGRESS_TIME_CADENCE
        {
            self.emit(ProgressEventKind::Progress, total_or_discovered, egraph_nodes, now);
            self.progress.last_emitted = now;
            self.progress.next_work_threshold =
                self.progress.processed.saturating_add(PROGRESS_WORK_CADENCE);
        }
        Ok(())
    }

    fn begin_phase(&mut self, phase: CheckerPhase) -> Result<Instant, OperationalSimulationError> {
        self.phase = phase;
        self.progress = ProgressState::new(Instant::now());
        self.check_deadline()?;
        self.emit(ProgressEventKind::Start, None, None, Instant::now());
        Ok(Instant::now())
    }

    fn complete_phase(
        &mut self,
        phase_started: Instant,
        total_or_discovered: Option<u64>,
        egraph_nodes: Option<u64>,
    ) -> Result<Duration, OperationalSimulationError> {
        self.check_deadline()?;
        let now = Instant::now();
        self.emit(ProgressEventKind::Complete, total_or_discovered, egraph_nodes, now);
        Ok(now.duration_since(phase_started))
    }

    fn check_counter(
        &mut self,
        kind: ResourceLimitKind,
        limit: usize,
        observed: usize,
    ) -> Result<(), OperationalSimulationError> {
        self.check_deadline()?;
        if observed > limit {
            return Err(self.resource_error(
                kind,
                ResourceObserved::Counter {
                    limit: limit.min(u64::MAX as usize) as u64,
                    observed: observed.min(u64::MAX as usize) as u64,
                },
            ));
        }
        Ok(())
    }

    fn emit(
        &mut self,
        event: ProgressEventKind,
        total_or_discovered: Option<u64>,
        egraph_nodes: Option<u64>,
        now: Instant,
    ) {
        (self.emit_progress)(ProgressEvent {
            phase: self.phase,
            event,
            processed: self.progress.processed,
            total_or_discovered,
            elapsed_ms: now.duration_since(self.started).as_millis() as u64,
            egraph_nodes,
            program: self.progress_site.as_ref().map(|(program, _, _)| program.clone()),
            scope: self.progress_site.as_ref().map(|(_, scope, _)| scope.clone()),
            node: self.progress_site.as_ref().map(|(_, _, node)| *node),
        });
    }

    fn resource_error(
        &self,
        kind: ResourceLimitKind,
        observed: ResourceObserved,
    ) -> OperationalSimulationError {
        OperationalSimulationError::ResourceLimitExceeded {
            phase: self.phase,
            kind,
            observed,
            diagnostics: self.diagnostics.clone(),
        }
    }

    fn enrich_error(&self, error: OperationalSimulationError) -> OperationalSimulationError {
        match error {
            OperationalSimulationError::ResourceLimitExceeded { phase, kind, observed, .. } => {
                OperationalSimulationError::ResourceLimitExceeded {
                    phase,
                    kind,
                    observed,
                    diagnostics: self.diagnostics.clone(),
                }
            }
            error => error,
        }
    }
}

/// Checks one closed protocol candidate using the production lowering,
/// relation, extraction and bound implementations.  Unsupported graph facts
/// remain typed errors; this entry point has no legacy or heuristic fallback.
pub fn check_operational_noise_candidate(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let (target, stage, wire) = resolve_target(protocol, request)?;
    let limits = CheckerLimits::production();
    let mut emit = |_| {};
    check_with_limits(
        target,
        &limits,
        &mut emit,
        |control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let mut lowerer = GraphLowerer::new(protocol, request, MxxAnalysis::default());
            let value = lowerer.lower_stage_wire(&stage, wire).map_err(|source| {
                OperationalSimulationError::Lower { site: site(&stage, wire, "lower"), source }
            })?;
            control.work(
                lowerer.lowered_wire_count() as u64,
                None,
                Some(lowerer.egraph.total_size() as u64),
            )?;
            control.diagnostics_mut().lowered_term_count = lowerer.lowered_wire_count() as u64;
            let LoweredValue::Term(root) = value else {
                return Err(OperationalSimulationError::Lower {
                    site: site(&stage, wire, "lower residual"),
                    source: super::error::LowerError::FamilyProducerNotResolved { family: wire },
                });
            };
            Ok((lowerer, root, stage.clone(), wire))
        },
        |(mut lowerer, root, stage, wire), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let registrations = lowerer.relation_registrations();
            let budget = control.rewrite_budget();
            let context = RewriteContext::new(budget);
            for registration in registrations {
                context.register(registration);
            }
            let rewrite = Rewrite::new(
                "operational-relation",
                RelationSearcher::new(context.clone()),
                RelationApplier::new(context.clone()),
            )
            .expect("closed relation rewrite name");
            let runner = Runner::default()
                .with_egraph(std::mem::take(&mut lowerer.egraph))
                .with_iter_limit(limits.iteration_limit)
                .with_node_limit(limits.node_limit)
                .with_time_limit(limits.total_time_limit)
                .run(&[rewrite]);
            lowerer.egraph = runner.egraph;
            control.check_rewrite_iterations(runner.iterations.len())?;
            control.check_egraph_nodes(lowerer.egraph.total_size())?;
            control.work(
                runner.iterations.len() as u64,
                Some(limits.iteration_limit as u64),
                Some(lowerer.egraph.total_size() as u64),
            )?;
            let counters = context.counters();
            control.diagnostics_mut().rewrite_iteration_count = runner.iterations.len() as u64;
            control.diagnostics_mut().egraph_node_count = lowerer.egraph.total_size() as u64;
            control.diagnostics_mut().egraph_class_count =
                lowerer.egraph.number_of_classes() as u64;
            control.diagnostics_mut().relation_candidate_count = counters.candidates;
            control.diagnostics_mut().relation_rewrite_count = counters.rewrites;
            if let Some(failure) = context.failure() {
                return Err(relation_error(&stage, wire, failure));
            }
            Ok((lowerer, root, stage, wire))
        },
        |(mut lowerer, root, stage, wire), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let control = RefCell::new(control);
            let mut check_node_count = |count| control.borrow_mut().check_egraph_nodes(count);
            let mut reserve = |count| control.borrow_mut().reserve_owned_elements(count);
            let mut deadline = || control.borrow_mut().check_deadline();
            let mut invalid_dag = |_| OperationalSimulationError::Lower {
                site: site(&stage, wire, "extract"),
                source: super::error::LowerError::CyclicGraphDependency { wire },
            };
            let proposal = extract_best_proposal(
                &lowerer.egraph,
                root,
                &mut ExtractionControl {
                    check_node_count: &mut check_node_count,
                    reserve_owned_elements: &mut reserve,
                    check_deadline: &mut deadline,
                    invalid_dag: &mut invalid_dag,
                },
                &mut |_, _, _| Ok(ProposalNodeClassification::default()),
            )?;
            control.borrow_mut().work(
                proposal.expression.as_ref().len() as u64,
                None,
                Some(lowerer.egraph.total_size() as u64),
            )?;
            control.borrow_mut().diagnostics_mut().final_term_count =
                proposal.expression.as_ref().len() as u64;
            let mut extracted_egraph = EGraph::new(lowerer.egraph.analysis.clone());
            let extracted_root = extracted_egraph.add_expr(&proposal.expression);
            lowerer.egraph = extracted_egraph;
            Ok((lowerer, extracted_root, stage, wire))
        },
        |(lowerer, root, stage, wire), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let view = lowerer.production_bound_view();
            let result = BoundEvaluator::new(&view).evaluate(root).map_err(|_| {
                OperationalSimulationError::Bound {
                    site: site(&stage, wire, "bound"),
                    source: super::error::BoundError::BoundExpressionNotEvaluable {
                        expression: IntExpr::constant(0),
                    },
                }
            })?;
            let bound =
                result.coefficient_class.maximum_absolute_coefficient().ok_or_else(|| {
                    OperationalSimulationError::Bound {
                        site: site(&stage, wire, "bound"),
                        source: super::error::BoundError::BoundExpressionNotEvaluable {
                            expression: IntExpr::constant(0),
                        },
                    }
                })?;
            control.work(1, None, None)?;
            Ok(bound)
        },
    )
}

fn resolve_target(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
) -> Result<(ResolvedAcceptanceTarget, StageId, WireRef), OperationalSimulationError> {
    let matches = protocol
        .bundle
        .operational_decoder_targets
        .iter()
        .filter(|candidate| candidate.target_id == request.target_id)
        .collect::<Vec<_>>();
    let [target] = matches.as_slice() else {
        return Err(OperationalSimulationError::Target(if matches.is_empty() {
            TargetError::MissingTargetId { target_id: request.target_id.clone() }
        } else {
            TargetError::DuplicateTargetId {
                target_id: request.target_id.clone(),
                declarations: Box::new([]),
            }
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
        actual => {
            return Err(OperationalSimulationError::Target(TargetError::InvalidResidualSort {
                target_id: target.target_id.clone(),
                residual: super::error::StageOutputRef {
                    stage: target.residual_stage.clone(),
                    output: target.residual_output.clone(),
                },
                actual: actual.clone(),
            }))
        }
    };
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
    Ok((
        ResolvedAcceptanceTarget { target_id: target.target_id.clone(), ciphertext_modulus, kind },
        stage.id.clone(),
        wire,
    ))
}

fn site(stage: &StageId, wire: WireRef, operation: &str) -> super::error::ErrorSite {
    super::error::ErrorSite {
        program: super::identity::ProgramKey::WorkflowStage(stage.clone()),
        scope_definition: FrozenGraphScopeId::Root,
        occurrence_path: Box::new([]),
        node: wire.node,
        output_port: Some(wire.port.0),
        operation: operation.to_owned(),
    }
}

fn relation_error(
    stage: &StageId,
    wire: WireRef,
    _: super::relation::RelationFailure,
) -> OperationalSimulationError {
    OperationalSimulationError::Relation {
        site: site(stage, wire, "relation rewrite"),
        source: super::error::RelationError::TransformedRelationOperand {
            operand: super::identity::WireSourceKey {
                scope: super::identity::OccurrenceScope {
                    program: super::identity::ProgramKey::WorkflowStage(stage.clone()),
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire,
            },
        },
    }
}

/// Drives the real lower/rewrite/extract/bound stages under one job-wide resource policy.
///
/// It is module-private because production callers use [`CheckerLimits::production`]; tests
/// inject small limits to exercise each typed resource boundary.  The callbacks own the actual
/// graph semantics and must call [`SimulationControl::work`] at their existing work boundaries.
pub(crate) fn check_with_limits<Lowered, Rewritten, Extracted>(
    target: ResolvedAcceptanceTarget,
    limits: &CheckerLimits,
    emit_progress: &mut dyn FnMut(ProgressEvent),
    mut lower: impl FnMut(&mut SimulationControl<'_>) -> Result<Lowered, OperationalSimulationError>,
    mut rewrite: impl FnMut(
        Lowered,
        &mut SimulationControl<'_>,
    ) -> Result<Rewritten, OperationalSimulationError>,
    mut extract: impl FnMut(
        Rewritten,
        &mut SimulationControl<'_>,
    ) -> Result<Extracted, OperationalSimulationError>,
    mut bound: impl FnMut(
        Extracted,
        &mut SimulationControl<'_>,
    ) -> Result<BigUint, OperationalSimulationError>,
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let mut control = SimulationControl::new(limits, emit_progress);

    let phase_started = control.begin_phase(CheckerPhase::Lower)?;
    let lowered = lower(&mut control).map_err(|error| control.enrich_error(error))?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.lowering_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Rewrite)?;
    let rewritten = rewrite(lowered, &mut control).map_err(|error| control.enrich_error(error))?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.rewrite_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Extract)?;
    let extracted =
        extract(rewritten, &mut control).map_err(|error| control.enrich_error(error))?;
    control.complete_phase(phase_started, None, None)?;

    let phase_started = control.begin_phase(CheckerPhase::Bound)?;
    let noise_bound =
        bound(extracted, &mut control).map_err(|error| control.enrich_error(error))?;
    control.check_integer_bits(&noise_bound, "final noise bound")?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.bound_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Acceptance)?;
    let (accepted, acceptance) = check_acceptance(&target, &noise_bound, &mut control)?;
    control.complete_phase(phase_started, None, None)?;
    control.diagnostics.total_milliseconds = control.started.elapsed().as_millis() as u64;

    Ok(OperationalSimulationReport {
        target_id: target.target_id,
        noise_bound,
        ciphertext_modulus: target.ciphertext_modulus,
        accepted,
        acceptance,
        diagnostics: control.diagnostics,
    })
}

fn check_acceptance(
    target: &ResolvedAcceptanceTarget,
    noise_bound: &BigUint,
    control: &mut SimulationControl<'_>,
) -> Result<(bool, OperationalAcceptanceReport), OperationalSimulationError> {
    control.check_deadline()?;
    match &target.kind {
        ResolvedDecoderKind::Threshold { plaintext_modulus } => {
            let threshold_left = BigUint::from(2_u8) * plaintext_modulus * noise_bound;
            control.check_integer_bits(&threshold_left, "decoder threshold left side")?;
            let margin = BigInt::from(target.ciphertext_modulus.clone()) -
                BigInt::from(threshold_left.clone());
            let accepted = threshold_left < target.ciphertext_modulus;
            Ok((
                accepted,
                OperationalAcceptanceReport::Threshold {
                    plaintext_modulus: plaintext_modulus.clone(),
                    threshold_left,
                    margin,
                },
            ))
        }
        ResolvedDecoderKind::BooleanInterval => {
            if target.ciphertext_modulus < BigUint::from(4_u8) {
                return Err(OperationalSimulationError::Target(
                    TargetError::BooleanIntervalModulusBelowFour {
                        target_id: target.target_id.clone(),
                        actual: BigInt::from(target.ciphertext_modulus.clone()),
                    },
                ));
            }
            let q = BigInt::from(target.ciphertext_modulus.clone());
            let noise = BigInt::from(noise_bound.clone());
            let quarter = shared_round_div(&(&q - BigInt::from(2_u8)), &BigInt::from(4_u8));
            let half = &q / BigInt::from(2_u8);
            let false_lower_margin = &quarter - &noise;
            let false_upper_margin = &q - (BigInt::from(3_u8) * &quarter + &noise);
            let true_lower_margin = &half - (&quarter + &noise);
            let true_upper_margin = BigInt::from(3_u8) * &quarter - (&half + &noise);
            let accepted = false_lower_margin > BigInt::zero() &&
                false_upper_margin > BigInt::zero() &&
                true_lower_margin >= BigInt::zero() &&
                true_upper_margin >= BigInt::zero();
            Ok((
                accepted,
                OperationalAcceptanceReport::BooleanInterval {
                    quarter,
                    false_lower_margin,
                    false_upper_margin,
                    true_lower_margin,
                    true_upper_margin,
                },
            ))
        }
    }
}

/// Delegates the tie rule to Graph IR rather than selecting a Rust division convention here.
fn shared_round_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    IntExpr::RoundDiv(
        Box::new(IntExpr::constant(numerator.clone())),
        Box::new(IntExpr::constant(denominator.clone())),
    )
    .evaluate(&ParamEnv::default())
    .expect("positive constant RoundDiv denominator")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boolean_target(q: u32) -> ResolvedAcceptanceTarget {
        ResolvedAcceptanceTarget {
            target_id: "interval".to_owned(),
            ciphertext_modulus: q.into(),
            kind: ResolvedDecoderKind::BooleanInterval,
        }
    }

    fn run_acceptance(target: ResolvedAcceptanceTarget, noise: u32) -> OperationalSimulationReport {
        check_with_limits(
            target,
            &CheckerLimits::production(),
            &mut |_| {},
            |_| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(noise.into()),
        )
        .expect("acceptance result")
    }

    #[test]
    fn production_limits_match_the_fixed_resource_policy() {
        assert_eq!(
            CheckerLimits::production(),
            CheckerLimits {
                iteration_limit: 32,
                node_limit: 2_000_000,
                total_owned_element_limit: 2_000_000,
                total_time_limit: Duration::from_secs(120),
                relation_sources_per_eclass: 64,
                switch_case_limit: 65_536,
                recurrence_step_limit: 10_000_000_u32.into(),
                max_integer_bits: 16_777_216_u32.into(),
            }
        );
    }

    #[test]
    fn boolean_interval_uses_graph_round_division_for_every_q_mod_four() {
        for (q, expected_quarter) in [(16, 4), (17, 4), (18, 4), (19, 4), (29, 7)] {
            let report = run_acceptance(boolean_target(q), 0);
            let OperationalAcceptanceReport::BooleanInterval { quarter, .. } = report.acceptance
            else {
                panic!("boolean target must report interval acceptance");
            };
            assert_eq!(quarter, BigInt::from(expected_quarter));
        }
    }

    #[test]
    fn boolean_interval_preserves_strict_and_nonstrict_inequalities() {
        let rejected = run_acceptance(boolean_target(17), 4);
        assert!(!rejected.accepted, "N == quarter must fail the strict lower condition");
        let accepted = run_acceptance(boolean_target(17), 3);
        assert!(accepted.accepted);
    }

    #[test]
    fn threshold_rejects_equality() {
        let report = run_acceptance(
            ResolvedAcceptanceTarget {
                target_id: "threshold".to_owned(),
                ciphertext_modulus: 12_u8.into(),
                kind: ResolvedDecoderKind::Threshold { plaintext_modulus: 2_u8.into() },
            },
            3,
        );
        assert!(!report.accepted);
        assert!(matches!(
            report.acceptance,
            OperationalAcceptanceReport::Threshold { margin, .. } if margin.is_zero()
        ));
    }

    #[test]
    fn progress_emits_at_the_shared_work_cadence() {
        let mut events = Vec::new();
        let report = check_with_limits(
            boolean_target(17),
            &CheckerLimits::production(),
            &mut |event| events.push(event),
            |control| {
                control.work(PROGRESS_WORK_CADENCE - 1, Some(PROGRESS_WORK_CADENCE), None)?;
                control.work(1, Some(PROGRESS_WORK_CADENCE), None)?;
                Ok(())
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(0_u8.into()),
        )
        .expect("successful simulation");
        assert!(report.accepted);
        assert!(events.iter().any(|event| {
            event.phase == CheckerPhase::Lower &&
                event.event == ProgressEventKind::Progress &&
                event.processed == PROGRESS_WORK_CADENCE
        }));
    }

    #[test]
    fn cumulative_owned_budget_is_not_refunded_between_allocations() {
        let mut limits = CheckerLimits::production();
        limits.total_owned_element_limit = 2;
        let error = check_with_limits(
            boolean_target(17),
            &limits,
            &mut |_| {},
            |control| {
                control.reserve_owned_elements(1)?;
                control.reserve_owned_elements(1)?;
                control.reserve_owned_elements(1)?;
                Ok(())
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(0_u8.into()),
        )
        .expect_err("third reservation must exceed the job-wide budget");
        assert!(matches!(
            error,
            OperationalSimulationError::ResourceLimitExceeded {
                phase: CheckerPhase::Lower,
                kind: ResourceLimitKind::TotalOwnedElements,
                observed: ResourceObserved::Counter { limit: 2, observed: 3 },
                ..
            }
        ));
    }
}
