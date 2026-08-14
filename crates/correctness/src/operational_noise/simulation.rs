//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The graph-specific stages remain injected here: this driver owns the one deadline,
//! cumulative allocation budget, diagnostics, and progress cadence for a checker job.
//! It deliberately does not manufacture a bound when lowering or relation rewriting is
//! unavailable.

use super::{
    OperationalAcceptanceReport, OperationalSimulationDiagnostics, OperationalSimulationReport,
    analysis::{MxxAnalysis, ResourceBudget},
    bound::{BoundEvaluationControl, BoundEvaluationError, BoundEvaluator},
    error::{
        CheckerPhase, OperationalSimulationError, ResourceLimitKind, ResourceObserved, TargetError,
    },
    extract::{ExtractionControl, ProposalNodeClassification, extract_best_proposal},
    lower::{GraphLowerer, LoweredValue, LoweringControl},
    relation::{RelationApplier, RelationSearcher, RewriteContext, SharedRewriteBudget},
};
use crate::{OperationalDecoderKind, ProtocolDecl, StageId};
use egg::{EGraph, Rewrite, Runner, StopReason};
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, ParamEnv, WireRef, WireType,
    node::{IntCompareOp, NodeKind},
};
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

/// Read-only bound callbacks still mutate the one job control through this
/// interior-mutability bridge; no phase gets its own deadline or allocation
/// counter.
struct BoundControlAdapter<'a, 'limits> {
    control: RefCell<&'a mut SimulationControl<'limits>>,
}

/// Shares the production job limits with graph lowering without giving it a
/// phase-local clock or allocation counter.
struct LoweringControlAdapter {
    deadline: Instant,
    owned_elements: Arc<AtomicUsize>,
    total_owned_element_limit: usize,
}

impl LoweringControl for LoweringControlAdapter {
    fn check_deadline(&self) -> Result<(), super::error::LowerError> {
        if Instant::now() >= self.deadline {
            return Err(super::error::LowerError::ResourceDeadlineExceeded);
        }
        Ok(())
    }

    fn reserve_owned_elements(&self, requested: usize) -> Result<(), super::error::LowerError> {
        self.check_deadline()?;
        let mut current = self.owned_elements.load(Ordering::Relaxed);
        loop {
            let observed = current.checked_add(requested).unwrap_or(usize::MAX);
            if observed > self.total_owned_element_limit {
                return Err(super::error::LowerError::ResourceAllocationExceeded { requested });
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
}

impl BoundEvaluationControl for BoundControlAdapter<'_, '_> {
    fn check_deadline(&self) -> Result<(), BoundEvaluationError> {
        self.control.borrow_mut().check_deadline().map_err(|_| {
            BoundEvaluationError::IntegerLimitExceeded {
                operation: "bound deadline",
                value: BigUint::zero(),
            }
        })
    }

    fn reserve_owned_elements(&self, requested: usize) -> Result<(), BoundEvaluationError> {
        self.control.borrow_mut().reserve_owned_elements(requested).map_err(|_| {
            BoundEvaluationError::IntegerLimitExceeded {
                operation: "bound allocation",
                value: BigUint::from(requested),
            }
        })
    }

    fn validate_integer_bits(
        &self,
        value: &BigUint,
        operation: &'static str,
    ) -> Result<(), BoundEvaluationError> {
        self.control.borrow_mut().check_integer_bits(value, operation).map_err(|_| {
            BoundEvaluationError::IntegerBitLimitExceeded {
                operation,
                bits: BigUint::from(value.bits()),
            }
        })
    }

    fn validate_pack(&self, term: egg::Id, _bit_count: usize) -> Result<(), BoundEvaluationError> {
        self.check_deadline().map_err(|_| BoundEvaluationError::InvalidPack { term })
    }

    fn recurrence_step_limit(&self) -> Option<BigUint> {
        Some(self.control.borrow().limits.recurrence_step_limit.clone())
    }
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

    fn lowering_control(&self) -> Arc<dyn LoweringControl> {
        Arc::new(LoweringControlAdapter {
            deadline: self.deadline,
            owned_elements: Arc::clone(&self.owned_elements),
            total_owned_element_limit: self.limits.total_owned_element_limit,
        })
    }

    fn analysis_budget(&self) -> ResourceBudget {
        ResourceBudget::from_shared(
            self.limits.total_owned_element_limit,
            Arc::clone(&self.owned_elements),
            self.started,
            self.deadline,
            self.limits.total_time_limit,
        )
    }

    fn check_switch_cases(&mut self, observed: usize) -> Result<(), OperationalSimulationError> {
        self.check_counter(ResourceLimitKind::SwitchCases, self.limits.switch_case_limit, observed)
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
    request
        .validate(protocol.params.iter().map(|parameter| parameter.name.clone()))
        .map_err(OperationalSimulationError::Request)?;
    let (target, stage, wire) = resolve_target(protocol, request)?;
    let limits = CheckerLimits::production();
    let mut emit = |_| {};
    check_with_limits(
        target,
        &limits,
        &mut emit,
        |control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let analysis = MxxAnalysis::with_resource_budget(
                Default::default(),
                limits.relation_sources_per_eclass,
                limits.switch_case_limit,
                control.analysis_budget(),
            );
            let mut lowerer = GraphLowerer::new_with_control(
                protocol,
                request,
                analysis,
                control.lowering_control(),
            );
            let value = lowerer.lower_stage_wire(&stage, wire).map_err(|source| {
                OperationalSimulationError::Lower { site: site(&stage, wire, "lower"), source }
            })?;
            control.work(
                lowerer.lowered_wire_count() as u64,
                None,
                Some(lowerer.egraph.total_size() as u64),
            )?;
            control.diagnostics_mut().lowered_term_count = lowerer.lowered_wire_count() as u64;
            let roots = match value {
                LoweredValue::Term(root) => {
                    control.reserve_owned_elements(1)?;
                    vec![root].into_boxed_slice()
                }
                LoweredValue::Family(family) => {
                    family.validate().map_err(|_| OperationalSimulationError::Lower {
                        site: site(&stage, wire, "validate residual family"),
                        source: super::error::LowerError::InvalidFamilyCount {
                            count: IntExpr::constant(0),
                        },
                    })?;
                    match family.storage {
                        super::family::FamilyCoverageStorage::ExactStored { elements } => {
                            control.check_switch_cases(elements.len())?;
                            elements
                        }
                        super::family::FamilyCoverageStorage::SharedTemplate { domain, .. } => {
                            return Err(OperationalSimulationError::Bound {
                                site: site(&stage, wire, "prove shared family maximum"),
                                source: super::error::BoundError::SharedFamilyMaximumNotProved {
                                    count: domain.logical_count,
                                },
                            });
                        }
                    }
                }
                LoweredValue::Trapdoor(_) => {
                    return Err(OperationalSimulationError::Lower {
                        site: site(&stage, wire, "lower residual"),
                        source: super::error::LowerError::FamilyProducerNotResolved {
                            family: wire,
                        },
                    });
                }
            };
            if let (Some(kind), Some(observed)) = (
                lowerer.egraph.analysis.resource_failure_kind.clone(),
                lowerer.egraph.analysis.resource_failure.clone(),
            ) {
                return Err(control.resource_error(kind, observed));
            }
            Ok((lowerer, roots, stage.clone(), wire))
        },
        |(mut lowerer, roots, stage, wire), control| {
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
            match runner.stop_reason.as_ref() {
                Some(StopReason::Saturated) => {}
                Some(StopReason::IterationLimit(limit)) => {
                    control.check_rewrite_iterations(limit.saturating_add(1))?;
                }
                Some(StopReason::NodeLimit(limit)) => {
                    control.check_egraph_nodes(limit.saturating_add(1))?
                }
                Some(StopReason::TimeLimit(_)) => control.check_deadline()?,
                Some(StopReason::Other(reason)) => {
                    return Err(OperationalSimulationError::Relation {
                        site: site(&stage, wire, "relation rewrite"),
                        source: super::error::RelationError::RewriteDidNotSaturate {
                            reason: reason.clone(),
                        },
                    });
                }
                None => {
                    return Err(OperationalSimulationError::Relation {
                        site: site(&stage, wire, "relation rewrite"),
                        source: super::error::RelationError::RewriteDidNotSaturate {
                            reason: "missing stop reason".to_owned(),
                        },
                    });
                }
            }
            lowerer.egraph = runner.egraph;
            if let (Some(kind), Some(observed)) = (
                lowerer.egraph.analysis.resource_failure_kind.clone(),
                lowerer.egraph.analysis.resource_failure.clone(),
            ) {
                return Err(control.resource_error(kind, observed));
            }
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
                return Err(match failure {
                    super::relation::RelationFailure::DeadlineExceeded => control.resource_error(
                        ResourceLimitKind::TotalTime,
                        ResourceObserved::Duration {
                            limit: limits.total_time_limit,
                            observed: control.started.elapsed(),
                        },
                    ),
                    super::relation::RelationFailure::OwnedElementLimitExceeded => {
                        let observed = control.owned_elements.load(Ordering::Relaxed);
                        control.resource_error(
                            ResourceLimitKind::TotalOwnedElements,
                            ResourceObserved::Counter {
                                limit: limits.total_owned_element_limit as u64,
                                observed: observed.saturating_add(1) as u64,
                            },
                        )
                    }
                    failure => {
                        relation_error(&stage, wire, &lowerer.egraph.analysis.symbols, failure)
                    }
                });
            }
            Ok((lowerer, roots, stage, wire, context))
        },
        |(mut lowerer, roots, stage, wire, context), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let control = RefCell::new(control);
            let mut check_node_count = |count| control.borrow_mut().check_egraph_nodes(count);
            let mut reserve = |count| control.borrow_mut().reserve_owned_elements(count);
            let mut deadline = || control.borrow_mut().check_deadline();
            let mut invalid_dag = |_| OperationalSimulationError::Lower {
                site: site(&stage, wire, "extract"),
                source: super::error::LowerError::CyclicGraphDependency { wire },
            };
            control.borrow_mut().reserve_owned_elements(roots.len())?;
            let mut proposals = Vec::with_capacity(roots.len());
            for root in roots {
                let proposal = extract_best_proposal(
                    &lowerer.egraph,
                    root,
                    &mut ExtractionControl {
                        check_node_count: &mut check_node_count,
                        reserve_owned_elements: &mut reserve,
                        check_deadline: &mut deadline,
                        invalid_dag: &mut invalid_dag,
                    },
                    &mut |_, node, egraph| {
                        let (relation_redex, large_atom) =
                            super::relation::classify_proposal_node(egraph, node, &context)
                                .map_err(|failure| {
                                    relation_error(&stage, wire, &egraph.analysis.symbols, failure)
                                })?;
                        Ok(ProposalNodeClassification { relation_redex, large_atom })
                    },
                )?;
                if proposal.cost.remaining_relation_redexes != 0 ||
                    proposal.cost.hidden_relation_redexes != 0 ||
                    proposal.cost.large_atom_count != 0
                {
                    return Err(OperationalSimulationError::Bound {
                        site: site(&stage, wire, "extract residual"),
                        source: super::error::BoundError::BoundExpressionNotEvaluable {
                            expression: IntExpr::constant(0),
                        },
                    });
                }
                proposals.push(proposal);
            }
            let term_count =
                proposals.iter().map(|proposal| proposal.expression.as_ref().len()).sum::<usize>();
            control.borrow_mut().work(
                term_count as u64,
                None,
                Some(lowerer.egraph.total_size() as u64),
            )?;
            control.borrow_mut().diagnostics_mut().final_term_count = term_count as u64;
            let mut extracted_egraph = EGraph::new(lowerer.egraph.analysis.clone());
            control.borrow_mut().reserve_owned_elements(proposals.len())?;
            let extracted_roots = proposals
                .iter()
                .map(|proposal| extracted_egraph.add_expr(&proposal.expression))
                .collect::<Vec<_>>();
            lowerer.egraph = extracted_egraph;
            Ok((lowerer, extracted_roots, stage, wire))
        },
        |(lowerer, roots, stage, wire), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let bound_control = BoundControlAdapter { control: RefCell::new(control) };
            let view = lowerer.production_bound_view_with_control(&bound_control);
            let mut bound = BigUint::zero();
            for root in roots {
                let result = BoundEvaluator::new(&view).evaluate(root).map_err(|_| {
                    OperationalSimulationError::Bound {
                        site: site(&stage, wire, "bound"),
                        source: super::error::BoundError::BoundExpressionNotEvaluable {
                            expression: IntExpr::constant(0),
                        },
                    }
                })?;
                let maximum =
                    result.coefficient_class.maximum_absolute_coefficient().ok_or_else(|| {
                        OperationalSimulationError::Bound {
                            site: site(&stage, wire, "bound"),
                            source: super::error::BoundError::BoundExpressionNotEvaluable {
                                expression: IntExpr::constant(0),
                            },
                        }
                    })?;
                bound = bound.max(maximum);
            }
            drop(view);
            drop(bound_control);
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
                declarations: matches
                    .iter()
                    .map(|target| super::error::TargetDeclarationSite {
                        target_id: target.target_id.clone(),
                        residual: stage_output(target),
                        decoder: super::error::DecoderWireRef {
                            stage: target.decoder_stage.clone(),
                            node: target.decoder_node,
                            port: 0,
                        },
                    })
                    .collect(),
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
    if target.decoder_stage != target.residual_stage ||
        !arguments.iter().copied().any(|input| wire_consumes(&decoder_stage.graph, input, wire))
    {
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

fn wire_consumes(graph: &mxx_ir_core::Graph, current: WireRef, target: WireRef) -> bool {
    let mut pending = vec![current];
    let mut visited = Vec::new();
    while let Some(wire) = pending.pop() {
        if wire == target {
            return true;
        }
        if visited.contains(&wire) {
            continue;
        }
        visited.push(wire);
        let Some(node) = graph.root_scope().node(wire.node) else {
            continue;
        };
        if let Some(arguments) = graph.root_scope().arguments(node) {
            pending.extend(arguments.iter().copied());
        }
    }
    false
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
    symbols: &super::identity::SymbolTables,
    failure: super::relation::RelationFailure,
) -> OperationalSimulationError {
    let key = |source: super::identity::AtomicSourceId| {
        symbols.atomic_sources.get(source.0).map(|descriptor| descriptor.key.clone())
    };
    let source = match failure {
        super::relation::RelationFailure::MissingRegistration { source } => key(source)
            .map(|source| super::error::RelationError::MissingRelationRegistration { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::InvalidRelationProducer { source } => key(source)
            .map(|source| super::error::RelationError::InvalidRelationSource { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedIndex { source } => key(source)
            .map(|source| super::error::RelationError::MismatchedRelationIndices { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedType { source } => key(source)
            .map(|source| super::error::RelationError::RelationTypeMismatch { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedLayout { source } => key(source)
            .map(|source| super::error::RelationError::RelationLayoutMismatch { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedPublic { source } => key(source)
            .map(|source| super::error::RelationError::RelationPublicMismatch { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedTrapdoor { source } => key(source)
            .map(|source| super::error::RelationError::RelationTrapdoorMismatch { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::MismatchedTarget { source } => key(source)
            .map(|source| super::error::RelationError::RelationTargetMismatch { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::UnavailableRelation { source } => key(source)
            .map(|source| super::error::RelationError::SmallDecompositionRangeNotProved { source })
            .unwrap_or(super::error::RelationError::UnknownRelationSource { source }),
        super::relation::RelationFailure::AmbiguousReplacement { sources } => {
            let mut candidates = Vec::with_capacity(sources.len());
            for source in sources {
                let Some(source) = key(source) else {
                    return OperationalSimulationError::Relation {
                        site: site(stage, wire, "relation rewrite"),
                        source: super::error::RelationError::UnknownRelationSource { source },
                    };
                };
                candidates.push(source);
            }
            super::error::RelationError::AmbiguousRelationSource { candidates: candidates.into() }
        }
        super::relation::RelationFailure::DifferentSelectorBlocked => {
            super::error::RelationError::BlockedRelationRewrite {
                reason: super::error::RelationRewriteBlockReason::DifferentSelector,
            }
        }
        super::relation::RelationFailure::TransformedOperand => {
            super::error::RelationError::BlockedRelationRewrite {
                reason: super::error::RelationRewriteBlockReason::TransformedOperand,
            }
        }
        super::relation::RelationFailure::InvalidAdditiveSort { expression } => {
            super::error::RelationError::InvalidAdditiveRelationSort {
                expression: usize::from(expression),
            }
        }
        super::relation::RelationFailure::DeadlineExceeded |
        super::relation::RelationFailure::OwnedElementLimitExceeded => unreachable!(
            "relation resource failures are mapped through the shared simulation control"
        ),
    };
    OperationalSimulationError::Relation { site: site(stage, wire, "relation rewrite"), source }
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
        let error = resolve_target(&protocol, &request)
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

    #[test]
    fn analysis_uses_the_same_cumulative_counter_as_the_driver() {
        let mut limits = CheckerLimits::production();
        limits.total_owned_element_limit = 2;
        let error = check_with_limits(
            boolean_target(17),
            &limits,
            &mut |_| {},
            |control| {
                control.reserve_owned_elements(1)?;
                let analysis = MxxAnalysis::with_resource_budget(
                    Default::default(),
                    limits.relation_sources_per_eclass,
                    limits.switch_case_limit,
                    control.analysis_budget(),
                );
                let mut egraph = EGraph::new(analysis);
                let left = egraph.add(super::super::language::MxxLang::IntConst(1.into()));
                let right = egraph.add(super::super::language::MxxLang::IntConst(2.into()));
                let _ = egraph
                    .add(super::super::language::MxxLang::MatrixAdd(vec![left, right].into()));
                let kind = egraph.analysis.resource_failure_kind.clone().expect("resource kind");
                let observed = egraph.analysis.resource_failure.clone().expect("resource value");
                Err(control.resource_error(kind, observed))
            },
            |_: (), _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(0_u8.into()),
        )
        .expect_err("analysis must observe the driver's earlier reservation");
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

    #[test]
    fn analysis_uses_the_driver_deadline_and_switch_limit() {
        let mut limits = CheckerLimits::production();
        limits.switch_case_limit = 1;
        let mut emit = |_| {};
        let mut control = SimulationControl::new(&limits, &mut emit);
        let analysis = MxxAnalysis::with_resource_budget(
            Default::default(),
            limits.relation_sources_per_eclass,
            limits.switch_case_limit,
            control.analysis_budget(),
        );
        let mut egraph = EGraph::new(analysis);
        let selector = egraph.add(super::super::language::MxxLang::IntConst(0.into()));
        let first = egraph.add(super::super::language::MxxLang::IntConst(1.into()));
        let second = egraph.add(super::super::language::MxxLang::IntConst(2.into()));
        let _ = egraph
            .add(super::super::language::MxxLang::Switch(vec![selector, first, second].into()));
        assert_eq!(egraph.analysis.resource_failure_kind, Some(ResourceLimitKind::SwitchCases));
        assert_eq!(
            egraph.analysis.resource_failure,
            Some(ResourceObserved::Counter { limit: 1, observed: 2 })
        );

        control.deadline = Instant::now();
        let analysis = MxxAnalysis::with_resource_budget(
            Default::default(),
            limits.relation_sources_per_eclass,
            limits.switch_case_limit,
            control.analysis_budget(),
        );
        let mut egraph = EGraph::new(analysis);
        let _ = egraph.add(super::super::language::MxxLang::IntConst(0.into()));
        assert_eq!(egraph.analysis.resource_failure_kind, Some(ResourceLimitKind::TotalTime));
        assert!(matches!(
            egraph.analysis.resource_failure,
            Some(ResourceObserved::Duration { .. })
        ));
    }

    #[test]
    fn exact_family_case_count_uses_the_production_switch_limit() {
        let mut limits = CheckerLimits::production();
        limits.switch_case_limit = 1;
        let error = check_with_limits(
            boolean_target(17),
            &limits,
            &mut |_| {},
            |control| {
                control.check_switch_cases(2)?;
                Ok(())
            },
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(0_u8.into()),
        )
        .expect_err("two family cases exceed the configured production limit");
        assert!(matches!(
            error,
            OperationalSimulationError::ResourceLimitExceeded {
                phase: CheckerPhase::Lower,
                kind: ResourceLimitKind::SwitchCases,
                observed: ResourceObserved::Counter { limit: 1, observed: 2 },
                ..
            }
        ));
    }

    #[test]
    fn relation_diagnostic_uses_the_registered_source_instead_of_the_residual_root() {
        let mut symbols = super::super::identity::SymbolTables::default();
        let source =
            symbols.atomic_sources.intern(super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from("relation-source"),
                ),
                sort: super::super::analysis::MxxSort::Int,
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            });
        let error = relation_error(
            &StageId("stage".to_owned()),
            WireRef { node: mxx_ir_core::NodeId(9), port: mxx_ir_core::Port(0) },
            &symbols,
            super::super::relation::RelationFailure::MissingRegistration {
                source: super::super::identity::AtomicSourceId(source),
            },
        );
        assert!(matches!(
            error,
            OperationalSimulationError::Relation {
                source: super::super::error::RelationError::MissingRelationRegistration {
                    source: super::super::identity::AtomicSourceKey::ProtocolInput(id),
                },
                ..
            } if id == crate::ProtocolInputId::from("relation-source")
        ));
    }
}
