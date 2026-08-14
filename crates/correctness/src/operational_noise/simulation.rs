//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The graph-specific stages remain injected here: this driver owns diagnostics and the
//! progress cadence for a checker job.
//! It deliberately does not manufacture a bound when lowering or relation rewriting is
//! unavailable.

use super::{
    OperationalAcceptanceReport, OperationalSimulationDiagnostics, OperationalSimulationReport,
    analysis::{MxxAnalysis, ResourceBudget},
    bound::{BoundEvaluationControl, BoundEvaluationError, BoundEvaluator},
    error::{OperationalSimulationError, TargetError},
    extract::{ExtractionControl, ProposalNodeClassification, extract_best_proposal},
    language::MxxLang,
    lower::{GraphLowerer, LoweredValue, LoweringControl},
    relation::{RelationApplier, RelationSearcher, RewriteContext, SharedRewriteBudget},
};
use crate::{OperationalDecoderKind, ProtocolDecl, StageId};
use egg::{EGraph, Language, Rewrite, Runner, StopReason};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, IntExpr, ParamEnv, Port, WireRef, WireType,
    node::{IntBinaryOp, IntCompareOp, NodeKind},
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
use tracing::{debug, error, info};

const PROGRESS_TIME_CADENCE: Duration = Duration::from_secs(1);

/// Logical checker phase reported by progress events and diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckerPhase {
    Target,
    Lower,
    Rewrite,
    Extract,
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

/// One structured, allocation-free progress record supplied at instrumented checker boundaries.
/// Progress is best-effort: opaque library operations such as an individual egg scan may run
/// between events, so this is not a wall-clock heartbeat guarantee.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressEvent {
    pub phase: CheckerPhase,
    pub event: ProgressEventKind,
    pub processed: u64,
    pub total_or_discovered: Option<u64>,
    pub elapsed_ms: u64,
    pub egraph_nodes: Option<u64>,
    pub owned_elements: u64,
    pub rewrite_iterations: u64,
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

/// Read-only bound callbacks still update the one job's progress owner through
/// this interior-mutability bridge.
struct BoundControlAdapter<'a, 'control> {
    control: RefCell<&'a mut SimulationControl<'control>>,
}

/// Shares the production progress owner with graph lowering.
struct LoweringControlAdapter<'a, 'control> {
    control: &'a mut SimulationControl<'control>,
}

impl LoweringControl for LoweringControlAdapter<'_, '_> {
    fn work(
        &mut self,
        scope: &super::identity::OccurrenceScope,
        node: mxx_ir_core::NodeId,
    ) -> Result<(), super::error::LowerError> {
        self.control.set_progress_site(
            format!("{:?}", scope.program),
            format!("{:?}", scope.definition),
            node.0 as u64,
        );
        let _ = self.control.work(1, None, None);
        Ok(())
    }
}

impl BoundEvaluationControl for BoundControlAdapter<'_, '_> {
    fn validate_pack(&self, term: egg::Id, _bit_count: usize) -> Result<(), BoundEvaluationError> {
        self.control
            .borrow_mut()
            .work(1, None, None)
            .map_err(|_| BoundEvaluationError::InvalidPack { term })
    }
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

    pub(crate) fn diagnostics_mut(&mut self) -> &mut OperationalSimulationDiagnostics {
        &mut self.diagnostics
    }

    /// Attaches the currently processed graph occurrence to cadence events.
    /// Callers update it at node/scope boundaries; it never requires a scan.
    pub(crate) fn set_progress_site(&mut self, program: String, scope: String, node: u64) {
        self.progress_site = Some((program, scope, node));
    }

    fn rewrite_budget(&self) -> SharedRewriteBudget {
        SharedRewriteBudget::from_shared(Arc::clone(&self.owned_elements))
    }

    fn analysis_budget(&self) -> ResourceBudget {
        ResourceBudget::from_shared(Arc::clone(&self.owned_elements))
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
        egraph_nodes: Option<u64>,
    ) -> Result<(), OperationalSimulationError> {
        self.progress.processed = self.progress.processed.saturating_add(units);
        let now = Instant::now();
        if now.duration_since(self.progress.last_emitted) >= PROGRESS_TIME_CADENCE {
            self.emit(ProgressEventKind::Progress, total_or_discovered, egraph_nodes, now);
            self.progress.last_emitted = now;
        }
        Ok(())
    }

    fn begin_phase(&mut self, phase: CheckerPhase) -> Result<Instant, OperationalSimulationError> {
        self.phase = phase;
        self.progress = ProgressState::new(Instant::now());
        self.emit(ProgressEventKind::Start, None, None, Instant::now());
        Ok(Instant::now())
    }

    fn complete_phase(
        &mut self,
        phase_started: Instant,
        total_or_discovered: Option<u64>,
        egraph_nodes: Option<u64>,
    ) -> Result<Duration, OperationalSimulationError> {
        let now = Instant::now();
        self.emit(ProgressEventKind::Complete, total_or_discovered, egraph_nodes, now);
        Ok(now.duration_since(phase_started))
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
            owned_elements: self.owned_elements.load(Ordering::Relaxed) as u64,
            rewrite_iterations: self.diagnostics.rewrite_iteration_count,
            program: self.progress_site.as_ref().map(|(program, _, _)| program.clone()),
            scope: self.progress_site.as_ref().map(|(_, scope, _)| scope.clone()),
            node: self.progress_site.as_ref().map(|(_, _, node)| *node),
        });
    }
}

/// Checks one closed protocol candidate using the production lowering,
/// relation, extraction and bound implementations.  Unsupported graph facts
/// remain typed errors; this entry point has no legacy or heuristic fallback.
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
                egraph_nodes = ?event.egraph_nodes,
                rewrite_iterations = event.rewrite_iterations,
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
                egraph_nodes = ?event.egraph_nodes,
                rewrite_iterations = event.rewrite_iterations,
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
    let (target, stage, wire) = resolve_target(protocol, request, &mut control)?;
    control.work(1, None, None)?;
    control.complete_phase(target_started, None, None)?;
    check_with_control(
        target,
        &mut control,
        |control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let analysis =
                MxxAnalysis::with_resource_budget(Default::default(), control.analysis_budget());
            let mut lowering_control = LoweringControlAdapter { control };
            let mut lowerer =
                GraphLowerer::new_with_control(protocol, request, analysis, &mut lowering_control);
            let lowered = lowerer.lower_stage_wire(&stage, wire);
            let lowerer = lowerer.into_uncontrolled();
            let value = match lowered {
                Ok(value) => value,
                Err(source) => {
                    return Err(OperationalSimulationError::Lower {
                        site: site(&stage, wire, "lower"),
                        source,
                    });
                }
            };
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
                        super::family::FamilyCoverageStorage::ExactStored { elements } => elements,
                        super::family::FamilyCoverageStorage::SharedTemplate {
                            domain,
                            representative,
                            binder_domains,
                        } => {
                            validate_shared_representative(
                                &lowerer.egraph,
                                representative,
                                &binder_domains,
                                control,
                                &stage,
                                wire,
                                domain.logical_count.clone(),
                            )?;
                            control.reserve_owned_elements(1)?;
                            vec![representative].into_boxed_slice()
                        }
                    }
                }
                LoweredValue::Trapdoor(_) | LoweredValue::TrapdoorFamily { .. } => {
                    return Err(OperationalSimulationError::Lower {
                        site: site(&stage, wire, "lower residual"),
                        source: super::error::LowerError::FamilyProducerNotResolved {
                            family: wire,
                        },
                    });
                }
            };
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
                .with_iter_limit(usize::MAX)
                .with_node_limit(usize::MAX)
                .with_time_limit(Duration::MAX)
                .run(&[rewrite]);
            match runner.stop_reason.as_ref() {
                Some(StopReason::Saturated) => {}
                Some(reason) => {
                    return Err(OperationalSimulationError::Relation {
                        site: site(&stage, wire, "relation rewrite"),
                        source: super::error::RelationError::RewriteDidNotSaturate {
                            reason: format!(
                                "internal inconsistency: explicitly unbounded egg runner stopped with {reason:?}"
                            ),
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
            control.work(
                runner.iterations.len() as u64,
                None,
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
                    &mut ExtractionControl { invalid_dag: &mut invalid_dag },
                    &mut |_, node, egraph| {
                        let (relation_redex, large_atom_witness) =
                            super::relation::classify_proposal_node(egraph, node, &context)
                                .map_err(|failure| {
                                    relation_error(&stage, wire, &egraph.analysis.symbols, failure)
                                })?;
                        Ok(ProposalNodeClassification {
                            relation_redex,
                            large_atom: large_atom_witness.is_some(),
                            large_atom_witness,
                        })
                    },
                )?;
                if proposal.cost.remaining_relation_redexes != 0 ||
                    proposal.cost.hidden_relation_redexes != 0 ||
                    proposal.cost.large_atom_count != 0
                {
                    return Err(OperationalSimulationError::Bound {
                        site: site(&stage, wire, "extract residual"),
                        source: super::error::BoundError::UnresolvedExtraction {
                            remaining_relation_redexes: proposal.cost.remaining_relation_redexes,
                            hidden_relation_redexes: proposal.cost.hidden_relation_redexes,
                            large_atom_count: proposal.cost.large_atom_count,
                            large_atom_witness: proposal.large_atom_witness.and_then(|source| {
                                lowerer
                                    .egraph
                                    .analysis
                                    .symbols
                                    .atomic_sources
                                    .get(source.0)
                                    .map(|descriptor| descriptor.key.clone())
                            }),
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
                let result = match BoundEvaluator::new(&view).evaluate(root) {
                    Ok(result) => result,
                    Err(source) => {
                        return Err(OperationalSimulationError::Bound {
                            site: site(&stage, wire, "bound"),
                            source: super::error::BoundError::EvaluationFailed { source },
                        });
                    }
                };
                let maximum = result
                    .coefficient_class
                    .maximum_absolute_coefficient()
                    .expect("BoundEvaluator rejects final Large roots");
                bound = bound.max(maximum);
            }
            drop(view);
            drop(bound_control);
            control.work(1, None, None)?;
            Ok(bound)
        },
    )
}

/// Validates the compact representative once. `BoundInput::scalar_maximum_absolute`
/// consumes each retained affine scalar's interval, so evaluating this matrix
/// root already selects the worst binder endpoint without enumerating lanes.
fn validate_shared_representative(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    representative: egg::Id,
    binder_domains: &[super::family::CoverageBinderDomain],
    control: &mut SimulationControl<'_>,
    stage: &StageId,
    wire: WireRef,
    logical_count: BigUint,
) -> Result<(), OperationalSimulationError> {
    let mut pending = vec![egraph.find(representative)];
    let mut visited = std::collections::BTreeSet::new();
    while let Some(term) = pending.pop() {
        let term = egraph.find(term);
        if !visited.insert(term) {
            continue;
        }
        control.reserve_owned_elements(1)?;
        control.work(1, None, Some(egraph.total_size() as u64))?;
        for node in &egraph[term].nodes {
            if let MxxLang::MatrixScale([scalar, _]) = node {
                let scalar = egraph.find(*scalar);
                let scalar_domain =
                    egraph[scalar].data.integer_domain.as_ref().ok_or_else(|| {
                        OperationalSimulationError::Bound {
                            site: site(stage, wire, "prove shared family maximum"),
                            source: super::error::BoundError::SharedFamilyMaximumNotProved {
                                count: logical_count.clone(),
                            },
                        }
                    })?;
                let relevant = match scalar_domain {
                    super::analysis::IntegerDomain::Affine { binders, .. } => binder_domains
                        .iter()
                        .filter(|retained| binders.contains_key(&retained.binder))
                        .cloned()
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                super::family::shared_affine_maximum(scalar_domain, &relevant).map_err(|_| {
                    OperationalSimulationError::Bound {
                        site: site(stage, wire, "prove shared family maximum"),
                        source: super::error::BoundError::SharedFamilyMaximumNotProved {
                            count: logical_count.clone(),
                        },
                    }
                })?;
            }
            pending.extend(node.children().iter().map(|child| egraph.find(*child)));
        }
    }
    Ok(())
}

fn resolve_target(
    protocol: &ProtocolDecl,
    request: &super::OperationalCheckRequest,
    control: &mut SimulationControl<'_>,
) -> Result<(ResolvedAcceptanceTarget, StageId, WireRef), OperationalSimulationError> {
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
    Ok((
        ResolvedAcceptanceTarget { target_id: target.target_id.clone(), ciphertext_modulus, kind },
        stage.id.clone(),
        wire,
    ))
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
    };
    OperationalSimulationError::Relation { site: site(stage, wire, "relation rewrite"), source }
}

/// Drives injected stages with the same progress and diagnostics owner used in production.
#[cfg(test)]
pub(crate) fn check_with_test_control<Lowered, Rewritten, Extracted>(
    target: ResolvedAcceptanceTarget,
    emit_progress: &mut dyn FnMut(ProgressEvent),
    lower: impl FnMut(&mut SimulationControl<'_>) -> Result<Lowered, OperationalSimulationError>,
    rewrite: impl FnMut(
        Lowered,
        &mut SimulationControl<'_>,
    ) -> Result<Rewritten, OperationalSimulationError>,
    extract: impl FnMut(
        Rewritten,
        &mut SimulationControl<'_>,
    ) -> Result<Extracted, OperationalSimulationError>,
    bound: impl FnMut(
        Extracted,
        &mut SimulationControl<'_>,
    ) -> Result<BigUint, OperationalSimulationError>,
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let mut control = SimulationControl::new(emit_progress);
    check_with_control(target, &mut control, lower, rewrite, extract, bound)
}

fn check_with_control<Lowered, Rewritten, Extracted>(
    target: ResolvedAcceptanceTarget,
    control: &mut SimulationControl<'_>,
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
    let phase_started = control.begin_phase(CheckerPhase::Lower)?;
    let lowered = lower(control)?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.lowering_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Rewrite)?;
    let rewritten = rewrite(lowered, control)?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.rewrite_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Extract)?;
    let extracted = extract(rewritten, control)?;
    control.complete_phase(phase_started, None, None)?;

    let phase_started = control.begin_phase(CheckerPhase::Bound)?;
    let noise_bound = bound(extracted, control)?;
    let elapsed = control.complete_phase(phase_started, None, None)?;
    control.diagnostics.bound_milliseconds = elapsed.as_millis() as u64;

    let phase_started = control.begin_phase(CheckerPhase::Acceptance)?;
    let (accepted, acceptance) = check_acceptance(&target, &noise_bound)?;
    control.complete_phase(phase_started, None, None)?;
    control.diagnostics.total_milliseconds = control.started.elapsed().as_millis() as u64;

    Ok(OperationalSimulationReport {
        target_id: target.target_id,
        noise_bound,
        ciphertext_modulus: target.ciphertext_modulus,
        accepted,
        acceptance,
        diagnostics: control.diagnostics.clone(),
    })
}

fn check_acceptance(
    target: &ResolvedAcceptanceTarget,
    noise_bound: &BigUint,
) -> Result<(bool, OperationalAcceptanceReport), OperationalSimulationError> {
    match &target.kind {
        ResolvedDecoderKind::Threshold { plaintext_modulus } => {
            let threshold_left = BigUint::from(2_u8) * plaintext_modulus * noise_bound;
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
        check_with_test_control(
            target,
            &mut |_| {},
            |_| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_, _| Ok(noise.into()),
        )
        .expect("acceptance result")
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
