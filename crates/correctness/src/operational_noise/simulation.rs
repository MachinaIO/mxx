//! Shared execution controls and exact decoder acceptance for operational simulation.
//!
//! The graph-specific stages remain injected here: this driver owns diagnostics and the
//! progress cadence for a checker job.
//! It deliberately does not manufacture a bound when lowering or relation rewriting is
//! unavailable.

use super::{
    OperationalAcceptanceReport, OperationalSimulationDiagnostics, OperationalSimulationReport,
    analysis::{MxxAnalysis, ResourceBudget},
    bound::BoundEvaluationError,
    error::{OperationalSimulationError, TargetError},
    extract::{
        ExtractedProposal, ExtractedProposalWithOrigins, ExtractionControl, ProposalCost,
        ProposalNodeClassification, evaluate_exact_selected_expression,
        extract_best_proposal_with_origins_and_structural_preference,
        extract_best_proposals_with_origins,
    },
    language::MxxLang,
    lower::{GraphLowerer, LoweredValue, LoweringControl},
    relation::{
        RelationApplier, RelationFailure, RelationRegistration, RelationSearcher, ReplacementPlan,
        RewriteContext, SharedRewriteBudget, classify_proposal_node,
        materialize_selected_polynomial_redex, replacement_plan_bounded_shape,
        replacement_plan_satisfied, selected_polynomial_monomials_with_context,
        selected_polynomial_normal_form_plan, selected_polynomial_redexes_mut,
    },
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
    collections::{BTreeSet, HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tracing::{debug, error, info};

const PROGRESS_TIME_CADENCE: Duration = Duration::from_secs(1);
const SHALLOW_ENODE_CHILD_LOG_LIMIT: usize = 16;
const EXPLICIT_LARGE_PRODUCT_FACTOR_LOG_LIMIT: usize = 32;

fn replacement_plan_existing_leaves(
    plan: &ReplacementPlan,
    leaves: &mut BTreeSet<egg::Id>,
) -> bool {
    let mut pending = vec![plan];
    while let Some(plan) = pending.pop() {
        match plan {
            ReplacementPlan::Existing(term) => {
                leaves.insert(*term);
            }
            ReplacementPlan::Product(children) | ReplacementPlan::Add(children) => {
                pending.extend(children.iter())
            }
            ReplacementPlan::Negate(child) => pending.push(child),
            ReplacementPlan::Switch(_) |
            ReplacementPlan::Concat { .. } |
            ReplacementPlan::Equivalent(_) => return false,
        }
    }
    true
}

fn append_exact_expression(
    output: &mut egg::RecExpr<MxxLang>,
    origins: &mut Vec<egg::Id>,
    extracted: &ExtractedProposalWithOrigins,
) -> egg::Id {
    let mut remap = Vec::with_capacity(extracted.proposal.expression.as_ref().len());
    for (index, node) in extracted.proposal.expression.as_ref().iter().enumerate() {
        let node = node.clone().map_children(|child| remap[usize::from(child)]);
        remap.push(output.add(node));
        origins.push(extracted.origins[index]);
    }
    *remap.last().expect("an extracted expression is nonempty")
}

/// Expands the shallow signed-polynomial recipe produced by
/// `signed_spines_replacement_plan` into one exact expression. Its only
/// recursive edges are Add/Negate/Product wrappers around Existing leaves;
/// Switches and other opaque operations remain inside an extracted leaf.
fn append_exact_final_plan(
    plan: &ReplacementPlan,
    composite_origin: egg::Id,
    output: &mut egg::RecExpr<MxxLang>,
    origins: &mut Vec<egg::Id>,
    first_large_source: &mut Option<super::identity::AtomicSourceId>,
    canonicalize_leaf: &mut dyn FnMut(egg::Id) -> egg::Id,
    unresolved: &mut dyn FnMut(u64, u64) -> OperationalSimulationError,
    extract_leaf: &mut dyn FnMut(
        egg::Id,
    ) -> Result<
        ExtractedProposalWithOrigins,
        OperationalSimulationError,
    >,
) -> Result<egg::Id, Option<OperationalSimulationError>> {
    let (node, children) = match plan {
        ReplacementPlan::Existing(term) => {
            let term = canonicalize_leaf(*term);
            let extracted = extract_leaf(term).map_err(Some)?;
            let relation = extracted.proposal.cost.unsatisfied_relation_redexes;
            let structural = extracted
                .proposal
                .cost
                .unsatisfied_structural_redexes
                .saturating_add(extracted.proposal.cost.hidden_structural_redexes);
            if relation != 0 || structural != 0 {
                return Err(Some(unresolved(relation, structural)));
            }
            if first_large_source.is_none() && extracted.proposal.cost.large_residual {
                *first_large_source = extracted.proposal.first_large_source;
            }
            return Ok(append_exact_expression(output, origins, &extracted));
        }
        ReplacementPlan::Product(children) => ("product", children.as_ref()),
        ReplacementPlan::Add(children) => ("add", children.as_ref()),
        ReplacementPlan::Negate(child) => {
            let child = append_exact_final_plan(
                child,
                composite_origin,
                output,
                origins,
                first_large_source,
                canonicalize_leaf,
                unresolved,
                extract_leaf,
            )?;
            let root = output.add(MxxLang::MatrixNegate([child]));
            origins.push(composite_origin);
            return Ok(root);
        }
        ReplacementPlan::Switch(_) |
        ReplacementPlan::Concat { .. } |
        ReplacementPlan::Equivalent(_) => {
            return Err(None);
        }
    };
    let children = children
        .iter()
        .map(|child| {
            append_exact_final_plan(
                child,
                composite_origin,
                output,
                origins,
                first_large_source,
                canonicalize_leaf,
                unresolved,
                extract_leaf,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let root = match node {
        "product" => output.add(MxxLang::MatrixMultiply(children.into_boxed_slice())),
        "add" => output.add(MxxLang::MatrixAdd(children.into_boxed_slice())),
        _ => unreachable!(),
    };
    origins.push(composite_origin);
    Ok(root)
}

/// Finds the nearest selected ordered product containing the requested atomic
/// source. Distances and the product spine are computed only from the already
/// selected `RecExpr`; multiplication is flattened without reordering.
fn nearest_selected_product_spine(
    expression: &egg::RecExpr<MxxLang>,
    target_source: super::identity::AtomicSourceId,
) -> Option<(usize, Vec<egg::Id>, usize)> {
    let mut source_distances = vec![None::<usize>; expression.as_ref().len()];
    let mut nearest_product = None::<(usize, usize)>;
    for (index, node) in expression.as_ref().iter().enumerate() {
        let distance = match node {
            MxxLang::Atom { source, .. } if *source == target_source => Some(0),
            _ => node
                .children()
                .iter()
                .filter_map(|child| source_distances[usize::from(*child)])
                .min()
                .map(|distance| distance.saturating_add(1)),
        };
        source_distances[index] = distance;
        if matches!(node, MxxLang::MatrixMultiply(_)) &&
            let Some(distance) = distance &&
            nearest_product.is_none_or(|current| (distance, index) < current)
        {
            nearest_product = Some((distance, index));
        }
    }

    let (_, product_index) = nearest_product?;
    let MxxLang::MatrixMultiply(factors) = &expression.as_ref()[product_index] else {
        unreachable!("nearest product index names a MatrixMultiply")
    };
    let mut pending = factors.iter().rev().copied().collect::<Vec<_>>();
    let mut spine = Vec::new();
    while let Some(factor) = pending.pop() {
        match &expression[factor] {
            MxxLang::MatrixMultiply(nested) => pending.extend(nested.iter().rev().copied()),
            _ => spine.push(factor),
        }
    }
    let source_position = spine.iter().position(|factor| {
        matches!(&expression[*factor], MxxLang::Atom { source, .. } if *source == target_source)
    })?;
    Some((product_index, spine, source_position))
}

fn direct_selected_atomic_source(
    expression: &egg::RecExpr<MxxLang>,
    factor: Option<egg::Id>,
) -> Option<super::identity::AtomicSourceId> {
    match factor.map(|factor| &expression[factor]) {
        Some(MxxLang::Atom { source, .. }) => Some(*source),
        _ => None,
    }
}

fn relation_canonical_ids(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    registrations: &[RelationRegistration],
) -> Vec<usize> {
    let relation_sources =
        registrations.iter().map(|registration| registration.source).collect::<BTreeSet<_>>();
    let mut canonical_ids = egraph
        .classes()
        .filter(|class| {
            class.nodes.iter().any(|node| {
                matches!(node, MxxLang::Atom { source, .. } if relation_sources.contains(source))
            })
        })
        .map(|class| usize::from(egraph.find(class.id)))
        .collect::<Vec<_>>();
    canonical_ids.sort_unstable();
    canonical_ids.dedup();
    canonical_ids
}

/// Emits bounded, failure-only context for an explicitly Large selected atom.
/// It reads the selected expression, its aligned origins, and the existing
/// relation registry; no result is persisted in checker state.
fn emit_explicit_large_product_context(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    expression: &egg::RecExpr<MxxLang>,
    origins: &[egg::Id],
    context: &RewriteContext,
    source: super::identity::AtomicSourceId,
) {
    if !tracing::enabled!(tracing::Level::INFO) {
        return;
    }
    let Some((product_index, spine, source_position)) =
        nearest_selected_product_spine(expression, source)
    else {
        tracing::info!(
            event = "operational_explicit_large_product_context",
            canonical_source_id = source.0,
            nearest_selected_product = false,
            "explicitly Large selected source has no selected MatrixMultiply ancestor"
        );
        return;
    };
    let source_factor = spine[source_position];
    let source_origin = egraph.find(origins[usize::from(source_factor)]);
    let left = source_position.checked_sub(1).map(|position| spine[position]);
    let right = spine.get(source_position + 1).copied();
    let left_source = direct_selected_atomic_source(expression, left);
    let right_source = direct_selected_atomic_source(expression, right);
    let source_role = |source: super::identity::AtomicSourceId| {
        egraph
            .analysis
            .symbols
            .atomic_sources
            .get(source.0)
            .and_then(|descriptor| descriptor.relation_role)
    };
    let registrations = context.diagnostic_registrations_for_expected_public(egraph, source_origin);
    let expected_relation_sources =
        registrations.iter().map(|registration| registration.source).collect::<BTreeSet<_>>();
    let expected_relation_canonical_ids = relation_canonical_ids(egraph, &registrations);
    let exact_adjacent_right_is_expected_relation =
        right_source.is_some_and(|right_source| expected_relation_sources.contains(&right_source));
    let ordered_canonical_factors = spine
        .iter()
        .take(EXPLICIT_LARGE_PRODUCT_FACTOR_LOG_LIMIT)
        .map(|factor| usize::from(egraph.find(origins[usize::from(*factor)])))
        .collect::<Vec<_>>();
    tracing::info!(
        event = "operational_explicit_large_product_context",
        canonical_source_id = source.0,
        canonical_source_eclass = usize::from(source_origin),
        nearest_selected_product = true,
        nearest_product_expression_index = product_index,
        factor_position = source_position,
        ordered_canonical_factors = ?ordered_canonical_factors,
        omitted_factor_count = spine.len().saturating_sub(EXPLICIT_LARGE_PRODUCT_FACTOR_LOG_LIMIT),
        immediate_left_operator = ?left.map(|factor| expression[factor].operator_name()),
        immediate_left_source_id = ?left_source.map(|source| source.0),
        immediate_left_relation_role = ?left_source.and_then(source_role),
        immediate_right_operator = ?right.map(|factor| expression[factor].operator_name()),
        immediate_right_source_id = ?right_source.map(|source| source.0),
        immediate_right_relation_role = ?right_source.and_then(source_role),
        expected_relation_registration_count = registrations.len(),
        expected_relation_canonical_ids = ?expected_relation_canonical_ids,
        expected_relation_canonical_count = expected_relation_canonical_ids.len(),
        exact_adjacent_right_is_expected_relation,
        "selected explicitly Large source in its nearest ordered product context"
    );
}

fn shallow_enode_log_shape(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    class: egg::Id,
) -> Option<(&'static str, Vec<usize>, usize)> {
    let node = egraph[class].nodes.first()?;
    let child_count = node.children().len();
    let children = node
        .children()
        .iter()
        .take(SHALLOW_ENODE_CHILD_LOG_LIMIT)
        .map(|child| usize::from(egraph.find(*child)))
        .collect();
    Some((
        node.operator_name(),
        children,
        child_count.saturating_sub(SHALLOW_ENODE_CHILD_LOG_LIMIT),
    ))
}

fn shallow_enode_directly_contains(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    class: egg::Id,
    target: egg::Id,
) -> bool {
    egraph[class]
        .nodes
        .first()
        .is_some_and(|node| node.children().iter().any(|child| egraph.find(*child) == target))
}

fn emit_detailed_large_diagnostic_for_epoch(epoch: u64) -> bool {
    epoch == 0 || epoch.is_power_of_two()
}

fn preference_only_reextract(redex_count: usize, preference_changed: bool) -> bool {
    redex_count == 0 && preference_changed
}

type ExactStructuralPreferences = HashMap<egg::Id, HashSet<MxxLang>>;

fn canonicalize_structural_preferences(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    preferences: &mut ExactStructuralPreferences,
) {
    let old = std::mem::take(preferences);
    for (class, nodes) in old {
        let class = egraph.find(class);
        for node in nodes {
            let node = node.map_children(|child| egraph.find(child));
            preferences.entry(class).or_default().insert(node);
        }
    }
}

fn commit_structural_preference_batch(
    preferences: &mut ExactStructuralPreferences,
    batch: ExactStructuralPreferences,
) -> bool {
    let mut changed = false;
    for (class, nodes) in batch {
        if preferences.get(&class) != Some(&nodes) {
            preferences.insert(class, nodes);
            changed = true;
        }
    }
    changed
}

fn preferred_batch_has_relation_redex(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    batch: &ExactStructuralPreferences,
    context: &RewriteContext,
) -> Result<bool, RelationFailure> {
    let mut has_relation_redex = false;
    for (origin, nodes) in batch {
        for node in nodes {
            has_relation_redex |=
                classify_proposal_node(egraph, *origin, node, context)?.relation_redex;
        }
    }
    Ok(has_relation_redex)
}

fn record_replacement_plan_preferences(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    plan: &ReplacementPlan,
    preferences: &mut ExactStructuralPreferences,
) -> Option<egg::Id> {
    let node = match plan {
        ReplacementPlan::Existing(id) => return Some(egraph.find(*id)),
        ReplacementPlan::Product(plans) => {
            let children = plans
                .iter()
                .map(|plan| record_replacement_plan_preferences(egraph, plan, preferences))
                .collect::<Option<Vec<_>>>()?;
            MxxLang::MatrixMultiply(children.into_boxed_slice())
        }
        ReplacementPlan::Add(plans) => {
            let children = plans
                .iter()
                .map(|plan| record_replacement_plan_preferences(egraph, plan, preferences))
                .collect::<Option<Vec<_>>>()?;
            MxxLang::MatrixAdd(children.into_boxed_slice())
        }
        ReplacementPlan::Negate(plan) => {
            let child = record_replacement_plan_preferences(egraph, plan, preferences)?;
            MxxLang::MatrixNegate([child])
        }
        ReplacementPlan::Switch(plans) => {
            let children = plans
                .iter()
                .map(|plan| record_replacement_plan_preferences(egraph, plan, preferences))
                .collect::<Option<Vec<_>>>()?;
            MxxLang::Switch(children.into_boxed_slice())
        }
        ReplacementPlan::Concat { axis, inputs } => {
            let children = inputs
                .iter()
                .map(|plan| record_replacement_plan_preferences(egraph, plan, preferences))
                .collect::<Option<Vec<_>>>()?;
            MxxLang::MatrixConcat { axis: *axis, inputs: children.into_boxed_slice() }
        }
        ReplacementPlan::Equivalent(plans) => {
            let mut roots = plans
                .iter()
                .map(|plan| record_replacement_plan_preferences(egraph, plan, preferences));
            let root = roots.next()??;
            return roots.try_fold(root, |root, candidate| {
                (egraph.find(candidate?) == egraph.find(root)).then_some(root)
            });
        }
    };
    let class = egraph.lookup(node.clone())?;
    let class = egraph.find(class);
    preferences.entry(class).or_default().insert(node);
    Some(class)
}

fn run_ordinary_relation_saturation(
    egraph: &mut EGraph<MxxLang, MxxAnalysis>,
    context: &RewriteContext,
) -> Result<usize, super::error::RelationError> {
    let rewrite = Rewrite::new(
        "operational-relation",
        RelationSearcher::new(context.clone()),
        RelationApplier::new(context.clone()),
    )
    .expect("closed relation rewrite name");
    let runner = Runner::default()
        .with_egraph(std::mem::take(egraph))
        .with_iter_limit(usize::MAX)
        .with_node_limit(usize::MAX)
        .with_time_limit(Duration::MAX)
        .run(&[rewrite]);
    let result = match runner.stop_reason.as_ref() {
        Some(StopReason::Saturated) => Ok(runner.iterations.len()),
        Some(reason) => Err(super::error::RelationError::RewriteDidNotSaturate {
            reason: format!(
                "internal inconsistency: explicitly unbounded egg runner stopped with {reason:?}"
            ),
        }),
        None => Err(super::error::RelationError::RewriteDidNotSaturate {
            reason: "missing stop reason".to_owned(),
        }),
    };
    *egraph = runner.egraph;
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectedPhasePostcondition {
    RestartOrdinary,
    Complete,
}

fn selected_phase_postcondition(
    selected_structural_unions: u64,
    unsatisfied_structural_redexes: u64,
    hidden_structural_redexes: u64,
) -> Result<SelectedPhasePostcondition, (u64, u64)> {
    if selected_structural_unions != 0 {
        Ok(SelectedPhasePostcondition::RestartOrdinary)
    } else if unsatisfied_structural_redexes == 0 && hidden_structural_redexes == 0 {
        Ok(SelectedPhasePostcondition::Complete)
    } else {
        Err((unsatisfied_structural_redexes, hidden_structural_redexes))
    }
}

/// Relation work is always staged before structural contraction. A structural
/// batch can expose an ordinary checked relation; in that case the selected
/// epoch ends without comparing unrelated structural measures, then ordinary
/// saturation rebuilds the available representation.
fn selected_relation_postcondition(
    selected_structural_unions: u64,
    unsatisfied_relation_redexes: u64,
) -> Result<Option<SelectedPhasePostcondition>, u64> {
    if unsatisfied_relation_redexes == 0 {
        Ok(None)
    } else if selected_structural_unions != 0 {
        Ok(Some(SelectedPhasePostcondition::RestartOrdinary))
    } else {
        Err(unsatisfied_relation_redexes)
    }
}

fn selected_relation_rewrite_delta(
    before: u64,
    after: u64,
    batch_size: usize,
) -> Result<u64, super::error::RelationError> {
    let delta = after.saturating_sub(before);
    (delta != 0)
        .then_some(delta)
        .ok_or(super::error::RelationError::SelectedNormalizationBatchDidNotUnion { batch_size })
}

fn accumulated_rewrite_iterations(current: u64, additional: usize) -> u64 {
    current.saturating_add(additional as u64)
}

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
            let iterations = run_ordinary_relation_saturation(&mut lowerer.egraph, &context)
                .map_err(|source| OperationalSimulationError::Relation {
                    site: site(&stage, wire, "relation rewrite"),
                    source,
                })?;
            control.work(iterations as u64, None, Some(lowerer.egraph.total_size() as u64))?;
            let counters = context.counters();
            control.diagnostics_mut().rewrite_iteration_count = iterations as u64;
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
            let mut bound_error = |source| OperationalSimulationError::Bound {
                site: site(&stage, wire, "extract semantic bound"),
                source: super::error::BoundError::EvaluationFailed { source },
            };
            let mut selected_roots =
                roots.iter().map(|root| lowerer.egraph.find(*root)).collect::<BTreeSet<_>>();
            // This is deliberately outside egg saturation: one deterministic
            // selected DAG is expanded to its canonical ordered polynomial,
            // including exact Switch scope hoisting from its stored cases.
            // This finite protocol makes no universal termination claim from
            // selected-redex counts: it freezes one selected snapshot,
            // materializes its complete batch, requires at least one new
            // union, rebuilds, and then extracts again.
            let mut normalization_epoch = 0_u64;
            let mut egraph_mutation_epoch = 0_u64;
            let mut structural_preferences = ExactStructuralPreferences::new();
            let (terminal_egraph_mutation_epoch, terminal_selected) = 'joint_fixed_point: loop {
                // Ordinary saturation may expose a fresh selected polynomial
                // or a Switch whose common terms or ordered factors can move
                // outside its stored cases. Recurrence is meaningful only
                // within one selected phase, so its epoch-local guard resets
                // at this boundary.
                let mut selected_phase_unions = 0_u64;
                loop {
                    let extraction_started = Instant::now();
                    let view = lowerer.production_bound_view();
                    let mut selected = Vec::with_capacity(selected_roots.len());
                    canonicalize_structural_preferences(
                        &lowerer.egraph,
                        &mut structural_preferences,
                    );
                    for root in &selected_roots {
                        let emit_detailed =
                            emit_detailed_large_diagnostic_for_epoch(normalization_epoch);
                        let mut diagnostic_preference_membership =
                            |origin: egg::Id, node: &MxxLang| {
                                structural_preferences
                                    .get(&origin)
                                    .is_some_and(|preferred| preferred.contains(node))
                            };
                        let proposal =
                            extract_best_proposal_with_origins_and_structural_preference(
                                &lowerer.egraph,
                                *root,
                                &view,
                                &mut ExtractionControl {
                                    invalid_dag: &mut invalid_dag,
                                    bound_error: &mut bound_error,
                                },
                                &mut |origin, node, egraph| {
                                    let classification = super::relation::classify_proposal_node(
                                        egraph, origin, node, &context,
                                    )
                                    .map_err(|failure| {
                                        relation_error(
                                            &stage,
                                            wire,
                                            &egraph.analysis.symbols,
                                            failure,
                                        )
                                    })?;
                                    Ok(ProposalNodeClassification {
                                        relation_redex: classification.relation_redex,
                                        local_checked_relation_count: classification
                                            .local_checked_relation_count,
                                    })
                                },
                                &mut |origin, node| {
                                    structural_preferences
                                        .get(&origin)
                                        .map_or(0, |preferred| u64::from(!preferred.contains(node)))
                                },
                                emit_detailed.then_some(
                                    &mut diagnostic_preference_membership
                                        as &mut dyn FnMut(egg::Id, &MxxLang) -> bool,
                                ),
                                emit_detailed,
                            )?;
                        selected.push((*root, proposal));
                    }
                    drop(view);

                    let unsatisfied_relation_redexes =
                        selected.iter().fold(0_u64, |total, (_, extracted)| {
                            total.saturating_add(
                                extracted.proposal.cost.unsatisfied_relation_redexes,
                            )
                        });
                    match selected_relation_postcondition(
                        selected_phase_unions,
                        unsatisfied_relation_redexes,
                    ) {
                        Ok(Some(SelectedPhasePostcondition::RestartOrdinary)) => {
                            let iterations =
                                run_ordinary_relation_saturation(&mut lowerer.egraph, &context)
                                    .map_err(|source| OperationalSimulationError::Relation {
                                        site: site(&stage, wire, "relation rewrite"),
                                        source,
                                    })?;
                            egraph_mutation_epoch = egraph_mutation_epoch.saturating_add(1);
                            {
                                let mut control = control.borrow_mut();
                                control.work(
                                    iterations as u64,
                                    None,
                                    Some(lowerer.egraph.total_size() as u64),
                                )?;
                                let diagnostics = control.diagnostics_mut();
                                diagnostics.rewrite_iteration_count =
                                    accumulated_rewrite_iterations(
                                        diagnostics.rewrite_iteration_count,
                                        iterations,
                                    );
                                diagnostics.egraph_node_count = lowerer.egraph.total_size() as u64;
                                diagnostics.egraph_class_count =
                                    lowerer.egraph.number_of_classes() as u64;
                            }
                            if let Some(failure) = context.failure() {
                                return Err(relation_error(
                                    &stage,
                                    wire,
                                    &lowerer.egraph.analysis.symbols,
                                    failure,
                                ));
                            }
                            selected_roots =
                                roots.iter().map(|root| lowerer.egraph.find(*root)).collect();
                            normalization_epoch = 0;
                            continue 'joint_fixed_point;
                        }
                        Ok(None) => {}
                        Ok(Some(SelectedPhasePostcondition::Complete)) => {
                            unreachable!("relation staging never completes a structural phase")
                        }
                        Err(unsatisfied_relation_redexes) => {
                            return Err(OperationalSimulationError::Bound {
                                site: site(&stage, wire, "extract residual"),
                                source: super::error::BoundError::UnresolvedExtraction {
                                    unsatisfied_relation_redexes,
                                    unsatisfied_structural_redexes: 0,
                                },
                            });
                        }
                    }

                    let mut selected_redexes = None;
                    let mut discovered_candidate_count = 0_usize;
                    let mut frontier_redex_count = 0_usize;
                    let mut frontier_preference_count = 0_usize;
                    let mut snapshot_preferred = Vec::new();
                    let mut selected_polynomial_evaluation_count = 0_usize;
                    for (_, extracted) in &selected {
                        debug_assert_eq!(
                            extracted.proposal.expression.as_ref().len(),
                            extracted.origins.len(),
                            "selected expression and origin records stay aligned"
                        );
                        selected_polynomial_evaluation_count += 1;
                        let mut selected_polynomial_progress =
                            || context.reserve(1).then_some(()).ok_or(());
                        let mut selected_polynomial = selected_polynomial_monomials_with_context(
                            &lowerer.egraph,
                            &extracted.proposal.expression,
                            &extracted.origins,
                            Some(&context),
                            &mut selected_polynomial_progress,
                        );
                        if let Some(failure) = context.failure() {
                            return Err(relation_error(
                                &stage,
                                wire,
                                &lowerer.egraph.analysis.symbols,
                                failure,
                            ));
                        }
                        let selected_polynomial_plans =
                            selected_polynomial.as_mut().and_then(|monomials| {
                                selected_polynomial_redexes_mut(
                                    &lowerer.egraph,
                                    &extracted.proposal.expression,
                                    &extracted.origins,
                                    usize::from(extracted.proposal.expression.root()),
                                    monomials,
                                    &mut selected_polynomial_progress,
                                )
                            });
                        if let Some(selected_polynomial_plans) = selected_polynomial_plans {
                            discovered_candidate_count = discovered_candidate_count.saturating_add(
                                selected_polynomial_plans.discovered_candidate_count,
                            );
                            frontier_redex_count = selected_polynomial_plans.redexes.len();
                            frontier_preference_count = frontier_preference_count
                                .saturating_add(selected_polynomial_plans.preferred.len());
                            snapshot_preferred.extend(selected_polynomial_plans.preferred);
                            if !selected_polynomial_plans.redexes.is_empty() {
                                selected_redexes = Some(selected_polynomial_plans.redexes);
                                break;
                            }
                        }
                    }
                    let batch_size = selected_redexes.as_ref().map_or(0, Vec::len);
                    info!(
                        epoch = normalization_epoch,
                        discovered_candidate_count,
                        selected_redex_count = frontier_redex_count,
                        frontier_preference_count,
                        batch_size,
                        selected_polynomial_evaluation_count,
                        egraph_nodes = lowerer.egraph.total_size(),
                        extraction_milliseconds = extraction_started.elapsed().as_millis() as u64,
                        "selected relation normalization epoch"
                    );
                    if selected_redexes.is_none() {
                        let mut preferred_batch = ExactStructuralPreferences::new();
                        for plan in &snapshot_preferred {
                            let Some(recorded_root) = record_replacement_plan_preferences(
                                &lowerer.egraph,
                                plan,
                                &mut preferred_batch,
                            ) else {
                                return Err(OperationalSimulationError::Relation {
                                    site: site(&stage, wire, "selected structural preference"),
                                    source: super::error::RelationError::
                                        SelectedNormalizationBatchDidNotUnion {
                                            batch_size: snapshot_preferred.len(),
                                    },
                                });
                            };
                            let _ = recorded_root;
                        }
                        let structural_preference_changed = commit_structural_preference_batch(
                            &mut structural_preferences,
                            preferred_batch,
                        );
                        canonicalize_structural_preferences(
                            &lowerer.egraph,
                            &mut structural_preferences,
                        );
                        let terminal_exposed_relation = preferred_batch_has_relation_redex(
                            &lowerer.egraph,
                            &structural_preferences,
                            &context,
                        )
                        .map_err(|failure| {
                            relation_error(&stage, wire, &lowerer.egraph.analysis.symbols, failure)
                        })?;
                        if terminal_exposed_relation {
                            let rewrites_before = context.counters().rewrites;
                            let iterations =
                                run_ordinary_relation_saturation(&mut lowerer.egraph, &context)
                                    .map_err(|source| OperationalSimulationError::Relation {
                                        site: site(&stage, wire, "relation rewrite"),
                                        source,
                                    })?;
                            if let Some(failure) = context.failure() {
                                return Err(relation_error(
                                    &stage,
                                    wire,
                                    &lowerer.egraph.analysis.symbols,
                                    failure,
                                ));
                            }
                            selected_relation_rewrite_delta(
                                rewrites_before,
                                context.counters().rewrites,
                                snapshot_preferred.len(),
                            )
                            .map_err(|source| {
                                OperationalSimulationError::Relation {
                                    site: site(&stage, wire, "relation rewrite"),
                                    source,
                                }
                            })?;
                            egraph_mutation_epoch = egraph_mutation_epoch.saturating_add(1);
                            canonicalize_structural_preferences(
                                &lowerer.egraph,
                                &mut structural_preferences,
                            );
                            {
                                let mut control = control.borrow_mut();
                                control.work(
                                    iterations as u64,
                                    None,
                                    Some(lowerer.egraph.total_size() as u64),
                                )?;
                                let diagnostics = control.diagnostics_mut();
                                diagnostics.rewrite_iteration_count =
                                    accumulated_rewrite_iterations(
                                        diagnostics.rewrite_iteration_count,
                                        iterations,
                                    );
                                diagnostics.egraph_node_count = lowerer.egraph.total_size() as u64;
                                diagnostics.egraph_class_count =
                                    lowerer.egraph.number_of_classes() as u64;
                            }
                            selected_roots =
                                roots.iter().map(|root| lowerer.egraph.find(*root)).collect();
                            normalization_epoch = 0;
                            continue 'joint_fixed_point;
                        }
                        if preference_only_reextract(
                            frontier_redex_count,
                            structural_preference_changed,
                        ) {
                            normalization_epoch = normalization_epoch.saturating_add(1);
                            continue;
                        }
                        let (unsatisfied_structural_redexes, hidden_structural_redexes) = selected
                            .iter()
                            .fold((0_u64, 0_u64), |(structural, hidden), (_, extracted)| {
                                (
                                    structural.saturating_add(
                                        extracted.proposal.cost.unsatisfied_structural_redexes,
                                    ),
                                    hidden.saturating_add(
                                        extracted.proposal.cost.hidden_structural_redexes,
                                    ),
                                )
                            });
                        if matches!(
                            selected_phase_postcondition(
                                selected_phase_unions,
                                unsatisfied_structural_redexes,
                                hidden_structural_redexes,
                            ),
                            Ok(SelectedPhasePostcondition::RestartOrdinary)
                        ) {
                            let iterations =
                                run_ordinary_relation_saturation(&mut lowerer.egraph, &context)
                                    .map_err(|source| OperationalSimulationError::Relation {
                                        site: site(&stage, wire, "relation rewrite"),
                                        source,
                                    })?;
                            egraph_mutation_epoch = egraph_mutation_epoch.saturating_add(1);
                            {
                                let mut control = control.borrow_mut();
                                control.work(
                                    iterations as u64,
                                    None,
                                    Some(lowerer.egraph.total_size() as u64),
                                )?;
                                let diagnostics = control.diagnostics_mut();
                                diagnostics.rewrite_iteration_count =
                                    accumulated_rewrite_iterations(
                                        diagnostics.rewrite_iteration_count,
                                        iterations,
                                    );
                                diagnostics.egraph_node_count = lowerer.egraph.total_size() as u64;
                                diagnostics.egraph_class_count =
                                    lowerer.egraph.number_of_classes() as u64;
                            }
                            if let Some(failure) = context.failure() {
                                return Err(relation_error(
                                    &stage,
                                    wire,
                                    &lowerer.egraph.analysis.symbols,
                                    failure,
                                ));
                            }
                            selected_roots =
                                roots.iter().map(|root| lowerer.egraph.find(*root)).collect();
                            normalization_epoch = 0;
                            continue 'joint_fixed_point;
                        }
                        if let Err((unsatisfied_structural_redexes, hidden_structural_redexes)) =
                            selected_phase_postcondition(
                                selected_phase_unions,
                                unsatisfied_structural_redexes,
                                hidden_structural_redexes,
                            )
                        {
                            return Err(OperationalSimulationError::Bound {
                                site: site(&stage, wire, "extract residual"),
                                source: super::error::BoundError::UnresolvedExtraction {
                                    unsatisfied_relation_redexes: 0,
                                    unsatisfied_structural_redexes: unsatisfied_structural_redexes
                                        .max(hidden_structural_redexes),
                                },
                            });
                        }
                        debug_assert_eq!(
                            selected_phase_postcondition(
                                selected_phase_unions,
                                unsatisfied_structural_redexes,
                                hidden_structural_redexes,
                            ),
                            Ok(SelectedPhasePostcondition::Complete)
                        );
                        // `selected` covers every canonical root and freezes
                        // the normalization fixed point. Final acceptance
                        // performs a separate preference-free extraction from
                        // this immutable e-graph snapshot below.
                        break 'joint_fixed_point (egraph_mutation_epoch, selected);
                    }
                    egraph_mutation_epoch = egraph_mutation_epoch.saturating_add(1);
                    let frontier_redexes = selected_redexes.expect("checked above");
                    let frontier_plan_total =
                        snapshot_preferred.len().saturating_add(frontier_redexes.len());
                    let mut frontier_changed = false;
                    let mut frontier_exposed_relation = false;
                    let mut frontier_union_count = 0_u64;
                    for (frontier_ordinal, (origin, plan)) in snapshot_preferred
                        .into_iter()
                        .map(|plan| (None, plan))
                        .chain(
                            frontier_redexes.into_iter().map(|(origin, plan)| (Some(origin), plan)),
                        )
                        .enumerate()
                    {
                        let (
                            plan_variant,
                            direct_children,
                            bounded_plan_nodes,
                            plan_nodes_truncated,
                        ) = replacement_plan_bounded_shape(&plan);
                        let canonical_origin = origin.map(|origin| lowerer.egraph.find(origin));
                        let canonical_existing_target = match &plan {
                            ReplacementPlan::Existing(target) => Some(lowerer.egraph.find(*target)),
                            _ => None,
                        };
                        let origin_enode = canonical_origin
                            .and_then(|id| shallow_enode_log_shape(&lowerer.egraph, id));
                        let existing_target_enode = canonical_existing_target
                            .and_then(|id| shallow_enode_log_shape(&lowerer.egraph, id));
                        let origin_directly_contains_target = canonical_origin
                            .zip(canonical_existing_target)
                            .map(|(origin, target)| {
                                shallow_enode_directly_contains(&lowerer.egraph, origin, target)
                            });
                        let target_directly_contains_origin = canonical_existing_target
                            .zip(canonical_origin)
                            .map(|(target, origin)| {
                                shallow_enode_directly_contains(&lowerer.egraph, target, origin)
                            });
                        info!(
                            event = "operational_selected_frontier_plan",
                            frontier_ordinal,
                            frontier_plan_total,
                            origin = ?canonical_origin.map(usize::from),
                            existing_target = ?canonical_existing_target.map(usize::from),
                            origin_physical_nodes = ?canonical_origin
                                .map(|id| lowerer.egraph[id].nodes.len()),
                            existing_target_physical_nodes = ?canonical_existing_target
                                .map(|id| lowerer.egraph[id].nodes.len()),
                            origin_parent_references = ?canonical_origin
                                .map(|id| lowerer.egraph[id].parents().len()),
                            existing_target_parent_references = ?canonical_existing_target
                                .map(|id| lowerer.egraph[id].parents().len()),
                            origin_sort = ?canonical_origin
                                .map(|id| &lowerer.egraph[id].data.sort),
                            existing_target_sort = ?canonical_existing_target
                                .map(|id| &lowerer.egraph[id].data.sort),
                            origin_enode_operator = ?origin_enode
                                .as_ref().map(|(operator, _, _)| operator),
                            origin_enode_children = ?origin_enode
                                .as_ref().map(|(_, children, _)| children),
                            origin_enode_children_omitted = ?origin_enode
                                .as_ref().map(|(_, _, omitted)| omitted),
                            existing_target_enode_operator = ?existing_target_enode
                                .as_ref().map(|(operator, _, _)| operator),
                            existing_target_enode_children = ?existing_target_enode
                                .as_ref().map(|(_, children, _)| children),
                            existing_target_enode_children_omitted = ?existing_target_enode
                                .as_ref().map(|(_, _, omitted)| omitted),
                            origin_directly_contains_target = ?origin_directly_contains_target,
                            target_directly_contains_origin = ?target_directly_contains_origin,
                            plan_variant,
                            direct_children,
                            bounded_plan_nodes,
                            plan_nodes_truncated,
                            egraph_nodes = lowerer.egraph.total_size(),
                            "selected frontier plan before materialization"
                        );
                        if let Some(origin) = origin {
                            let origin = lowerer.egraph.find(origin);
                            if !replacement_plan_satisfied(&lowerer.egraph, origin, &plan) {
                                let Some((origin, replacement)) =
                                    materialize_selected_polynomial_redex(
                                        &mut lowerer.egraph,
                                        (origin, &plan),
                                        &context,
                                    )
                                else {
                                    if let Some(failure) = context.failure() {
                                        return Err(relation_error(
                                            &stage,
                                            wire,
                                            &lowerer.egraph.analysis.symbols,
                                            failure,
                                        ));
                                    }
                                    return Err(OperationalSimulationError::Relation {
                                        site: site(
                                            &stage,
                                            wire,
                                            "selected relation normalization",
                                        ),
                                        source: super::error::RelationError::
                                            SelectedNormalizationBatchDidNotUnion { batch_size },
                                    });
                                };
                                if canonical_existing_target.is_some() {
                                    info!(
                                        event = "operational_selected_existing_union_phase",
                                        phase = "before_union",
                                        origin = usize::from(lowerer.egraph.find(origin)),
                                        replacement = usize::from(lowerer.egraph.find(replacement)),
                                        egraph_nodes = lowerer.egraph.total_size(),
                                        "selected Existing plan immediately before union"
                                    );
                                }
                                let outer_union = lowerer.egraph.union(origin, replacement);
                                if canonical_existing_target.is_some() {
                                    info!(
                                        event = "operational_selected_existing_union_phase",
                                        phase = "after_union",
                                        origin = usize::from(lowerer.egraph.find(origin)),
                                        replacement = usize::from(lowerer.egraph.find(replacement)),
                                        outer_union,
                                        egraph_nodes = lowerer.egraph.total_size(),
                                        "selected Existing plan immediately after union"
                                    );
                                    info!(
                                        event = "operational_selected_existing_union_phase",
                                        phase = "before_rebuild",
                                        origin = usize::from(lowerer.egraph.find(origin)),
                                        replacement = usize::from(lowerer.egraph.find(replacement)),
                                        outer_union,
                                        egraph_nodes = lowerer.egraph.total_size(),
                                        "selected Existing plan immediately before rebuild"
                                    );
                                }
                                lowerer.egraph.rebuild();
                                canonicalize_structural_preferences(
                                    &lowerer.egraph,
                                    &mut structural_preferences,
                                );
                                if let Some(failure) = context.failure() {
                                    return Err(relation_error(
                                        &stage,
                                        wire,
                                        &lowerer.egraph.analysis.symbols,
                                        failure,
                                    ));
                                }
                                if !replacement_plan_satisfied(
                                    &lowerer.egraph,
                                    lowerer.egraph.find(origin),
                                    &plan,
                                ) {
                                    return Err(OperationalSimulationError::Relation {
                                        site: site(
                                            &stage,
                                            wire,
                                            "selected relation normalization",
                                        ),
                                        source: super::error::RelationError::
                                        SelectedNormalizationBatchDidNotUnion { batch_size },
                                    });
                                }
                                let unioned =
                                    outer_union || matches!(plan, ReplacementPlan::Equivalent(_));
                                if unioned {
                                    frontier_changed = true;
                                    frontier_union_count = frontier_union_count.saturating_add(1);
                                    selected_phase_unions = selected_phase_unions.saturating_add(1);
                                    context.note_rewrite(false);
                                }
                            }
                        }

                        let mut plan_preferences = ExactStructuralPreferences::new();
                        if record_replacement_plan_preferences(
                            &lowerer.egraph,
                            &plan,
                            &mut plan_preferences,
                        )
                        .is_none()
                        {
                            return Err(OperationalSimulationError::Relation {
                                site: site(&stage, wire, "selected relation normalization"),
                                source: super::error::RelationError::
                                    SelectedNormalizationBatchDidNotUnion { batch_size },
                            });
                        }
                        frontier_changed |= commit_structural_preference_batch(
                            &mut structural_preferences,
                            plan_preferences,
                        );
                    }
                    if !frontier_changed {
                        return Err(OperationalSimulationError::Relation {
                            site: site(&stage, wire, "selected relation normalization"),
                            source:
                                super::error::RelationError::SelectedNormalizationBatchDidNotUnion {
                                    batch_size,
                                },
                        });
                    }
                    control.borrow_mut().work(
                        frontier_union_count,
                        None,
                        Some(lowerer.egraph.total_size() as u64),
                    )?;
                    canonicalize_structural_preferences(
                        &lowerer.egraph,
                        &mut structural_preferences,
                    );
                    frontier_exposed_relation |= preferred_batch_has_relation_redex(
                        &lowerer.egraph,
                        &structural_preferences,
                        &context,
                    )
                    .map_err(|failure| {
                        relation_error(&stage, wire, &lowerer.egraph.analysis.symbols, failure)
                    })?;
                    if frontier_exposed_relation {
                        let rewrites_before = context.counters().rewrites;
                        let iterations =
                            run_ordinary_relation_saturation(&mut lowerer.egraph, &context)
                                .map_err(|source| OperationalSimulationError::Relation {
                                    site: site(&stage, wire, "relation rewrite"),
                                    source,
                                })?;
                        if let Some(failure) = context.failure() {
                            return Err(relation_error(
                                &stage,
                                wire,
                                &lowerer.egraph.analysis.symbols,
                                failure,
                            ));
                        }
                        selected_relation_rewrite_delta(
                            rewrites_before,
                            context.counters().rewrites,
                            batch_size,
                        )
                        .map_err(|source| {
                            OperationalSimulationError::Relation {
                                site: site(&stage, wire, "relation rewrite"),
                                source,
                            }
                        })?;
                        egraph_mutation_epoch = egraph_mutation_epoch.saturating_add(1);
                        canonicalize_structural_preferences(
                            &lowerer.egraph,
                            &mut structural_preferences,
                        );
                        {
                            let mut control = control.borrow_mut();
                            control.work(
                                iterations as u64,
                                None,
                                Some(lowerer.egraph.total_size() as u64),
                            )?;
                            let diagnostics = control.diagnostics_mut();
                            diagnostics.rewrite_iteration_count = accumulated_rewrite_iterations(
                                diagnostics.rewrite_iteration_count,
                                iterations,
                            );
                            diagnostics.egraph_node_count = lowerer.egraph.total_size() as u64;
                            diagnostics.egraph_class_count =
                                lowerer.egraph.number_of_classes() as u64;
                        }
                        selected_roots =
                            roots.iter().map(|root| lowerer.egraph.find(*root)).collect();
                        normalization_epoch = 0;
                        continue 'joint_fixed_point;
                    }
                    selected_roots = roots.iter().map(|root| lowerer.egraph.find(*root)).collect();
                    normalization_epoch += 1;
                }
            };
            let counters = context.counters();
            control.borrow_mut().diagnostics_mut().relation_candidate_count = counters.candidates;
            control.borrow_mut().diagnostics_mut().relation_rewrite_count = counters.rewrites;
            let final_selected_roots =
                roots.iter().map(|root| lowerer.egraph.find(*root)).collect::<BTreeSet<_>>();
            let terminal_snapshot_is_current = terminal_egraph_mutation_epoch ==
                egraph_mutation_epoch &&
                selected_roots == final_selected_roots;
            debug_assert!(terminal_snapshot_is_current);
            if !terminal_snapshot_is_current {
                let root = lowerer.egraph.find(
                    *roots.first().expect("a lowered residual has at least one selected root"),
                );
                return Err(OperationalSimulationError::Bound {
                    site: site(&stage, wire, "validate terminal snapshot"),
                    source: super::error::BoundError::EvaluationFailed {
                        source: BoundEvaluationError::MissingExtractedTerm { term: root },
                    },
                });
            }
            // Retain the exact checked-relation-normalized polynomial recipe.
            // Materializing it into this e-graph would not retain its syntax:
            // hash-consing can return the original class and expose its raw
            // relation-product alternatives again.
            let mut normalized_plans = Vec::with_capacity(terminal_selected.len());
            for (root, extracted) in &terminal_selected {
                let mut progress = || context.reserve(1).then_some(()).ok_or(());
                let plan = selected_polynomial_normal_form_plan(
                    &mut lowerer.egraph,
                    &extracted.proposal.expression,
                    &extracted.origins,
                    &context,
                    &mut progress,
                )
                .map_err(|reason| {
                    tracing::info!(
                        root = usize::from(*root),
                        ?reason,
                        "final polynomial normalization failed"
                    );
                    if let Some(failure) = context.failure() {
                        relation_error(&stage, wire, &lowerer.egraph.analysis.symbols, failure)
                    } else {
                        OperationalSimulationError::Bound {
                            site: site(&stage, wire, "materialize final polynomial"),
                            source: super::error::BoundError::EvaluationFailed {
                                source: BoundEvaluationError::MissingExtractedTerm { term: *root },
                            },
                        }
                    }
                })?;
                normalized_plans.push((*root, plan));
            }
            lowerer.egraph.rebuild();

            let view = lowerer.production_bound_view();
            let mut distinct_leaves = BTreeSet::new();
            for (_, plan) in &normalized_plans {
                if !replacement_plan_existing_leaves(plan, &mut distinct_leaves) {
                    return Err(OperationalSimulationError::Bound {
                        site: site(&stage, wire, "materialize final polynomial"),
                        source: super::error::BoundError::EvaluationFailed {
                            source: BoundEvaluationError::MissingExtractedTerm {
                                term: *final_selected_roots
                                    .first()
                                    .expect("a lowered residual has a final root"),
                            },
                        },
                    });
                }
            }
            let distinct_leaves = distinct_leaves
                .into_iter()
                .map(|leaf| lowerer.egraph.find(leaf))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let extracted_leaves = extract_best_proposals_with_origins(
                &lowerer.egraph,
                &distinct_leaves,
                &view,
                &mut ExtractionControl {
                    invalid_dag: &mut invalid_dag,
                    bound_error: &mut bound_error,
                },
                &mut |origin, node, egraph| {
                    let classification =
                        super::relation::classify_proposal_node(egraph, origin, node, &context)
                            .map_err(|failure| {
                                relation_error(&stage, wire, &egraph.analysis.symbols, failure)
                            })?;
                    Ok(ProposalNodeClassification {
                        relation_redex: classification.relation_redex,
                        local_checked_relation_count: classification.local_checked_relation_count,
                    })
                },
            )?;
            let extracted_leaves =
                distinct_leaves.iter().copied().zip(extracted_leaves).collect::<HashMap<_, _>>();
            let mut final_selected = Vec::with_capacity(final_selected_roots.len());
            for root in &final_selected_roots {
                let normalized_index = normalized_plans
                    .binary_search_by_key(root, |(root, _)| *root)
                    .map_err(|_| OperationalSimulationError::Bound {
                        site: site(&stage, wire, "reuse final polynomial"),
                        source: super::error::BoundError::EvaluationFailed {
                            source: BoundEvaluationError::MissingExtractedTerm { term: *root },
                        },
                    })?;
                let plan = &normalized_plans[normalized_index].1;
                let mut expression = egg::RecExpr::default();
                let mut origins = Vec::new();
                let mut first_large_source = None;
                let mut canonicalize_leaf = |leaf| lowerer.egraph.find(leaf);
                let mut unresolved =
                    |unsatisfied_relation_redexes, unsatisfied_structural_redexes| {
                        OperationalSimulationError::Bound {
                            site: site(&stage, wire, "extract final polynomial leaf"),
                            source: super::error::BoundError::UnresolvedExtraction {
                                unsatisfied_relation_redexes,
                                unsatisfied_structural_redexes,
                            },
                        }
                    };
                let mut extract_leaf = |leaf| {
                    extracted_leaves.get(&leaf).cloned().ok_or_else(|| {
                        OperationalSimulationError::Bound {
                            site: site(&stage, wire, "reuse final polynomial leaf"),
                            source: super::error::BoundError::EvaluationFailed {
                                source: BoundEvaluationError::MissingExtractedTerm { term: leaf },
                            },
                        }
                    })
                };
                append_exact_final_plan(
                    plan,
                    *root,
                    &mut expression,
                    &mut origins,
                    &mut first_large_source,
                    &mut canonicalize_leaf,
                    &mut unresolved,
                    &mut extract_leaf,
                )
                .map_err(|source| {
                    source.unwrap_or_else(|| OperationalSimulationError::Bound {
                        site: site(&stage, wire, "materialize final polynomial"),
                        source: super::error::BoundError::EvaluationFailed {
                            source: BoundEvaluationError::MissingExtractedTerm { term: *root },
                        },
                    })
                })?;
                let semantic_bound =
                    evaluate_exact_selected_expression(&view, &expression, &origins)
                        .map_err(&mut bound_error)?;
                let proposal = ExtractedProposal {
                    cost: ProposalCost {
                        large_residual: matches!(
                            semantic_bound.coefficient_class,
                            super::bound::BoundClass::Large
                        ),
                        node_count: expression.as_ref().len() as u64,
                        ..Default::default()
                    },
                    semantic_bound: Some(semantic_bound),
                    first_large_source,
                    expression,
                };
                final_selected.push((
                    *root,
                    ExtractedProposalWithOrigins { proposal, origins: origins.into_boxed_slice() },
                ));
            }
            drop(view);
            control.borrow_mut().reserve_owned_elements(final_selected_roots.len())?;
            let mut proposals = Vec::with_capacity(roots.len());
            let mut bounds = Vec::with_capacity(roots.len());
            for root in roots {
                let root = lowerer.egraph.find(root);
                let selected_index = final_selected
                    .binary_search_by_key(&root, |(root, _)| *root)
                    .map_err(|_| OperationalSimulationError::Bound {
                        site: site(&stage, wire, "reuse final extraction"),
                        source: super::error::BoundError::EvaluationFailed {
                            source: BoundEvaluationError::MissingExtractedTerm { term: root },
                        },
                    })?;
                let extracted = &final_selected[selected_index].1;
                let proposal = &extracted.proposal;
                if proposal.cost.unsatisfied_relation_redexes != 0 ||
                    proposal.cost.unsatisfied_structural_redexes != 0
                {
                    return Err(OperationalSimulationError::Bound {
                        site: site(&stage, wire, "extract residual"),
                        source: super::error::BoundError::UnresolvedExtraction {
                            unsatisfied_relation_redexes: proposal
                                .cost
                                .unsatisfied_relation_redexes,
                            unsatisfied_structural_redexes: proposal
                                .cost
                                .unsatisfied_structural_redexes,
                        },
                    });
                }
                let semantic_bound = proposal.semantic_bound.clone().ok_or_else(|| {
                    OperationalSimulationError::Bound {
                        site: site(&stage, wire, "extract semantic bound"),
                        source: super::error::BoundError::EvaluationFailed {
                            source: BoundEvaluationError::NonMatrixTerm { term: root },
                        },
                    }
                })?;
                if matches!(semantic_bound.coefficient_class, super::bound::BoundClass::Large) {
                    let source = proposal.first_large_source.and_then(|source| {
                        lowerer
                            .egraph
                            .analysis
                            .symbols
                            .atomic_sources
                            .get(source.0)
                            .map(|descriptor| descriptor.key.clone())
                    });
                    if let Some(super::identity::AtomicSourceKey::ExplicitLarge(graph_source)) =
                        source.as_ref()
                    {
                        if let Some(source_id) = proposal.first_large_source {
                            emit_explicit_large_product_context(
                                &lowerer.egraph,
                                &proposal.expression,
                                &extracted.origins,
                                &context,
                                source_id,
                            );
                        }
                        let binding = lowerer.graph_wire_binding_diagnostic(graph_source);
                        tracing::info!(
                            event = "operational_explicit_large_binding",
                            selected_root = usize::from(root),
                            selected_expression_nodes = proposal.expression.as_ref().len(),
                            source = ?graph_source,
                            producer_stage = ?binding.stage,
                            producer_outputs = ?binding.output_names,
                            artifact_consumers = ?binding.artifact_consumers,
                            "selected residual retains an explicitly Large workflow artifact"
                        );
                    }
                    return Err(OperationalSimulationError::Bound {
                        site: site(&stage, wire, "extract residual"),
                        source: source.map_or_else(
                            || super::error::BoundError::EvaluationFailed {
                                source: BoundEvaluationError::UnconsumedLargeTerm { term: root },
                            },
                            |source| super::error::BoundError::UnconsumedLargeTerm { source },
                        ),
                    });
                }
                bounds.push((root, semantic_bound));
                proposals.push(proposal.clone());
            }
            let term_count =
                proposals.iter().map(|proposal| proposal.expression.as_ref().len()).sum::<usize>();
            control.borrow_mut().work(
                term_count as u64,
                None,
                Some(lowerer.egraph.total_size() as u64),
            )?;
            control.borrow_mut().diagnostics_mut().final_term_count = term_count as u64;
            drop(lowerer);
            Ok((bounds, stage, wire))
        },
        |(bounds, stage, wire), control| {
            control.set_progress_site(stage.0.clone(), "root".to_owned(), wire.node.0 as u64);
            let mut bound = BigUint::zero();
            for (root, result) in bounds {
                let maximum =
                    result.coefficient_class.maximum_absolute_coefficient().ok_or_else(|| {
                        OperationalSimulationError::Bound {
                            site: site(&stage, wire, "bound"),
                            source: super::error::BoundError::EvaluationFailed {
                                source: BoundEvaluationError::UnconsumedLargeTerm { term: root },
                            },
                        }
                    })?;
                bound = bound.max(maximum);
            }
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

    fn final_leaf_fixture(term: egg::Id, cost: ProposalCost) -> ExtractedProposalWithOrigins {
        let mut expression = egg::RecExpr::default();
        expression.add(MxxLang::MatrixConstant(super::super::identity::MatrixConstantSpecId(0)));
        ExtractedProposalWithOrigins {
            proposal: ExtractedProposal {
                cost,
                semantic_bound: Some(super::super::bound::MatrixBound {
                    matrix_type: mxx_ir_core::types::ConcreteMatrixType {
                        modulus: 17.into(),
                        ring_dimension: 1,
                        rows: 1,
                        columns: 1,
                    },
                    coefficient_class: super::super::bound::BoundClass::bounded(1_u8.into()),
                    metadata: super::super::bound::MatrixMetadata::unknown(),
                }),
                first_large_source: None,
                expression,
            },
            origins: vec![term].into_boxed_slice(),
        }
    }

    fn final_leaf_error(relation: u64, structural: u64) -> OperationalSimulationError {
        OperationalSimulationError::Bound {
            site: site(
                &StageId("final-leaf-test".to_owned()),
                WireRef { node: mxx_ir_core::NodeId(0), port: Port(0) },
                "extract final polynomial leaf",
            ),
            source: super::super::error::BoundError::UnresolvedExtraction {
                unsatisfied_relation_redexes: relation,
                unsatisfied_structural_redexes: structural,
            },
        }
    }

    #[test]
    fn exact_final_plan_prepass_deduplicates_repeated_leaves() {
        let leaf = egg::Id::from(7);
        let plan = ReplacementPlan::Add(
            vec![ReplacementPlan::Existing(leaf), ReplacementPlan::Existing(leaf)]
                .into_boxed_slice(),
        );
        let mut leaves = BTreeSet::new();
        assert!(replacement_plan_existing_leaves(&plan, &mut leaves));
        assert_eq!(leaves, BTreeSet::from([leaf]));
    }

    #[test]
    fn exact_final_plan_rejects_bounded_leaf_with_unresolved_checked_relation() {
        let leaf = egg::Id::from(9);
        let mut output = egg::RecExpr::default();
        let mut origins = Vec::new();
        let mut first_large = None;
        let error = append_exact_final_plan(
            &ReplacementPlan::Existing(leaf),
            leaf,
            &mut output,
            &mut origins,
            &mut first_large,
            &mut |term| term,
            &mut final_leaf_error,
            &mut |term| {
                Ok(final_leaf_fixture(
                    term,
                    ProposalCost {
                        unsatisfied_relation_redexes: 1,
                        unsatisfied_structural_redexes: 2,
                        hidden_structural_redexes: 3,
                        ..Default::default()
                    },
                ))
            },
        )
        .expect_err("an unresolved checked relation is never hidden by a finite bound");
        assert!(matches!(
            error,
            Some(OperationalSimulationError::Bound {
                source: super::super::error::BoundError::UnresolvedExtraction {
                    unsatisfied_relation_redexes: 1,
                    unsatisfied_structural_redexes: 5,
                },
                ..
            })
        ));
    }

    fn test_matrix_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        relation_role: Option<super::super::identity::AtomicRelationRole>,
    ) -> (egg::Id, super::super::identity::AtomicSourceId) {
        let matrix_type = super::super::identity::ResolvedMatrixType {
            modulus: super::super::identity::ResolvedIntExpr::Const(17.into()),
            ring_dimension: super::super::identity::ResolvedIntExpr::Const(1.into()),
            rows: super::super::identity::ResolvedIntExpr::Const(1.into()),
            columns: super::super::identity::ResolvedIntExpr::Const(1.into()),
        };
        let source = egraph.analysis.symbols.atomic_sources.intern(
            super::super::identity::AtomicSourceDescriptor {
                key: super::super::identity::AtomicSourceKey::ProtocolInput(
                    crate::ProtocolInputId::from(name),
                ),
                sort: super::super::analysis::MxxSort::Matrix(matrix_type),
                integer_domain: None,
                canonical_residue_convention: Some(
                    super::super::identity::CanonicalResidueConvention::Nonnegative,
                ),
                relation_role,
            },
        );
        let source = super::super::identity::AtomicSourceId(source);
        let term = egraph.add(MxxLang::Atom { source, indices: Box::new([]) });
        (term, source)
    }

    fn preferred_relation_fixture()
    -> (EGraph<MxxLang, MxxAnalysis>, RewriteContext, ExactStructuralPreferences, egg::Id, egg::Id)
    {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = test_matrix_atom(&mut egraph, "preferred-public", None);
        let (relation, source) = test_matrix_atom(
            &mut egraph,
            "preferred-relation",
            Some(super::super::identity::AtomicRelationRole::Preimage),
        );
        let (target, _) = test_matrix_atom(&mut egraph, "preferred-target", None);
        let product_node = MxxLang::MatrixMultiply(vec![public, relation].into_boxed_slice());
        let product = egraph.add(product_node.clone());
        egraph.rebuild();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(super::super::relation::RelationRegistration {
            source,
            expected_public: public,
            target,
            trapdoor: None,
            indices: Box::new([]),
        });
        let mut preferences = ExactStructuralPreferences::new();
        preferences.entry(egraph.find(product)).or_default().insert(product_node);
        (egraph, context, preferences, product, target)
    }

    #[test]
    fn explicit_large_context_uses_nearest_ordered_spine_and_expected_public_registry() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, left_source) = test_matrix_atom(&mut egraph, "diagnostic-left", None);
        let (explicit, explicit_source) =
            test_matrix_atom(&mut egraph, "diagnostic-explicit", None);
        let (relation, relation_source) = test_matrix_atom(
            &mut egraph,
            "diagnostic-relation",
            Some(super::super::identity::AtomicRelationRole::Preimage),
        );
        let (tail, tail_source) = test_matrix_atom(&mut egraph, "diagnostic-tail", None);
        let inner = egraph
            .add(MxxLang::MatrixMultiply(vec![left, explicit, relation, tail].into_boxed_slice()));
        let outer = egraph.add(MxxLang::MatrixMultiply(vec![left, inner].into_boxed_slice()));
        egraph.rebuild();

        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(super::super::relation::RelationRegistration {
            source: relation_source,
            expected_public: explicit,
            target: tail,
            trapdoor: None,
            indices: Box::new([]),
        });

        let mut expression = egg::RecExpr::default();
        let selected_left =
            expression.add(MxxLang::Atom { source: left_source, indices: Box::new([]) });
        let selected_explicit =
            expression.add(MxxLang::Atom { source: explicit_source, indices: Box::new([]) });
        let selected_relation =
            expression.add(MxxLang::Atom { source: relation_source, indices: Box::new([]) });
        let selected_tail =
            expression.add(MxxLang::Atom { source: tail_source, indices: Box::new([]) });
        let selected_inner = expression.add(MxxLang::MatrixMultiply(
            vec![selected_left, selected_explicit, selected_relation, selected_tail]
                .into_boxed_slice(),
        ));
        expression
            .add(MxxLang::MatrixMultiply(vec![selected_left, selected_inner].into_boxed_slice()));
        let origins = [left, explicit, relation, tail, inner, outer];

        let (product_index, spine, source_position) =
            nearest_selected_product_spine(&expression, explicit_source)
                .expect("selected explicit source has a product ancestor");
        assert_eq!(product_index, usize::from(selected_inner));
        assert_eq!(source_position, 1);
        assert_eq!(spine, vec![selected_left, selected_explicit, selected_relation, selected_tail]);
        assert_eq!(
            spine
                .iter()
                .map(|factor| usize::from(egraph.find(origins[usize::from(*factor)])))
                .collect::<Vec<_>>(),
            vec![left, explicit, relation, tail]
                .into_iter()
                .map(|factor| usize::from(egraph.find(factor)))
                .collect::<Vec<_>>()
        );

        let registrations = context.diagnostic_registrations_for_expected_public(&egraph, explicit);
        assert_eq!(registrations.len(), 1);
        assert_eq!(relation_canonical_ids(&egraph, &registrations), vec![usize::from(relation)]);
        let right_source =
            direct_selected_atomic_source(&expression, spine.get(source_position + 1).copied());
        assert_eq!(right_source, Some(relation_source));
        assert!(registrations.iter().any(|registration| {
            right_source == Some(registration.source) &&
                egraph.find(registration.expected_public) == egraph.find(explicit)
        }));
    }

    #[test]
    fn preferred_batch_exposes_relation_before_next_extraction() {
        let (egraph, context, preferences, _, _) = preferred_relation_fixture();
        assert!(
            preferred_batch_has_relation_redex(&egraph, &preferences, &context)
                .expect("registered preferred relation classifies")
        );
    }

    #[test]
    fn preferred_batch_does_not_retrigger_after_saturation() {
        let (mut egraph, context, mut preferences, product, target) = preferred_relation_fixture();
        assert!(
            preferred_batch_has_relation_redex(&egraph, &preferences, &context)
                .expect("registered preferred relation classifies")
        );

        run_ordinary_relation_saturation(&mut egraph, &context)
            .expect("preferred relation saturates");
        canonicalize_structural_preferences(&egraph, &mut preferences);

        assert_eq!(egraph.find(product), egraph.find(target));
        assert!(
            !preferred_batch_has_relation_redex(&egraph, &preferences, &context)
                .expect("satisfied preferred relation classifies")
        );
    }

    #[test]
    fn frontier_exposure_is_or_accumulated_and_saturated_once_after_all_siblings() {
        let (mut egraph, context, relation_preferences, product, target) =
            preferred_relation_fixture();
        let (left, _) = test_matrix_atom(&mut egraph, "frontier-later-left", None);
        let (right, _) = test_matrix_atom(&mut egraph, "frontier-later-right", None);
        let later_node = MxxLang::MatrixAdd(vec![left, right].into_boxed_slice());
        let later = egraph.add(later_node.clone());
        egraph.rebuild();
        let mut later_preferences = ExactStructuralPreferences::new();
        later_preferences.entry(egraph.find(later)).or_default().insert(later_node.clone());

        let mut exposed = false;
        exposed |= preferred_batch_has_relation_redex(&egraph, &relation_preferences, &context)
            .expect("the first sibling exposes a relation");
        assert!(
            egraph[egraph.find(later)].nodes.contains(&later_node),
            "the later sibling is present before the deferred runner"
        );
        exposed |= preferred_batch_has_relation_redex(&egraph, &later_preferences, &context)
            .expect("the later sibling classifies without a relation");
        assert!(exposed, "one exposed sibling keeps the frontier flag set");

        let mut saturation_count = 0;
        let iterations = if exposed {
            saturation_count += 1;
            run_ordinary_relation_saturation(&mut egraph, &context)
                .expect("the frontier relation saturates once")
        } else {
            0
        };
        assert_eq!(saturation_count, 1);
        assert!(iterations > 0);
        assert_eq!(egraph.find(product), egraph.find(target));
        assert!(egraph[egraph.find(later)].nodes.contains(&later_node));
    }

    #[test]
    fn preferred_batch_without_relation_does_not_restart_or_reorder_preferences() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = test_matrix_atom(&mut egraph, "preferred-left", None);
        let (right, _) = test_matrix_atom(&mut egraph, "preferred-right", None);
        let add_node = MxxLang::MatrixAdd(vec![left, right].into_boxed_slice());
        let add = egraph.add(add_node.clone());
        egraph.rebuild();
        let mut preferences = ExactStructuralPreferences::new();
        preferences.entry(egraph.find(add)).or_default().insert(add_node);
        let original = preferences.clone();
        let context = RewriteContext::new(SharedRewriteBudget::new());

        assert!(
            !preferred_batch_has_relation_redex(&egraph, &preferences, &context)
                .expect("ordinary preferred composite classifies")
        );
        assert_eq!(preferences, original, "classification does not alter extraction cost order");
    }

    #[test]
    fn preferred_batch_relation_classification_failure_propagates() {
        let (egraph, _, preferences, _, _) = preferred_relation_fixture();
        let context = RewriteContext::new(SharedRewriteBudget::new());
        assert!(matches!(
            preferred_batch_has_relation_redex(&egraph, &preferences, &context),
            Err(RelationFailure::MissingRegistration { .. })
        ));
    }

    #[test]
    fn complete_preference_scan_catches_relation_exposed_after_a_later_union() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (public, _) = test_matrix_atom(&mut egraph, "complete-scan-public", None);
        let (ordinary, _) = test_matrix_atom(&mut egraph, "complete-scan-ordinary", None);
        let (relation, source) = test_matrix_atom(
            &mut egraph,
            "complete-scan-relation",
            Some(super::super::identity::AtomicRelationRole::Preimage),
        );
        let (target, _) = test_matrix_atom(&mut egraph, "complete-scan-target", None);
        let preferred_node = MxxLang::MatrixMultiply(vec![public, ordinary].into_boxed_slice());
        let preferred = egraph.add(preferred_node.clone());
        let (later_left, _) = test_matrix_atom(&mut egraph, "complete-scan-later-left", None);
        let (later_right, _) = test_matrix_atom(&mut egraph, "complete-scan-later-right", None);
        let later_node = MxxLang::MatrixAdd(vec![later_left, later_right].into_boxed_slice());
        let later = egraph.add(later_node.clone());
        egraph.rebuild();

        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(super::super::relation::RelationRegistration {
            source,
            expected_public: public,
            target,
            trapdoor: None,
            indices: Box::new([]),
        });
        let mut preferences = ExactStructuralPreferences::from([(
            egraph.find(preferred),
            HashSet::from([preferred_node]),
        )]);
        let later_batch =
            ExactStructuralPreferences::from([(egraph.find(later), HashSet::from([later_node]))]);

        egraph.union(ordinary, relation);
        egraph.rebuild();
        canonicalize_structural_preferences(&egraph, &mut preferences);
        assert!(
            !preferred_batch_has_relation_redex(&egraph, &later_batch, &context)
                .expect("the later batch itself has no relation")
        );
        assert!(
            preferred_batch_has_relation_redex(&egraph, &preferences, &context)
                .expect("the earlier preference becomes relation-applicable")
        );

        let rewrites_before = context.counters().rewrites;
        run_ordinary_relation_saturation(&mut egraph, &context)
            .expect("the complete preference scan routes to ordinary saturation");
        let rewrites_after = context.counters().rewrites;
        assert!(
            selected_relation_rewrite_delta(rewrites_before, rewrites_after, 1).is_ok(),
            "a terminal exposure restarts only after an actual rewrite"
        );
        assert_eq!(egraph.find(preferred), egraph.find(target));
    }

    #[test]
    fn replacement_preference_records_composites_but_not_existing_leaves() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let composite = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        egraph.rebuild();
        let plan = ReplacementPlan::Add(
            vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)]
                .into_boxed_slice(),
        );
        let mut preferences = ExactStructuralPreferences::new();
        let mut first_batch = ExactStructuralPreferences::new();
        let first = record_replacement_plan_preferences(&egraph, &plan, &mut first_batch);
        assert_eq!(first, Some(egraph.find(composite)));
        let first_changed = commit_structural_preference_batch(&mut preferences, first_batch);
        assert!(preference_only_reextract(0, first_changed));
        let mut duplicate_batch = ExactStructuralPreferences::new();
        let duplicate = record_replacement_plan_preferences(&egraph, &plan, &mut duplicate_batch);
        assert_eq!(duplicate, Some(egraph.find(composite)));
        let duplicate_changed =
            commit_structural_preference_batch(&mut preferences, duplicate_batch);
        assert!(!preference_only_reextract(0, duplicate_changed));
        assert!(!preference_only_reextract(1, true));
        assert_eq!(preferences.len(), 1);
        assert!(preferences.contains_key(&egraph.find(composite)));
        assert!(!preferences.contains_key(&egraph.find(left)));
        assert!(!preferences.contains_key(&egraph.find(right)));
    }

    #[test]
    fn existing_plan_leaves_prior_composite_preference_unchanged() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (left, _) = test_matrix_atom(&mut egraph, "existing-preference-left", None);
        let (right, _) = test_matrix_atom(&mut egraph, "existing-preference-right", None);
        let composite_node = MxxLang::MatrixAdd(vec![left, right].into_boxed_slice());
        let composite = egraph.add(composite_node.clone());
        egraph.union(left, composite);
        egraph.rebuild();
        let class = egraph.find(left);
        let canonical_composite = composite_node.map_children(|child| egraph.find(child));
        let mut preferences = ExactStructuralPreferences::from([(
            class,
            HashSet::from([canonical_composite.clone()]),
        )]);
        let mut batch = ExactStructuralPreferences::new();

        assert_eq!(
            record_replacement_plan_preferences(
                &egraph,
                &ReplacementPlan::Existing(class),
                &mut batch,
            ),
            Some(class)
        );
        assert!(batch.is_empty(), "an Existing plan has no physical-node preference");
        assert!(!commit_structural_preference_batch(&mut preferences, batch));
        assert_eq!(preferences[&class], HashSet::from([canonical_composite]));
    }

    #[test]
    fn later_batch_replaces_only_its_explicitly_constrained_class() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let first_node = MxxLang::MatrixAdd(vec![left, right].into_boxed_slice());
        let second_node = MxxLang::MatrixAdd(vec![right, left].into_boxed_slice());
        let first = egraph.add(first_node.clone());
        let second = egraph.add(second_node.clone());
        egraph.union(first, second);
        egraph.rebuild();
        let first_plan = ReplacementPlan::Add(
            vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)]
                .into_boxed_slice(),
        );
        let second_plan = ReplacementPlan::Add(
            vec![ReplacementPlan::Existing(right), ReplacementPlan::Existing(left)]
                .into_boxed_slice(),
        );
        let child_class = egraph.find(left);
        let mut preferences = ExactStructuralPreferences::new();
        preferences.entry(child_class).or_default().insert(MxxLang::IntConst(1.into()));

        let mut first_batch = ExactStructuralPreferences::new();
        record_replacement_plan_preferences(&egraph, &first_plan, &mut first_batch)
            .expect("first batch compiles");
        assert!(commit_structural_preference_batch(&mut preferences, first_batch));
        assert_eq!(preferences[&egraph.find(first)].len(), 1);
        assert!(preferences[&egraph.find(first)].contains(&first_node));

        let mut second_batch = ExactStructuralPreferences::new();
        record_replacement_plan_preferences(&egraph, &second_plan, &mut second_batch)
            .expect("second batch compiles");
        assert!(commit_structural_preference_batch(&mut preferences, second_batch));
        assert_eq!(preferences[&egraph.find(first)].len(), 1);
        assert!(preferences[&egraph.find(first)].contains(&second_node));
        assert!(
            preferences[&child_class].contains(&MxxLang::IntConst(1.into())),
            "an Existing leaf does not replace its child's prior preference"
        );
    }

    #[test]
    fn replacement_preference_keeps_multiple_exact_enodes_across_rebuild() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let first = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        let second = egraph.add(MxxLang::MatrixAdd(vec![right, left].into_boxed_slice()));
        egraph.union(first, second);
        egraph.rebuild();
        let plan = ReplacementPlan::Equivalent(
            vec![
                ReplacementPlan::Add(
                    vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)]
                        .into_boxed_slice(),
                ),
                ReplacementPlan::Add(
                    vec![ReplacementPlan::Existing(right), ReplacementPlan::Existing(left)]
                        .into_boxed_slice(),
                ),
            ]
            .into_boxed_slice(),
        );
        let mut preferences = ExactStructuralPreferences::new();

        assert!(record_replacement_plan_preferences(&egraph, &plan, &mut preferences).is_some());
        egraph.rebuild();
        canonicalize_structural_preferences(&egraph, &mut preferences);
        let desired = &preferences[&egraph.find(first)];
        assert_eq!(desired.len(), 2);
        assert!(desired.contains(&MxxLang::MatrixAdd(
            vec![egraph.find(left), egraph.find(right)].into_boxed_slice()
        )));
        assert!(desired.contains(&MxxLang::MatrixAdd(
            vec![egraph.find(right), egraph.find(left)].into_boxed_slice()
        )));
    }

    #[test]
    fn equivalent_plan_can_become_satisfied_when_its_outer_union_is_false() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![left, right].into_boxed_slice()));
        egraph.rebuild();
        let plan = ReplacementPlan::Equivalent(
            vec![
                ReplacementPlan::Add(
                    vec![ReplacementPlan::Existing(left), ReplacementPlan::Existing(right)]
                        .into_boxed_slice(),
                ),
                ReplacementPlan::Add(
                    vec![ReplacementPlan::Existing(right), ReplacementPlan::Existing(left)]
                        .into_boxed_slice(),
                ),
            ]
            .into_boxed_slice(),
        );
        assert!(!replacement_plan_satisfied(&egraph, root, &plan));

        let context = RewriteContext::new(SharedRewriteBudget::new());
        let (origin, replacement) =
            materialize_selected_polynomial_redex(&mut egraph, (root, &plan), &context)
                .expect("equivalent frontier plan materializes");
        assert!(!egraph.union(origin, replacement), "the first equivalent is already the root");
        egraph.rebuild();
        assert!(
            replacement_plan_satisfied(&egraph, root, &plan),
            "the internal Equivalent union satisfies the complete plan"
        );
    }

    #[test]
    fn detailed_large_diagnostic_uses_a_power_of_two_epoch_cadence() {
        let emitted = (0_u64..=140)
            .filter(|epoch| emit_detailed_large_diagnostic_for_epoch(*epoch))
            .collect::<Vec<_>>();
        assert_eq!(emitted, vec![0, 1, 2, 4, 8, 16, 32, 64, 128]);
        assert!(!emit_detailed_large_diagnostic_for_epoch(3));
        assert!(!emit_detailed_large_diagnostic_for_epoch(140));
    }

    #[test]
    fn selected_phase_postcondition_restarts_completes_or_fails_without_spinning() {
        assert_eq!(
            selected_phase_postcondition(1, 7, 9),
            Ok(SelectedPhasePostcondition::RestartOrdinary)
        );
        assert_eq!(selected_phase_postcondition(0, 0, 0), Ok(SelectedPhasePostcondition::Complete));
        assert_eq!(selected_phase_postcondition(0, 3, 0), Err((3, 0)));
        assert_eq!(
            selected_phase_postcondition(0, 0, 5),
            Err((0, 5)),
            "a hidden-only unresolved extraction must fail"
        );
    }

    #[test]
    fn relation_exposed_by_structural_work_restarts_before_structural_contraction() {
        assert_eq!(
            selected_relation_postcondition(1, 1),
            Ok(Some(SelectedPhasePostcondition::RestartOrdinary)),
            "a structural batch must return to ordinary saturation before comparing measures"
        );
        assert_eq!(selected_relation_postcondition(0, 0), Ok(None));
        assert_eq!(selected_relation_postcondition(0, 1), Err(1));
    }

    #[test]
    fn selected_relation_restart_requires_actual_rewrite_progress() {
        assert!(matches!(
            selected_relation_rewrite_delta(7, 7, 3),
            Err(super::super::error::RelationError::SelectedNormalizationBatchDidNotUnion {
                batch_size: 3,
            })
        ));
        assert_eq!(
            selected_relation_rewrite_delta(7, 9, 3),
            Ok(2),
            "a positive rewrite delta permits the fixed-point restart"
        );
    }

    #[test]
    fn rewrite_iteration_diagnostics_accumulate_saturatingly() {
        assert_eq!(accumulated_rewrite_iterations(4, 3), 7);
        assert_eq!(accumulated_rewrite_iterations(u64::MAX - 1, 3), u64::MAX);
    }

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
