//! Polynomial-time extraction of one deterministic operational-noise proposal.
//!
//! Relation applicability comes from the checked relation registry.  Every
//! selected matrix candidate is classified by the same authoritative source
//! resolver and node transfer used by final bound evaluation.

use super::{
    analysis::{
        MxxAnalysis, MxxSort, RelationProvenanceVisit, RelationUnavailableReason,
        resolved_constant, try_visit_relation_provenance,
    },
    bound::{
        BoundClass, BoundEvaluationError, BoundEvaluator, BoundInput, MatrixBound,
        SelectedChildBounds,
    },
    error::OperationalSimulationError,
    identity::{AtomicRelationRole, AtomicSourceId, AtomicSourceKey},
    language::MxxLang,
    relation::{
        PointwiseAddSwitchProbe, PointwiseAddSwitchReject, PointwiseDirectProbe,
        pointwise_add_switch_probe,
    },
};
use egg::{EGraph, Id, Language, RecExpr};
use std::{
    collections::{BTreeSet, HashSet},
    fmt,
};

/// The exact lexicographic preference used to select a final expression.
///
/// This cost is not a noise estimate.  Saturating arithmetic keeps it
/// monotone even when a compact e-graph represents an exponentially large AST.
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct ProposalCost {
    /// Checked relation rewrites that remain in this exact representative.
    /// This is an exact obligation that must be discharged before structural
    /// normalization is considered.
    pub unsatisfied_relation_redexes: u64,
    /// Checked relation boundaries in this exact raw e-node. This is
    /// deliberately not aggregated from children: each selected child chooses
    /// its materialized replacements locally.
    pub local_checked_relation_count: u64,
    /// Checked selected structural normalizations in this exact raw e-node.
    /// This root-local preference includes a satisfied raw recipe, while the
    /// applied normalized representation has no recipe at all. It is never
    /// propagated from children or across Switch cases.
    pub local_checked_structural_count: u64,
    /// Selected structural normalizations that remain in this exact
    /// representative. This is a separate exact obligation, ordered after
    /// relation work and its root-local preference.
    pub unsatisfied_structural_redexes: u64,
    /// Structural work made nonlocal by an Add. Relation work never enters
    /// this accounting: only selected structural normalization may be hidden.
    pub hidden_structural_redexes: u64,
    /// Whether this whole selected matrix expression is semantically Large.
    /// This is deliberately root-local so proved-zero operations annihilate
    /// Large children before proposal ordering.
    pub large_residual: bool,
    pub node_count: u64,
}

/// Facts whose authoritative owners live outside extraction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProposalNodeClassification {
    /// This exact e-node is a relation redex which a checked rewrite could consume.
    pub relation_redex: bool,
    /// Checked relation boundaries in this exact e-node, including boundaries
    /// whose replacement is already canonically equal to the enclosing class.
    /// It is an extraction preference only, never a final obligation.
    pub local_checked_relation_count: u64,
}

/// The extracted DAG and the cost that selected it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtractedProposal {
    pub cost: ProposalCost,
    /// Exact coefficient bound of the selected matrix root.  Generic
    /// non-matrix extraction users receive `None`; production rejects that
    /// case before final acceptance.
    pub semantic_bound: Option<MatrixBound>,
    /// Ephemeral diagnostic for a selected Large residual.  It is never
    /// stored in the e-graph, analysis data, or a source registry.
    pub first_large_source: Option<AtomicSourceId>,
    pub expression: RecExpr<MxxLang>,
}

/// Extraction-only provenance for the selected DAG.  Each entry has the same
/// index as `proposal.expression` and names the canonical e-class from which
/// that expression-local node was selected.  It is discarded before the
/// public simulation result and is never stored in the e-graph or analysis.
pub(crate) struct ExtractedProposalWithOrigins {
    pub(crate) proposal: ExtractedProposal,
    pub(crate) origins: Box<[Id]>,
}

/// Maps extraction failures that require the simulation driver's source site.
pub struct ExtractionControl<'a> {
    /// Maps an e-class with no finite DAG representative to the existing
    /// site-bearing analysis error owned by the driver.
    pub invalid_dag: &'a mut dyn FnMut(Id) -> OperationalSimulationError,
    /// Attaches the driver's graph site to a semantic transfer failure.
    pub bound_error: &'a mut dyn FnMut(BoundEvaluationError) -> OperationalSimulationError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExtractionState {
    Pending,
    Visiting,
    Complete,
}

#[derive(Clone, Debug)]
struct Candidate {
    cost: ProposalCost,
    semantic_bound: Option<MatrixBound>,
    first_large_source: Option<AtomicSourceId>,
    node: MxxLang,
    state: ExtractionState,
    output: Option<Id>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuildFrame {
    Enter(Id),
    Finish(Id),
}

/// Selects and materializes the best representative of `root`.
///
/// The classification callback is the sole relation integration hook.  It is
/// called with the containing canonical e-class and may inspect existing
/// `AnalysisData::relation_provenance` through `egraph`; the relation stage
/// must return `relation_redex = true` only after its complete typed identity
/// checks. It separately counts exact checked raw boundaries, even when their
/// replacements are already canonically satisfied. Matrix semantics are not
/// part of this callback: they are computed
/// from `bound_input` and already-selected child candidates with the final
/// evaluator's exact zero-first transfer.
/// The callback must be idempotent: relaxation may classify the same e-node in
/// several passes, and diagnostics must not count callback invocations.
///
/// Relaxation performs at most one pass per canonical e-class.  Every finite
/// optimum is cycle-free because `node_count` strictly increases across an
/// edge, so its height is at most the number of classes.  The resulting bound
/// is `O(C * N)` classification/cost work for `C` classes and `N` e-nodes.
pub fn extract_best_proposal<I: BoundInput>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    bound_input: &I,
    control: &mut ExtractionControl<'_>,
    classify: &mut dyn FnMut(
        Id,
        &MxxLang,
        &EGraph<MxxLang, MxxAnalysis>,
    ) -> Result<ProposalNodeClassification, OperationalSimulationError>,
) -> Result<ExtractedProposal, OperationalSimulationError> {
    extract_best_proposal_with_origins(egraph, root, bound_input, control, classify, true)
        .map(|extracted| extracted.proposal)
}

/// Internal form of [`extract_best_proposal`] which retains the selected
/// e-class origin beside every emitted expression-local node for one local
/// normalization epoch. `emit_detailed_large_diagnostic` controls only the
/// read-only path trace; the cheap Large summary and extraction result are
/// independent of it.
pub(crate) fn extract_best_proposal_with_origins<I: BoundInput>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    bound_input: &I,
    control: &mut ExtractionControl<'_>,
    classify: &mut dyn FnMut(
        Id,
        &MxxLang,
        &EGraph<MxxLang, MxxAnalysis>,
    ) -> Result<ProposalNodeClassification, OperationalSimulationError>,
    emit_detailed_large_diagnostic: bool,
) -> Result<ExtractedProposalWithOrigins, OperationalSimulationError> {
    let class_count = egraph.number_of_classes();
    let mut classes = Vec::with_capacity(class_count);
    for class in egraph.classes() {
        classes.push(class.id);
    }
    classes.sort_unstable();

    // Canonical egg ids are bounded by the number of inserted e-nodes.  One
    // indexed table holds both dynamic-programming choices and DAG build state.
    let slot_count = egraph.nodes().len();
    let mut candidates = vec![None::<Candidate>; slot_count];

    for _ in 0..class_count {
        let mut changed = false;
        for &class_id in &classes {
            let canonical = egraph.find(class_id);
            let index = usize::from(canonical);
            let class = &egraph[canonical];
            for node in class.iter() {
                let Some((cost, semantic_bound, first_large_source)) = proposal_cost(
                    egraph,
                    canonical,
                    node,
                    &candidates,
                    bound_input,
                    control,
                    classify,
                )?
                else {
                    continue;
                };
                let replace = candidates[index].as_ref().is_none_or(|current| {
                    cost < current.cost || (cost == current.cost && node < &current.node)
                });
                // A selected node can acquire a different semantic bound when
                // one of its selected children is refreshed at the same public
                // cost.  Keep that node current even if the derived ordering
                // cost worsens; a later scan then compares every alternative
                // against the refreshed candidate.
                let refresh = candidates[index].as_ref().is_some_and(|current| {
                    node == &current.node &&
                        (cost != current.cost ||
                            semantic_bound != current.semantic_bound ||
                            first_large_source != current.first_large_source)
                });
                if replace || refresh {
                    candidates[index] = Some(Candidate {
                        cost,
                        semantic_bound,
                        first_large_source,
                        node: node.clone(),
                        state: ExtractionState::Pending,
                        output: None,
                    });
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let root = egraph.find(root);
    let root_index = usize::from(root);
    if candidates.get(root_index).and_then(Option::as_ref).is_none() {
        return Err((control.invalid_dag)(root));
    }

    let mut work = vec![BuildFrame::Enter(root)];
    let mut nodes = Vec::<MxxLang>::new();
    let mut origins = Vec::<Id>::new();
    while let Some(frame) = work.pop() {
        let class = match frame {
            BuildFrame::Enter(class) | BuildFrame::Finish(class) => egraph.find(class),
        };
        let index = usize::from(class);
        match frame {
            BuildFrame::Enter(_) => {
                let Some(candidate) = candidates.get_mut(index).and_then(Option::as_mut) else {
                    return Err((control.invalid_dag)(class));
                };
                match candidate.state {
                    ExtractionState::Complete => continue,
                    ExtractionState::Visiting => return Err((control.invalid_dag)(class)),
                    ExtractionState::Pending => candidate.state = ExtractionState::Visiting,
                }
                work.push(BuildFrame::Finish(class));
                for &child in candidate.node.children().iter().rev() {
                    work.push(BuildFrame::Enter(egraph.find(child)));
                }
            }
            BuildFrame::Finish(_) => {
                let Some(candidate) = candidates.get(index).and_then(Option::as_ref) else {
                    return Err((control.invalid_dag)(class));
                };
                let mut missing_child = None;
                let output_node = candidate.node.clone().map_children(|child| {
                    let child = egraph.find(child);
                    let child_candidate =
                        candidates.get(usize::from(child)).and_then(Option::as_ref);
                    match child_candidate.and_then(|candidate| candidate.output) {
                        Some(output) => output,
                        None => {
                            missing_child = Some(child);
                            Id::from(0)
                        }
                    }
                });
                if let Some(child) = missing_child {
                    return Err((control.invalid_dag)(child));
                }
                // Relaxation selects by public lexicographic cost.  A child can
                // change to an equally priced finite alternative without changing
                // an ancestor's cost, so refresh the selected node only after its
                // selected children are complete.  This is the same zero-first
                // transfer used for final evaluation, not a second bound cache.
                let semantic_bound = matches!(&egraph[class].data.sort, Ok(MxxSort::Matrix(_)))
                    .then(|| {
                        let children = CandidateChildBounds { egraph, candidates: &candidates };
                        BoundEvaluator::evaluate_selected_node(
                            bound_input,
                            class,
                            &candidate.node,
                            &children,
                        )
                    })
                    .transpose()
                    .map_err(|source| (control.bound_error)(source))?;
                let first_large_source = selected_first_large_source(
                    egraph,
                    &candidate.node,
                    semantic_bound.as_ref(),
                    &candidates,
                );
                let output = Id::from(nodes.len());
                nodes.push(output_node);
                origins.push(class);
                let Some(candidate) = candidates[index].as_mut() else {
                    return Err((control.invalid_dag)(class));
                };
                candidate.semantic_bound = semantic_bound;
                candidate.cost.large_residual = candidate
                    .semantic_bound
                    .as_ref()
                    .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large));
                candidate.first_large_source = first_large_source;
                candidate.output = Some(output);
                candidate.state = ExtractionState::Complete;
            }
        }
    }

    let expression = RecExpr::from(nodes);
    if !expression.is_dag() {
        return Err((control.invalid_dag)(root));
    }
    let root_candidate = candidates
        .get(root_index)
        .and_then(Option::as_ref)
        .ok_or_else(|| (control.invalid_dag)(root))?;
    let selected_large = root_candidate
        .semantic_bound
        .as_ref()
        .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large));
    if selected_large {
        tracing::info!(
            event = "operational_selected_large_residual",
            selected_root = usize::from(egraph.find(root)),
            selected_expression_nodes = expression.as_ref().len(),
            selected_large_source_id = ?root_candidate.first_large_source.map(|source| source.0),
            "selected Large residual reaches final extraction"
        );
    }
    let source_less_info_enabled = selected_large &&
        root_candidate.first_large_source.is_none() &&
        tracing::enabled!(tracing::Level::INFO);
    let debug_enabled = selected_large && tracing::enabled!(tracing::Level::DEBUG);
    let detailed_diagnostic = (emit_detailed_large_diagnostic &&
        (source_less_info_enabled || debug_enabled))
        .then(|| {
            selected_large_diagnostic(
                egraph,
                root,
                root_candidate.first_large_source,
                &candidates,
                bound_input,
            )
        });
    if let Some(diagnostic) = detailed_diagnostic.as_ref() {
        if source_less_info_enabled {
            tracing::info!(
                event = "operational_selected_large_without_atomic_source",
                selected_root = usize::from(egraph.find(root)),
                selected_large_path = ?diagnostic,
                "selected Large residual has no atomic-source witness"
            );
        }
        if debug_enabled {
            tracing::debug!(
                selected_cost = ?root_candidate.cost,
                selected_large_source_id = ?root_candidate.first_large_source.map(|source| source.0),
                selected_large_source_kind = ?root_candidate
                    .first_large_source
                    .and_then(|source| selected_atomic_source_kind(egraph, source)),
                selected_large_path = ?diagnostic,
                egraph_classes = egraph.number_of_classes(),
                egraph_nodes = egraph.nodes().len(),
                "selected Large residual"
            );
        }
    }
    Ok(ExtractedProposalWithOrigins {
        proposal: ExtractedProposal {
            cost: root_candidate.cost.clone(),
            semantic_bound: root_candidate.semantic_bound.clone(),
            first_large_source: root_candidate.first_large_source,
            expression,
        },
        origins: origins.into_boxed_slice(),
    })
}

/// A bounded, read-only trace of one selected Large path.  It exists only in
/// the debug logging branch above; extraction never stores it in analysis or
/// proposal state.
struct SelectedLargePathStep {
    operator: &'static str,
    selected_cost: ProposalCost,
    /// The bounded outcome for every inspected eligible structure.  This is
    /// the authoritative pointwise diagnostic; it never substitutes one
    /// direct candidate's rejection for the whole e-class result.
    pointwise_add_switch_probe: Option<PointwiseAddSwitchProbe>,
    /// Detail-only first direct rejection used by the older product/sampler
    /// views below.  Its field name makes the limited scope explicit.
    pointwise_first_direct_reject: Option<PointwiseAddSwitchReject>,
    pointwise_negative_fixed_views: Option<Box<[SelectedClassView]>>,
    pointwise_negative_fixed_paths: Option<Box<[SelectedNegativeFixedPath]>>,
    /// Read-only stored-case evidence for a sampler-index mismatch across one
    /// eligible pointwise Add/Switch boundary.
    pointwise_switch_sampler_cases: Option<SelectedPointwiseSwitchSamplerCases>,
    /// Product leaves for the fixed signed terms at this selected Add.  This
    /// is failure-only evidence for an associativity mismatch; it is not a
    /// relation rewrite or an e-graph summary.
    pointwise_fixed_product_spines: Option<Box<[SelectedFixedProductSpine]>>,
    add_selected_large_child_count: Option<usize>,
    add_direct_child_inputs: Option<Box<[SelectedDiagnosticInput]>>,
    add_direct_child_omitted_child_count: usize,
    multiply_factors: Option<Box<[SelectedClassView]>>,
    multiply_omitted_factor_count: usize,
    /// The complete bounded ordered spine for this selected product.  This
    /// appears on both the root (positive) path and retained negative paths.
    product_spine: Option<SelectedProductSpine>,
    add_product_boundary: Option<SelectedAddProductBoundary>,
    matrix_concat: Option<SelectedMatrixConcat>,
    slice: Option<SelectedSlice>,
    hash_plain: Option<SelectedHashPlain>,
    following_switch_cases: Option<usize>,
    selected_switch_case: Option<usize>,
}

impl fmt::Debug for SelectedLargePathStep {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedLargePathStep")
            .field("operator", &self.operator)
            .field("selected_cost", &self.selected_cost)
            .field("pointwise_add_switch_probe", &self.pointwise_add_switch_probe)
            .field("pointwise_first_direct_reject", &self.pointwise_first_direct_reject)
            .field("pointwise_negative_fixed_views", &self.pointwise_negative_fixed_views)
            .field("pointwise_negative_fixed_paths", &self.pointwise_negative_fixed_paths)
            .field("pointwise_switch_sampler_cases", &self.pointwise_switch_sampler_cases)
            .field("pointwise_fixed_product_spines", &self.pointwise_fixed_product_spines)
            .field("add_selected_large_child_count", &self.add_selected_large_child_count)
            .field("add_direct_child_inputs", &self.add_direct_child_inputs)
            .field(
                "add_direct_child_omitted_child_count",
                &self.add_direct_child_omitted_child_count,
            )
            .field("multiply_factors", &self.multiply_factors)
            .field("multiply_omitted_factor_count", &self.multiply_omitted_factor_count)
            .field("product_spine", &self.product_spine)
            .field("add_product_boundary", &self.add_product_boundary)
            .field("matrix_concat", &self.matrix_concat)
            .field("slice", &self.slice)
            .field("hash_plain", &self.hash_plain)
            .field("following_switch_cases", &self.following_switch_cases)
            .field("selected_switch_case", &self.selected_switch_case)
            .finish()
    }
}

/// One retained negative fixed identity and the selected Large path below its
/// canonical base.  This is bounded, failure-only logging; it is not an
/// additional relation analysis or an e-graph cache.
struct SelectedNegativeFixedPath {
    canonical_eclass: usize,
    multiplicity: usize,
    steps: Box<[SelectedLargePathStep]>,
}

/// The selector and stored cases of the one eligible Switch used by a rejected
/// pointwise Add.  Each sampler search is bounded and failure-only.
struct SelectedPointwiseSwitchSamplerCases {
    selector: SelectedIntegerEClassView,
    fixed_sampler_occurrences: SelectedSamplerOccurrenceSet,
    cases: Box<[SelectedPointwiseSwitchSamplerCase]>,
    omitted_case_count: usize,
    common_sources: Box<[SelectedPointwiseCommonSamplerSource]>,
    omitted_common_source_count: usize,
    rotation_evidence: SelectedPointwiseRotationEvidence,
}

/// One exact stored case and the bounded result of looking for the source
/// observed below the retained negative fixed term.
struct SelectedPointwiseSwitchSamplerCase {
    case_index: usize,
    sampler_occurrences: SelectedSamplerOccurrenceSet,
}

/// One selected sampler occurrence.  Stored sampler coordinates and actual
/// Atom coordinates remain separate so a rotated family is visible without
/// changing identity or selection semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedSamplerOccurrence {
    sampler_source_key: super::identity::GraphWireSourceKey,
    sampler_contract: SelectedSamplerNonIndexContract,
    stored_canonical_index_views: Box<[SelectedIntegerEClassView]>,
    stored_omitted_index_count: usize,
    actual_canonical_index_views: Box<[SelectedIntegerEClassView]>,
    actual_omitted_index_count: usize,
}

/// A bounded selected-DAG sampler traversal retains sampler occurrences in
/// discovery order.  Duplicates and truncation make rotation evidence
/// inconclusive, but remain visible in the failure report.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedSamplerOccurrenceSet {
    occurrences: Box<[SelectedSamplerOccurrence]>,
    duplicate_source_keys: bool,
    truncated: bool,
    omitted_occurrence_count: usize,
}

/// The sampler meaning that must agree independently of the graph-wire
/// source and coordinate identities.
#[derive(Clone, Debug, Eq, PartialEq)]
enum SelectedSamplerNonIndexContract {
    Gaussian {
        max_coefficient_bound: super::identity::ResolvedIntExpr,
    },
    UniformInterval {
        minimum: super::identity::ResolvedIntExpr,
        maximum: super::identity::ResolvedIntExpr,
    },
    Preimage {
        public_eclass: usize,
        trapdoor_id: u32,
        target_eclass: usize,
        cutoff: super::identity::ResolvedIntExpr,
    },
    DecomposedHash {
        public_eclass: usize,
        target_eclass: usize,
        arguments: SelectedCanonicalEclasses,
        matrix_type: super::identity::ResolvedMatrixType,
        base: super::identity::ResolvedIntExpr,
        digit_count: super::identity::ResolvedIntExpr,
        small: bool,
        range_proved: bool,
    },
    GadgetDecomposition {
        public_eclass: usize,
        target_eclass: usize,
        base: super::identity::ResolvedIntExpr,
        digit_count: super::identity::ResolvedIntExpr,
        small: bool,
        range_proved: bool,
    },
}

struct SelectedPointwiseCommonSamplerSource {
    sampler_source_key: super::identity::GraphWireSourceKey,
    fixed: SelectedSamplerOccurrence,
    cases: Box<[SelectedSamplerOccurrence]>,
}

enum SelectedPointwiseRotationEvidence {
    Evidence(SelectedPointwiseRotationCandidate),
    Inconclusive { candidate_count: usize, duplicate_source_keys: bool, truncated: bool },
}

struct SelectedPointwiseRotationCandidate {
    sampler_source_key: super::identity::GraphWireSourceKey,
    sampler_contract: SelectedSamplerNonIndexContract,
    rotated_coordinate: usize,
    fixed: SelectedSamplerOccurrence,
    cases: Box<[SelectedSamplerOccurrence]>,
}

/// One signed fixed Add term and its bounded ordered product leaves.  A
/// product is associative only when both spines have the same ordered leaves;
/// this report deliberately does not make that proof or rewrite the graph.
struct SelectedFixedProductSpine {
    canonical_eclass: usize,
    negative: bool,
    multiplicity: usize,
    spine: SelectedProductSpine,
}

/// A read-only, iterative flattening of the selected physical product spine.
/// `ambiguous_competing_product` and `cycle` make an incomplete spine unusable
/// as evidence for a future exact relation rewrite.  `omitted_subtrees` is an
/// exact count of pending product positions left unexplored after the cap,
/// rather than a guessed count of their leaves.
struct SelectedProductSpine {
    leaves: Box<[SelectedClassView]>,
    ambiguous_competing_product: bool,
    cycle: bool,
    truncated: bool,
    omitted_subtrees: usize,
}

/// A bounded, selected-node-only view of a MatrixConcat.  It retains the
/// physical selected product spine for product inputs, and expands a direct
/// selected Add input by exactly one bounded child layer.  This is diagnostic
/// evidence only and never changes e-graph state or extraction selection.
struct SelectedMatrixConcat {
    axis: super::identity::Axis,
    inputs: Box<[SelectedDiagnosticInput]>,
    omitted_input_count: usize,
}

struct SelectedDiagnosticInput {
    canonical_eclass: usize,
    selected: SelectedDiagnosticInputView,
}

enum SelectedDiagnosticInputView {
    Product(SelectedProductSpine),
    /// Only a concat input may expand an Add.  Its retained children never
    /// expand nested Adds, so this diagnostic remains bounded by one layer.
    Add(SelectedDiagnosticAddChildren),
    Class(SelectedClassView),
}

struct SelectedDiagnosticAddChildren {
    inputs: Box<[SelectedDiagnosticInput]>,
    omitted_input_count: usize,
}

/// The first direct Add factor in one selected product, together with the
/// selected product spines on its two sides.  It deliberately skips the Add's
/// own siblings: this is only the multiplication context needed to compare
/// differently-associated signal products.
struct SelectedAddProductBoundary {
    add_eclass: usize,
    prefix: SelectedProductSpine,
    suffix: SelectedProductSpine,
}

impl fmt::Debug for SelectedFixedProductSpine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedFixedProductSpine")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("negative", &self.negative)
            .field("multiplicity", &self.multiplicity)
            .field("spine", &self.spine)
            .finish()
    }
}

impl fmt::Debug for SelectedProductSpine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedProductSpine")
            .field("leaves", &self.leaves)
            .field("ambiguous_competing_product", &self.ambiguous_competing_product)
            .field("cycle", &self.cycle)
            .field("truncated", &self.truncated)
            .field("omitted_subtrees", &self.omitted_subtrees)
            .finish()
    }
}

impl fmt::Debug for SelectedMatrixConcat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedMatrixConcat")
            .field("axis", &self.axis)
            .field("inputs", &self.inputs)
            .field("omitted_input_count", &self.omitted_input_count)
            .finish()
    }
}

impl fmt::Debug for SelectedDiagnosticInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedDiagnosticInput")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("selected", &self.selected)
            .finish()
    }
}

impl fmt::Debug for SelectedDiagnosticInputView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product(spine) => formatter.debug_tuple("Product").field(spine).finish(),
            Self::Add(children) => formatter.debug_tuple("Add").field(children).finish(),
            Self::Class(view) => formatter.debug_tuple("Class").field(view).finish(),
        }
    }
}

impl fmt::Debug for SelectedDiagnosticAddChildren {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedDiagnosticAddChildren")
            .field("inputs", &self.inputs)
            .field("omitted_input_count", &self.omitted_input_count)
            .finish()
    }
}

impl fmt::Debug for SelectedAddProductBoundary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedAddProductBoundary")
            .field("add_eclass", &self.add_eclass)
            .field("prefix", &self.prefix)
            .field("suffix", &self.suffix)
            .finish()
    }
}

impl fmt::Debug for SelectedNegativeFixedPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedNegativeFixedPath")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("multiplicity", &self.multiplicity)
            .field("steps", &self.steps)
            .finish()
    }
}

impl fmt::Debug for SelectedPointwiseSwitchSamplerCases {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedPointwiseSwitchSamplerCases")
            .field("selector", &self.selector)
            .field("fixed_sampler_occurrences", &self.fixed_sampler_occurrences)
            .field("cases", &self.cases)
            .field("omitted_case_count", &self.omitted_case_count)
            .field("common_sources", &self.common_sources)
            .field("omitted_common_source_count", &self.omitted_common_source_count)
            .field("rotation_evidence", &self.rotation_evidence)
            .finish()
    }
}

impl fmt::Debug for SelectedPointwiseSwitchSamplerCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedPointwiseSwitchSamplerCase")
            .field("case_index", &self.case_index)
            .field("sampler_occurrences", &self.sampler_occurrences)
            .finish()
    }
}

impl fmt::Debug for SelectedPointwiseCommonSamplerSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedPointwiseCommonSamplerSource")
            .field("sampler_source_key", &self.sampler_source_key)
            .field("fixed", &self.fixed)
            .field("cases", &self.cases)
            .finish()
    }
}

impl fmt::Debug for SelectedPointwiseRotationEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evidence(candidate) => {
                formatter.debug_tuple("Evidence").field(candidate).finish()
            }
            Self::Inconclusive { candidate_count, duplicate_source_keys, truncated } => formatter
                .debug_struct("Inconclusive")
                .field("candidate_count", candidate_count)
                .field("duplicate_source_keys", duplicate_source_keys)
                .field("truncated", truncated)
                .finish(),
        }
    }
}

impl fmt::Debug for SelectedPointwiseRotationCandidate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedPointwiseRotationCandidate")
            .field("sampler_source_key", &self.sampler_source_key)
            .field("sampler_contract", &self.sampler_contract)
            .field("rotated_coordinate", &self.rotated_coordinate)
            .field("fixed", &self.fixed)
            .field("cases", &self.cases)
            .finish()
    }
}

const MAX_SELECTED_LARGE_DIAGNOSTIC_STEPS: usize = 64;
const MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN: usize = 16;
const MAX_SELECTED_LARGE_PROVENANCE_VISITS: usize = 16;
const MAX_SELECTED_PRODUCT_SPINE_LEAVES: usize = 64;
const MAX_SELECTED_PRODUCT_SPINE_STEPS: usize = 64;

/// A bounded view of one already-selected e-class.  It deliberately records
/// only the selected e-node, rather than searching alternate e-nodes.
struct SelectedClassView {
    canonical_eclass: usize,
    selected_operator: &'static str,
    atom_source: Option<SelectedAtomSource>,
    relation_provenance: SelectedRelationProvenance,
    slice: Option<SelectedSlice>,
    hash_plain: Option<SelectedHashPlain>,
    gadget_decomposition: Option<SelectedGadgetDecomposition>,
    negate_bases: Option<SelectedNegateBases>,
}

struct SelectedNegateBases {
    bases: Box<[SelectedNegateBase]>,
    omitted_base_count: usize,
}

struct SelectedNegateBase {
    canonical_eclass: usize,
    nodes: Box<[SelectedShallowNode]>,
    omitted_node_count: usize,
    add_children: Box<[SelectedNegateBaseChild]>,
    omitted_add_child_count: usize,
}

struct SelectedNegateBaseChild {
    canonical_eclass: usize,
    selected_operator: &'static str,
    selected_direct_children: Box<[usize]>,
    omitted_child_count: usize,
    product_spine: Option<Box<SelectedProductSpine>>,
    nested_negate_base: Option<Box<SelectedNestedNegateBase>>,
}

struct SelectedNestedNegateBase {
    canonical_eclass: usize,
    physical_nodes: Box<[SelectedShallowNode]>,
    omitted_node_count: usize,
    selected: SelectedDiagnosticInputView,
}

impl fmt::Debug for SelectedNegateBases {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectedNegateBases")
            .field("bases", &self.bases)
            .field("omitted_base_count", &self.omitted_base_count)
            .finish()
    }
}
impl fmt::Debug for SelectedNegateBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectedNegateBase")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("nodes", &self.nodes)
            .field("omitted_node_count", &self.omitted_node_count)
            .field("add_children", &self.add_children)
            .field("omitted_add_child_count", &self.omitted_add_child_count)
            .finish()
    }
}
impl fmt::Debug for SelectedNegateBaseChild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectedNegateBaseChild")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("selected_operator", &self.selected_operator)
            .field("selected_direct_children", &self.selected_direct_children)
            .field("omitted_child_count", &self.omitted_child_count)
            .field("product_spine", &self.product_spine)
            .field("nested_negate_base", &self.nested_negate_base)
            .finish()
    }
}
impl fmt::Debug for SelectedNestedNegateBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SelectedNestedNegateBase")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("physical_nodes", &self.physical_nodes)
            .field("omitted_node_count", &self.omitted_node_count)
            .field("selected", &self.selected)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedShallowNode {
    operator_name: &'static str,
    canonical_children: Box<[usize]>,
    omitted_child_count: usize,
}

/// The already-interned authority behind a selected Atom leaf.  This records
/// identity metadata only: it deliberately excludes integer domains and
/// runtime/candidate values.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedAtomSource {
    source_id: u32,
    key: super::identity::AtomicSourceKey,
    sort: MxxSort,
    relation_role: Option<AtomicRelationRole>,
    canonical_indices: Box<[usize]>,
    canonical_index_views: Box<[SelectedIntegerEClassView]>,
    omitted_index_count: usize,
    sampler: Option<SelectedSamplerIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedCanonicalEclasses {
    retained: Box<[usize]>,
    views: Box<[SelectedIntegerEClassView]>,
    omitted_count: usize,
}

/// One nonrecursive view of a retained integer coordinate.  The selected
/// e-node is preferred; if extraction has no candidate for this scalar class,
/// the minimum physical e-node gives a deterministic diagnostic fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedIntegerEClassView {
    canonical_eclass: usize,
    operator: &'static str,
    binder: Option<super::identity::BinderDescriptor>,
    direct_canonical_children: Box<[usize]>,
    omitted_child_count: usize,
    integer_domain: Option<super::analysis::IntegerDomain>,
    scalar_provenance: Option<super::analysis::ScalarProvenance>,
}

/// Read-only expansion of an existing sampler descriptor.  Occurrence source
/// metadata stays separate from canonical semantic operands so two compact
/// sampler IDs can be compared without changing interner identity.
#[derive(Clone, Debug, Eq, PartialEq)]
enum SelectedSamplerIdentity {
    Gaussian {
        source_key: super::identity::GraphWireSourceKey,
        indices: SelectedCanonicalEclasses,
        max_coefficient_bound: super::identity::ResolvedIntExpr,
    },
    UniformInterval {
        source_key: super::identity::GraphWireSourceKey,
        indices: SelectedCanonicalEclasses,
        minimum: super::identity::ResolvedIntExpr,
        maximum: super::identity::ResolvedIntExpr,
    },
    Preimage {
        source_key: super::identity::GraphWireSourceKey,
        indices: SelectedCanonicalEclasses,
        public_eclass: usize,
        trapdoor_id: u32,
        target_eclass: usize,
        cutoff: super::identity::ResolvedIntExpr,
    },
    DecomposedHash {
        source_key: super::identity::GraphWireSourceKey,
        indices: SelectedCanonicalEclasses,
        public_eclass: usize,
        target_eclass: usize,
        arguments: SelectedCanonicalEclasses,
        matrix_type: super::identity::ResolvedMatrixType,
        base: super::identity::ResolvedIntExpr,
        digit_count: super::identity::ResolvedIntExpr,
        small: bool,
        range_proved: bool,
    },
    GadgetDecomposition(SelectedGadgetDecomposition),
}

/// The semantic part of a deterministic gadget-decomposition sampler.  The
/// graph-wire source remains separate so this diagnostic can distinguish two
/// occurrences that share a sampler meaning from one occurrence used twice.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedGadgetDecomposition {
    source_key: super::identity::GraphWireSourceKey,
    semantic: SelectedGadgetDecompositionSemanticKey,
}

/// The part of a gadget-decomposition identity that must agree for two
/// deterministic occurrences to denote the same sampled decomposition.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SelectedGadgetDecompositionSemanticKey {
    public_eclass: usize,
    target_eclass: usize,
    base: super::identity::ResolvedIntExpr,
    digit_count: super::identity::ResolvedIntExpr,
    small: bool,
    range_proved: bool,
    ordered_coordinate_eclasses: SelectedCanonicalEclasses,
}

struct SelectedRelationProvenance {
    observed_direct_sources: Box<[SelectedDirectRelationSource]>,
    observed_direct_source_count: usize,
    observed_unavailable_sources: Box<[SelectedUnavailableRelationSource]>,
    observed_unavailable_count: usize,
    observed_switch_count: usize,
    traversal_truncated: bool,
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SelectedDirectRelationSource {
    source_id: u32,
    relation_role: Option<AtomicRelationRole>,
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SelectedUnavailableRelationSource {
    source_id: u32,
    relation_role: Option<AtomicRelationRole>,
    reason: RelationUnavailableReason,
}

struct SelectedSlice {
    spec_id: u32,
    spec: Option<super::identity::SliceSpec>,
    canonical_input: usize,
    /// Resolved dimensions of the input sort already inferred by the e-graph.
    /// These are diagnostic context only; unresolved parameters remain absent.
    input_rows: Option<num_bigint::BigInt>,
    input_columns: Option<num_bigint::BigInt>,
}

struct SelectedHashPlain {
    query_id: u32,
    canonical_arguments: Box<[usize]>,
    omitted_argument_count: usize,
    physical_nodes: Box<[SelectedShallowNode]>,
    omitted_node_count: usize,
}

impl fmt::Debug for SelectedClassView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedClassView")
            .field("canonical_eclass", &self.canonical_eclass)
            .field("selected_operator", &self.selected_operator)
            .field("atom_source", &self.atom_source)
            .field("relation_provenance", &self.relation_provenance)
            .field("slice", &self.slice)
            .field("hash_plain", &self.hash_plain)
            .field("gadget_decomposition", &self.gadget_decomposition)
            .field("negate_bases", &self.negate_bases)
            .finish()
    }
}

impl fmt::Debug for SelectedRelationProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedRelationProvenance")
            .field("observed_direct_sources", &self.observed_direct_sources)
            .field("observed_direct_source_count", &self.observed_direct_source_count)
            .field("observed_unavailable_sources", &self.observed_unavailable_sources)
            .field("observed_unavailable_count", &self.observed_unavailable_count)
            .field("observed_switch_count", &self.observed_switch_count)
            .field("traversal_truncated", &self.traversal_truncated)
            .finish()
    }
}

impl fmt::Debug for SelectedSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedSlice")
            .field("spec_id", &self.spec_id)
            .field("spec", &self.spec)
            .field("canonical_input", &self.canonical_input)
            .field("input_rows", &self.input_rows)
            .field("input_columns", &self.input_columns)
            .finish()
    }
}

impl fmt::Debug for SelectedHashPlain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SelectedHashPlain")
            .field("query_id", &self.query_id)
            .field("canonical_arguments", &self.canonical_arguments)
            .field("omitted_argument_count", &self.omitted_argument_count)
            .field("physical_nodes", &self.physical_nodes)
            .field("omitted_node_count", &self.omitted_node_count)
            .finish()
    }
}

fn selected_relation_provenance(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    term: Id,
) -> SelectedRelationProvenance {
    let mut observed_direct_sources = Vec::new();
    let mut observed_direct_source_count = 0;
    let mut observed_unavailable_sources = Vec::new();
    let mut observed_unavailable_count = 0;
    let mut observed_switch_count = 0;
    let mut visits = 0;
    let completed = try_visit_relation_provenance(
        &egraph[egraph.find(term)].data.relation_provenance,
        || {
            if visits == MAX_SELECTED_LARGE_PROVENANCE_VISITS {
                false
            } else {
                visits += 1;
                true
            }
        },
        |visit| match visit {
            RelationProvenanceVisit::Direct(source) => {
                observed_direct_source_count += 1;
                if observed_direct_sources.len() < MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN {
                    observed_direct_sources.push(SelectedDirectRelationSource {
                        source_id: source.source.0,
                        relation_role: egraph
                            .analysis
                            .symbols
                            .atomic_sources
                            .get(source.source.0)
                            .and_then(|descriptor| descriptor.relation_role),
                    });
                }
            }
            RelationProvenanceVisit::Unavailable { source, reason } => {
                observed_unavailable_count += 1;
                if observed_unavailable_sources.len() < MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN {
                    observed_unavailable_sources.push(SelectedUnavailableRelationSource {
                        source_id: source.source.0,
                        relation_role: egraph
                            .analysis
                            .symbols
                            .atomic_sources
                            .get(source.source.0)
                            .and_then(|descriptor| descriptor.relation_role),
                        reason,
                    });
                }
            }
            RelationProvenanceVisit::Switch { .. } => observed_switch_count += 1,
        },
    );
    observed_direct_sources.sort_unstable();
    observed_direct_sources.dedup();
    observed_unavailable_sources.sort_unstable();
    observed_unavailable_sources.dedup();
    SelectedRelationProvenance {
        observed_direct_sources: observed_direct_sources.into_boxed_slice(),
        observed_direct_source_count,
        observed_unavailable_sources: observed_unavailable_sources.into_boxed_slice(),
        observed_unavailable_count,
        observed_switch_count,
        traversal_truncated: !completed,
    }
}

fn selected_slice(egraph: &EGraph<MxxLang, MxxAnalysis>, spec: u32, input: Id) -> SelectedSlice {
    let input = egraph.find(input);
    let (input_rows, input_columns) = match &egraph[input].data.sort {
        Ok(MxxSort::Matrix(matrix)) => {
            (resolved_constant(&matrix.rows), resolved_constant(&matrix.columns))
        }
        _ => (None, None),
    };
    SelectedSlice {
        spec_id: spec,
        spec: egraph.analysis.symbols.slices.get(spec).cloned(),
        canonical_input: usize::from(input),
        input_rows,
        input_columns,
    }
}

fn selected_hash_plain(
    query: u32,
    arguments: &[Id],
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    canonical: Id,
) -> SelectedHashPlain {
    let canonical_arguments = arguments
        .iter()
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .map(|argument| usize::from(egraph.find(*argument)))
        .collect();
    SelectedHashPlain {
        query_id: query,
        canonical_arguments,
        omitted_argument_count: arguments
            .len()
            .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
        physical_nodes: egraph[egraph.find(canonical)]
            .nodes
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|node| {
                let children = node.children();
                SelectedShallowNode {
                    operator_name: node.operator_name(),
                    canonical_children: children
                        .iter()
                        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                        .map(|child| usize::from(egraph.find(*child)))
                        .collect(),
                    omitted_child_count: children
                        .len()
                        .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                }
            })
            .collect(),
        omitted_node_count: egraph[egraph.find(canonical)]
            .nodes
            .len()
            .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
    }
}

fn selected_class_view(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    term: Id,
) -> Option<SelectedClassView> {
    let canonical = egraph.find(term);
    let node = &candidates.get(usize::from(canonical))?.as_ref()?.node;
    let (slice, hash_plain) = match node {
        MxxLang::MatrixSlice { spec, input } => {
            (Some(selected_slice(egraph, spec.0, input[0])), None)
        }
        MxxLang::HashPlain { query, arguments } => {
            (None, Some(selected_hash_plain(query.0, arguments, egraph, canonical)))
        }
        _ => (None, None),
    };
    let atom_source = selected_atom_source(egraph, candidates, node);
    let gadget_decomposition = atom_source.as_ref().and_then(|atom| match &atom.sampler {
        Some(SelectedSamplerIdentity::GadgetDecomposition(gadget)) => Some(gadget.clone()),
        _ => None,
    });
    let negate_bases = matches!(node, MxxLang::MatrixNegate(_)).then(|| {
        let bases = egraph[canonical]
            .nodes
            .iter()
            .filter_map(|node| match node {
                MxxLang::MatrixNegate([input]) => Some(usize::from(egraph.find(*input))),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let omitted_base_count = bases.len().saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
        SelectedNegateBases {
            bases: bases
                .into_iter()
                .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                .map(|base| {
                    let base_id = Id::from(base);
                    let nodes = egraph[base_id]
                        .nodes
                        .iter()
                        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                        .map(|node| {
                            let children = node.children();
                            SelectedShallowNode {
                                operator_name: node.operator_name(),
                                canonical_children: children
                                    .iter()
                                    .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                                    .map(|child| usize::from(egraph.find(*child)))
                                    .collect(),
                                omitted_child_count: children
                                    .len()
                                    .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                            }
                        })
                        .collect::<Vec<_>>();
                    let direct_children = egraph[base_id]
                        .nodes
                        .iter()
                        .filter_map(|node| match node {
                            MxxLang::MatrixAdd(children) => Some(children.iter().copied()),
                            _ => None,
                        })
                        .flatten()
                        .collect::<Vec<_>>();
                    let add_children = direct_children
                        .iter()
                        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                        .filter_map(|child| {
                            let canonical = egraph.find(*child);
                            let selected = &candidates.get(usize::from(canonical))?.as_ref()?.node;
                            let children = selected.children();
                            let nested_negate_base = match selected {
                                MxxLang::MatrixNegate([base]) => {
                                    let base = egraph.find(*base);
                                    let input =
                                        selected_diagnostic_input(egraph, candidates, base, true)?;
                                    Some(Box::new(SelectedNestedNegateBase {
                                        canonical_eclass: usize::from(base),
                                        physical_nodes: egraph[base]
                                            .nodes
                                            .iter()
                                            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                                            .map(|node| {
                                                let children = node.children();
                                                SelectedShallowNode {
                                                    operator_name: node.operator_name(),
                                                    canonical_children: children
                                                        .iter()
                                                        .take(
                                                            MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN,
                                                        )
                                                        .map(|id| usize::from(egraph.find(*id)))
                                                        .collect(),
                                                    omitted_child_count: children
                                                        .len()
                                                        .saturating_sub(
                                                            MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN,
                                                        ),
                                                }
                                            })
                                            .collect(),
                                        omitted_node_count: egraph[base]
                                            .nodes
                                            .len()
                                            .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                                        selected: input.selected,
                                    }))
                                }
                                _ => None,
                            };
                            Some(SelectedNegateBaseChild {
                                canonical_eclass: usize::from(canonical),
                                selected_operator: selected.operator_name(),
                                selected_direct_children: children
                                    .iter()
                                    .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                                    .map(|child| usize::from(egraph.find(*child)))
                                    .collect(),
                                omitted_child_count: children
                                    .len()
                                    .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                                product_spine: matches!(selected, MxxLang::MatrixMultiply(_)).then(
                                    || {
                                        Box::new(selected_product_spine(
                                            egraph, candidates, canonical,
                                        ))
                                    },
                                ),
                                nested_negate_base,
                            })
                        })
                        .collect::<Vec<_>>();
                    SelectedNegateBase {
                        canonical_eclass: base,
                        omitted_node_count: egraph[base_id]
                            .nodes
                            .len()
                            .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                        nodes: nodes.into_boxed_slice(),
                        omitted_add_child_count: direct_children
                            .len()
                            .saturating_sub(add_children.len()),
                        add_children: add_children.into_boxed_slice(),
                    }
                })
                .collect(),
            omitted_base_count,
        }
    });
    Some(SelectedClassView {
        canonical_eclass: usize::from(canonical),
        selected_operator: node.operator_name(),
        atom_source,
        relation_provenance: selected_relation_provenance(egraph, canonical),
        slice,
        hash_plain,
        gadget_decomposition,
        negate_bases,
    })
}

/// Reads one Atom descriptor from the existing symbol table.  It is kept next
/// to the selected leaf rather than derived from the e-class provenance so a
/// nonrelation source remains distinguishable in a product-spine diagnostic.
fn selected_atom_source(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    node: &MxxLang,
) -> Option<SelectedAtomSource> {
    let MxxLang::Atom { source, indices } = node else { return None };
    let descriptor = egraph.analysis.symbols.atomic_sources.get(source.0)?;
    let sampler = match &descriptor.key {
        super::identity::AtomicSourceKey::Sampler(sampler_id) => egraph
            .analysis
            .symbols
            .samplers
            .get(sampler_id.0)
            .map(|sampler| selected_sampler_identity(egraph, candidates, sampler)),
        _ => None,
    };
    Some(SelectedAtomSource {
        source_id: source.0,
        key: descriptor.key.clone(),
        sort: descriptor.sort.clone(),
        relation_role: descriptor.relation_role,
        canonical_indices: indices
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|index| usize::from(egraph.find(*index)))
            .collect(),
        canonical_index_views: indices
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|index| selected_integer_eclass_view(egraph, candidates, *index))
            .collect(),
        omitted_index_count: indices.len().saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
        sampler,
    })
}

fn selected_canonical_eclasses(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    ids: &[Id],
) -> SelectedCanonicalEclasses {
    SelectedCanonicalEclasses {
        retained: ids
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|id| usize::from(egraph.find(*id)))
            .collect(),
        views: ids
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|id| selected_integer_eclass_view(egraph, candidates, *id))
            .collect(),
        omitted_count: ids.len().saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
    }
}

fn selected_integer_eclass_view(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    id: Id,
) -> SelectedIntegerEClassView {
    let canonical = egraph.find(id);
    let node = candidates
        .get(usize::from(canonical))
        .and_then(Option::as_ref)
        .map(|candidate| &candidate.node)
        .unwrap_or_else(|| {
            egraph[canonical]
                .nodes
                .iter()
                .min()
                .expect("an e-class always contains at least one physical node")
        });
    let binder = match node {
        MxxLang::IntBinder(binder) => egraph.analysis.symbols.binders.get(binder.0).cloned(),
        _ => None,
    };
    SelectedIntegerEClassView {
        canonical_eclass: usize::from(canonical),
        operator: node.operator_name(),
        binder,
        direct_canonical_children: node
            .children()
            .iter()
            .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
            .map(|child| usize::from(egraph.find(*child)))
            .collect(),
        omitted_child_count: node
            .children()
            .len()
            .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
        integer_domain: egraph[canonical].data.integer_domain.clone(),
        scalar_provenance: egraph[canonical].data.scalar_provenance,
    }
}

fn selected_sampler_identity(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    sampler: &super::identity::SamplerIdentity,
) -> SelectedSamplerIdentity {
    use super::identity::SamplerIdentity;
    match sampler {
        SamplerIdentity::Gaussian { source, indices, max_coefficient_bound } => {
            SelectedSamplerIdentity::Gaussian {
                source_key: source.clone(),
                indices: selected_canonical_eclasses(egraph, candidates, indices),
                max_coefficient_bound: max_coefficient_bound.clone(),
            }
        }
        SamplerIdentity::UniformInterval { source, indices, minimum, maximum } => {
            SelectedSamplerIdentity::UniformInterval {
                source_key: source.clone(),
                indices: selected_canonical_eclasses(egraph, candidates, indices),
                minimum: minimum.clone(),
                maximum: maximum.clone(),
            }
        }
        SamplerIdentity::Preimage { source, indices, public, trapdoor, target, cutoff } => {
            SelectedSamplerIdentity::Preimage {
                source_key: source.clone(),
                indices: selected_canonical_eclasses(egraph, candidates, indices),
                public_eclass: usize::from(egraph.find(*public)),
                trapdoor_id: trapdoor.0,
                target_eclass: usize::from(egraph.find(*target)),
                cutoff: cutoff.clone(),
            }
        }
        SamplerIdentity::DecomposedHash {
            source,
            indices,
            public,
            target,
            arguments,
            matrix_type,
            base,
            digit_count,
            small,
            range_proved,
        } => SelectedSamplerIdentity::DecomposedHash {
            source_key: source.clone(),
            indices: selected_canonical_eclasses(egraph, candidates, indices),
            public_eclass: usize::from(egraph.find(*public)),
            target_eclass: usize::from(egraph.find(*target)),
            arguments: selected_canonical_eclasses(egraph, candidates, arguments),
            matrix_type: matrix_type.clone(),
            base: base.clone(),
            digit_count: digit_count.clone(),
            small: *small,
            range_proved: *range_proved,
        },
        SamplerIdentity::GadgetDecomposition {
            source,
            indices,
            public,
            target,
            base,
            digit_count,
            small,
            range_proved,
        } => SelectedSamplerIdentity::GadgetDecomposition(SelectedGadgetDecomposition {
            source_key: source.0.clone(),
            semantic: SelectedGadgetDecompositionSemanticKey {
                public_eclass: usize::from(egraph.find(*public)),
                target_eclass: usize::from(egraph.find(*target)),
                base: base.clone(),
                digit_count: digit_count.clone(),
                small: *small,
                range_proved: *range_proved,
                ordered_coordinate_eclasses: selected_canonical_eclasses(
                    egraph, candidates, indices,
                ),
            },
        }),
    }
}

fn negative_fixed_term_views(
    rejection: Option<&PointwiseAddSwitchReject>,
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
) -> Option<Box<[SelectedClassView]>> {
    let PointwiseAddSwitchReject::UnmatchedFixedTerms { fixed_terms, .. } = rejection? else {
        return None;
    };
    let views = fixed_terms
        .iter()
        .filter(|term| term.negative)
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .filter_map(|term| selected_class_view(egraph, candidates, Id::from(term.eclass)))
        .collect::<Vec<_>>();
    (!views.is_empty()).then(|| views.into_boxed_slice())
}

fn negative_fixed_term_paths<I: BoundInput>(
    rejection: Option<&PointwiseAddSwitchReject>,
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    bound_input: &I,
) -> Option<Box<[SelectedNegativeFixedPath]>> {
    let PointwiseAddSwitchReject::UnmatchedFixedTerms { fixed_terms, .. } = rejection? else {
        return None;
    };
    let paths = fixed_terms
        .iter()
        .filter(|term| term.negative)
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .map(|term| SelectedNegativeFixedPath {
            canonical_eclass: term.eclass,
            multiplicity: term.multiplicity,
            steps: selected_large_path_from(
                egraph,
                Id::from(term.eclass),
                candidates
                    .get(usize::from(egraph.find(Id::from(term.eclass))))
                    .and_then(Option::as_ref)
                    .and_then(|candidate| candidate.first_large_source),
                candidates,
                bound_input,
                false,
            )
            .into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    (!paths.is_empty()).then(|| paths.into_boxed_slice())
}

/// Reports every retained signed fixed identity at one rejected pointwise Add
/// boundary.  The multiplicity summary is already bounded by the relation
/// diagnostic; product flattening is separately capped and never mutates the
/// graph.  Keeping both polarities is essential when the mismatch is a
/// positive product versus its negative, differently associated counterpart.
fn fixed_product_spines(
    rejection: Option<&PointwiseAddSwitchReject>,
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
) -> Option<Box<[SelectedFixedProductSpine]>> {
    let PointwiseAddSwitchReject::UnmatchedFixedTerms { fixed_terms, .. } = rejection? else {
        return None;
    };
    (!fixed_terms.is_empty()).then(|| {
        fixed_terms
            .iter()
            .map(|term| SelectedFixedProductSpine {
                canonical_eclass: term.eclass,
                negative: term.negative,
                multiplicity: term.multiplicity,
                spine: selected_product_spine(egraph, candidates, Id::from(term.eclass)),
            })
            .collect()
    })
}

fn sampler_source_key(sampler: &SelectedSamplerIdentity) -> &super::identity::GraphWireSourceKey {
    match sampler {
        SelectedSamplerIdentity::Gaussian { source_key, .. } |
        SelectedSamplerIdentity::UniformInterval { source_key, .. } |
        SelectedSamplerIdentity::Preimage { source_key, .. } |
        SelectedSamplerIdentity::DecomposedHash { source_key, .. } => source_key,
        SelectedSamplerIdentity::GadgetDecomposition(gadget) => &gadget.source_key,
    }
}

fn sampler_stored_index_views(sampler: &SelectedSamplerIdentity) -> &[SelectedIntegerEClassView] {
    match sampler {
        SelectedSamplerIdentity::Gaussian { indices, .. } |
        SelectedSamplerIdentity::UniformInterval { indices, .. } |
        SelectedSamplerIdentity::Preimage { indices, .. } |
        SelectedSamplerIdentity::DecomposedHash { indices, .. } => &indices.views,
        SelectedSamplerIdentity::GadgetDecomposition(gadget) => {
            &gadget.semantic.ordered_coordinate_eclasses.views
        }
    }
}

fn sampler_stored_omitted_index_count(sampler: &SelectedSamplerIdentity) -> usize {
    match sampler {
        SelectedSamplerIdentity::Gaussian { indices, .. } |
        SelectedSamplerIdentity::UniformInterval { indices, .. } |
        SelectedSamplerIdentity::Preimage { indices, .. } |
        SelectedSamplerIdentity::DecomposedHash { indices, .. } => indices.omitted_count,
        SelectedSamplerIdentity::GadgetDecomposition(gadget) => {
            gadget.semantic.ordered_coordinate_eclasses.omitted_count
        }
    }
}

/// Iteratively examines selected e-classes.  It shares the selected-DAG rule
/// of the existing Large-path and product-spine diagnostics, but explores
/// every retained child so an enclosing Add or Multiply cannot hide the
/// sampler occurrence.  Work is bounded by 64 popped classes
/// and 16 retained child positions per class; callers additionally cap stored
/// Switch cases at 16.
fn sampler_non_index_contract(
    sampler: &SelectedSamplerIdentity,
) -> SelectedSamplerNonIndexContract {
    match sampler {
        SelectedSamplerIdentity::Gaussian { max_coefficient_bound, .. } => {
            SelectedSamplerNonIndexContract::Gaussian {
                max_coefficient_bound: max_coefficient_bound.clone(),
            }
        }
        SelectedSamplerIdentity::UniformInterval { minimum, maximum, .. } => {
            SelectedSamplerNonIndexContract::UniformInterval {
                minimum: minimum.clone(),
                maximum: maximum.clone(),
            }
        }
        SelectedSamplerIdentity::Preimage {
            public_eclass,
            trapdoor_id,
            target_eclass,
            cutoff,
            ..
        } => SelectedSamplerNonIndexContract::Preimage {
            public_eclass: *public_eclass,
            trapdoor_id: *trapdoor_id,
            target_eclass: *target_eclass,
            cutoff: cutoff.clone(),
        },
        SelectedSamplerIdentity::DecomposedHash {
            public_eclass,
            target_eclass,
            arguments,
            matrix_type,
            base,
            digit_count,
            small,
            range_proved,
            ..
        } => SelectedSamplerNonIndexContract::DecomposedHash {
            public_eclass: *public_eclass,
            target_eclass: *target_eclass,
            arguments: arguments.clone(),
            matrix_type: matrix_type.clone(),
            base: base.clone(),
            digit_count: digit_count.clone(),
            small: *small,
            range_proved: *range_proved,
        },
        SelectedSamplerIdentity::GadgetDecomposition(gadget) => {
            let semantic = &gadget.semantic;
            SelectedSamplerNonIndexContract::GadgetDecomposition {
                public_eclass: semantic.public_eclass,
                target_eclass: semantic.target_eclass,
                base: semantic.base.clone(),
                digit_count: semantic.digit_count.clone(),
                small: semantic.small,
                range_proved: semantic.range_proved,
            }
        }
    }
}

/// Every non-index contract field is fully retained except DecomposedHash
/// arguments, whose existing view is independently capped.  An omitted
/// argument makes equality evidence incomplete even when its retained prefix
/// happens to match.
fn sampler_non_index_contract_truncated(sampler: &SelectedSamplerIdentity) -> bool {
    match sampler {
        SelectedSamplerIdentity::DecomposedHash { arguments, .. } => arguments.omitted_count > 0,
        SelectedSamplerIdentity::Gaussian { .. } |
        SelectedSamplerIdentity::UniformInterval { .. } |
        SelectedSamplerIdentity::Preimage { .. } |
        SelectedSamplerIdentity::GadgetDecomposition(_) => false,
    }
}

fn selected_sampler_occurrences(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    start: Id,
) -> SelectedSamplerOccurrenceSet {
    let mut work = vec![egraph.find(start)];
    let mut visited = HashSet::new();
    let mut occurrences = Vec::new();
    let mut duplicate_source_keys = false;
    let mut truncated = false;
    let mut omitted_occurrence_count = 0;
    let mut steps = 0;
    while let Some(current) = work.pop() {
        if steps == MAX_SELECTED_LARGE_DIAGNOSTIC_STEPS {
            truncated = true;
            break;
        }
        let current = egraph.find(current);
        if !visited.insert(current) {
            continue;
        }
        steps += 1;
        let Some(candidate) = candidates.get(usize::from(current)).and_then(Option::as_ref) else {
            truncated = true;
            break;
        };
        if let Some(atom) =
            selected_class_view(egraph, candidates, current).and_then(|view| view.atom_source)
        {
            if let Some(sampler) = atom.sampler {
                let source_key = sampler_source_key(&sampler);
                duplicate_source_keys |=
                    occurrences.iter().any(|occurrence: &SelectedSamplerOccurrence| {
                        occurrence.sampler_source_key == *source_key
                    });
                if occurrences.len() == MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN {
                    truncated = true;
                    omitted_occurrence_count += 1;
                } else {
                    occurrences.push(SelectedSamplerOccurrence {
                        sampler_source_key: source_key.clone(),
                        sampler_contract: sampler_non_index_contract(&sampler),
                        stored_canonical_index_views: sampler_stored_index_views(&sampler)
                            .to_vec()
                            .into_boxed_slice(),
                        stored_omitted_index_count: sampler_stored_omitted_index_count(&sampler),
                        actual_canonical_index_views: atom.canonical_index_views,
                        actual_omitted_index_count: atom.omitted_index_count,
                    });
                    let occurrence = occurrences.last().expect("just pushed");
                    truncated |= occurrence.stored_omitted_index_count > 0 ||
                        occurrence.actual_omitted_index_count > 0 ||
                        sampler_non_index_contract_truncated(&sampler);
                }
            }
            continue;
        }
        let children =
            candidate.node.children().iter().map(|child| egraph.find(*child)).collect::<Vec<_>>();
        if children.len() > MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN {
            truncated = true;
            break;
        }
        work.extend(children.into_iter().rev());
    }
    SelectedSamplerOccurrenceSet {
        occurrences: occurrences.into_boxed_slice(),
        duplicate_source_keys,
        truncated,
        omitted_occurrence_count,
    }
}

/// Reports the exact stored cases retained by the pointwise relation failure.
/// The relation checker supplies these coordinates only after it has rejected
/// competing fixed Switches and nested or ambiguous case shapes.
fn pointwise_switch_sampler_cases(
    rejection: Option<&PointwiseAddSwitchReject>,
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
) -> Option<SelectedPointwiseSwitchSamplerCases> {
    let PointwiseAddSwitchReject::UnmatchedFixedTerms {
        fixed_terms, selector, switch_cases, ..
    } = rejection?
    else {
        return None;
    };
    let fixed_sampler_occurrences = fixed_terms
        .iter()
        .find(|term| term.negative)
        .map(|term| selected_sampler_occurrences(egraph, candidates, Id::from(term.eclass)))
        .unwrap_or(SelectedSamplerOccurrenceSet {
            occurrences: Box::new([]),
            duplicate_source_keys: false,
            truncated: false,
            omitted_occurrence_count: 0,
        });
    let selector = selected_integer_eclass_view(egraph, candidates, *selector);
    let case_count = switch_cases.len().saturating_sub(1);
    let cases: Vec<SelectedPointwiseSwitchSamplerCase> = switch_cases[1..]
        .iter()
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .enumerate()
        .map(|(case_index, case)| SelectedPointwiseSwitchSamplerCase {
            case_index,
            sampler_occurrences: selected_sampler_occurrences(egraph, candidates, *case),
        })
        .collect();
    let common_sources = fixed_sampler_occurrences
        .occurrences
        .iter()
        .filter_map(|fixed| {
            let cases = cases
                .iter()
                .map(|case| {
                    case.sampler_occurrences
                        .occurrences
                        .iter()
                        .find(|occurrence| {
                            occurrence.sampler_source_key == fixed.sampler_source_key
                        })
                        .cloned()
                })
                .collect::<Option<Vec<_>>>()?;
            Some(SelectedPointwiseCommonSamplerSource {
                sampler_source_key: fixed.sampler_source_key.clone(),
                fixed: fixed.clone(),
                cases: cases.into_boxed_slice(),
            })
        })
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .collect::<Vec<_>>();
    let omitted_common_source_count = fixed_sampler_occurrences
        .occurrences
        .iter()
        .filter(|fixed| {
            cases.iter().all(|case| {
                case.sampler_occurrences
                    .occurrences
                    .iter()
                    .any(|occurrence| occurrence.sampler_source_key == fixed.sampler_source_key)
            })
        })
        .count()
        .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
    let selector_eclass = selector.canonical_eclass;
    let mut candidates_for_rotation = Vec::new();
    let traversal_inconclusive = case_count > MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN ||
        fixed_sampler_occurrences.truncated ||
        fixed_sampler_occurrences.duplicate_source_keys ||
        cases.iter().any(|case| {
            case.sampler_occurrences.truncated || case.sampler_occurrences.duplicate_source_keys
        });
    if !traversal_inconclusive {
        for common in &common_sources {
            let matching_coordinates = common
                .fixed
                .actual_canonical_index_views
                .iter()
                .enumerate()
                .filter_map(|(index, view)| {
                    (view.canonical_eclass == selector_eclass).then_some(index)
                })
                .collect::<Vec<_>>();
            let [rotated_coordinate] = matching_coordinates.as_slice() else {
                continue;
            };
            if common.cases.iter().enumerate().all(|(case_index, case)| {
                case.sampler_contract == common.fixed.sampler_contract &&
                    case.actual_canonical_index_views.len() ==
                        common.fixed.actual_canonical_index_views.len() &&
                    case.actual_canonical_index_views.get(*rotated_coordinate).is_some_and(
                        |view| {
                            view.operator == "int-const" &&
                                view.integer_domain ==
                                    Some(super::analysis::IntegerDomain::Exact(
                                        case_index.into(),
                                    ))
                        },
                    ) &&
                    case.actual_canonical_index_views.iter().enumerate().all(|(index, view)| {
                        index == *rotated_coordinate ||
                            view.canonical_eclass ==
                                common.fixed.actual_canonical_index_views[index]
                                    .canonical_eclass
                    })
            }) {
                candidates_for_rotation.push(SelectedPointwiseRotationCandidate {
                    sampler_source_key: common.sampler_source_key.clone(),
                    sampler_contract: common.fixed.sampler_contract.clone(),
                    rotated_coordinate: *rotated_coordinate,
                    fixed: common.fixed.clone(),
                    cases: common.cases.clone(),
                });
            }
        }
    }
    let rotation_evidence = if !traversal_inconclusive && candidates_for_rotation.len() == 1 {
        SelectedPointwiseRotationEvidence::Evidence(candidates_for_rotation.pop().expect("one"))
    } else {
        SelectedPointwiseRotationEvidence::Inconclusive {
            candidate_count: candidates_for_rotation.len(),
            duplicate_source_keys: fixed_sampler_occurrences.duplicate_source_keys ||
                cases.iter().any(|case| case.sampler_occurrences.duplicate_source_keys),
            truncated: case_count > MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN ||
                fixed_sampler_occurrences.truncated ||
                cases.iter().any(|case| case.sampler_occurrences.truncated),
        }
    };
    Some(SelectedPointwiseSwitchSamplerCases {
        selector,
        fixed_sampler_occurrences,
        cases: cases.into_boxed_slice(),
        omitted_case_count: case_count.saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
        common_sources: common_sources.into_boxed_slice(),
        omitted_common_source_count,
        rotation_evidence,
    })
}

#[derive(Clone, Copy)]
enum ProductSpineFrame {
    Enter(Id),
    Exit(Id),
}

/// Iteratively expands the selected multiply node of each product position.
/// A class with two different physical multiply layouts is explicitly marked
/// ambiguous instead of choosing one.  The active-path set permits repeated
/// factors in a DAG while still rejecting a true product cycle.
fn selected_product_spine(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    start: Id,
) -> SelectedProductSpine {
    selected_product_spine_from_roots(egraph, candidates, &[start])
}

fn selected_product_spine_from_roots(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    roots: &[Id],
) -> SelectedProductSpine {
    let mut leaves = Vec::new();
    let mut stack = roots
        .iter()
        .rev()
        .map(|root| ProductSpineFrame::Enter(egraph.find(*root)))
        .collect::<Vec<_>>();
    let mut active = HashSet::new();
    let mut steps = 0;
    let mut ambiguous_competing_product = false;
    let mut cycle = false;
    let mut truncated = false;
    while let Some(frame) = stack.pop() {
        if steps == MAX_SELECTED_PRODUCT_SPINE_STEPS ||
            leaves.len() == MAX_SELECTED_PRODUCT_SPINE_LEAVES
        {
            truncated = true;
            stack.push(frame);
            break;
        }
        match frame {
            ProductSpineFrame::Exit(id) => {
                active.remove(&id);
            }
            ProductSpineFrame::Enter(id) => {
                steps += 1;
                let id = egraph.find(id);
                if !active.insert(id) {
                    cycle = true;
                    break;
                }
                let Some(selected) = candidates.get(usize::from(id)).and_then(Option::as_ref)
                else {
                    active.remove(&id);
                    truncated = true;
                    break;
                };
                let selected_factors = match &selected.node {
                    MxxLang::MatrixMultiply(factors) => factors,
                    _ => {
                        if let Some(view) = selected_class_view(egraph, candidates, id) {
                            leaves.push(view);
                        } else {
                            truncated = true;
                        }
                        active.remove(&id);
                        continue;
                    }
                };
                let selected_factors =
                    selected_factors.iter().map(|factor| egraph.find(*factor)).collect::<Vec<_>>();
                let mut physical = None;
                for node in &egraph[id].nodes {
                    let MxxLang::MatrixMultiply(factors) = node else { continue };
                    let factors =
                        factors.iter().map(|factor| egraph.find(*factor)).collect::<Vec<_>>();
                    match &physical {
                        Some(previous) if previous != &factors => {
                            ambiguous_competing_product = true;
                            break;
                        }
                        Some(_) => {}
                        None => physical = Some(factors),
                    }
                }
                if ambiguous_competing_product || physical.as_ref() != Some(&selected_factors) {
                    ambiguous_competing_product = true;
                    active.remove(&id);
                    break;
                }
                stack.push(ProductSpineFrame::Exit(id));
                for factor in selected_factors.into_iter().rev() {
                    stack.push(ProductSpineFrame::Enter(factor));
                }
            }
        }
    }
    SelectedProductSpine {
        leaves: leaves.into_boxed_slice(),
        ambiguous_competing_product,
        cycle,
        truncated,
        omitted_subtrees: stack
            .iter()
            .filter(|frame| matches!(frame, ProductSpineFrame::Enter(_)))
            .count(),
    }
}

fn selected_add_product_boundary(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    node: &MxxLang,
) -> Option<SelectedAddProductBoundary> {
    let MxxLang::MatrixMultiply(factors) = node else { return None };
    let add_index = factors.iter().position(|factor| {
        candidates
            .get(usize::from(egraph.find(*factor)))
            .and_then(Option::as_ref)
            .is_some_and(|candidate| matches!(candidate.node, MxxLang::MatrixAdd(_)))
    })?;
    Some(SelectedAddProductBoundary {
        add_eclass: usize::from(egraph.find(factors[add_index])),
        prefix: selected_product_spine_from_roots(egraph, candidates, &factors[..add_index]),
        suffix: selected_product_spine_from_roots(egraph, candidates, &factors[add_index + 1..]),
    })
}

fn selected_diagnostic_inputs(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    inputs: &[Id],
    expand_direct_add: bool,
) -> Box<[SelectedDiagnosticInput]> {
    inputs
        .iter()
        .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
        .filter_map(|input| {
            selected_diagnostic_input(egraph, candidates, *input, expand_direct_add)
        })
        .collect()
}

fn selected_diagnostic_input(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    input: Id,
    expand_direct_add: bool,
) -> Option<SelectedDiagnosticInput> {
    let canonical = egraph.find(input);
    let candidate = candidates.get(usize::from(canonical))?.as_ref()?;
    let selected = match &candidate.node {
        MxxLang::MatrixMultiply(_) => SelectedDiagnosticInputView::Product(selected_product_spine(
            egraph, candidates, canonical,
        )),
        MxxLang::MatrixAdd(children) if expand_direct_add => {
            SelectedDiagnosticInputView::Add(SelectedDiagnosticAddChildren {
                inputs: selected_diagnostic_inputs(egraph, candidates, children, false),
                omitted_input_count: children
                    .len()
                    .saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
            })
        }
        _ => {
            SelectedDiagnosticInputView::Class(selected_class_view(egraph, candidates, canonical)?)
        }
    };
    Some(SelectedDiagnosticInput { canonical_eclass: usize::from(canonical), selected })
}

fn selected_matrix_concat(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    axis: super::identity::Axis,
    inputs: &[Id],
) -> SelectedMatrixConcat {
    let omitted_input_count = inputs.len().saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
    let inputs = selected_diagnostic_inputs(egraph, candidates, inputs, true);
    SelectedMatrixConcat { axis, inputs, omitted_input_count }
}

fn selected_large_diagnostic<I: BoundInput>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
    target_source: Option<AtomicSourceId>,
    candidates: &[Option<Candidate>],
    bound_input: &I,
) -> Vec<SelectedLargePathStep> {
    selected_large_path_from(egraph, root, target_source, candidates, bound_input, true)
}

fn selected_atomic_source_kind(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    source: AtomicSourceId,
) -> Option<&'static str> {
    egraph.analysis.symbols.atomic_sources.get(source.0).map(|descriptor| match &descriptor.key {
        AtomicSourceKey::ProtocolInput(_) => "protocol-input",
        AtomicSourceKey::GraphWire(_) => "graph-wire",
        AtomicSourceKey::ExplicitLarge(_) => "explicit-large",
        AtomicSourceKey::SequentialState(_) => "sequential-state",
        AtomicSourceKey::SequentialRecurrence { .. } => "sequential-recurrence",
        AtomicSourceKey::Sampler(_) => "sampler",
    })
}

/// Follows only the selected Large child at every edge.  The root trace and
/// each retained negative fixed-term trace share this walker so diagnostics
/// cannot silently use different expression-selection rules.  When extraction
/// recorded a source, every selected edge must retain that same source; a
/// missing matching child terminates the diagnostic rather than changing
/// provenance.
fn selected_large_path_from<I: BoundInput>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    start: Id,
    target_source: Option<AtomicSourceId>,
    candidates: &[Option<Candidate>],
    bound_input: &I,
    include_negative_fixed_paths: bool,
) -> Vec<SelectedLargePathStep> {
    let mut steps = Vec::new();
    let mut visited = HashSet::new();
    let mut current = egraph.find(start);
    for _ in 0..MAX_SELECTED_LARGE_DIAGNOSTIC_STEPS {
        if !visited.insert(current) {
            break;
        }
        let Some(candidate) = candidates.get(usize::from(current)).and_then(Option::as_ref) else {
            break;
        };
        let node = &candidate.node;
        let pointwise_add_switch_probe = matches!(node, MxxLang::MatrixAdd(_))
            .then(|| pointwise_add_switch_probe(egraph, current));
        let pointwise_first_direct_reject = pointwise_add_switch_probe.as_ref().and_then(|probe| {
            probe.outcomes.iter().find_map(|outcome| match &outcome.direct {
                PointwiseDirectProbe::Rejected(reject) => Some(reject.clone()),
                PointwiseDirectProbe::Ready => None,
            })
        });
        let pointwise_negative_fixed_views =
            negative_fixed_term_views(pointwise_first_direct_reject.as_ref(), egraph, candidates);
        let pointwise_negative_fixed_paths = include_negative_fixed_paths
            .then(|| {
                negative_fixed_term_paths(
                    pointwise_first_direct_reject.as_ref(),
                    egraph,
                    candidates,
                    bound_input,
                )
            })
            .flatten();
        let pointwise_switch_sampler_cases = pointwise_switch_sampler_cases(
            pointwise_first_direct_reject.as_ref(),
            egraph,
            candidates,
        );
        let pointwise_fixed_product_spines =
            fixed_product_spines(pointwise_first_direct_reject.as_ref(), egraph, candidates);
        let (
            add_selected_large_child_count,
            add_direct_child_inputs,
            add_direct_child_omitted_child_count,
        ) = match node {
            MxxLang::MatrixAdd(children) => {
                let selected_large_child_count = children
                    .iter()
                    .filter(|child| candidate_has_large_bound(egraph, candidates, **child))
                    .count();
                let inputs = selected_diagnostic_inputs(egraph, candidates, children, false);
                (
                    Some(selected_large_child_count),
                    Some(inputs),
                    children.len().saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                )
            }
            _ => (None, None, 0),
        };
        let (multiply_factors, multiply_omitted_factor_count) = match node {
            MxxLang::MatrixMultiply(factors) => {
                let factor_count = factors.len();
                let views = factors
                    .iter()
                    .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                    .filter_map(|factor| selected_class_view(egraph, candidates, *factor))
                    .collect::<Vec<_>>();
                (
                    Some(views.into_boxed_slice()),
                    factor_count.saturating_sub(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN),
                )
            }
            _ => (None, 0),
        };
        let add_product_boundary = selected_add_product_boundary(egraph, candidates, node);
        let product_spine = matches!(node, MxxLang::MatrixMultiply(_))
            .then(|| selected_product_spine(egraph, candidates, current));
        let matrix_concat = match node {
            MxxLang::MatrixConcat { axis, inputs } => {
                Some(selected_matrix_concat(egraph, candidates, *axis, inputs))
            }
            _ => None,
        };
        let (slice, hash_plain) = match node {
            MxxLang::MatrixSlice { spec, input } => {
                (Some(selected_slice(egraph, spec.0, input[0])), None)
            }
            MxxLang::HashPlain { query, arguments } => {
                (None, Some(selected_hash_plain(query.0, arguments, egraph, current)))
            }
            _ => (None, None),
        };
        let selected_large_child = selected_large_child(egraph, candidates, node, target_source);
        let following_switch_cases = selected_large_child.and_then(|child| {
            match candidates
                .get(usize::from(egraph.find(child)))
                .and_then(Option::as_ref)
                .map(|candidate| &candidate.node)
            {
                Some(MxxLang::Switch(children)) => children.len().checked_sub(1),
                _ => None,
            }
        });
        let selected_switch_case = match node {
            MxxLang::Switch(children) => selected_large_child.and_then(|child| {
                children.iter().position(|candidate| *candidate == child)?.checked_sub(1)
            }),
            _ => None,
        };
        steps.push(SelectedLargePathStep {
            operator: node.operator_name(),
            selected_cost: candidate.cost.clone(),
            pointwise_add_switch_probe,
            pointwise_first_direct_reject,
            pointwise_negative_fixed_views,
            pointwise_negative_fixed_paths,
            pointwise_switch_sampler_cases,
            pointwise_fixed_product_spines,
            add_selected_large_child_count,
            add_direct_child_inputs,
            add_direct_child_omitted_child_count,
            multiply_factors,
            multiply_omitted_factor_count,
            product_spine,
            add_product_boundary,
            matrix_concat,
            slice,
            hash_plain,
            following_switch_cases,
            selected_switch_case,
        });
        let Some(next) = selected_large_child else { break };
        current = egraph.find(next);
    }
    steps
}

fn candidate_has_large_bound(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    child: Id,
) -> bool {
    candidates
        .get(usize::from(egraph.find(child)))
        .and_then(Option::as_ref)
        .and_then(|candidate| candidate.semantic_bound.as_ref())
        .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large))
}

fn selected_large_child(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    candidates: &[Option<Candidate>],
    node: &MxxLang,
    target_source: Option<AtomicSourceId>,
) -> Option<Id> {
    node.children().iter().copied().find(|child| {
        let canonical = egraph.find(*child);
        let Some(candidate) = candidates.get(usize::from(canonical)).and_then(Option::as_ref)
        else {
            return false;
        };
        candidate
            .semantic_bound
            .as_ref()
            .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large)) &&
            match target_source {
                Some(source) => candidate.first_large_source == Some(source),
                None => true,
            }
    })
}

/// Aggregates already-selected child rewrite obligations without allocation.
/// Switch cases are mutually exclusive at runtime: the selector is always
/// evaluated, while every encoded case remains conservatively covered by the
/// maximum case obligation. Physical DAG size is still the sum of all stored
/// children and remains an independent final tie-breaker.
fn selected_child_obligations(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    node: &MxxLang,
    candidates: &[Option<Candidate>],
) -> Option<(u64, u64, u64, u64)> {
    let cost = |child: Id| {
        candidates.get(usize::from(egraph.find(child)))?.as_ref().map(|candidate| &candidate.cost)
    };
    let mut node_count = 1_u64;
    for child in node.children() {
        node_count = node_count.saturating_add(cost(*child)?.node_count);
    }
    if let MxxLang::Switch(children) = node {
        let (&selector, cases) = children.split_first()?;
        if cases.is_empty() {
            return None;
        }
        let selector = cost(selector)?;
        let mut case_relation = 0_u64;
        let mut case_structural = 0_u64;
        let mut case_hidden_structural = 0_u64;
        for case in cases {
            let case = cost(*case)?;
            case_relation = case_relation.max(case.unsatisfied_relation_redexes);
            case_structural = case_structural.max(case.unsatisfied_structural_redexes);
            case_hidden_structural = case_hidden_structural.max(case.hidden_structural_redexes);
        }
        Some((
            selector.unsatisfied_relation_redexes.saturating_add(case_relation),
            selector.unsatisfied_structural_redexes.saturating_add(case_structural),
            selector.hidden_structural_redexes.saturating_add(case_hidden_structural),
            node_count,
        ))
    } else {
        let mut relation = 0_u64;
        let mut structural = 0_u64;
        let mut hidden_structural = 0_u64;
        for child in node.children() {
            let child = cost(*child)?;
            relation = relation.saturating_add(child.unsatisfied_relation_redexes);
            structural = structural.saturating_add(child.unsatisfied_structural_redexes);
            hidden_structural = hidden_structural.saturating_add(child.hidden_structural_redexes);
        }
        Some((relation, structural, hidden_structural, node_count))
    }
}

struct CandidateChildBounds<'a> {
    egraph: &'a EGraph<MxxLang, MxxAnalysis>,
    candidates: &'a [Option<Candidate>],
}

impl SelectedChildBounds for CandidateChildBounds<'_> {
    fn child_bound(&self, term: Id) -> Option<&MatrixBound> {
        let term = self.egraph.find(term);
        self.candidates
            .get(usize::from(term))
            .and_then(Option::as_ref)
            .and_then(|candidate| candidate.semantic_bound.as_ref())
    }
}

fn proposal_cost<I: BoundInput>(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    class: Id,
    node: &MxxLang,
    candidates: &[Option<Candidate>],
    bound_input: &I,
    control: &mut ExtractionControl<'_>,
    classify: &mut dyn FnMut(
        Id,
        &MxxLang,
        &EGraph<MxxLang, MxxAnalysis>,
    ) -> Result<ProposalNodeClassification, OperationalSimulationError>,
) -> Result<
    Option<(ProposalCost, Option<MatrixBound>, Option<AtomicSourceId>)>,
    OperationalSimulationError,
> {
    let Some((child_relation, child_structural, child_hidden_structural, node_count)) =
        selected_child_obligations(egraph, node, candidates)
    else {
        return Ok(None);
    };
    let classification = classify(class, node, egraph)?;
    let semantic_bound = matches!(&egraph[class].data.sort, Ok(MxxSort::Matrix(_)))
        .then(|| {
            let children = CandidateChildBounds { egraph, candidates };
            BoundEvaluator::evaluate_selected_node(bound_input, class, node, &children)
        })
        .transpose()
        .map_err(|source| (control.bound_error)(source))?;
    let large_residual = semantic_bound
        .as_ref()
        .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large));
    let first_large_source =
        selected_first_large_source(egraph, node, semantic_bound.as_ref(), candidates);
    Ok(Some((
        ProposalCost {
            unsatisfied_relation_redexes: child_relation
                .saturating_add(u64::from(classification.relation_redex)),
            local_checked_relation_count: classification.local_checked_relation_count,
            local_checked_structural_count: 0,
            unsatisfied_structural_redexes: child_structural,
            // At an addition all selected structural work below it is hidden exactly once;
            // an enclosing addition replaces this value with the same descendant count.
            hidden_structural_redexes: if matches!(node, MxxLang::MatrixAdd(_)) {
                child_structural
            } else {
                child_hidden_structural
            },
            large_residual,
            node_count,
        },
        semantic_bound,
        first_large_source,
    )))
}

fn selected_first_large_source(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    node: &MxxLang,
    semantic_bound: Option<&MatrixBound>,
    candidates: &[Option<Candidate>],
) -> Option<AtomicSourceId> {
    semantic_bound
        .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large))
        .then(|| match node {
            MxxLang::Atom { source, .. } => Some(*source),
            _ => node.children().iter().find_map(|child| {
                let child = egraph.find(*child);
                candidates
                    .get(usize::from(child))
                    .and_then(Option::as_ref)
                    .filter(|candidate| {
                        candidate.semantic_bound.as_ref().is_some_and(|bound| {
                            matches!(bound.coefficient_class, BoundClass::Large)
                        })
                    })
                    .and_then(|candidate| candidate.first_large_source)
            }),
        })
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        analysis::{MxxAnalysis, MxxSort, resolved_constant},
        bound::{BoundEvaluationError, ResolvedMatrixConstant},
        identity::{
            AtomicRelationRole, AtomicSourceDescriptor, AtomicSourceKey, BinderDescriptor,
            BinderId, BinderKey, CanonicalResidueConvention, CrtSpecId, GraphWireSourceKey,
            HashQuerySpec, HashTagPart, MatrixConstantSpecId, OccurrenceScope, ProgramKey,
            ResolvedIndexRange, ResolvedIntExpr, ResolvedMatrixType, SamplerDescriptorId,
            SamplerIdentity, SliceSpec, SliceSpecId, WireSourceKey,
        },
        relation::{
            PointwiseAddSwitchReject, RelationApplier, RelationRegistration, RelationSearcher,
            RewriteContext, SharedRewriteBudget,
        },
    };
    use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, WireRef, types::ConcreteMatrixType};
    use num_bigint::{BigInt, BigUint};
    use num_traits::ToPrimitive;
    use std::{
        cell::Cell,
        collections::BTreeMap,
        fmt,
        sync::{Arc, Mutex},
    };
    use tracing::{
        Event, Subscriber,
        field::{Field, Visit},
        level_filters::LevelFilter,
    };
    use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

    #[derive(Default)]
    struct DiagnosticEventFields {
        fields: BTreeMap<String, String>,
    }

    impl Visit for DiagnosticEventFields {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.fields.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    #[derive(Clone)]
    struct SelectedLargeDiagnosticCapture {
        level: LevelFilter,
        events: Arc<Mutex<Vec<BTreeMap<String, String>>>>,
    }

    impl SelectedLargeDiagnosticCapture {
        fn new(level: LevelFilter) -> Self {
            Self { level, events: Arc::default() }
        }
    }

    impl<S> Layer<S> for SelectedLargeDiagnosticCapture
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn max_level_hint(&self) -> Option<LevelFilter> {
            Some(self.level)
        }

        fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
            let mut event_fields = DiagnosticEventFields::default();
            event.record(&mut event_fields);
            if event_fields.fields.contains_key("selected_large_path") {
                self.events.lock().expect("diagnostic capture lock").push(event_fields.fields);
            }
        }
    }

    struct NoBounds;

    impl BoundInput for NoBounds {
        fn node(&self, _: Id) -> Option<&MxxLang> {
            None
        }
        fn matrix_type(&self, term: Id) -> Result<ConcreteMatrixType, BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn atom_bound(
            &self,
            _: AtomicSourceId,
            term: Id,
        ) -> Result<MatrixBound, BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn matrix_constant(
            &self,
            _: MatrixConstantSpecId,
            term: Id,
        ) -> Result<(ConcreteMatrixType, ResolvedMatrixConstant), BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn scalar_maximum_absolute(&self, term: Id) -> Result<BigUint, BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn lift_constant_polynomial_class(
            &self,
            term: Id,
            _: Id,
        ) -> Result<BoundClass, BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn crt_coefficients(
            &self,
            _: CrtSpecId,
            term: Id,
        ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
        fn validate_pack(&self, term: Id, _: usize) -> Result<(), BoundEvaluationError> {
            Err(BoundEvaluationError::NonMatrixTerm { term })
        }
    }

    #[derive(Default)]
    struct SemanticInput {
        nodes: BTreeMap<Id, MxxLang>,
        matrix_types: BTreeMap<Id, ConcreteMatrixType>,
        atom_classes: BTreeMap<AtomicSourceId, BoundClass>,
        missing: Option<AtomicSourceId>,
        reachable_cases: BTreeMap<Id, Box<[bool]>>,
    }

    fn scalar_matrix_type() -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(1.into()),
            columns: ResolvedIntExpr::Const(1.into()),
        }
    }

    fn concrete_scalar_matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: 17.into(), ring_dimension: 1, rows: 1, columns: 1 }
    }

    fn matrix_atom(egraph: &mut EGraph<MxxLang, MxxAnalysis>, name: &str) -> (Id, AtomicSourceId) {
        let source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            }));
        (egraph.add(MxxLang::Atom { source, indices: Box::new([]) }), source)
    }

    impl BoundInput for SemanticInput {
        fn node(&self, term: Id) -> Option<&MxxLang> {
            self.nodes.get(&term)
        }
        fn matrix_type(&self, term: Id) -> Result<ConcreteMatrixType, BoundEvaluationError> {
            Ok(self.matrix_types.get(&term).cloned().unwrap_or_else(concrete_scalar_matrix_type))
        }
        fn atom_bound(
            &self,
            source: AtomicSourceId,
            term: Id,
        ) -> Result<MatrixBound, BoundEvaluationError> {
            if self.missing == Some(source) {
                return Err(BoundEvaluationError::MissingInputBoundContract { term });
            }
            Ok(MatrixBound {
                matrix_type: self.matrix_type(term)?,
                coefficient_class: self.atom_classes[&source].clone(),
                metadata: super::super::bound::MatrixMetadata::unknown(),
            })
        }
        fn matrix_constant(
            &self,
            _: MatrixConstantSpecId,
            term: Id,
        ) -> Result<(ConcreteMatrixType, ResolvedMatrixConstant), BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidMatrixConstant { term })
        }
        fn scalar_maximum_absolute(&self, term: Id) -> Result<BigUint, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidMatrixScale { term })
        }
        fn lift_constant_polynomial_class(
            &self,
            term: Id,
            _: Id,
        ) -> Result<BoundClass, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidMatrixScale { term })
        }
        fn crt_coefficients(
            &self,
            _: CrtSpecId,
            term: Id,
        ) -> Result<Box<[BigInt]>, BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidCrtRecompose { term })
        }
        fn validate_pack(&self, term: Id, _: usize) -> Result<(), BoundEvaluationError> {
            Err(BoundEvaluationError::InvalidPack { term })
        }
        fn switch_reachable_cases(
            &self,
            term: Id,
            _: Id,
            _: usize,
        ) -> Result<Box<[bool]>, BoundEvaluationError> {
            self.reachable_cases
                .get(&term)
                .cloned()
                .ok_or(BoundEvaluationError::InvalidSwitchReachability { term })
        }
    }

    fn resolved_matrix_type(rows: i64, columns: i64) -> ResolvedMatrixType {
        ResolvedMatrixType {
            modulus: ResolvedIntExpr::Const(17.into()),
            ring_dimension: ResolvedIntExpr::Const(1.into()),
            rows: ResolvedIntExpr::Const(rows.into()),
            columns: ResolvedIntExpr::Const(columns.into()),
        }
    }

    fn typed_matrix_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        name: &str,
        rows: i64,
        columns: i64,
        relation_role: Option<AtomicRelationRole>,
    ) -> (Id, AtomicSourceId) {
        let source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
                sort: MxxSort::Matrix(resolved_matrix_type(rows, columns)),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role,
            }));
        (egraph.add(MxxLang::Atom { source, indices: Box::new([]) }), source)
    }

    fn gadget_decomposition_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        wire_node: u32,
        public: Id,
        target: Id,
        coordinate: Id,
        base: i64,
    ) -> (Id, GraphWireSourceKey) {
        let source = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(wire_node.into()), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let sampler =
            egraph.analysis.symbols.samplers.intern(SamplerIdentity::GadgetDecomposition {
                source: source.clone().into(),
                indices: vec![coordinate].into(),
                public,
                target,
                base: ResolvedIntExpr::Const(base.into()),
                digit_count: ResolvedIntExpr::Const(2.into()),
                small: false,
                range_proved: false,
            });
        let atom_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: Some(AtomicRelationRole::GadgetDecomposition),
            }));
        (
            egraph.add(MxxLang::Atom { source: atom_source, indices: vec![coordinate].into() }),
            source,
        )
    }

    fn nonrelation_sampler_atom(
        egraph: &mut EGraph<MxxLang, MxxAnalysis>,
        sampler: SamplerIdentity,
        atom_indices: Box<[Id]>,
    ) -> Id {
        let sampler = egraph.analysis.symbols.samplers.intern(sampler);
        let source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::Sampler(SamplerDescriptorId(sampler)),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            }));
        egraph.add(MxxLang::Atom { source, indices: atom_indices })
    }

    fn populate_matrix_types(input: &mut SemanticInput, egraph: &EGraph<MxxLang, MxxAnalysis>) {
        for class in egraph.classes() {
            let Ok(MxxSort::Matrix(matrix)) = &class.data.sort else { continue };
            let (Some(modulus), Some(ring_dimension), Some(rows), Some(columns)) = (
                resolved_constant(&matrix.modulus),
                resolved_constant(&matrix.ring_dimension),
                resolved_constant(&matrix.rows),
                resolved_constant(&matrix.columns),
            ) else {
                panic!("typed extraction fixture only uses resolved matrix dimensions");
            };
            input.matrix_types.insert(
                class.id,
                ConcreteMatrixType {
                    modulus,
                    ring_dimension: ring_dimension.to_usize().expect("small ring dimension"),
                    rows: rows.to_usize().expect("small row count"),
                    columns: columns.to_usize().expect("small column count"),
                },
            );
        }
    }

    fn extract_with_input(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        root: Id,
        input: &SemanticInput,
    ) -> ExtractedProposal {
        let mut invalid = |_| panic!("valid test graph must have a finite DAG representative");
        let mut bound_error = |error| panic!("valid semantic candidate failed: {error:?}");
        extract_best_proposal(
            egraph,
            root,
            input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |_, _, _| Ok(ProposalNodeClassification::default()),
        )
        .unwrap()
    }

    fn diagnostic_candidates(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        large_classes: &[Id],
    ) -> Vec<Option<Candidate>> {
        let mut candidates = vec![None; egraph.nodes().len()];
        for class in egraph.classes() {
            let class = egraph.find(class.id);
            let node = egraph[class].nodes[0].clone();
            let semantic_bound =
                matches!(egraph[class].data.sort, Ok(MxxSort::Matrix(_))).then(|| MatrixBound {
                    matrix_type: concrete_scalar_matrix_type(),
                    coefficient_class: if large_classes.iter().any(|id| egraph.find(*id) == class) {
                        BoundClass::Large
                    } else {
                        BoundClass::bounded(1_u8.into())
                    },
                    metadata: super::super::bound::MatrixMetadata::unknown(),
                });
            candidates[usize::from(class)] = Some(Candidate {
                cost: ProposalCost {
                    large_residual: semantic_bound
                        .as_ref()
                        .is_some_and(|bound| matches!(bound.coefficient_class, BoundClass::Large)),
                    node_count: 1,
                    ..Default::default()
                },
                semantic_bound,
                first_large_source: None,
                node,
                state: ExtractionState::Complete,
                output: None,
            });
        }
        candidates
    }

    #[test]
    fn selected_large_diagnostic_follows_one_switch_case_and_caps_depth() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (first, _) = matrix_atom(&mut egraph, "first");
        let (second, _) = matrix_atom(&mut egraph, "second");
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch].into_boxed_slice()));
        egraph.rebuild();
        let candidates = diagnostic_candidates(&egraph, &[root, switch, first, second]);
        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);

        assert!(steps.iter().any(|step| step.following_switch_cases == Some(2)));
        assert!(steps.iter().any(|step| step.selected_switch_case == Some(0)));

        let mut root = first;
        let mut large = vec![first];
        for _ in 0..64 {
            root = egraph.add(MxxLang::MatrixAdd(vec![root].into_boxed_slice()));
            large.push(root);
        }
        egraph.rebuild();
        let candidates = diagnostic_candidates(&egraph, &large);
        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);

        assert_eq!(steps.len(), MAX_SELECTED_LARGE_DIAGNOSTIC_STEPS);
    }

    #[test]
    fn selected_large_diagnostic_follows_the_recorded_large_source() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (first, first_source) = matrix_atom(&mut egraph, "first-large");
        let (second, second_source) = matrix_atom(&mut egraph, "second-large");
        let switch = egraph.add(MxxLang::Switch(vec![selector, first, second].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch].into_boxed_slice()));
        egraph.rebuild();

        let mut candidates = diagnostic_candidates(&egraph, &[root, switch, first, second]);
        for (class, source) in [
            (root, second_source),
            (switch, second_source),
            (first, first_source),
            (second, second_source),
        ] {
            candidates[usize::from(egraph.find(class))]
                .as_mut()
                .expect("diagnostic candidate")
                .first_large_source = Some(source);
        }
        let before = egraph.total_size();

        let steps =
            selected_large_diagnostic(&egraph, root, Some(second_source), &candidates, &NoBounds);

        assert_eq!(egraph.total_size(), before, "diagnostic selection is read-only");
        assert!(steps.len() <= MAX_SELECTED_LARGE_DIAGNOSTIC_STEPS);
        assert!(steps.iter().any(|step| step.selected_switch_case == Some(1)));
        assert!(steps.iter().all(|step| step.selected_switch_case != Some(0)));
    }

    #[test]
    fn selected_product_spine_is_associative_but_not_commutative() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (scale, _) = matrix_atom(&mut egraph, "spine-scale");
        let (public, _) = matrix_atom(&mut egraph, "spine-public");
        let (digit, _) = matrix_atom(&mut egraph, "spine-digit");
        let left = egraph.add(MxxLang::MatrixMultiply(vec![scale, public].into()));
        let left_associated = egraph.add(MxxLang::MatrixMultiply(vec![left, digit].into()));
        let right = egraph.add(MxxLang::MatrixMultiply(vec![public, digit].into()));
        let right_associated = egraph.add(MxxLang::MatrixMultiply(vec![scale, right].into()));
        let reordered = egraph.add(MxxLang::MatrixMultiply(vec![public, scale, digit].into()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &[left_associated, right_associated, reordered, scale, public, digit],
        );
        let left = selected_product_spine(&egraph, &candidates, left_associated);
        let right = selected_product_spine(&egraph, &candidates, right_associated);
        let reordered = selected_product_spine(&egraph, &candidates, reordered);
        let ids = |spine: &SelectedProductSpine| {
            spine.leaves.iter().map(|leaf| leaf.canonical_eclass).collect::<Vec<_>>()
        };
        assert_eq!(ids(&left), ids(&right));
        assert_ne!(ids(&left), ids(&reordered));
        assert!(!left.ambiguous_competing_product && !left.cycle && !left.truncated);
        assert_eq!(egraph.total_size(), before, "diagnostic is read-only");
    }

    #[test]
    fn selected_product_spine_rejects_ambiguous_cycles_and_caps_work() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (first, _) = matrix_atom(&mut egraph, "spine-first");
        let (second, _) = matrix_atom(&mut egraph, "spine-second");
        let (third, _) = matrix_atom(&mut egraph, "spine-third");
        let first_product = egraph.add(MxxLang::MatrixMultiply(vec![first, second].into()));
        let second_product = egraph.add(MxxLang::MatrixMultiply(vec![first, third].into()));
        egraph.union(first_product, second_product);
        egraph.rebuild();
        let candidates = diagnostic_candidates(&egraph, &[first_product, first, second, third]);
        let ambiguous = selected_product_spine(&egraph, &candidates, first_product);
        assert!(ambiguous.ambiguous_competing_product);

        let cyclic = egraph.add(MxxLang::MatrixMultiply(vec![first].into()));
        egraph.union(cyclic, first);
        let cyclic = egraph.find(cyclic);
        let selected_cycle = egraph[cyclic]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("union retains physical product")
            .clone();
        let mut candidates = diagnostic_candidates(&egraph, &[cyclic]);
        candidates[usize::from(cyclic)].as_mut().expect("candidate").node = selected_cycle;
        let cycle = selected_product_spine(&egraph, &candidates, cyclic);
        assert!(cycle.cycle);

        let mut capped_egraph = EGraph::new(MxxAnalysis::default());
        let leaves = (0..65)
            .map(|index| matrix_atom(&mut capped_egraph, &format!("spine-cap-{index}")).0)
            .collect::<Vec<_>>();
        let capped = capped_egraph.add(MxxLang::MatrixMultiply(leaves.into()));
        capped_egraph.rebuild();
        let candidates = diagnostic_candidates(&capped_egraph, &[capped]);
        let capped = selected_product_spine(&capped_egraph, &candidates, capped);
        assert!(capped.truncated);
        assert!(!capped.leaves.is_empty());
        assert!(capped.leaves.len() <= MAX_SELECTED_PRODUCT_SPINE_LEAVES);
        assert!(capped.omitted_subtrees > 0);
    }

    #[test]
    fn selected_product_boundary_reports_only_the_add_context() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (prefix, _) = matrix_atom(&mut egraph, "boundary-prefix");
        let (left, _) = matrix_atom(&mut egraph, "boundary-left");
        let (right, _) = matrix_atom(&mut egraph, "boundary-right");
        let (suffix, _) = matrix_atom(&mut egraph, "boundary-suffix");
        let add = egraph.add(MxxLang::MatrixAdd(vec![left, right].into()));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![prefix, add, suffix].into()));
        egraph.rebuild();
        let candidates = diagnostic_candidates(&egraph, &[product]);
        let node = candidates[usize::from(egraph.find(product))]
            .as_ref()
            .expect("product candidate")
            .node
            .clone();
        let boundary =
            selected_add_product_boundary(&egraph, &candidates, &node).expect("direct Add factor");
        assert_eq!(boundary.add_eclass, usize::from(egraph.find(add)));
        assert_eq!(boundary.prefix.leaves.len(), 1);
        assert_eq!(boundary.suffix.leaves.len(), 1);
        assert_eq!(boundary.prefix.leaves[0].canonical_eclass, usize::from(egraph.find(prefix)));
        assert_eq!(boundary.suffix.leaves[0].canonical_eclass, usize::from(egraph.find(suffix)));
    }

    #[test]
    fn product_spine_keeps_gadget_semantics_and_retained_wire_audit_separate() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let coordinate = egraph.add(MxxLang::IntConst(0.into()));
        let (public, _) = matrix_atom(&mut egraph, "gadget-spine-public");
        let (target, _) = matrix_atom(&mut egraph, "gadget-spine-target");
        let (other_target, _) = matrix_atom(&mut egraph, "gadget-spine-other-target");
        let (first, first_source) =
            gadget_decomposition_atom(&mut egraph, 31_106, public, target, coordinate, 4);
        let (second, second_source) =
            gadget_decomposition_atom(&mut egraph, 31_108, public, target, coordinate, 4);
        let (different_target, _) =
            gadget_decomposition_atom(&mut egraph, 31_109, public, other_target, coordinate, 4);
        let (different_base, _) =
            gadget_decomposition_atom(&mut egraph, 31_110, public, target, coordinate, 8);
        egraph.rebuild();
        let candidates =
            diagnostic_candidates(&egraph, &[first, second, different_target, different_base]);
        let view = |term| {
            selected_class_view(&egraph, &candidates, term)
                .and_then(|view| view.gadget_decomposition)
                .expect("gadget decomposition view")
        };
        let first = view(first);
        let second = view(second);
        let different_target = view(different_target);
        let different_base = view(different_base);
        assert_eq!(first.semantic, second.semantic);
        assert_ne!(first_source, second_source, "fixture uses distinct producing occurrences");
        assert_eq!(
            first.source_key, second.source_key,
            "semantic interning retains the first occurrence only as audit metadata"
        );
        assert_eq!(first.source_key, first_source);
        assert_ne!(first.semantic, different_target.semantic);
        assert_ne!(first.semantic, different_base.semantic);
    }

    #[test]
    fn product_spine_atom_views_keep_authoritative_source_kinds_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (protocol, protocol_source) = matrix_atom(&mut egraph, "atom-view-protocol");
        let graph_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(44), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let graph_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::GraphWire(graph_key.clone()),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            }));
        let graph = egraph.add(MxxLang::Atom { source: graph_source, indices: Box::new([]) });
        let coordinate = egraph.add(MxxLang::IntConst(0.into()));
        let (public, _) = matrix_atom(&mut egraph, "atom-view-public");
        let (target, _) = matrix_atom(&mut egraph, "atom-view-target");
        let (sampler, _) =
            gadget_decomposition_atom(&mut egraph, 45, public, target, coordinate, 4);
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(&egraph, &[protocol, graph, sampler]);
        let source = |term| {
            selected_class_view(&egraph, &candidates, term)
                .and_then(|view| view.atom_source)
                .expect("atom source view")
        };
        let protocol_view = source(protocol);
        let graph_view = source(graph);
        let sampler_view = source(sampler);
        assert_eq!(protocol_view.source_id, protocol_source.0);
        assert!(matches!(
            protocol_view.key,
            AtomicSourceKey::ProtocolInput(ref input) if input == &crate::ProtocolInputId::from("atom-view-protocol")
        ));
        assert_eq!(graph_view.key, AtomicSourceKey::GraphWire(graph_key));
        assert!(matches!(sampler_view.key, AtomicSourceKey::Sampler(_)));
        assert!(protocol_view.sampler.is_none());
        assert!(graph_view.sampler.is_none());
        assert!(sampler_view.sampler.is_some());
        assert_eq!(protocol_view.sort, MxxSort::Matrix(scalar_matrix_type()));
        assert_eq!(graph_view.sort, MxxSort::Matrix(scalar_matrix_type()));
        assert_eq!(sampler_view.sort, MxxSort::Matrix(scalar_matrix_type()));
        assert_eq!(
            egraph.total_size(),
            before,
            "atom-source diagnosis must not mutate the e-graph"
        );
    }

    #[test]
    fn atom_source_view_expands_nonrelation_sampler_semantics() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let first_index = egraph.add(MxxLang::IntConst(3.into()));
        let second_index = egraph.add(MxxLang::IntConst(5.into()));
        let source_key = |wire_node| GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(wire_node), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let gaussian_source = source_key(51);
        let uniform_source = source_key(52);
        let gaussian = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::Gaussian {
                source: gaussian_source.clone(),
                indices: vec![first_index, second_index].into(),
                max_coefficient_bound: ResolvedIntExpr::Const(7.into()),
            },
            vec![first_index, second_index].into(),
        );
        let uniform = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::UniformInterval {
                source: uniform_source.clone(),
                indices: vec![second_index].into(),
                minimum: ResolvedIntExpr::Const((-2).into()),
                maximum: ResolvedIntExpr::Const(9.into()),
            },
            vec![second_index].into(),
        );
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(&egraph, &[gaussian, uniform]);
        let sampler = |term| {
            selected_class_view(&egraph, &candidates, term)
                .and_then(|view| view.atom_source)
                .and_then(|atom| atom.sampler)
                .expect("expanded sampler identity")
        };
        let SelectedSamplerIdentity::Gaussian { source_key, indices, max_coefficient_bound } =
            sampler(gaussian)
        else {
            panic!("Gaussian variant is retained")
        };
        assert_eq!(source_key, gaussian_source);
        assert_eq!(
            indices.retained.as_ref(),
            &[usize::from(egraph.find(first_index)), usize::from(egraph.find(second_index))]
        );
        assert_eq!(indices.omitted_count, 0);
        assert_eq!(max_coefficient_bound, ResolvedIntExpr::Const(7.into()));
        let SelectedSamplerIdentity::UniformInterval { source_key, indices, minimum, maximum } =
            sampler(uniform)
        else {
            panic!("UniformInterval variant is retained")
        };
        assert_eq!(source_key, uniform_source);
        assert_eq!(indices.retained.as_ref(), &[usize::from(egraph.find(second_index))]);
        assert_eq!(indices.omitted_count, 0);
        assert_eq!(minimum, ResolvedIntExpr::Const((-2).into()));
        assert_eq!(maximum, ResolvedIntExpr::Const(9.into()));
        assert_eq!(egraph.total_size(), before, "sampler diagnosis must not mutate the e-graph");
    }

    #[test]
    fn decomposed_hash_non_index_argument_cap_is_incomplete_even_when_prefixes_match() {
        let source_key = |wire_node| GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(wire_node), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let prefix = SelectedCanonicalEclasses {
            retained: (0..MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN).collect(),
            views: Box::new([]),
            omitted_count: 1,
        };
        let sampler = |source_key| SelectedSamplerIdentity::DecomposedHash {
            source_key,
            indices: SelectedCanonicalEclasses {
                retained: Box::new([]),
                views: Box::new([]),
                omitted_count: 0,
            },
            public_eclass: 1,
            target_eclass: 2,
            arguments: prefix.clone(),
            matrix_type: scalar_matrix_type(),
            base: ResolvedIntExpr::Const(2.into()),
            digit_count: ResolvedIntExpr::Const(3.into()),
            small: false,
            range_proved: true,
        };
        let first = sampler(source_key(79));
        let second = sampler(source_key(80));
        assert_eq!(
            sampler_non_index_contract(&first),
            sampler_non_index_contract(&second),
            "the retained argument prefix alone cannot distinguish omitted arguments"
        );
        assert!(sampler_non_index_contract_truncated(&first));
        assert!(sampler_non_index_contract_truncated(&second));
    }

    #[test]
    fn sampler_diagnostic_distinguishes_stored_and_actual_binder_owners() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let producer_key = BinderKey { loop_scope: scope.clone(), loop_node: NodeId(61), slot: 0 };
        let consumer_key = BinderKey { loop_scope: scope, loop_node: NodeId(62), slot: 0 };
        let producer = egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: producer_key.clone(),
            minimum: 0.into(),
            maximum: 7.into(),
        });
        let consumer = egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: consumer_key.clone(),
            minimum: 0.into(),
            maximum: 7.into(),
        });
        let producer_index = egraph.add(MxxLang::IntBinder(BinderId(producer)));
        let consumer_index = egraph.add(MxxLang::IntBinder(BinderId(consumer)));
        let sampler_source = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(63), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let atom = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::Gaussian {
                source: sampler_source,
                indices: vec![producer_index].into(),
                max_coefficient_bound: ResolvedIntExpr::Const(1.into()),
            },
            vec![consumer_index].into(),
        );
        egraph.rebuild();
        let before_nodes = egraph.total_size();
        let before_classes = egraph.number_of_classes();
        let candidates = diagnostic_candidates(&egraph, &[atom]);
        let atom = selected_class_view(&egraph, &candidates, atom)
            .and_then(|view| view.atom_source)
            .expect("selected sampler Atom view");
        let actual = atom.canonical_index_views.first().expect("actual consumer index");
        let SelectedSamplerIdentity::Gaussian { indices: stored, .. } =
            atom.sampler.expect("stored sampler identity")
        else {
            panic!("Gaussian sampler")
        };
        let stored = stored.views.first().expect("stored producer index");
        assert_eq!(actual.operator, "int-binder");
        assert_eq!(stored.operator, "int-binder");
        assert_eq!(actual.binder.as_ref().expect("consumer binder").key, consumer_key);
        assert_eq!(stored.binder.as_ref().expect("producer binder").key, producer_key);
        assert_ne!(actual.canonical_eclass, stored.canonical_eclass);
        assert_ne!(
            actual.integer_domain, stored.integer_domain,
            "affine domains retain their distinct BinderKey owners"
        );
        assert_eq!(actual.scalar_provenance, stored.scalar_provenance);
        assert_eq!(egraph.total_size(), before_nodes, "diagnostic must not add e-nodes");
        assert_eq!(egraph.number_of_classes(), before_classes, "diagnostic must not add classes");
    }

    #[test]
    fn atom_source_view_keeps_ordered_indices_and_caps_them() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "indexed-atom-view",
                )),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: None,
            }));
        let first_index = egraph.add(MxxLang::IntConst(0.into()));
        let second_index = egraph.add(MxxLang::IntConst(1.into()));
        let first = egraph.add(MxxLang::Atom { source, indices: vec![first_index].into() });
        let second = egraph.add(MxxLang::Atom { source, indices: vec![second_index].into() });
        let many_indices =
            (0..18).map(|index| egraph.add(MxxLang::IntConst(index.into()))).collect::<Vec<_>>();
        let capped = egraph.add(MxxLang::Atom { source, indices: many_indices.clone().into() });
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(&egraph, &[first, second, capped]);
        let view = |term| {
            selected_class_view(&egraph, &candidates, term)
                .and_then(|view| view.atom_source)
                .expect("selected Atom source")
        };
        let first = view(first);
        let second = view(second);
        let capped = view(capped);
        assert_eq!(first.source_id, second.source_id);
        assert_ne!(first.canonical_indices, second.canonical_indices);
        assert_eq!(first.canonical_indices.as_ref(), &[usize::from(egraph.find(first_index))]);
        assert_eq!(second.canonical_indices.as_ref(), &[usize::from(egraph.find(second_index))]);
        assert_eq!(capped.canonical_indices.len(), MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
        assert_eq!(capped.omitted_index_count, 2);
        assert_eq!(
            capped.canonical_indices.as_ref(),
            many_indices[..MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN]
                .iter()
                .map(|index| usize::from(egraph.find(*index)))
                .collect::<Vec<_>>()
        );
        assert_eq!(egraph.total_size(), before, "index diagnosis must not mutate the e-graph");
    }

    #[test]
    fn selected_large_diagnostic_records_selected_relation_slice_and_hash_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let key_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "diagnostic-hash-key",
                )),
                sort: MxxSort::Bytes(ResolvedIntExpr::Const(1.into())),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let key = egraph.add(MxxLang::Atom { source: key_source, indices: Box::new([]) });
        let query = egraph
            .analysis
            .symbols
            .hash_queries
            .intern(HashQuerySpec { matrix_type: scalar_matrix_type(), tag_program: Box::new([]) });
        let hash = egraph.add(MxxLang::HashPlain {
            query: super::super::identity::HashQuerySpecId(query),
            arguments: vec![key].into_boxed_slice(),
        });
        let slice = egraph.analysis.symbols.slices.intern(SliceSpec {
            rows: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
            columns: Some(ResolvedIndexRange {
                start: ResolvedIntExpr::Const(0.into()),
                end: ResolvedIntExpr::Const(1.into()),
            }),
        });
        let sliced = egraph.add(MxxLang::MatrixSlice { spec: SliceSpecId(slice), input: [hash] });
        let (preimage, preimage_source) = typed_matrix_atom(
            &mut egraph,
            "diagnostic-preimage",
            1,
            1,
            Some(AtomicRelationRole::Preimage),
        );
        let (decomposed_hash, decomposed_hash_source) = typed_matrix_atom(
            &mut egraph,
            "diagnostic-decomposed-hash",
            1,
            1,
            Some(AtomicRelationRole::DecomposedHash),
        );
        let mut factors = vec![sliced, preimage, decomposed_hash];
        for index in 0..15 {
            let (extra, _) = typed_matrix_atom(
                &mut egraph,
                &format!("diagnostic-extra-{index}"),
                1,
                1,
                Some(AtomicRelationRole::Preimage),
            );
            factors.push(extra);
        }
        let root = egraph.add(MxxLang::MatrixMultiply(factors.into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &egraph.classes().map(|class| class.id).collect::<Vec<_>>(),
        );
        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);

        let multiply = steps.first().expect("root multiply step");
        assert!(multiply.product_spine.is_some(), "selected positive product has a spine");
        let factors = multiply.multiply_factors.as_ref().expect("multiply factor views");
        assert_eq!(factors.len(), MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
        assert_eq!(multiply.multiply_omitted_factor_count, 2);
        assert_eq!(
            factors[1].relation_provenance.observed_direct_sources.as_ref(),
            &[SelectedDirectRelationSource {
                source_id: preimage_source.0,
                relation_role: Some(AtomicRelationRole::Preimage),
            }]
        );
        assert_eq!(
            factors[2].relation_provenance.observed_direct_sources.as_ref(),
            &[SelectedDirectRelationSource {
                source_id: decomposed_hash_source.0,
                relation_role: Some(AtomicRelationRole::DecomposedHash),
            }]
        );
        assert_eq!(factors[1].relation_provenance.observed_direct_source_count, 1);
        let slice_view = steps[1].slice.as_ref().expect("slice view");
        assert_eq!(slice_view.spec_id, slice);
        assert_eq!(slice_view.input_rows, Some(1.into()));
        assert_eq!(slice_view.input_columns, Some(1.into()));
        assert_eq!(steps[2].hash_plain.as_ref().expect("hash view").query_id, query);
        assert_eq!(egraph.total_size(), before, "failure diagnostic must not mutate the e-graph");
    }

    #[test]
    fn selected_large_diagnostic_records_capped_affine_concat_inputs_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (scale, scale_source) = typed_matrix_atom(&mut egraph, "concat-scale", 1, 1, None);
        let (public, _) = typed_matrix_atom(&mut egraph, "concat-public", 1, 2, None);
        let (residual, residual_source) =
            typed_matrix_atom(&mut egraph, "concat-residual", 1, 1, None);
        let (third_residual, _) =
            typed_matrix_atom(&mut egraph, "concat-third-residual", 1, 1, None);
        let (fourth_residual, _) =
            typed_matrix_atom(&mut egraph, "concat-fourth-residual", 1, 1, None);
        let slice = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, start: i64, end: i64| {
            let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
                rows: None,
                columns: Some(ResolvedIndexRange {
                    start: ResolvedIntExpr::Const(start.into()),
                    end: ResolvedIntExpr::Const(end.into()),
                }),
            }));
            (spec, egraph.add(MxxLang::MatrixSlice { spec, input: [public] }))
        };
        let (first_spec, first_slice) = slice(&mut egraph, 0, 1);
        let (second_spec, second_slice) = slice(&mut egraph, 1, 2);
        let first_product =
            egraph.add(MxxLang::MatrixMultiply(vec![scale, first_slice].into_boxed_slice()));
        let second_product =
            egraph.add(MxxLang::MatrixMultiply(vec![scale, second_slice].into_boxed_slice()));
        let residual_product =
            egraph.add(MxxLang::MatrixMultiply(vec![scale, residual].into_boxed_slice()));
        let third_product =
            egraph.add(MxxLang::MatrixMultiply(vec![scale, third_residual].into_boxed_slice()));
        let fourth_product =
            egraph.add(MxxLang::MatrixMultiply(vec![scale, fourth_residual].into_boxed_slice()));
        let first_chunk = egraph.add(MxxLang::MatrixAdd(
            vec![first_product, residual_product, residual_product].into_boxed_slice(),
        ));
        let second_chunk = egraph.add(MxxLang::MatrixAdd(
            vec![second_product, residual_product, residual_product].into_boxed_slice(),
        ));
        let third_chunk = egraph.add(MxxLang::MatrixAdd(
            vec![third_product, third_product, third_product].into_boxed_slice(),
        ));
        let fourth_chunk = egraph.add(MxxLang::MatrixAdd(
            vec![fourth_product, fourth_product, fourth_product].into_boxed_slice(),
        ));
        let mut inputs = vec![first_chunk, second_chunk, third_chunk, fourth_chunk];
        while inputs.len() <= MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN {
            inputs.push(fourth_chunk);
        }
        let concat = egraph.add(MxxLang::MatrixConcat {
            axis: super::super::identity::Axis::Columns,
            inputs: inputs.into_boxed_slice(),
        });
        let root = egraph.add(MxxLang::MatrixAdd(vec![concat].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &egraph.classes().map(|class| class.id).collect::<Vec<_>>(),
        );

        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let concat = steps
            .iter()
            .find_map(|step| step.matrix_concat.as_ref())
            .expect("selected concat step");
        let selected_chunk_add = steps
            .iter()
            .find(|step| {
                step.add_direct_child_inputs.as_ref().is_some_and(|inputs| inputs.len() == 3)
            })
            .expect("selected chunk Add step");
        assert!(
            selected_chunk_add
                .add_direct_child_inputs
                .as_ref()
                .expect("selected chunk Add inputs")
                .iter()
                .all(|input| matches!(input.selected, SelectedDiagnosticInputView::Product(_)))
        );
        assert_eq!(concat.axis, super::super::identity::Axis::Columns);
        assert_eq!(
            [first_chunk, second_chunk, third_chunk, fourth_chunk]
                .map(|chunk| egraph.find(chunk))
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4,
            "fixture keeps four physical Add eclasses"
        );
        assert_eq!(concat.inputs.len(), MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
        assert_eq!(concat.omitted_input_count, 1);
        assert_eq!(
            concat.inputs.iter().map(|input| input.canonical_eclass).collect::<Vec<_>>(),
            [first_chunk, second_chunk, third_chunk, fourth_chunk]
                .into_iter()
                .chain(std::iter::repeat(fourth_chunk))
                .take(MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN)
                .map(|input| usize::from(egraph.find(input)))
                .collect::<Vec<_>>(),
            "concat inputs retain physical order before the diagnostic cap"
        );

        for (input, expected_first_product) in [
            (&concat.inputs[0], first_product),
            (&concat.inputs[1], second_product),
            (&concat.inputs[2], third_product),
            (&concat.inputs[3], fourth_product),
        ] {
            let SelectedDiagnosticInputView::Add(children) = &input.selected else {
                panic!("concat input retains its direct selected Add children");
            };
            assert_eq!(children.inputs.len(), 3);
            assert_eq!(children.omitted_input_count, 0);
            assert_eq!(
                children.inputs[0].canonical_eclass,
                usize::from(egraph.find(expected_first_product))
            );
            for child in children.inputs.iter() {
                let SelectedDiagnosticInputView::Product(spine) = &child.selected else {
                    panic!("direct Add child retains its selected product spine");
                };
                assert_eq!(spine.leaves.len(), 2);
                assert_eq!(spine.leaves[0].canonical_eclass, usize::from(egraph.find(scale)));
                assert_eq!(
                    spine.leaves[0].atom_source.as_ref().map(|source| source.source_id),
                    Some(scale_source.0)
                );
            }
        }
        for (input, slice_spec, sliced) in [
            (&concat.inputs[0], first_spec, first_slice),
            (&concat.inputs[1], second_spec, second_slice),
        ] {
            let SelectedDiagnosticInputView::Add(children) = &input.selected else {
                unreachable!("above asserts that this concat input is an Add");
            };
            let SelectedDiagnosticInputView::Product(spine) = &children.inputs[0].selected else {
                unreachable!("above asserts that direct Add children are products");
            };
            assert_eq!(spine.leaves[1].canonical_eclass, usize::from(egraph.find(sliced)));
            assert_eq!(
                spine.leaves[1].slice.as_ref().map(|slice| slice.spec_id),
                Some(slice_spec.0)
            );
            assert_eq!(
                spine.leaves[1].slice.as_ref().map(|slice| slice.canonical_input),
                Some(usize::from(egraph.find(public)))
            );
        }
        let SelectedDiagnosticInputView::Add(children) = &concat.inputs[0].selected else {
            unreachable!("above asserts that the first concat input is an Add");
        };
        let SelectedDiagnosticInputView::Product(residual_spine) = &children.inputs[1].selected
        else {
            unreachable!("above asserts that direct Add children are products");
        };
        assert_eq!(residual_spine.leaves[1].canonical_eclass, usize::from(egraph.find(residual)));
        assert_eq!(
            residual_spine.leaves[1].atom_source.as_ref().map(|source| source.source_id),
            Some(residual_source.0)
        );
        assert_eq!(egraph.total_size(), before, "concat diagnosis must not mutate the e-graph");
    }

    #[test]
    fn selected_slice_diagnostic_keeps_unresolved_input_dimensions() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let matrix_type = ResolvedMatrixType {
            modulus: ResolvedIntExpr::Parameter("q".into()),
            ring_dimension: ResolvedIntExpr::Parameter("n".into()),
            rows: ResolvedIntExpr::Parameter("rows".into()),
            columns: ResolvedIntExpr::Parameter("columns".into()),
        };
        let source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "diagnostic-unresolved-slice-input",
                )),
                sort: MxxSort::Matrix(matrix_type),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let input = egraph.add(MxxLang::Atom { source, indices: Box::new([]) });
        let spec = egraph.analysis.symbols.slices.intern(SliceSpec { rows: None, columns: None });
        let sliced = egraph.add(MxxLang::MatrixSlice { spec: SliceSpecId(spec), input: [input] });
        egraph.rebuild();
        let before = egraph.total_size();

        let MxxLang::MatrixSlice { spec, input } = &egraph[egraph.find(sliced)].nodes[0] else {
            panic!("fixture retains its slice node");
        };
        let view = selected_slice(&egraph, spec.0, input[0]);
        assert_eq!(view.input_rows, None);
        assert_eq!(view.input_columns, None);
        assert_eq!(egraph.total_size(), before, "slice diagnostic must not mutate the e-graph");
    }

    #[test]
    fn selected_large_diagnostic_records_typed_pointwise_rejection() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (signal, _) = matrix_atom(&mut egraph, "signal");
        let (first_case, _) = matrix_atom(&mut egraph, "first-case");
        let (second_case, _) = matrix_atom(&mut egraph, "second-case");
        let negated_signal = egraph.add(MxxLang::MatrixNegate([signal]));
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negated_signal].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();

        let candidates = diagnostic_candidates(&egraph, &[root, switch, first_case, second_case]);
        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);

        let probe = steps
            .first()
            .and_then(|step| step.pointwise_add_switch_probe.as_ref())
            .expect("selected Add uses the shared pointwise probe");
        assert_eq!(probe.outcomes.len(), 1);
        assert!(matches!(
            &probe.outcomes[0].direct,
            PointwiseDirectProbe::Rejected(PointwiseAddSwitchReject::UnmatchedFixedTerms {
                case_index: 0,
                matched: 0,
                required: 1,
                direct_terms: 1,
                negated_terms: 0,
                ..
            })
        ));
        assert_eq!(egraph.total_size(), before);
    }

    #[test]
    fn selected_large_diagnostic_reports_stored_switch_sampler_indices_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let scope = OccurrenceScope {
            program: ProgramKey::Ideal,
            definition: FrozenGraphScopeId::Root,
            path: Box::new([]),
        };
        let selector_binder = egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: BinderKey { loop_scope: scope.clone(), loop_node: NodeId(70), slot: 0 },
            minimum: 0.into(),
            maximum: 1.into(),
        });
        let producer_binder = egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: BinderKey { loop_scope: scope, loop_node: NodeId(71), slot: 0 },
            minimum: 0.into(),
            maximum: 2.into(),
        });
        let selector = egraph.add(MxxLang::IntBinder(BinderId(selector_binder)));
        let producer = egraph.add(MxxLang::IntBinder(BinderId(producer_binder)));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let kth = egraph.add(MxxLang::IntConst(2.into()));
        let source_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(72), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let sampler = || SamplerIdentity::UniformInterval {
            source: source_key.clone(),
            indices: vec![producer].into(),
            minimum: ResolvedIntExpr::Const((-1).into()),
            maximum: ResolvedIntExpr::Const(1.into()),
        };
        let first_case = nonrelation_sampler_atom(&mut egraph, sampler(), vec![zero].into());
        let second_case = nonrelation_sampler_atom(&mut egraph, sampler(), vec![one].into());
        let fixed = nonrelation_sampler_atom(&mut egraph, sampler(), vec![kth].into());
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let negative_fixed = egraph.add(MxxLang::MatrixNegate([fixed]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negative_fixed].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &[root, switch, first_case, second_case, fixed, negative_fixed],
        );

        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let root_step = steps.first().expect("root Add diagnostic");
        let report = root_step
            .pointwise_switch_sampler_cases
            .as_ref()
            .expect("eligible Switch sampler report");
        assert_eq!(report.selector.operator, "int-binder");
        assert_eq!(report.selector.binder.as_ref().expect("bounded selector").maximum, 1.into());
        assert_eq!(report.cases.len(), 2);
        assert_eq!(report.omitted_case_count, 0);
        let fixed = report
            .fixed_sampler_occurrences
            .occurrences
            .first()
            .expect("fixed sampler is found through the selected expression");
        assert_eq!(fixed.sampler_source_key, source_key);
        assert_eq!(
            fixed.actual_canonical_index_views[0].canonical_eclass,
            usize::from(egraph.find(kth)),
        );
        for (case, actual) in report.cases.iter().zip([zero, one]) {
            let case = case
                .sampler_occurrences
                .occurrences
                .first()
                .expect("matching sampler is found below each selected case");
            assert_eq!(case.sampler_source_key, source_key);
            assert_eq!(
                case.stored_canonical_index_views[0].canonical_eclass,
                usize::from(egraph.find(producer)),
            );
            assert_eq!(
                case.actual_canonical_index_views[0].canonical_eclass,
                usize::from(egraph.find(actual)),
            );
        }
        assert_eq!(egraph.total_size(), before, "sampler report must not mutate the e-graph");
    }

    #[test]
    fn selected_large_diagnostic_finds_nested_sampler_cases_and_fails_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let producer = egraph.add(MxxLang::IntConst(7.into()));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let source_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(74), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let sampler = || SamplerIdentity::UniformInterval {
            source: source_key.clone(),
            indices: vec![producer].into(),
            minimum: ResolvedIntExpr::Const((-1).into()),
            maximum: ResolvedIntExpr::Const(1.into()),
        };
        let global_secret = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::Gaussian {
                source: GraphWireSourceKey {
                    wire: WireSourceKey {
                        scope: OccurrenceScope {
                            program: ProgramKey::Ideal,
                            definition: FrozenGraphScopeId::Root,
                            path: Box::new([]),
                        },
                        wire: WireRef { node: NodeId(76), port: Port(0) },
                    },
                    coordinate_binders: Box::new([]),
                },
                indices: Box::new([]),
                max_coefficient_bound: ResolvedIntExpr::Const(1.into()),
            },
            Box::new([]),
        );
        let (scale, _) = matrix_atom(&mut egraph, "nested-sampler-scale");
        let nested = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, index| {
            let sample = nonrelation_sampler_atom(egraph, sampler(), vec![index].into());
            let product = egraph.add(MxxLang::MatrixMultiply(
                vec![scale, sample, global_secret].into_boxed_slice(),
            ));
            egraph.add(MxxLang::MatrixAdd(vec![product].into_boxed_slice()))
        };
        let first_case = nested(&mut egraph, zero);
        let second_case = nested(&mut egraph, one);
        let fixed = nested(&mut egraph, selector);
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let negative_fixed = egraph.add(MxxLang::MatrixNegate([fixed]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negative_fixed].into_boxed_slice()));

        let (missing, _) = matrix_atom(&mut egraph, "nested-sampler-missing");
        let other_source_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(75), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let other = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::UniformInterval {
                source: other_source_key,
                indices: vec![producer].into(),
                minimum: ResolvedIntExpr::Const((-1).into()),
                maximum: ResolvedIntExpr::Const(1.into()),
            },
            vec![zero].into(),
        );
        let ambiguous = egraph.add(MxxLang::MatrixAdd(vec![first_case, other].into_boxed_slice()));
        let truncated_children = (0..MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN + 1)
            .map(|index| matrix_atom(&mut egraph, &format!("nested-sampler-cap-{index}")).0)
            .collect::<Vec<_>>();
        let truncated = egraph.add(MxxLang::MatrixAdd(truncated_children.into_boxed_slice()));
        let over_cap_indices =
            std::iter::repeat_n(producer, MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN + 1)
                .collect::<Vec<_>>();
        let stored_over_cap = nonrelation_sampler_atom(
            &mut egraph,
            SamplerIdentity::UniformInterval {
                source: source_key.clone(),
                indices: over_cap_indices.clone().into_boxed_slice(),
                minimum: ResolvedIntExpr::Const((-1).into()),
                maximum: ResolvedIntExpr::Const(1.into()),
            },
            vec![zero].into(),
        );
        let actual_over_cap =
            nonrelation_sampler_atom(&mut egraph, sampler(), over_cap_indices.into_boxed_slice());
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &[
                root,
                switch,
                first_case,
                second_case,
                fixed,
                negative_fixed,
                ambiguous,
                truncated,
                stored_over_cap,
                actual_over_cap,
            ],
        );
        let direct_fixed_search = selected_sampler_occurrences(&egraph, &candidates, fixed);
        assert!(
            direct_fixed_search.occurrences.len() == 2 && !direct_fixed_search.truncated,
            "{direct_fixed_search:?}"
        );

        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let report = steps
            .first()
            .and_then(|step| step.pointwise_switch_sampler_cases.as_ref())
            .expect("nested pointwise sampler report");
        let fixed = report
            .fixed_sampler_occurrences
            .occurrences
            .iter()
            .find(|occurrence| occurrence.sampler_source_key == source_key)
            .expect("nested indexed fixed sampler search");
        assert_eq!(fixed.sampler_source_key, source_key);
        assert_eq!(
            fixed.actual_canonical_index_views[0].canonical_eclass,
            usize::from(egraph.find(selector))
        );
        for (case, actual) in report.cases.iter().zip([zero, one]) {
            let occurrence = case
                .sampler_occurrences
                .occurrences
                .iter()
                .find(|occurrence| occurrence.sampler_source_key == source_key)
                .expect("nested indexed case sampler is found");
            assert_eq!(occurrence.sampler_source_key, source_key);
            assert_eq!(
                occurrence.actual_canonical_index_views[0].canonical_eclass,
                usize::from(egraph.find(actual)),
            );
        }
        assert!(
            report.common_sources.iter().any(|common| common.sampler_source_key == source_key),
            "indexed sampler remains a common source beside unindexed samplers"
        );
        let SelectedPointwiseRotationEvidence::Evidence(rotation) = &report.rotation_evidence
        else {
            panic!(
                "exactly one indexed sampler rotation is evidenced: {:?}",
                report.rotation_evidence
            );
        };
        assert_eq!(rotation.sampler_source_key, source_key);
        assert_eq!(rotation.rotated_coordinate, 0);
        let missing = selected_sampler_occurrences(&egraph, &candidates, missing);
        assert!(missing.occurrences.is_empty() && !missing.truncated);
        let ambiguous = selected_sampler_occurrences(&egraph, &candidates, ambiguous);
        assert!(ambiguous.occurrences.len() >= 2);
        let truncated = selected_sampler_occurrences(&egraph, &candidates, truncated);
        assert!(truncated.truncated);
        let stored_over_cap = selected_sampler_occurrences(&egraph, &candidates, stored_over_cap);
        assert!(stored_over_cap.truncated);
        assert_eq!(stored_over_cap.occurrences[0].stored_omitted_index_count, 1);
        let actual_over_cap = selected_sampler_occurrences(&egraph, &candidates, actual_over_cap);
        assert!(actual_over_cap.truncated);
        assert_eq!(actual_over_cap.occurrences[0].actual_omitted_index_count, 1);
        assert_eq!(egraph.total_size(), before, "nested sampler diagnosis must not mutate");
    }

    #[test]
    fn selected_large_diagnostic_uses_relation_selected_cases_with_competing_shapes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let producer = egraph.add(MxxLang::IntConst(7.into()));
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let kth = egraph.add(MxxLang::IntConst(2.into()));
        let source_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(73), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let sampler = || SamplerIdentity::UniformInterval {
            source: source_key.clone(),
            indices: vec![producer].into(),
            minimum: ResolvedIntExpr::Const((-1).into()),
            maximum: ResolvedIntExpr::Const(1.into()),
        };
        let first_case = nonrelation_sampler_atom(&mut egraph, sampler(), vec![zero].into());
        let second_case = nonrelation_sampler_atom(&mut egraph, sampler(), vec![one].into());
        let fixed = nonrelation_sampler_atom(&mut egraph, sampler(), vec![kth].into());
        let selected_switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let negative_fixed = egraph.add(MxxLang::MatrixNegate([fixed]));
        let root = egraph
            .add(MxxLang::MatrixAdd(vec![selected_switch, negative_fixed].into_boxed_slice()));

        let (other_first, _) = matrix_atom(&mut egraph, "ambiguous-switch-first");
        let (other_second, _) = matrix_atom(&mut egraph, "ambiguous-switch-second");
        let ambiguous_first = egraph
            .add(MxxLang::Switch(vec![selector, other_first, other_first].into_boxed_slice()));
        let ambiguous_second = egraph
            .add(MxxLang::Switch(vec![selector, other_first, other_second].into_boxed_slice()));
        egraph.union(ambiguous_first, ambiguous_second);
        let competing_root = egraph
            .add(MxxLang::MatrixAdd(vec![ambiguous_first, negative_fixed].into_boxed_slice()));
        egraph.union(root, competing_root);
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &[root, selected_switch, first_case, second_case, fixed, negative_fixed],
        );

        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let root_step = steps.first().expect("root Add diagnostic");
        let (rejection_selector, rejection_cases) =
            match root_step.pointwise_first_direct_reject.as_ref().expect("pointwise rejection") {
                PointwiseAddSwitchReject::UnmatchedFixedTerms {
                    physical_root_adds,
                    eligible_single_switch_adds,
                    selector,
                    switch_cases,
                    ..
                } => {
                    assert_eq!((*physical_root_adds, *eligible_single_switch_adds), (2, 1));
                    (*selector, switch_cases)
                }
                other => panic!("unexpected pointwise rejection: {other:?}"),
            };
        let report = root_step
            .pointwise_switch_sampler_cases
            .as_ref()
            .expect("only the planner-selected Switch is reported");
        assert_eq!(report.selector.canonical_eclass, usize::from(egraph.find(rejection_selector)));
        assert_eq!(
            report
                .cases
                .iter()
                .map(|case| {
                    case.sampler_occurrences.occurrences[0].actual_canonical_index_views[0]
                        .canonical_eclass
                })
                .collect::<Vec<_>>(),
            rejection_cases[1..]
                .iter()
                .map(|case| {
                    selected_class_view(&egraph, &candidates, *case)
                        .and_then(|view| view.atom_source)
                        .expect("selected sampler case")
                        .canonical_index_views[0]
                        .canonical_eclass
                })
                .collect::<Vec<_>>(),
        );
        assert_eq!(egraph.total_size(), before, "competing-shape diagnosis must not mutate");
    }

    #[test]
    fn selected_large_diagnostic_rejects_rotation_evidence_with_omitted_switch_cases() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector_binder = egraph.analysis.symbols.binders.intern(BinderDescriptor {
            key: BinderKey {
                loop_scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                loop_node: NodeId(77),
                slot: 0,
            },
            minimum: 0.into(),
            maximum: 16.into(),
        });
        let selector = egraph.add(MxxLang::IntBinder(BinderId(selector_binder)));
        let producer = egraph.add(MxxLang::IntConst(9.into()));
        let source_key = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: OccurrenceScope {
                    program: ProgramKey::Ideal,
                    definition: FrozenGraphScopeId::Root,
                    path: Box::new([]),
                },
                wire: WireRef { node: NodeId(78), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let sampler = || SamplerIdentity::UniformInterval {
            source: source_key.clone(),
            indices: vec![producer].into(),
            minimum: ResolvedIntExpr::Const((-1).into()),
            maximum: ResolvedIntExpr::Const(1.into()),
        };
        let fixed = nonrelation_sampler_atom(&mut egraph, sampler(), vec![selector].into());
        let mut switch_terms = vec![selector];
        for index in 0..17 {
            let actual =
                egraph.add(MxxLang::IntConst((if index == 16 { 99 } else { index }).into()));
            switch_terms.push(nonrelation_sampler_atom(
                &mut egraph,
                sampler(),
                vec![actual].into(),
            ));
        }
        let switch = egraph.add(MxxLang::Switch(switch_terms.into_boxed_slice()));
        let negative_fixed = egraph.add(MxxLang::MatrixNegate([fixed]));
        let root = egraph.add(MxxLang::MatrixAdd(vec![switch, negative_fixed].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(&egraph, &[root, switch, fixed, negative_fixed]);
        let report = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds)
            .into_iter()
            .next()
            .and_then(|step| step.pointwise_switch_sampler_cases)
            .expect("pointwise sampler report");
        assert_eq!(report.cases.len(), MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN);
        assert_eq!(report.omitted_case_count, 1);
        assert!(matches!(
            report.rotation_evidence,
            SelectedPointwiseRotationEvidence::Inconclusive { truncated: true, .. }
        ));
        assert_eq!(egraph.total_size(), before, "omitted-case diagnosis must not mutate");
    }

    #[test]
    fn selected_large_diagnostic_traces_each_negative_fixed_hash_product_without_mutation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let key_one_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "negative-key-one",
                )),
                sort: MxxSort::Bytes(ResolvedIntExpr::Const(1.into())),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let key_two_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "negative-key-two",
                )),
                sort: MxxSort::Bytes(ResolvedIntExpr::Const(1.into())),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let key_one = egraph.add(MxxLang::Atom { source: key_one_source, indices: Box::new([]) });
        let key_two = egraph.add(MxxLang::Atom { source: key_two_source, indices: Box::new([]) });
        let query_one = egraph
            .analysis
            .symbols
            .hash_queries
            .intern(HashQuerySpec { matrix_type: scalar_matrix_type(), tag_program: Box::new([]) });
        let query_two = egraph.analysis.symbols.hash_queries.intern(HashQuerySpec {
            matrix_type: scalar_matrix_type(),
            tag_program: vec![HashTagPart::Literal(vec![1].into_boxed_slice())].into_boxed_slice(),
        });
        let hash_one = egraph.add(MxxLang::HashPlain {
            query: super::super::identity::HashQuerySpecId(query_one),
            arguments: vec![key_one].into_boxed_slice(),
        });
        let hash_two = egraph.add(MxxLang::HashPlain {
            query: super::super::identity::HashQuerySpecId(query_two),
            arguments: vec![key_two].into_boxed_slice(),
        });
        let (prefix_one, _) = matrix_atom(&mut egraph, "negative-prefix-one");
        let (prefix_two, _) = matrix_atom(&mut egraph, "negative-prefix-two");
        let product_one =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix_one, hash_one].into_boxed_slice()));
        let product_two =
            egraph.add(MxxLang::MatrixMultiply(vec![prefix_two, hash_two].into_boxed_slice()));
        let negative_one = egraph.add(MxxLang::MatrixNegate([product_one]));
        let negative_two = egraph.add(MxxLang::MatrixNegate([product_two]));
        let (first_case, _) = matrix_atom(&mut egraph, "negative-first-case");
        let (second_case, _) = matrix_atom(&mut egraph, "negative-second-case");
        let switch =
            egraph.add(MxxLang::Switch(vec![selector, first_case, second_case].into_boxed_slice()));
        let root = egraph
            .add(MxxLang::MatrixAdd(vec![switch, negative_one, negative_two].into_boxed_slice()));
        egraph.rebuild();
        let before = egraph.total_size();
        let candidates = diagnostic_candidates(
            &egraph,
            &[
                root,
                switch,
                first_case,
                second_case,
                negative_one,
                negative_two,
                product_one,
                product_two,
                hash_one,
                hash_two,
            ],
        );

        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let root_step = steps.first().expect("root diagnostic step");
        assert_eq!(root_step.add_selected_large_child_count, Some(3));
        assert_eq!(root_step.add_direct_child_inputs.as_ref().expect("direct Add inputs").len(), 3);
        let paths = root_step
            .pointwise_negative_fixed_paths
            .as_ref()
            .expect("unmatched negative fixed paths");
        assert_eq!(paths.len(), 2);

        let path_for = |product: Id| {
            paths
                .iter()
                .find(|path| path.canonical_eclass == usize::from(egraph.find(product)))
                .expect("retained negative product path")
        };
        let first = path_for(product_one);
        let second = path_for(product_two);
        assert_eq!(first.multiplicity, 1);
        assert_eq!(second.multiplicity, 1);
        for (path, prefix, hash, query, key) in [
            (first, prefix_one, hash_one, query_one, key_one),
            (second, prefix_two, hash_two, query_two, key_two),
        ] {
            let product = path.steps.first().expect("product step");
            assert_eq!(product.operator, "matrix-multiply");
            let spine = product.product_spine.as_ref().expect("negative product spine");
            assert_eq!(spine.leaves.len(), 2);
            let factors = product.multiply_factors.as_ref().expect("product factors");
            assert_eq!(factors[0].canonical_eclass, usize::from(egraph.find(prefix)));
            assert_eq!(factors[1].canonical_eclass, usize::from(egraph.find(hash)));
            let hash =
                path.steps.get(1).and_then(|step| step.hash_plain.as_ref()).expect("hash step");
            assert_eq!(hash.query_id, query);
            assert_eq!(hash.canonical_arguments.as_ref(), &[usize::from(egraph.find(key))]);
        }
        assert_ne!(query_one, query_two, "query identity remains part of the trace");
        assert_ne!(
            egraph.find(key_one),
            egraph.find(key_two),
            "argument identity remains part of the trace"
        );
        assert_eq!(
            egraph.total_size(),
            before,
            "negative-path diagnosis must not mutate the e-graph"
        );
    }

    #[test]
    fn selected_large_diagnostic_caps_add_children_and_stops_on_selected_cycle() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let mut terms = Vec::new();
        for index in 0..17 {
            terms.push(matrix_atom(&mut egraph, &format!("capped-add-child-{index}")).0);
        }
        let root = egraph.add(MxxLang::MatrixAdd(terms.clone().into_boxed_slice()));
        egraph.rebuild();
        let mut large = terms.clone();
        large.push(root);
        let candidates = diagnostic_candidates(&egraph, &large);
        let steps = selected_large_diagnostic(&egraph, root, None, &candidates, &NoBounds);
        let root_step = steps.first().expect("root Add step");
        assert_eq!(root_step.add_selected_large_child_count, Some(17));
        assert_eq!(
            root_step.add_direct_child_inputs.as_ref().expect("capped direct inputs").len(),
            MAX_SELECTED_LARGE_DIAGNOSTIC_CHILDREN
        );
        assert_eq!(root_step.add_direct_child_omitted_child_count, 1);

        let cyclic = egraph.add(MxxLang::MatrixAdd(vec![terms[0]].into_boxed_slice()));
        egraph.union(cyclic, terms[0]);
        let canonical = egraph.find(cyclic);
        let selected_cycle = egraph[canonical]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixAdd(_)))
            .expect("union retains the physical cyclic Add")
            .clone();
        let mut cycle_candidates = diagnostic_candidates(&egraph, &[canonical]);
        cycle_candidates[usize::from(canonical)] = Some(Candidate {
            cost: ProposalCost { large_residual: true, node_count: 1, ..Default::default() },
            semantic_bound: Some(MatrixBound {
                matrix_type: concrete_scalar_matrix_type(),
                coefficient_class: BoundClass::Large,
                metadata: super::super::bound::MatrixMetadata::unknown(),
            }),
            first_large_source: None,
            node: selected_cycle,
            state: ExtractionState::Complete,
            output: None,
        });
        let cycle_steps =
            selected_large_path_from(&egraph, canonical, None, &cycle_candidates, &NoBounds, false);
        assert_eq!(cycle_steps.len(), 1, "local visited guard stops a selected self-cycle");
    }

    #[test]
    fn extraction_prefers_two_chunk_affine_preimage_boundary_over_public_large_sources() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (s, s_source) = typed_matrix_atom(&mut egraph, "s", 1, 1, None);
        let (t, t_source) = typed_matrix_atom(&mut egraph, "t", 1, 1, None);
        let (b0, b0_source) = typed_matrix_atom(&mut egraph, "b0", 1, 1, None);
        let (b1, b1_source) = typed_matrix_atom(&mut egraph, "b1", 1, 2, None);
        let (e0, e0_source) = typed_matrix_atom(&mut egraph, "e0", 1, 1, None);
        let (e_left, e_left_source) = typed_matrix_atom(&mut egraph, "e-left", 1, 1, None);
        let (e_right, e_right_source) = typed_matrix_atom(&mut egraph, "e-right", 1, 1, None);
        let (k0_left, k0_left_source) =
            typed_matrix_atom(&mut egraph, "k0-left", 1, 1, Some(AtomicRelationRole::Preimage));
        let (k0_right, k0_right_source) =
            typed_matrix_atom(&mut egraph, "k0-right", 1, 1, Some(AtomicRelationRole::Preimage));
        let (k1, k1_source) =
            typed_matrix_atom(&mut egraph, "k1", 2, 1, Some(AtomicRelationRole::Preimage));
        let (large_target, large_target_source) =
            typed_matrix_atom(&mut egraph, "large-target", 1, 1, None);

        let scaled_b0 = egraph.add(MxxLang::MatrixMultiply(vec![s, b0].into_boxed_slice()));
        let c_b0 = egraph.add(MxxLang::MatrixAdd(vec![scaled_b0, e0].into_boxed_slice()));
        let slice = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, start: i64, end: i64| {
            let spec = SliceSpecId(egraph.analysis.symbols.slices.intern(SliceSpec {
                rows: None,
                columns: Some(ResolvedIndexRange {
                    start: ResolvedIntExpr::Const(start.into()),
                    end: ResolvedIntExpr::Const(end.into()),
                }),
            }));
            egraph.add(MxxLang::MatrixSlice { spec, input: [b1] })
        };
        let target = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, start, end, error| {
            let selected_columns = slice(egraph, start, end);
            let signal =
                egraph.add(MxxLang::MatrixMultiply(vec![t, selected_columns].into_boxed_slice()));
            egraph.add(MxxLang::MatrixAdd(vec![signal, error].into_boxed_slice()))
        };
        let target_left = target(&mut egraph, 0, 1, e_left);
        let target_right = target(&mut egraph, 1, 2, e_right);
        let chunk_left =
            egraph.add(MxxLang::MatrixMultiply(vec![c_b0, k0_left].into_boxed_slice()));
        let chunk_right =
            egraph.add(MxxLang::MatrixMultiply(vec![c_b0, k0_right].into_boxed_slice()));
        let chunks = egraph.add(MxxLang::MatrixConcat {
            axis: super::super::identity::Axis::Columns,
            inputs: vec![chunk_left, chunk_right].into_boxed_slice(),
        });
        let root = egraph.add(MxxLang::MatrixMultiply(vec![chunks, k1].into_boxed_slice()));

        let context = RewriteContext::new(SharedRewriteBudget::new());
        for (source, target) in [(k0_left_source, target_left), (k0_right_source, target_right)] {
            context.register(RelationRegistration {
                source,
                expected_public: b0,
                target,
                trapdoor: None,
                indices: Box::new([]),
            });
        }
        context.register(RelationRegistration {
            source: k1_source,
            expected_public: b1,
            target: large_target,
            trapdoor: None,
            indices: Box::new([]),
        });
        let rewrite = egg::Rewrite::new(
            "test-extraction-two-chunk-affine-preimage-boundary",
            RelationSearcher::new(context.clone()),
            RelationApplier::new(context.clone()),
        )
        .expect("closed relation rewrite");
        let egraph = egg::Runner::default().with_egraph(egraph).run(&[rewrite]).egraph;
        assert_eq!(context.failure(), None);

        let mut input = SemanticInput {
            atom_classes: BTreeMap::from([
                (b0_source, BoundClass::Large),
                (b1_source, BoundClass::Large),
                (large_target_source, BoundClass::Large),
                (s_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (t_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (e0_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (e_left_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (e_right_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (k0_left_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (
                    k0_right_source,
                    BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() },
                ),
                (k1_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
            ]),
            ..Default::default()
        };
        populate_matrix_types(&mut input, &egraph);
        let mut invalid = |_| panic!("fixture has a finite selected DAG");
        let mut bound_error = |error| panic!("fixture has valid bounds: {error:?}");
        let result = extract_best_proposal(
            &egraph,
            root,
            &input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |origin, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, origin, node, &context)
                    .map(|classification| ProposalNodeClassification {
                        relation_redex: classification.relation_redex,
                        local_checked_relation_count: classification.local_checked_relation_count,
                    })
                    .map_err(|failure| panic!("fixture relation is valid: {failure:?}"))
            },
        )
        .expect("two-level normalized residual extracts");

        assert_eq!(result.cost.unsatisfied_relation_redexes, 0);
        assert_eq!(result.cost.unsatisfied_structural_redexes, 0);
        assert!(result.cost.large_residual);
        assert_eq!(result.first_large_source, Some(large_target_source));

        input.atom_classes.insert(
            large_target_source,
            BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() },
        );
        let mut invalid = |_| panic!("fixture has a finite selected DAG");
        let mut bound_error = |error| panic!("fixture has valid bounds: {error:?}");
        let finite = extract_best_proposal(
            &egraph,
            root,
            &input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |origin, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, origin, node, &context)
                    .map(|classification| ProposalNodeClassification {
                        relation_redex: classification.relation_redex,
                        local_checked_relation_count: classification.local_checked_relation_count,
                    })
                    .map_err(|failure| panic!("fixture relation is valid: {failure:?}"))
            },
        )
        .expect("finite two-level normalized residual extracts");
        assert!(!finite.cost.large_residual);
        assert_eq!(finite.first_large_source, None);
        assert!(matches!(
            finite.semantic_bound.map(|bound| bound.coefficient_class),
            Some(BoundClass::Bounded { .. }) | Some(BoundClass::ExactZero)
        ));
    }

    fn extract(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        root: Id,
        classify: &mut dyn FnMut(
            Id,
            &MxxLang,
            &EGraph<MxxLang, MxxAnalysis>,
        )
            -> Result<ProposalNodeClassification, OperationalSimulationError>,
    ) -> Result<ExtractedProposal, OperationalSimulationError> {
        let mut invalid = |_| panic!("valid test graph must have a finite DAG representative");
        let mut bound_error = |error| panic!("non-matrix test must not evaluate bounds: {error:?}");
        extract_best_proposal(
            egraph,
            root,
            &NoBounds,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            classify,
        )
    }

    #[test]
    fn lexicographic_cost_prefers_relation_then_size_for_nonmatrices() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let zero = egraph.add(MxxLang::IntConst(0.into()));
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let compact_large = egraph.add(MxxLang::IntAdd([zero, one]));
        let larger_small = egraph.add(MxxLang::IntMul([zero, one]));
        egraph.union(compact_large, larger_small);
        egraph.rebuild();

        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification {
                relation_redex: matches!(node, MxxLang::IntAdd(_)),
                ..Default::default()
            })
        };
        let result = extract(&egraph, compact_large, &mut classify).unwrap();

        assert_eq!(result.cost.unsatisfied_relation_redexes, 0);
        assert!(!result.cost.large_residual);
        assert!(matches!(result.expression[result.expression.root()], MxxLang::IntMul(_)));
    }

    #[test]
    fn satisfied_raw_relation_precedes_two_structural_obligations() {
        let raw_satisfied_relation =
            ProposalCost { local_checked_relation_count: 1, ..Default::default() };
        let replacement_with_structural_work =
            ProposalCost { unsatisfied_structural_redexes: 2, ..Default::default() };

        assert!(
            replacement_with_structural_work < raw_satisfied_relation,
            "the local relation preference is decided before structural work"
        );
    }

    #[test]
    fn checked_relation_count_selects_a_larger_satisfied_replacement() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let public = egraph.add(MxxLang::IntConst(2.into()));
        let relation = egraph.add(MxxLang::IntConst(3.into()));
        let signal = egraph.add(MxxLang::IntConst(5.into()));
        let error = egraph.add(MxxLang::IntConst(7.into()));
        let raw = egraph.add(MxxLang::IntMul([public, relation]));
        let replacement_product = egraph.add(MxxLang::IntMul([signal, public]));
        let replacement = egraph.add(MxxLang::IntAdd([replacement_product, error]));
        egraph.union(raw, replacement);
        egraph.rebuild();
        let raw = egraph.find(raw);
        let original_nodes = egraph.total_size();
        let classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification {
                local_checked_relation_count: u64::from(matches!(node, MxxLang::IntMul(factors)
                    if factors.len() == 2 && factors[0] == public && factors[1] == relation)),
                ..Default::default()
            })
        };

        let mut first_classify = classify;
        let first =
            extract(&egraph, raw, &mut first_classify).expect("satisfied replacement extracts");
        let mut second_classify = classify;
        let second =
            extract(&egraph, raw, &mut second_classify).expect("repeat extraction is stable");

        assert_eq!(egraph.total_size(), original_nodes, "extraction does not mutate the e-graph");
        assert_eq!(first, second, "local preference has deterministic extraction output");
        assert_eq!(first.cost.local_checked_relation_count, 0);
        assert!(matches!(first.expression[first.expression.root()], MxxLang::IntAdd(_)));
    }

    #[test]
    fn missing_relation_replacement_remains_unresolved_until_materialized() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let public = egraph.add(MxxLang::IntConst(2.into()));
        let relation = egraph.add(MxxLang::IntConst(3.into()));
        let raw = egraph.add(MxxLang::IntMul([public, relation]));
        egraph.rebuild();
        let classify = |origin: Id, node: &MxxLang, egraph: &EGraph<MxxLang, MxxAnalysis>| {
            let raw = matches!(node, MxxLang::IntMul(factors)
                if factors.len() == 2 && factors[0] == public && factors[1] == relation);
            let materialized = egraph[egraph.find(origin)]
                .nodes
                .iter()
                .any(|node| matches!(node, MxxLang::IntAdd(_)));
            Ok(ProposalNodeClassification {
                relation_redex: raw && !materialized,
                local_checked_relation_count: u64::from(raw),
            })
        };

        let mut before_classify = classify;
        let before = extract(&egraph, raw, &mut before_classify).expect("raw relation extracts");
        assert_eq!(before.cost.unsatisfied_relation_redexes, 1);
        assert_eq!(before.cost.local_checked_relation_count, 1);

        let error = egraph.add(MxxLang::IntConst(7.into()));
        let replacement = egraph.add(MxxLang::IntAdd([public, error]));
        egraph.union(raw, replacement);
        egraph.rebuild();
        let mut after_classify = classify;
        let after = extract(&egraph, raw, &mut after_classify)
            .expect("materialized relation replacement extracts");
        assert_eq!(after.cost.unsatisfied_relation_redexes, 0);
        assert_eq!(after.cost.local_checked_relation_count, 0);
        assert!(matches!(after.expression[after.expression.root()], MxxLang::IntAdd(_)));
    }

    #[test]
    fn checked_relation_count_exposes_a_selected_add_to_its_outer_product() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let scale = egraph.add(MxxLang::IntConst(2.into()));
        let public = egraph.add(MxxLang::IntConst(3.into()));
        let relation = egraph.add(MxxLang::IntConst(5.into()));
        let error = egraph.add(MxxLang::IntConst(7.into()));
        let raw = egraph.add(MxxLang::IntMul([public, relation]));
        let replacement = egraph.add(MxxLang::IntAdd([public, error]));
        let outer = egraph.add(MxxLang::IntMul([scale, raw]));
        egraph.union(raw, replacement);
        egraph.rebuild();
        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification {
                local_checked_relation_count: u64::from(matches!(node, MxxLang::IntMul(factors)
                    if factors.len() == 2 && factors[0] == public && factors[1] == relation)),
                ..Default::default()
            })
        };

        let result = extract(&egraph, outer, &mut classify).expect("outer product extracts");
        let MxxLang::IntMul(factors) = &result.expression[result.expression.root()] else {
            panic!("outer product remains visible");
        };
        assert!(matches!(result.expression[factors[1]], MxxLang::IntAdd(_)));
        assert!(
            result.cost.local_checked_relation_count == 0,
            "the outer cost does not inherit its selected child's checked relation count"
        );
    }

    #[test]
    fn checked_relation_count_is_not_aggregated_from_children() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(2.into()));
        let right = egraph.add(MxxLang::IntConst(3.into()));
        let raw = egraph.add(MxxLang::IntAdd([left, right]));
        let outer = egraph.add(MxxLang::IntMul([left, raw]));
        egraph.rebuild();
        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification {
                local_checked_relation_count: u64::from(matches!(node, MxxLang::IntAdd(_))),
                ..Default::default()
            })
        };

        let result = extract(&egraph, outer, &mut classify).expect("outer expression extracts");
        assert_eq!(result.cost.local_checked_relation_count, 0);
    }

    #[test]
    fn addition_keeps_relation_obligations_separate_from_structural_hiding() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let redex = egraph.add(MxxLang::IntMul([left, right]));
        let inner = egraph.add(MxxLang::MatrixAdd(vec![redex].into_boxed_slice()));
        let outer = egraph.add(MxxLang::MatrixAdd(vec![inner].into_boxed_slice()));
        egraph.rebuild();
        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification {
                relation_redex: matches!(node, MxxLang::IntMul(_)),
                ..Default::default()
            })
        };

        let result = extract(&egraph, outer, &mut classify).unwrap();
        assert_eq!(result.cost.unsatisfied_relation_redexes, 1);
        assert_eq!(result.cost.unsatisfied_structural_redexes, 0);
        assert_eq!(result.cost.hidden_structural_redexes, 0);
    }

    #[test]
    fn zero_times_large_is_exact_zero_during_extraction() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (zero, zero_source) = matrix_atom(&mut egraph, "zero");
        let (large, large_source) = matrix_atom(&mut egraph, "large");
        let root = egraph.add(MxxLang::MatrixMultiply(vec![zero, large].into_boxed_slice()));
        egraph.rebuild();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([
                (zero_source, BoundClass::ExactZero),
                (large_source, BoundClass::Large),
            ]),
            ..Default::default()
        };

        let result = extract_with_input(&egraph, root, &input);

        assert!(!result.cost.large_residual);
        assert_eq!(result.first_large_source, None);
    }

    #[test]
    fn source_less_large_emits_the_bounded_selected_path_at_info() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let key_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("hash-key")),
                sort: MxxSort::Bytes(ResolvedIntExpr::Const(1.into())),
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let key = egraph.add(MxxLang::Atom { source: key_source, indices: Box::new([]) });
        let query = egraph
            .analysis
            .symbols
            .hash_queries
            .intern(HashQuerySpec { matrix_type: scalar_matrix_type(), tag_program: Box::new([]) });
        let root = egraph.add(MxxLang::HashPlain {
            query: super::super::identity::HashQuerySpecId(query),
            arguments: vec![key].into_boxed_slice(),
        });
        egraph.rebuild();
        let mut input = SemanticInput::default();
        populate_matrix_types(&mut input, &egraph);

        let capture = SelectedLargeDiagnosticCapture::new(LevelFilter::INFO);
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let result = tracing::subscriber::with_default(subscriber, || {
            extract_with_input(&egraph, root, &input)
        });

        assert!(result.cost.large_residual);
        assert_eq!(result.first_large_source, None);
        let events = capture.events.lock().expect("diagnostic capture lock");
        assert_eq!(events.len(), 1);
        assert!(
            events[0]
                .get("selected_large_path")
                .is_some_and(|path| path.contains("SelectedLargePathStep"))
        );
    }

    #[test]
    fn nonzero_times_large_retains_the_first_large_source() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (bounded, bounded_source) = matrix_atom(&mut egraph, "bounded");
        let (large, large_source) = matrix_atom(&mut egraph, "large");
        let root = egraph.add(MxxLang::MatrixMultiply(vec![bounded, large].into_boxed_slice()));
        egraph.rebuild();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([
                (bounded_source, BoundClass::Bounded { maximum_absolute_coefficient: 2_u8.into() }),
                (large_source, BoundClass::Large),
            ]),
            ..Default::default()
        };

        let debug_capture = SelectedLargeDiagnosticCapture::new(LevelFilter::DEBUG);
        let debug_subscriber = tracing_subscriber::registry().with(debug_capture.clone());
        let result = tracing::subscriber::with_default(debug_subscriber, || {
            extract_with_input(&egraph, root, &input)
        });

        assert!(result.cost.large_residual);
        assert_eq!(result.first_large_source, Some(large_source));
        let events = debug_capture.events.lock().expect("diagnostic capture lock");
        assert_eq!(events.len(), 1);
        let expected_source = format!("Some({})", large_source.0);
        assert_eq!(
            events[0].get("selected_large_source_id").map(String::as_str),
            Some(expected_source.as_str()),
        );
        assert_eq!(
            events[0].get("selected_large_source_kind").map(String::as_str),
            Some("Some(\"protocol-input\")"),
        );
        assert!(
            events[0]
                .get("selected_large_path")
                .is_some_and(|path| path.contains("SelectedLargePathStep"))
        );
        drop(events);

        let info_capture = SelectedLargeDiagnosticCapture::new(LevelFilter::INFO);
        let info_subscriber = tracing_subscriber::registry().with(info_capture.clone());
        tracing::subscriber::with_default(info_subscriber, || {
            let result = extract_with_input(&egraph, root, &input);
            assert_eq!(result.first_large_source, Some(large_source));
        });
        assert!(
            info_capture.events.lock().expect("diagnostic capture lock").is_empty(),
            "the diagnostic path is not built or emitted below DEBUG"
        );
    }

    #[test]
    fn finite_eclass_alternative_is_selected_over_large() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (large, large_source) = matrix_atom(&mut egraph, "large");
        let (bounded, bounded_source) = matrix_atom(&mut egraph, "bounded");
        egraph.union(large, bounded);
        egraph.rebuild();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([
                (large_source, BoundClass::Large),
                (bounded_source, BoundClass::Bounded { maximum_absolute_coefficient: 3_u8.into() }),
            ]),
            ..Default::default()
        };

        let result = extract_with_input(&egraph, large, &input);

        assert!(!result.cost.large_residual);
        assert!(matches!(
            result.expression[result.expression.root()],
            MxxLang::Atom { source, .. } if source == bounded_source
        ));
        assert!(matches!(
            result.semantic_bound.map(|bound| bound.coefficient_class),
            Some(BoundClass::Bounded { maximum_absolute_coefficient })
                if maximum_absolute_coefficient == BigUint::from(3_u8)
        ));
    }

    #[test]
    fn extraction_prefers_finite_pointwise_normalized_switch_over_large_add_switch() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (large_case, large_source) = matrix_atom(&mut egraph, "large-case");
        let (bounded_case, bounded_source) = matrix_atom(&mut egraph, "bounded-case");
        let original_switch =
            egraph.add(MxxLang::Switch(vec![selector, large_case, large_case].into_boxed_slice()));
        let original = egraph.add(MxxLang::MatrixAdd(vec![original_switch].into_boxed_slice()));
        let normalized = egraph
            .add(MxxLang::Switch(vec![selector, bounded_case, bounded_case].into_boxed_slice()));
        egraph.union(original, normalized);
        egraph.rebuild();
        let mut input = SemanticInput {
            atom_classes: BTreeMap::from([
                (large_source, BoundClass::Large),
                (bounded_source, BoundClass::bounded(1_u8.into())),
            ]),
            reachable_cases: BTreeMap::from([
                (egraph.find(original_switch), vec![true, true].into_boxed_slice()),
                (egraph.find(normalized), vec![true, true].into_boxed_slice()),
            ]),
            ..Default::default()
        };
        populate_matrix_types(&mut input, &egraph);
        let result = extract_with_input(&egraph, original, &input);

        assert!(!result.cost.large_residual);
        assert!(matches!(result.expression[result.expression.root()], MxxLang::Switch(_)));
    }

    #[test]
    fn extraction_prefers_pointwise_preimage_normalization_over_the_original_large_public() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let (scale, scale_source) = matrix_atom(&mut egraph, "scale");
        let (public, public_source) = matrix_atom(&mut egraph, "public");
        let (residual, residual_source) = matrix_atom(&mut egraph, "residual");
        let (left_target, left_target_source) = matrix_atom(&mut egraph, "left-target");
        let (right_target, right_target_source) = matrix_atom(&mut egraph, "right-target");
        let relation_atom = |egraph: &mut EGraph<MxxLang, MxxAnalysis>, name| {
            let source = AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(
                AtomicSourceDescriptor {
                    key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(name)),
                    sort: MxxSort::Matrix(scalar_matrix_type()),
                    integer_domain: None,
                    canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                    relation_role: Some(AtomicRelationRole::Preimage),
                },
            ));
            let term =
                egraph.add(MxxLang::Atom { source, indices: vec![selector].into_boxed_slice() });
            (term, source)
        };
        let (left_relation, left_source) = relation_atom(&mut egraph, "left-relation");
        let (right_relation, right_source) = relation_atom(&mut egraph, "right-relation");
        let relation = egraph
            .add(MxxLang::Switch(vec![selector, left_relation, right_relation].into_boxed_slice()));
        let matching = egraph.add(MxxLang::MatrixMultiply(vec![scale, public].into_boxed_slice()));
        let additive = egraph.add(MxxLang::MatrixAdd(vec![matching, residual].into_boxed_slice()));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![additive, relation].into_boxed_slice()));
        let context = RewriteContext::new(SharedRewriteBudget::new());
        for (source, target) in [(left_source, left_target), (right_source, right_target)] {
            context.register(RelationRegistration {
                source,
                expected_public: public,
                target,
                trapdoor: None,
                indices: vec![selector].into_boxed_slice(),
            });
        }
        let rewrite = egg::Rewrite::new(
            "test-pointwise-preimage",
            RelationSearcher::new(context.clone()),
            RelationApplier::new(context.clone()),
        )
        .expect("closed test rewrite");
        let runner = egg::Runner::default().with_egraph(egraph).run(&[rewrite]);
        let egraph = runner.egraph;
        assert_eq!(context.failure(), None);
        assert_eq!(context.counters().selector_distributions, 1);
        assert!(context.counters().rewrites >= 3);

        let reachable_cases = egraph
            .classes()
            .filter_map(|class| {
                class.nodes.iter().find_map(|node| match node {
                    MxxLang::Switch(cases) => Some((class.id, vec![true; cases.len() - 1].into())),
                    _ => None,
                })
            })
            .collect();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([
                (scale_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (public_source, BoundClass::Large),
                (
                    residual_source,
                    BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() },
                ),
                (left_target_source, BoundClass::Large),
                (right_target_source, BoundClass::Large),
                (left_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (right_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
            ]),
            reachable_cases,
            ..Default::default()
        };
        let mut invalid = |_| panic!("fixture has a finite selected DAG");
        let mut bound_error = |error| panic!("fixture has valid matrix bounds: {error:?}");
        let result = extract_best_proposal(
            &egraph,
            egraph.find(root),
            &input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |origin, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, origin, node, &context)
                    .map(|classification| ProposalNodeClassification {
                        relation_redex: classification.relation_redex,
                        local_checked_relation_count: classification.local_checked_relation_count,
                    })
                    .map_err(|failure| panic!("fixture relation is valid: {failure:?}"))
            },
        )
        .expect("pointwise relation candidate extracts");

        assert!(result.cost.large_residual);
        assert!(matches!(
            result.first_large_source,
            Some(source) if source == left_target_source || source == right_target_source
        ));
        assert!(!result.expression.as_ref().iter().any(|node| {
            matches!(node, MxxLang::Atom { source, .. } if *source == public_source)
        }));
    }

    #[test]
    fn different_selector_relation_does_not_hide_a_reachable_large_residual() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left_selector = egraph.add(MxxLang::IntConst(0.into()));
        let right_selector = egraph.add(MxxLang::IntConst(1.into()));
        let (public_case, public_source) = matrix_atom(&mut egraph, "public-large");
        let (target, target_source) = matrix_atom(&mut egraph, "registered-target");
        let relation_source =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from(
                    "registered-preimage",
                )),
                sort: MxxSort::Matrix(scalar_matrix_type()),
                integer_domain: None,
                canonical_residue_convention: Some(CanonicalResidueConvention::Nonnegative),
                relation_role: Some(AtomicRelationRole::Preimage),
            }));
        let relation_case =
            egraph.add(MxxLang::Atom { source: relation_source, indices: Box::new([]) });
        let public = egraph
            .add(MxxLang::Switch(vec![left_selector, public_case, public_case].into_boxed_slice()));
        let relation = egraph.add(MxxLang::Switch(
            vec![right_selector, relation_case, relation_case].into_boxed_slice(),
        ));
        let root = egraph.add(MxxLang::MatrixMultiply(vec![public, relation].into_boxed_slice()));
        egraph.rebuild();

        assert!(
            !egraph[egraph.find(relation_case)].data.relation_provenance.is_empty(),
            "the relation operand has production provenance"
        );
        let context = RewriteContext::new(SharedRewriteBudget::new());
        context.register(RelationRegistration {
            source: relation_source,
            expected_public: public_case,
            target,
            trapdoor: None,
            indices: Box::new([]),
        });
        let root = egraph.find(root);
        let node = egraph[root]
            .nodes
            .iter()
            .find(|node| matches!(node, MxxLang::MatrixMultiply(_)))
            .expect("matrix product representative");
        assert!(
            !super::super::relation::classify_proposal_node(&egraph, root, node, &context)
                .expect("different selectors are locally inapplicable")
                .relation_redex
        );
        assert_eq!(context.failure(), None);

        let mut input = SemanticInput {
            atom_classes: BTreeMap::from([
                (public_source, BoundClass::Large),
                (relation_source, BoundClass::bounded(1_u8.into())),
                (target_source, BoundClass::bounded(1_u8.into())),
            ]),
            reachable_cases: BTreeMap::from([
                (egraph.find(public), vec![true, true].into_boxed_slice()),
                (egraph.find(relation), vec![true, true].into_boxed_slice()),
            ]),
            ..Default::default()
        };
        for class in egraph.classes() {
            let class = egraph.find(class.id);
            input.nodes.entry(class).or_insert_with(|| egraph[class].nodes[0].clone());
        }
        populate_matrix_types(&mut input, &egraph);

        let mut invalid = |_| panic!("fixture has a finite selected DAG");
        let mut bound_error = |error| panic!("fixture has valid bound transfers: {error:?}");
        let extracted = extract_best_proposal(
            &egraph,
            root,
            &input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |origin, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, origin, node, &context)
                    .map(|classification| ProposalNodeClassification {
                        relation_redex: classification.relation_redex,
                        local_checked_relation_count: classification.local_checked_relation_count,
                    })
                    .map_err(|failure| panic!("fixture relation is valid: {failure:?}"))
            },
        )
        .expect("the selected residual extracts");
        assert!(extracted.cost.large_residual);
        assert_eq!(extracted.first_large_source, Some(public_source));
        assert!(matches!(
            extracted.semantic_bound.map(|bound| bound.coefficient_class),
            Some(BoundClass::Large)
        ));
        assert_eq!(
            BoundEvaluator::new(&input).evaluate(root),
            Err(BoundEvaluationError::UnconsumedLargeTerm { term: root })
        );
    }

    #[test]
    fn shared_large_child_has_a_deterministic_witness() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (large, large_source) = matrix_atom(&mut egraph, "large");
        let root = egraph.add(MxxLang::MatrixAdd(vec![large, large].into_boxed_slice()));
        egraph.rebuild();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([(large_source, BoundClass::Large)]),
            ..Default::default()
        };

        let result = extract_with_input(&egraph, root, &input);

        assert!(result.cost.large_residual);
        assert_eq!(result.first_large_source, Some(large_source));
        assert_eq!(result.expression.as_ref().len(), 2);
        assert!(matches!(
            result.semantic_bound.map(|bound| bound.coefficient_class),
            Some(BoundClass::Large)
        ));
    }

    #[test]
    fn same_node_refresh_propagates_zero_over_a_large_child() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        // The e-class order deliberately makes `delayed` unavailable while
        // `descendant` first selects its bounded transpose alternative.
        let (root_placeholder, root_source) = matrix_atom(&mut egraph, "root");
        let (ready, ready_source) = matrix_atom(&mut egraph, "ready");
        let (descendant, descendant_source) = matrix_atom(&mut egraph, "descendant");
        let (delayed, delayed_source) = matrix_atom(&mut egraph, "delayed");
        let (large, large_source) = matrix_atom(&mut egraph, "large");
        let transpose = egraph.add(MxxLang::MatrixTranspose([ready]));
        let negate = egraph.add(MxxLang::MatrixNegate([delayed]));
        let product = egraph.add(MxxLang::MatrixMultiply(vec![descendant, large].into()));
        egraph.union(descendant, transpose);
        egraph.union(descendant, negate);
        egraph.union(root_placeholder, product);
        egraph.rebuild();
        let input = SemanticInput {
            atom_classes: BTreeMap::from([
                (root_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (ready_source, BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() }),
                (
                    descendant_source,
                    BoundClass::Bounded { maximum_absolute_coefficient: 1_u8.into() },
                ),
                (delayed_source, BoundClass::ExactZero),
                (large_source, BoundClass::Large),
            ]),
            ..Default::default()
        };
        let mut invalid = |_| panic!("fixture has a finite selected DAG");
        let mut bound_error = |error| panic!("fixture has valid matrix bounds: {error:?}");
        let result = extract_best_proposal(
            &egraph,
            root_placeholder,
            &input,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |_, node, _| {
                Ok(ProposalNodeClassification {
                    relation_redex: matches!(
                        node,
                        MxxLang::Atom { source, .. }
                            if *source == root_source || *source == descendant_source
                    ),
                    ..Default::default()
                })
            },
        )
        .unwrap();

        assert!(!result.cost.large_residual);
        assert_eq!(
            result.semantic_bound.map(|bound| bound.coefficient_class),
            Some(BoundClass::ExactZero)
        );
    }

    struct NoSelectedChildren;

    impl SelectedChildBounds for NoSelectedChildren {
        fn child_bound(&self, _: Id) -> Option<&MatrixBound> {
            None
        }
    }

    #[test]
    fn missing_contract_error_agrees_with_final_evaluation() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let (atom, source) = matrix_atom(&mut egraph, "missing");
        egraph.rebuild();
        let atom = egraph.find(atom);
        let node = egraph[atom].nodes.first().unwrap().clone();
        let input = SemanticInput {
            nodes: BTreeMap::from([(atom, node.clone())]),
            missing: Some(source),
            ..Default::default()
        };

        let extraction =
            BoundEvaluator::evaluate_selected_node(&input, atom, &node, &NoSelectedChildren);
        let final_evaluation = BoundEvaluator::new(&input).evaluate(atom);

        assert_eq!(extraction, Err(BoundEvaluationError::MissingInputBoundContract { term: atom }));
        assert_eq!(final_evaluation, extraction);
    }

    #[test]
    fn deterministic_tie_uses_language_order_without_changing_public_cost() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let one = egraph.add(MxxLang::IntConst(1.into()));
        let two = egraph.add(MxxLang::IntConst(2.into()));
        egraph.union(two, one);
        egraph.rebuild();
        let mut classify = |_: Id, _: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification::default())
        };

        let result = extract(&egraph, two, &mut classify).unwrap();
        assert_eq!(result.cost, ProposalCost { node_count: 1, ..Default::default() });
        assert_eq!(result.expression[result.expression.root()], MxxLang::IntConst(1.into()));
    }

    #[test]
    fn switch_obligations_add_selector_take_case_max_and_sum_node_count() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let selector = egraph.add(MxxLang::IntConst(0.into()));
        let low = egraph.add(MxxLang::IntConst(1.into()));
        let high = egraph.add(MxxLang::IntConst(2.into()));
        let switch = MxxLang::Switch(
            std::iter::once(selector)
                .chain([low, low, low, high, low, low, low, low])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        egraph.add(switch.clone());
        egraph.rebuild();
        let mut candidates = vec![None; egraph.nodes().len()];
        let candidate = |node: MxxLang, cost: ProposalCost| Candidate {
            cost,
            semantic_bound: None,
            first_large_source: None,
            node,
            state: ExtractionState::Pending,
            output: None,
        };
        candidates[usize::from(egraph.find(selector))] = Some(candidate(
            MxxLang::IntConst(0.into()),
            ProposalCost {
                unsatisfied_relation_redexes: 1,
                unsatisfied_structural_redexes: 2,
                hidden_structural_redexes: 3,
                node_count: 3,
                ..Default::default()
            },
        ));
        candidates[usize::from(egraph.find(low))] = Some(candidate(
            MxxLang::IntConst(1.into()),
            ProposalCost {
                unsatisfied_relation_redexes: 2,
                unsatisfied_structural_redexes: 4,
                hidden_structural_redexes: 5,
                node_count: 5,
                ..Default::default()
            },
        ));
        candidates[usize::from(egraph.find(high))] = Some(candidate(
            MxxLang::IntConst(2.into()),
            ProposalCost {
                unsatisfied_relation_redexes: 7,
                unsatisfied_structural_redexes: 1,
                hidden_structural_redexes: 11,
                node_count: 11,
                ..Default::default()
            },
        ));

        assert_eq!(
            selected_child_obligations(&egraph, &switch, &candidates),
            Some((8, 6, 14, 50)),
            "each category adds the selector and takes the case maximum, while all stored nodes count"
        );
        let all_zero = MxxLang::Switch(vec![selector, selector, selector].into_boxed_slice());
        candidates[usize::from(egraph.find(selector))].as_mut().unwrap().cost =
            ProposalCost { node_count: 3, ..Default::default() };
        assert_eq!(
            selected_child_obligations(&egraph, &all_zero, &candidates),
            Some((0, 0, 0, 10))
        );
        candidates[usize::from(egraph.find(selector))].as_mut().unwrap().cost = ProposalCost {
            unsatisfied_relation_redexes: u64::MAX,
            unsatisfied_structural_redexes: u64::MAX,
            hidden_structural_redexes: u64::MAX,
            node_count: u64::MAX,
            ..Default::default()
        };
        assert_eq!(
            selected_child_obligations(&egraph, &all_zero, &candidates),
            Some((u64::MAX, u64::MAX, u64::MAX, u64::MAX)),
            "every accumulated category saturates independently"
        );
    }

    #[test]
    fn shared_child_is_materialized_once_as_a_dag() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let child = egraph.add(MxxLang::IntConst(7.into()));
        let root = egraph.add(MxxLang::IntAdd([child, child]));
        egraph.rebuild();
        let mut classify = |_: Id, _: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification::default())
        };

        let result = extract(&egraph, root, &mut classify).unwrap();
        assert_eq!(result.expression.as_ref().len(), 2);
        assert!(result.expression.is_dag());
    }

    #[test]
    fn internal_extraction_origins_align_with_the_shared_selected_dag() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let child = egraph.add(MxxLang::IntConst(7.into()));
        let root = egraph.add(MxxLang::IntAdd([child, child]));
        egraph.rebuild();
        let mut invalid = |_| panic!("fixture has a finite DAG");
        let mut bound_error = |error| panic!("non-matrix fixture has no bounds: {error:?}");
        let extracted = extract_best_proposal_with_origins(
            &egraph,
            root,
            &NoBounds,
            &mut ExtractionControl { invalid_dag: &mut invalid, bound_error: &mut bound_error },
            &mut |_, _, _| Ok(ProposalNodeClassification::default()),
            true,
        )
        .expect("shared DAG extracts");

        assert_eq!(extracted.origins.len(), extracted.proposal.expression.as_ref().len());
        assert_eq!(
            extracted.origins[usize::from(extracted.proposal.expression.root())],
            egraph.find(root)
        );
        assert!(extracted.origins.iter().all(|origin| *origin == egraph.find(*origin)));
    }

    #[test]
    fn local_structural_preference_precedes_propagated_work() {
        let satisfied_local_recipe =
            ProposalCost { local_checked_structural_count: 1, node_count: 3, ..Default::default() };
        let expanded_add_with_descendant_work =
            ProposalCost { unsatisfied_structural_redexes: 2, node_count: 7, ..Default::default() };

        assert_eq!(satisfied_local_recipe.unsatisfied_structural_redexes, 0);
        assert_eq!(satisfied_local_recipe.local_checked_structural_count, 1);
        assert_eq!(expanded_add_with_descendant_work.local_checked_structural_count, 0);
        assert_eq!(expanded_add_with_descendant_work.unsatisfied_structural_redexes, 2);
        assert!(
            expanded_add_with_descendant_work < satisfied_local_recipe,
            "local structural preference precedes propagated structural obligations"
        );
    }

    #[test]
    fn local_structural_work_precedes_propagated_structural_work() {
        let local = ProposalCost {
            local_checked_structural_count: 1,
            unsatisfied_structural_redexes: 1,
            ..Default::default()
        };
        let propagated = ProposalCost { unsatisfied_structural_redexes: 2, ..Default::default() };

        assert!(
            propagated < local,
            "local structural work is ordered before propagated structural obligations"
        );
    }

    #[test]
    fn classification_work_is_bounded_by_classes_times_nodes() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let leaf = egraph.add(MxxLang::IntConst(1.into()));
        let mut root = leaf;
        for _ in 0..16 {
            root = egraph.add(MxxLang::IntAdd([root, leaf]));
        }
        egraph.rebuild();
        let calls = Cell::new(0_usize);
        let mut classify = |_: Id, _: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            calls.set(calls.get() + 1);
            Ok(ProposalNodeClassification::default())
        };

        let result = extract(&egraph, root, &mut classify).unwrap();
        assert!(result.expression.is_dag());
        assert!(calls.get() <= egraph.number_of_classes() * egraph.total_size());
    }
}
