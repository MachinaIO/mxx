//! Polynomial-time extraction of one deterministic operational-noise proposal.
//!
//! Relation applicability comes from the checked relation registry.  Every
//! selected matrix candidate is classified by the same authoritative source
//! resolver and node transfer used by final bound evaluation.

use super::{
    analysis::{MxxAnalysis, MxxSort},
    bound::{
        BoundClass, BoundEvaluationError, BoundEvaluator, BoundInput, MatrixBound,
        SelectedChildBounds,
    },
    error::OperationalSimulationError,
    identity::AtomicSourceId,
    language::MxxLang,
};
use egg::{EGraph, Id, Language, RecExpr};

/// The exact lexicographic preference used to select a final expression.
///
/// This cost is not a noise estimate.  Saturating arithmetic keeps it
/// monotone even when a compact e-graph represents an exponentially large AST.
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct ProposalCost {
    pub remaining_relation_redexes: u64,
    pub hidden_relation_redexes: u64,
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
/// checks.  Matrix semantics are not part of this callback: they are computed
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
    Ok(ExtractedProposal {
        cost: root_candidate.cost.clone(),
        semantic_bound: root_candidate.semantic_bound.clone(),
        first_large_source: root_candidate.first_large_source,
        expression,
    })
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
    let mut child_remaining = 0_u64;
    let mut child_hidden = 0_u64;
    let mut node_count = 1_u64;
    for &child in node.children() {
        let child = egraph.find(child);
        let Some(child) = candidates.get(usize::from(child)).and_then(Option::as_ref) else {
            return Ok(None);
        };
        child_remaining = child_remaining.saturating_add(child.cost.remaining_relation_redexes);
        child_hidden = child_hidden.saturating_add(child.cost.hidden_relation_redexes);
        node_count = node_count.saturating_add(child.cost.node_count);
    }
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
            remaining_relation_redexes: child_remaining
                .saturating_add(u64::from(classification.relation_redex)),
            // At an addition all relation redexes below it are hidden exactly once;
            // an enclosing addition replaces this value with the same descendant count.
            hidden_relation_redexes: if matches!(node, MxxLang::MatrixAdd(_)) {
                child_remaining
            } else {
                child_hidden
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
            AtomicRelationRole, AtomicSourceDescriptor, AtomicSourceKey,
            CanonicalResidueConvention, CrtSpecId, MatrixConstantSpecId, ResolvedIndexRange,
            ResolvedIntExpr, ResolvedMatrixType, SliceSpec, SliceSpecId,
        },
        relation::{
            RelationApplier, RelationRegistration, RelationSearcher, RewriteContext,
            SharedRewriteBudget,
        },
    };
    use mxx_ir_core::types::ConcreteMatrixType;
    use num_bigint::{BigInt, BigUint};
    use num_traits::ToPrimitive;
    use std::{cell::Cell, collections::BTreeMap};

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
            &mut |_, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, node, &context)
                    .map(|relation_redex| ProposalNodeClassification { relation_redex })
                    .map_err(|failure| panic!("fixture relation is valid: {failure:?}"))
            },
        )
        .expect("two-level normalized residual extracts");

        assert_eq!(result.cost.remaining_relation_redexes, 0);
        assert_eq!(result.cost.hidden_relation_redexes, 0);
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
            &mut |_, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, node, &context)
                    .map(|relation_redex| ProposalNodeClassification { relation_redex })
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
            Ok(ProposalNodeClassification { relation_redex: matches!(node, MxxLang::IntAdd(_)) })
        };
        let result = extract(&egraph, compact_large, &mut classify).unwrap();

        assert_eq!(result.cost.remaining_relation_redexes, 0);
        assert!(!result.cost.large_residual);
        assert!(matches!(result.expression[result.expression.root()], MxxLang::IntMul(_)));
    }

    #[test]
    fn addition_counts_descendant_redexes_as_hidden_once() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let left = egraph.add(MxxLang::IntConst(1.into()));
        let right = egraph.add(MxxLang::IntConst(2.into()));
        let redex = egraph.add(MxxLang::IntMul([left, right]));
        let inner = egraph.add(MxxLang::MatrixAdd(vec![redex].into_boxed_slice()));
        let outer = egraph.add(MxxLang::MatrixAdd(vec![inner].into_boxed_slice()));
        egraph.rebuild();
        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification { relation_redex: matches!(node, MxxLang::IntMul(_)) })
        };

        let result = extract(&egraph, outer, &mut classify).unwrap();
        assert_eq!(result.cost.remaining_relation_redexes, 1);
        assert_eq!(result.cost.hidden_relation_redexes, 1);
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

        let result = extract_with_input(&egraph, root, &input);

        assert!(result.cost.large_residual);
        assert_eq!(result.first_large_source, Some(large_source));
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
            &mut |_, node, egraph| {
                super::super::relation::classify_proposal_node(egraph, node, &context)
                    .map(|relation_redex| ProposalNodeClassification { relation_redex })
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
