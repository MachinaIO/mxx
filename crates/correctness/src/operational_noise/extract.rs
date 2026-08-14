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
                let improves = candidates[index].as_ref().is_none_or(|current| {
                    cost < current.cost || (cost == current.cost && node < &current.node)
                });
                if improves {
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
    let Some(root_candidate) = candidates.get(root_index).and_then(Option::as_ref) else {
        return Err((control.invalid_dag)(root));
    };
    let root_cost = root_candidate.cost.clone();

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
                let output = Id::from(nodes.len());
                nodes.push(output_node);
                let Some(candidate) = candidates[index].as_mut() else {
                    return Err((control.invalid_dag)(class));
                };
                candidate.output = Some(output);
                candidate.state = ExtractionState::Complete;
            }
        }
    }

    let expression = RecExpr::from(nodes);
    if !expression.is_dag() {
        return Err((control.invalid_dag)(root));
    }
    let first_large_source = candidates
        .get(root_index)
        .and_then(Option::as_ref)
        .and_then(|candidate| candidate.first_large_source);
    Ok(ExtractedProposal { cost: root_cost, first_large_source, expression })
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
    let first_large_source = large_residual
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
        .flatten();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        analysis::{MxxAnalysis, MxxSort},
        bound::{BoundEvaluationError, ResolvedMatrixConstant},
        identity::{
            AtomicSourceDescriptor, AtomicSourceKey, CanonicalResidueConvention, CrtSpecId,
            MatrixConstantSpecId, ResolvedIntExpr, ResolvedMatrixType,
        },
    };
    use mxx_ir_core::types::ConcreteMatrixType;
    use num_bigint::{BigInt, BigUint};
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
        atom_classes: BTreeMap<AtomicSourceId, BoundClass>,
        missing: Option<AtomicSourceId>,
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
        fn matrix_type(&self, _: Id) -> Result<ConcreteMatrixType, BoundEvaluationError> {
            Ok(concrete_scalar_matrix_type())
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
                matrix_type: concrete_scalar_matrix_type(),
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
