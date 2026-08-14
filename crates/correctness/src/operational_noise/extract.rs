//! Polynomial-time extraction of one deterministic operational-noise proposal.
//!
//! Relation applicability and the `Large` role of an atom are owned by the
//! relation and producer registries, respectively.  The extractor therefore
//! accepts one classification callback instead of copying either registry or
//! guessing from an atom's matrix type.

use super::{
    analysis::MxxAnalysis, error::OperationalSimulationError, identity::AtomicSourceId,
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
    pub large_atom_count: u64,
    pub node_count: u64,
}

/// Facts whose authoritative owners live outside extraction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProposalNodeClassification {
    /// This exact e-node is a relation redex which a checked rewrite could consume.
    pub relation_redex: bool,
    /// This exact e-node is an atom whose producer role is `Large`.
    pub large_atom: bool,
    /// The first Large atom key on this candidate path, retained only for an
    /// extraction failure witness and excluded from proposal ordering.
    pub large_atom_witness: Option<AtomicSourceId>,
}

/// The extracted DAG and the cost that selected it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtractedProposal {
    pub cost: ProposalCost,
    pub large_atom_witness: Option<AtomicSourceId>,
    pub expression: RecExpr<MxxLang>,
}

/// Maps extraction failures that require the simulation driver's source site.
pub struct ExtractionControl<'a> {
    /// Maps an e-class with no finite DAG representative to the existing
    /// site-bearing analysis error owned by the driver.
    pub invalid_dag: &'a mut dyn FnMut(Id) -> OperationalSimulationError,
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
    /// Classification witness for this exact selected e-node.  This is stable
    /// while relaxation compares costs; the aggregate witness is recomputed
    /// only after all selected children have completed materialization.
    own_large_atom_witness: Option<AtomicSourceId>,
    large_atom_witness: Option<AtomicSourceId>,
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
/// checks.  Likewise, `large_atom` must come from the closed producer role,
/// not from observed values or a full-residue fallback.
/// The callback must be idempotent: relaxation may classify the same e-node in
/// several passes, and diagnostics must not count callback invocations.
///
/// Relaxation performs at most one pass per canonical e-class.  Every finite
/// optimum is cycle-free because `node_count` strictly increases across an
/// edge, so its height is at most the number of classes.  The resulting bound
/// is `O(C * N)` classification/cost work for `C` classes and `N` e-nodes.
pub fn extract_best_proposal(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    root: Id,
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
                let Some((cost, own_large_atom_witness)) =
                    proposal_cost(egraph, canonical, node, &candidates, classify)?
                else {
                    continue;
                };
                let improves = candidates[index].as_ref().is_none_or(|current| {
                    cost < current.cost || (cost == current.cost && node < &current.node)
                });
                if improves {
                    candidates[index] = Some(Candidate {
                        cost,
                        own_large_atom_witness,
                        large_atom_witness: None,
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
                let mut large_atom_witness = candidate.own_large_atom_witness.clone();
                let output_node = candidate.node.clone().map_children(|child| {
                    let child = egraph.find(child);
                    let child_candidate =
                        candidates.get(usize::from(child)).and_then(Option::as_ref);
                    if large_atom_witness.is_none() {
                        large_atom_witness = child_candidate
                            .and_then(|candidate| candidate.large_atom_witness.clone());
                    }
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
                candidate.large_atom_witness = large_atom_witness;
                candidate.output = Some(output);
                candidate.state = ExtractionState::Complete;
            }
        }
    }

    let expression = RecExpr::from(nodes);
    if !expression.is_dag() {
        return Err((control.invalid_dag)(root));
    }
    let root_large_atom_witness = candidates
        .get(root_index)
        .and_then(Option::as_ref)
        .and_then(|candidate| candidate.large_atom_witness.clone());
    Ok(ExtractedProposal {
        cost: root_cost,
        large_atom_witness: root_large_atom_witness,
        expression,
    })
}

fn proposal_cost(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    class: Id,
    node: &MxxLang,
    candidates: &[Option<Candidate>],
    classify: &mut dyn FnMut(
        Id,
        &MxxLang,
        &EGraph<MxxLang, MxxAnalysis>,
    ) -> Result<ProposalNodeClassification, OperationalSimulationError>,
) -> Result<Option<(ProposalCost, Option<AtomicSourceId>)>, OperationalSimulationError> {
    let mut child_remaining = 0_u64;
    let mut child_hidden = 0_u64;
    let mut large_atom_count = 0_u64;
    let mut node_count = 1_u64;
    for &child in node.children() {
        let child = egraph.find(child);
        let Some(child) = candidates.get(usize::from(child)).and_then(Option::as_ref) else {
            return Ok(None);
        };
        child_remaining = child_remaining.saturating_add(child.cost.remaining_relation_redexes);
        child_hidden = child_hidden.saturating_add(child.cost.hidden_relation_redexes);
        large_atom_count = large_atom_count.saturating_add(child.cost.large_atom_count);
        node_count = node_count.saturating_add(child.cost.node_count);
    }
    let classification = classify(class, node, egraph)?;
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
            large_atom_count: large_atom_count.saturating_add(u64::from(classification.large_atom)),
            node_count,
        },
        classification.large_atom.then_some(classification.large_atom_witness).flatten(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        analysis::{MxxAnalysis, MxxSort},
        identity::{AtomicSourceDescriptor, AtomicSourceKey},
    };
    use std::cell::Cell;

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
        extract_best_proposal(
            egraph,
            root,
            &mut ExtractionControl { invalid_dag: &mut invalid },
            classify,
        )
    }

    #[test]
    fn lexicographic_cost_prefers_relation_then_large_then_size() {
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
                large_atom: matches!(node, MxxLang::IntMul(_)),
                large_atom_witness: None,
            })
        };
        let result = extract(&egraph, compact_large, &mut classify).unwrap();

        assert_eq!(result.cost.remaining_relation_redexes, 0);
        assert_eq!(result.cost.large_atom_count, 1);
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
            Ok(ProposalNodeClassification {
                relation_redex: matches!(node, MxxLang::IntMul(_)),
                large_atom: false,
                large_atom_witness: None,
            })
        };

        let result = extract(&egraph, outer, &mut classify).unwrap();
        assert_eq!(result.cost.remaining_relation_redexes, 1);
        assert_eq!(result.cost.hidden_relation_redexes, 1);
    }

    #[test]
    fn equal_cost_child_replacement_uses_the_final_witness() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let stale = egraph.add(MxxLang::IntConst(2.into()));
        let final_child = egraph.add(MxxLang::IntConst(1.into()));
        egraph.union(stale, final_child);
        let root = egraph.add(MxxLang::IntAdd([stale, stale]));
        egraph.rebuild();

        let stale_witness =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("stale-input")),
                sort: MxxSort::Int,
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let final_witness =
            AtomicSourceId(egraph.analysis.symbols.atomic_sources.intern(AtomicSourceDescriptor {
                key: AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("final-input")),
                sort: MxxSort::Int,
                integer_domain: None,
                canonical_residue_convention: None,
                relation_role: None,
            }));
        let mut classify = |_: Id, node: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            let large_atom_witness = match node {
                MxxLang::IntConst(value) if value == &num_bigint::BigInt::from(2) => {
                    Some(stale_witness.clone())
                }
                MxxLang::IntConst(value) if value == &num_bigint::BigInt::from(1) => {
                    Some(final_witness.clone())
                }
                _ => None,
            };
            Ok(ProposalNodeClassification {
                relation_redex: false,
                large_atom: large_atom_witness.is_some(),
                large_atom_witness,
            })
        };

        let result = extract(&egraph, root, &mut classify).unwrap();
        assert_eq!(result.cost.large_atom_count, 2);
        assert_eq!(result.large_atom_witness, Some(final_witness));
        assert_eq!(
            result
                .large_atom_witness
                .and_then(|source| egraph.analysis.symbols.atomic_sources.get(source.0))
                .map(|descriptor| descriptor.key.clone()),
            Some(AtomicSourceKey::ProtocolInput(crate::ProtocolInputId::from("final-input"))),
        );
        assert!(
            result
                .expression
                .as_ref()
                .iter()
                .any(|node| *node == MxxLang::IntConst(num_bigint::BigInt::from(1)))
        );
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
