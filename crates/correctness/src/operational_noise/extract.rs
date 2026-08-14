//! Polynomial-time extraction of one deterministic operational-noise proposal.
//!
//! Relation applicability and the `Large` role of an atom are owned by the
//! relation and producer registries, respectively.  The extractor therefore
//! accepts one classification callback instead of copying either registry or
//! guessing from an atom's matrix type.

use super::{analysis::MxxAnalysis, error::OperationalSimulationError, language::MxxLang};
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProposalNodeClassification {
    /// This exact e-node is a relation redex which a checked rewrite could consume.
    pub relation_redex: bool,
    /// This exact e-node is an atom whose producer role is `Large`.
    pub large_atom: bool,
}

/// The extracted DAG and the cost that selected it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtractedProposal {
    pub cost: ProposalCost,
    pub expression: RecExpr<MxxLang>,
}

/// Job-owned resource checks injected by the simulation driver.
///
/// These callbacks are the only extraction boundary to `CheckerLimits`, the
/// cumulative owned-element budget, and the one shared deadline.  They return
/// the driver's fully populated typed resource error, so extraction does not
/// create a second limits object or diagnostics owner.
pub struct ExtractionControl<'a> {
    pub check_node_count: &'a mut dyn FnMut(usize) -> Result<(), OperationalSimulationError>,
    pub reserve_owned_elements: &'a mut dyn FnMut(usize) -> Result<(), OperationalSimulationError>,
    pub check_deadline: &'a mut dyn FnMut() -> Result<(), OperationalSimulationError>,
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
    (control.check_deadline)()?;
    let node_count = egraph.total_size();
    (control.check_deadline)()?;
    (control.check_node_count)(node_count)?;

    let class_count = egraph.number_of_classes();
    (control.reserve_owned_elements)(class_count)?;
    let mut classes = Vec::with_capacity(class_count);
    for class in egraph.classes() {
        (control.check_deadline)()?;
        classes.push(class.id);
    }
    (control.check_deadline)()?;
    classes.sort_unstable();
    (control.check_deadline)()?;

    // Canonical egg ids are bounded by the number of inserted e-nodes.  One
    // indexed table holds both dynamic-programming choices and DAG build state.
    let slot_count = egraph.nodes().len();
    (control.reserve_owned_elements)(slot_count)?;
    let mut candidates = vec![None::<Candidate>; slot_count];

    for _ in 0..class_count {
        (control.check_deadline)()?;
        let mut changed = false;
        for &class_id in &classes {
            (control.check_deadline)()?;
            let canonical = egraph.find(class_id);
            let index = usize::from(canonical);
            let class = &egraph[canonical];
            for node in class.iter() {
                (control.check_deadline)()?;
                let Some(cost) =
                    proposal_cost(egraph, canonical, node, &candidates, control, classify)?
                else {
                    continue;
                };
                let improves = candidates[index].as_ref().is_none_or(|current| {
                    cost < current.cost || (cost == current.cost && node < &current.node)
                });
                if improves {
                    // A cloned variadic e-node owns these child slots.  A later
                    // replacement reserves again; the cumulative budget is not refunded.
                    (control.reserve_owned_elements)(node.children().len())?;
                    candidates[index] = Some(Candidate {
                        cost,
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

    (control.reserve_owned_elements)(1)?;
    let mut work = vec![BuildFrame::Enter(root)];
    let mut nodes = Vec::<MxxLang>::new();
    while let Some(frame) = work.pop() {
        (control.check_deadline)()?;
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
                let child_count = candidate.node.children().len();
                (control.reserve_owned_elements)(1usize.saturating_add(child_count))?;
                work.push(BuildFrame::Finish(class));
                for &child in candidate.node.children().iter().rev() {
                    (control.check_deadline)()?;
                    work.push(BuildFrame::Enter(egraph.find(child)));
                }
            }
            BuildFrame::Finish(_) => {
                let Some(candidate) = candidates.get(index).and_then(Option::as_ref) else {
                    return Err((control.invalid_dag)(class));
                };
                (control.reserve_owned_elements)(
                    1usize.saturating_add(candidate.node.children().len()),
                )?;
                let mut missing_child = None;
                let mut control_error = None;
                let output_node = candidate.node.clone().map_children(|child| {
                    if control_error.is_none() {
                        if let Err(error) = (control.check_deadline)() {
                            control_error = Some(error);
                        }
                    }
                    let child = egraph.find(child);
                    let output = candidates
                        .get(usize::from(child))
                        .and_then(Option::as_ref)
                        .and_then(|candidate| candidate.output);
                    match output {
                        Some(output) => output,
                        None => {
                            missing_child = Some(child);
                            Id::from(0)
                        }
                    }
                });
                if let Some(error) = control_error {
                    return Err(error);
                }
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

    (control.check_deadline)()?;
    (control.check_node_count)(node_count)?;
    let expression = RecExpr::from(nodes);
    if !expression.is_dag() {
        return Err((control.invalid_dag)(root));
    }
    Ok(ExtractedProposal { cost: root_cost, expression })
}

fn proposal_cost(
    egraph: &EGraph<MxxLang, MxxAnalysis>,
    class: Id,
    node: &MxxLang,
    candidates: &[Option<Candidate>],
    control: &mut ExtractionControl<'_>,
    classify: &mut dyn FnMut(
        Id,
        &MxxLang,
        &EGraph<MxxLang, MxxAnalysis>,
    ) -> Result<ProposalNodeClassification, OperationalSimulationError>,
) -> Result<Option<ProposalCost>, OperationalSimulationError> {
    let mut child_remaining = 0_u64;
    let mut child_hidden = 0_u64;
    let mut large_atom_count = 0_u64;
    let mut node_count = 1_u64;
    for &child in node.children() {
        (control.check_deadline)()?;
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
    Ok(Some(ProposalCost {
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
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        analysis::MxxAnalysis,
        error::{CheckerPhase, ResourceLimitKind, ResourceObserved},
    };
    use std::{cell::Cell, time::Duration};

    fn resource_error(
        kind: ResourceLimitKind,
        limit: u64,
        observed: u64,
    ) -> OperationalSimulationError {
        OperationalSimulationError::ResourceLimitExceeded {
            phase: CheckerPhase::Extract,
            kind,
            observed: ResourceObserved::Counter { limit, observed },
            diagnostics: Default::default(),
        }
    }

    fn extract_with_limits(
        egraph: &EGraph<MxxLang, MxxAnalysis>,
        root: Id,
        node_limit: usize,
        owned_limit: usize,
        classify: &mut dyn FnMut(
            Id,
            &MxxLang,
            &EGraph<MxxLang, MxxAnalysis>,
        )
            -> Result<ProposalNodeClassification, OperationalSimulationError>,
    ) -> Result<ExtractedProposal, OperationalSimulationError> {
        let owned = Cell::new(0_usize);
        let mut check_nodes = |observed: usize| {
            if observed > node_limit {
                Err(resource_error(
                    ResourceLimitKind::EGraphNodes,
                    node_limit as u64,
                    observed as u64,
                ))
            } else {
                Ok(())
            }
        };
        let mut reserve = |amount: usize| {
            let observed = owned.get().checked_add(amount).unwrap_or(usize::MAX);
            if observed > owned_limit {
                Err(resource_error(
                    ResourceLimitKind::TotalOwnedElements,
                    owned_limit as u64,
                    observed as u64,
                ))
            } else {
                owned.set(observed);
                Ok(())
            }
        };
        let mut deadline = || Ok(());
        let mut invalid = |_| {
            resource_error(ResourceLimitKind::EGraphNodes, node_limit as u64, node_limit as u64)
        };
        extract_best_proposal(
            egraph,
            root,
            &mut ExtractionControl {
                check_node_count: &mut check_nodes,
                reserve_owned_elements: &mut reserve,
                check_deadline: &mut deadline,
                invalid_dag: &mut invalid,
            },
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
            })
        };
        let result = extract_with_limits(&egraph, compact_large, 32, 1_000, &mut classify).unwrap();

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
            })
        };

        let result = extract_with_limits(&egraph, outer, 32, 1_000, &mut classify).unwrap();
        assert_eq!(result.cost.remaining_relation_redexes, 1);
        assert_eq!(result.cost.hidden_relation_redexes, 1);
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

        let result = extract_with_limits(&egraph, two, 8, 128, &mut classify).unwrap();
        assert_eq!(result.cost, ProposalCost { node_count: 1, ..Default::default() });
        assert_eq!(result.expression[result.expression.root()], MxxLang::IntConst(1.into()));
    }

    #[test]
    fn node_and_owned_element_limits_fail_closed() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let root = egraph.add(MxxLang::IntConst(1.into()));
        egraph.rebuild();
        let mut classify = |_: Id, _: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification::default())
        };

        assert!(matches!(
            extract_with_limits(&egraph, root, 0, 128, &mut classify),
            Err(OperationalSimulationError::ResourceLimitExceeded {
                phase: CheckerPhase::Extract,
                kind: ResourceLimitKind::EGraphNodes,
                ..
            })
        ));
        assert!(matches!(
            extract_with_limits(&egraph, root, 8, 0, &mut classify),
            Err(OperationalSimulationError::ResourceLimitExceeded {
                phase: CheckerPhase::Extract,
                kind: ResourceLimitKind::TotalOwnedElements,
                ..
            })
        ));
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

        let result = extract_with_limits(&egraph, root, 8, 256, &mut classify).unwrap();
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

        let result = extract_with_limits(&egraph, root, 64, 10_000, &mut classify).unwrap();
        assert!(result.expression.is_dag());
        assert!(calls.get() <= egraph.number_of_classes() * egraph.total_size());
    }

    #[test]
    fn shared_deadline_failure_is_forwarded_without_fallback() {
        let mut egraph = EGraph::new(MxxAnalysis::default());
        let root = egraph.add(MxxLang::IntConst(1.into()));
        egraph.rebuild();
        let calls = Cell::new(0_usize);
        let mut check_nodes = |_| Ok(());
        let mut reserve = |_| Ok(());
        let mut deadline = || {
            let observed = calls.get() + 1;
            calls.set(observed);
            if observed >= 3 {
                Err(OperationalSimulationError::ResourceLimitExceeded {
                    phase: CheckerPhase::Extract,
                    kind: ResourceLimitKind::TotalTime,
                    observed: ResourceObserved::Duration {
                        limit: Duration::from_secs(1),
                        observed: Duration::from_secs(2),
                    },
                    diagnostics: Default::default(),
                })
            } else {
                Ok(())
            }
        };
        let mut invalid = |_| resource_error(ResourceLimitKind::EGraphNodes, 1, 1);
        let mut classify = |_: Id, _: &MxxLang, _: &EGraph<MxxLang, MxxAnalysis>| {
            Ok(ProposalNodeClassification::default())
        };

        assert!(matches!(
            extract_best_proposal(
                &egraph,
                root,
                &mut ExtractionControl {
                    check_node_count: &mut check_nodes,
                    reserve_owned_elements: &mut reserve,
                    check_deadline: &mut deadline,
                    invalid_dag: &mut invalid,
                },
                &mut classify,
            ),
            Err(OperationalSimulationError::ResourceLimitExceeded {
                phase: CheckerPhase::Extract,
                kind: ResourceLimitKind::TotalTime,
                ..
            })
        ));
    }
}
