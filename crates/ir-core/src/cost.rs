//! Application-independent work assigned to existing frozen operations.
//!
//! A site charges an opaque event when its producer executes, not when an operand references
//! that producer. The graph's ordinary structural traversal handles shared nodes, repeated
//! calls, and index-dependent loop bounds. Units are constant within a scope invocation;
//! applications must split or refine their accounting if the units themselves depend on data.
//! This counts additive work, not parallel latency or backend kernel launches.

use crate::{FrozenGraphScopeId, Graph, NodeId, ParamEnv, expr::ExprError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// One event on one producer. Ports are deliberately absent: reading multiple outputs of an
/// operation does not execute it multiple times. Event interpretation belongs to the caller.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkSite {
    pub scope: FrozenGraphScopeId,
    pub node: NodeId,
    pub event: String,
    pub units: u128,
}

pub(crate) type NodeWeights =
    BTreeMap<FrozenGraphScopeId, BTreeMap<NodeId, BTreeMap<String, u128>>>;

#[derive(Debug, Error)]
pub enum WorkError {
    #[error("work site does not identify a frozen producer: {scope:?}, {node:?}")]
    MissingProducer { scope: FrozenGraphScopeId, node: NodeId },
    #[error("duplicate work event {event} on {scope:?}, {node:?}")]
    DuplicateEvent { scope: FrozenGraphScopeId, node: NodeId, event: String },
    #[error(transparent)]
    Expression(#[from] ExprError),
}

impl Graph {
    /// Counts application-supplied work using the same traversal as [`Graph::operation_counts`].
    ///
    /// Sharing a producer within one scope charges it once. Each call executes its child, and
    /// each loop iteration executes its body with the actual bindings. A proven index-invariant
    /// body is counted once and multiplied; analysis-cache hits never erase execution work.
    /// Missing producers, duplicate ownership of an event and arithmetic overflow are errors.
    pub fn execution_work(
        &self,
        env: &ParamEnv,
        sites: &[WorkSite],
    ) -> Result<BTreeMap<String, u128>, WorkError> {
        let mut weights = NodeWeights::new();
        for site in sites {
            if self.scope(&site.scope).and_then(|scope| scope.node(site.node)).is_none() {
                return Err(WorkError::MissingProducer {
                    scope: site.scope.clone(),
                    node: site.node,
                });
            }
            if weights
                .entry(site.scope.clone())
                .or_default()
                .entry(site.node)
                .or_default()
                .insert(site.event.clone(), site.units)
                .is_some()
            {
                return Err(WorkError::DuplicateEvent {
                    scope: site.scope.clone(),
                    node: site.node,
                    event: site.event.clone(),
                });
            }
        }
        Ok(self.weighted_operation_counts(env, Some(&weights))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GraphOutput, IntExpr, NodeHandle, SubgraphHandle, ValueHandle, WireType,
        graph::with_new_construction_scope,
        node::{LoopInputMode, NodeKind, ParallelLoop, SequentialLoop},
        types::MatrixType,
    };

    fn matrix() -> WireType {
        WireType::Matrix(MatrixType {
            ring: crate::ring::test_ring(17, 8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        })
    }

    fn input(name: &str) -> ValueHandle {
        NodeHandle::new(
            NodeKind::Input { name: name.to_owned(), wire_type: matrix(), artifact: None },
            Vec::new(),
            vec![matrix()],
        )
        .output(0)
        .unwrap()
    }

    fn scale(value: ValueHandle) -> ValueHandle {
        NodeHandle::new(
            NodeKind::MatrixScale { scalar: IntExpr::constant(2) },
            vec![value],
            vec![matrix()],
        )
        .output(0)
        .unwrap()
    }

    fn freeze(outputs: Vec<ValueHandle>) -> (Graph, crate::FreezeMap) {
        Graph::freeze(
            "work-counts",
            Vec::new(),
            outputs
                .into_iter()
                .enumerate()
                .map(|(i, value)| (format!("out-{i}"), GraphOutput { value, availability: None }))
                .collect(),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
    }

    fn site(map: &crate::FreezeMap, value: &ValueHandle, units: u128) -> WorkSite {
        let location = map.resolve_unique(value).unwrap();
        WorkSite {
            scope: location.scope.clone(),
            node: location.wire.node,
            event: "example-work".to_owned(),
            units,
        }
    }

    fn export_counted_fixture(graph: &Graph, sites: &[WorkSite], name: &str) {
        let validated = crate::ring::test_validate(graph, &ParamEnv::default()).unwrap();
        let options = crate::lean::ExportOptions {
            module_name: name.to_owned(),
            namespace: name.to_owned(),
            ..Default::default()
        };
        let plain = crate::lean::export(&validated, &options).unwrap();
        let counted = crate::lean::cost::export(&validated, &options, sites).unwrap();
        assert_eq!(plain.source, counted.source);
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_execution_cost");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(format!("{name}.lean")), &counted.source).unwrap();
        std::fs::write(
            directory.join(format!("{name}_Cost.lean")),
            counted.cost_source.as_ref().unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn shared_fanout_and_separately_constructed_producers_have_distinct_work() {
        let source = input("source");
        let producer = scale(source.clone());
        let separate = scale(source);
        let (graph, map) =
            freeze(vec![scale(producer.clone()), scale(producer.clone()), separate.clone()]);
        assert_eq!(graph.operation_counts(&ParamEnv::default()).unwrap()["MatrixScale"], 4);
        assert_eq!(
            graph.execution_work(&ParamEnv::default(), &[site(&map, &producer, 7)]).unwrap(),
            BTreeMap::from([("example-work".to_owned(), 7)]),
        );
        assert_eq!(
            graph
                .execution_work(
                    &ParamEnv::default(),
                    &[site(&map, &producer, 7), site(&map, &separate, 7)]
                )
                .unwrap()["example-work"],
            14,
        );
        let assigned = site(&map, &producer, 7);
        assert!(matches!(
            graph.execution_work(&ParamEnv::default(), &[assigned.clone(), assigned.clone()]),
            Err(WorkError::DuplicateEvent { .. })
        ));
        let bad = WorkSite { node: NodeId(u64::MAX), ..assigned };
        assert!(matches!(
            graph.execution_work(&ParamEnv::default(), &[bad]),
            Err(WorkError::MissingProducer { .. })
        ));
        export_counted_fixture(
            &graph,
            &[site(&map, &producer, 7), site(&map, &separate, 7)],
            "SharedWork",
        );
    }

    #[test]
    fn named_calls_with_different_bindings_are_charged_separately() {
        let (definition, leaf) = with_new_construction_scope(|scope| {
            let argument = input("argument");
            let (body, leaf) = with_new_construction_scope(|child_scope| {
                let state = input("state");
                let leaf = scale(state.clone());
                (
                    SubgraphHandle::new("inner", child_scope, vec![state], vec![leaf.clone()])
                        .unwrap(),
                    leaf,
                )
            });
            let output = NodeHandle::sequential_loop(
                body,
                vec![argument.clone()],
                vec![matrix()],
                SequentialLoop {
                    count: IntExpr::Var("iterations".to_owned()),
                    index_slot: 0,
                    bindings: Vec::new(),
                    carried_count: 1,
                },
            )
            .output(0)
            .unwrap();
            (SubgraphHandle::new("named", scope, vec![argument], vec![output]).unwrap(), leaf)
        });
        let source = input("source");
        let outputs = [2, 3].map(|count| {
            NodeHandle::subgraph_call(
                definition.clone(),
                vec![source.clone()],
                vec![("iterations".to_owned(), IntExpr::constant(count))],
                vec![None],
            )
            .output(0)
            .unwrap()
        });
        let (graph, map) = freeze(outputs.to_vec());
        let assigned = site(&map, &leaf, 7);
        assert_eq!(
            graph.execution_work(&ParamEnv::default(), &[assigned.clone()]).unwrap()["example-work"],
            35
        );
        assert!(matches!(
            graph
                .execution_work(&ParamEnv::default(), &[WorkSite { units: u128::MAX, ..assigned }]),
            Err(WorkError::Expression(ExprError::CountOverflow))
        ));
    }

    #[test]
    fn nested_loop_bounds_follow_the_actual_outer_index_and_zero_loops_do_no_work() {
        for count in [0, 3] {
            let (outer_body, leaf) = with_new_construction_scope(|scope| {
                let argument = input("argument");
                let (inner_body, leaf) = with_new_construction_scope(|inner_scope| {
                    let state = input("state");
                    let leaf = scale(state.clone());
                    (
                        SubgraphHandle::new("inner", inner_scope, vec![state], vec![leaf.clone()])
                            .unwrap(),
                        leaf,
                    )
                });
                let output = NodeHandle::sequential_loop(
                    inner_body,
                    vec![argument.clone()],
                    vec![matrix()],
                    SequentialLoop {
                        count: IntExpr::Add(
                            Box::new(IntExpr::LoopIndex(0)),
                            Box::new(IntExpr::constant(1)),
                        ),
                        index_slot: 1,
                        bindings: Vec::new(),
                        carried_count: 1,
                    },
                )
                .output(0)
                .unwrap();
                (SubgraphHandle::new("outer", scope, vec![argument], vec![output]).unwrap(), leaf)
            });
            let output = NodeHandle::parallel_loop(
                outer_body,
                vec![input("source")],
                vec![WireType::IndexedFamily {
                    element: Box::new(matrix()),
                    count: IntExpr::constant(count),
                }],
                ParallelLoop {
                    count: IntExpr::constant(count),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: Vec::new(),
                    input_modes: vec![LoopInputMode::Broadcast],
                },
            )
            .output(0)
            .unwrap();
            let (graph, map) = freeze(vec![output]);
            let result =
                graph.execution_work(&ParamEnv::default(), &[site(&map, &leaf, 7)]).unwrap();
            assert_eq!(
                result.get("example-work").copied().unwrap_or(0),
                if count == 0 { 0 } else { 42 }
            );
        }
    }
}
