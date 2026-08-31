//! Stable, machine-readable inventories of frozen graph structure.
//!
//! An inventory records the graph that a checker is asked to analyze before
//! lowering begins.  It is intentionally a structural view: it does not
//! evaluate expressions or infer any operational property.

use crate::{
    FrozenGraphScopeId, Graph, NodeId, OutputRoot, Port, WireRef, WireType,
    encoding::{EncodingError, hash_canonical},
    node::NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// One stable snapshot of every frozen scope and node in a graph.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GraphInventory {
    /// SHA-256 of the canonical symbolic graph, encoded as lower-case hex.
    pub symbolic_graph_digest: String,
    pub graph_name: String,
    pub root_outputs: BTreeMap<String, OutputRoot>,
    pub effect_roots: Vec<WireRef>,
    pub scopes: Vec<ScopeInventory>,
    /// One edge for each structural owner node, without expanding call paths.
    pub structural_edges: Vec<StructuralEdge>,
}

/// The frozen nodes and boundary wires belonging to one concrete scope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScopeInventory {
    pub scope: FrozenGraphScopeId,
    pub inputs: Vec<WireRef>,
    pub outputs: Vec<WireRef>,
    pub nodes: Vec<NodeInventory>,
}

/// One node, including its exact predecessor wires and every output port.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeInventory {
    pub id: NodeId,
    pub kind: NodeKind,
    pub predecessors: Vec<WireRef>,
    pub outputs: Vec<OutputPortInventory>,
}

/// The type of one node output port.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputPortInventory {
    pub port: Port,
    pub wire_type: WireType,
}

/// One direct structural edge from an owner node to its child scope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StructuralEdge {
    pub parent_scope: FrozenGraphScopeId,
    pub owner: NodeId,
    pub child_scope: FrozenGraphScopeId,
}

#[derive(Debug, Error)]
pub enum InventoryError {
    #[error(transparent)]
    Encoding(#[from] EncodingError),
    #[error("structural scope {scope:?} references a missing child scope")]
    MissingChildScope { scope: FrozenGraphScopeId },
}

/// Produces a deterministic, non-evaluating inventory for a frozen graph.
///
/// Structural edges are recorded once per owner node.  This is linear in the
/// frozen graph size and does not expand repeated subgraph calls into paths.
pub fn inventory(graph: &Graph) -> Result<GraphInventory, InventoryError> {
    let symbolic_graph_digest = hex_digest(hash_canonical(graph)?);
    let scopes = graph
        .scopes()
        .values()
        .map(|scope| ScopeInventory {
            scope: scope.id().clone(),
            inputs: scope.inputs().to_vec(),
            outputs: scope.outputs().to_vec(),
            nodes: scope
                .nodes()
                .iter()
                .enumerate()
                .map(|(index, node)| {
                    let id = NodeId(index as u64);
                    NodeInventory {
                        id,
                        kind: node.kind().clone(),
                        predecessors: scope.arguments(node).expect("frozen node belongs to scope"),
                        outputs: node
                            .output_types()
                            .iter()
                            .enumerate()
                            .map(|(port, wire_type)| OutputPortInventory {
                                port: Port(port as u32),
                                wire_type: wire_type.clone(),
                            })
                            .collect(),
                    }
                })
                .collect(),
        })
        .collect();
    let structural_edges = graph
        .scopes()
        .iter()
        .flat_map(|(scope_id, scope)| {
            scope.nodes().iter().enumerate().filter_map(move |(index, _)| {
                let owner = NodeId(index as u64);
                graph.child_scope_id(scope_id, owner).map(|child_scope| StructuralEdge {
                    parent_scope: scope_id.clone(),
                    owner,
                    child_scope,
                })
            })
        })
        .collect::<Vec<_>>();
    for edge in &structural_edges {
        if graph.scope(&edge.child_scope).is_none() {
            return Err(InventoryError::MissingChildScope { scope: edge.child_scope.clone() });
        }
    }
    Ok(GraphInventory {
        symbolic_graph_digest,
        graph_name: graph.name().to_owned(),
        root_outputs: graph.outputs().clone(),
        effect_roots: graph.effect_roots().to_vec(),
        scopes,
        structural_edges,
    })
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GraphOutput, NodeHandle, SubgraphHandle, WireType,
        node::{LoopInputMode, NodeKind, ParallelLoop},
        with_new_construction_scope,
    };

    fn input(name: &str, wire_type: WireType) -> crate::ValueHandle {
        NodeHandle::new(
            NodeKind::Input { name: name.to_owned(), wire_type: wire_type.clone(), artifact: None },
            Vec::new(),
            vec![wire_type],
        )
        .output(0)
        .expect("input output")
    }

    fn graph() -> Graph {
        let pair = with_new_construction_scope(|scope| {
            let integer = input("integer", WireType::Int);
            let boolean = input("boolean", WireType::Bool);
            SubgraphHandle::new(
                "pair",
                scope,
                vec![integer.clone(), boolean.clone()],
                vec![integer, boolean],
            )
            .expect("pair subgraph")
        });
        let loop_body = with_new_construction_scope(|scope| {
            let value = input("value", WireType::Int);
            SubgraphHandle::new("loop-body", scope, vec![value.clone()], vec![value])
                .expect("loop body")
        });
        let integer = input("input-integer", WireType::Int);
        let boolean = input("input-boolean", WireType::Bool);
        let first = NodeHandle::subgraph_call(
            pair.clone(),
            vec![integer.clone(), boolean.clone()],
            Vec::new(),
            vec![None, None],
        );
        let second = NodeHandle::subgraph_call(
            pair,
            vec![integer.clone(), boolean],
            Vec::new(),
            vec![None, None],
        );
        let loop_output = NodeHandle::parallel_loop(
            loop_body,
            vec![integer],
            vec![WireType::Int],
            ParallelLoop {
                count: crate::IntExpr::constant(1),
                minimum_count: 0,
                index_slot: 0,
                bindings: Vec::new(),
                input_modes: vec![LoopInputMode::Broadcast],
                output_mode: crate::node::ParallelOutputMode::Family,
            },
        )
        .output(0)
        .expect("loop output");
        let first_integer = first.output(0).expect("first integer output");
        let first_boolean = first.output(1).expect("first boolean output");
        let second_integer = second.output(0).expect("second integer output");
        let second_boolean = second.output(1).expect("second boolean output");
        Graph::freeze(
            "inventory",
            Vec::new(),
            BTreeMap::from([
                (
                    "first-integer".to_owned(),
                    GraphOutput { value: first_integer, confidentiality: None },
                ),
                (
                    "first-boolean".to_owned(),
                    GraphOutput { value: first_boolean, confidentiality: None },
                ),
                (
                    "second-integer".to_owned(),
                    GraphOutput { value: second_integer, confidentiality: None },
                ),
                ("loop".to_owned(), GraphOutput { value: loop_output, confidentiality: None }),
            ]),
            Vec::new(),
            vec![second_boolean],
            BTreeMap::new(),
        )
        .expect("graph freezes")
        .0
    }

    #[test]
    fn inventory_records_shared_subgraphs_loops_outputs_effects_and_boundaries() {
        let inventory = inventory(&graph()).expect("inventory succeeds");
        assert_eq!(inventory.graph_name, "inventory");
        assert_eq!(inventory.symbolic_graph_digest.len(), 64);
        assert_eq!(inventory.root_outputs.len(), 4);
        assert_eq!(inventory.effect_roots.len(), 1);
        assert_eq!(inventory.scopes.len(), 3);
        assert_eq!(inventory.structural_edges.len(), 3);
        assert_eq!(
            inventory
                .structural_edges
                .iter()
                .filter(|edge| {
                    matches!(&edge.child_scope, FrozenGraphScopeId::Subgraph { canonical_name } if canonical_name == "pair")
                })
                .count(),
            2
        );
        assert!(
            inventory.structural_edges.iter().any(|edge| {
                matches!(edge.child_scope, FrozenGraphScopeId::ParallelBody { .. })
            })
        );
        let root = inventory
            .scopes
            .iter()
            .find(|scope| scope.scope == FrozenGraphScopeId::Root)
            .expect("root scope is recorded");
        let pair = inventory
            .scopes
            .iter()
            .find(|scope| {
                matches!(&scope.scope, FrozenGraphScopeId::Subgraph { canonical_name } if canonical_name == "pair")
            })
            .expect("shared subgraph is recorded");
        assert_eq!(pair.inputs.len(), 2);
        assert_eq!(pair.outputs.len(), 2);
        let nodes = &root.nodes;
        let input = nodes
            .iter()
            .find(|node| matches!(&node.kind, NodeKind::Input { name, .. } if name == "input-integer"))
            .expect("input node is recorded");
        assert!(input.predecessors.is_empty());
        assert_eq!(input.outputs[0].port, Port(0));
        assert_eq!(input.outputs[0].wire_type, WireType::Int);
    }

    #[test]
    fn inventory_is_stable_and_round_trips_through_json() {
        let first = inventory(&graph()).expect("first inventory");
        let second = inventory(&graph()).expect("second inventory");
        assert_eq!(first, second);
        let encoded = serde_json::to_vec(&first).expect("inventory serializes");
        let decoded = serde_json::from_slice(&encoded).expect("inventory deserializes");
        assert_eq!(first, decoded);
    }
}
