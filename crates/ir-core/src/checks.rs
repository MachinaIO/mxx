use crate::{
    graph::Graph,
    types::{ConcreteMatrixType, NodeId},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WarningKind {
    RuntimeSelectBoundsCheck,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElaborationWarning {
    pub node: NodeId,
    pub kind: WarningKind,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum CheckError {
    #[error("duplicate node id {0:?}")]
    DuplicateNode(NodeId),
    #[error("node {node:?} refers to unavailable node {dependency:?}")]
    NotTopological { node: NodeId, dependency: NodeId },
    #[error("graph output {name} refers to unavailable node {node:?}")]
    InvalidOutput { name: String, node: NodeId },
    #[error("matrix modulus or ring dimension mismatch")]
    RingMismatch,
    #[error("matrix shape mismatch: left {left:?}, right {right:?}")]
    ShapeMismatch { left: ConcreteMatrixType, right: ConcreteMatrixType },
}

pub fn check_topological(graph: &Graph) -> Result<(), CheckError> {
    let mut seen = BTreeSet::new();
    for node in &graph.nodes {
        if !seen.insert(node.id) {
            return Err(CheckError::DuplicateNode(node.id));
        }
        for argument in &node.args {
            if !seen.contains(&argument.node) {
                return Err(CheckError::NotTopological { node: node.id, dependency: argument.node });
            }
        }
    }
    for (name, wire) in &graph.outputs {
        if !seen.contains(&wire.node) {
            return Err(CheckError::InvalidOutput { name: name.clone(), node: wire.node });
        }
    }
    Ok(())
}

pub fn check_same_ring(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<(), CheckError> {
    if left.modulus != right.modulus || left.ring_dimension != right.ring_dimension {
        return Err(CheckError::RingMismatch);
    }
    Ok(())
}

pub fn check_add_shape(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<(), CheckError> {
    check_same_ring(left, right)?;
    if left.rows != right.rows || left.columns != right.columns {
        return Err(CheckError::ShapeMismatch { left: left.clone(), right: right.clone() });
    }
    Ok(())
}

pub fn multiplication_type(
    left: &ConcreteMatrixType,
    right: &ConcreteMatrixType,
) -> Result<ConcreteMatrixType, CheckError> {
    check_same_ring(left, right)?;
    let (rows, columns) = if left.is_scalar() {
        (right.rows, right.columns)
    } else if right.is_scalar() {
        (left.rows, left.columns)
    } else if left.columns == right.rows {
        (left.rows, right.columns)
    } else {
        return Err(CheckError::ShapeMismatch { left: left.clone(), right: right.clone() });
    };
    Ok(ConcreteMatrixType {
        modulus: left.modulus.clone(),
        ring_dimension: left.ring_dimension,
        rows,
        columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::Graph,
        node::{Node, NodeKind},
        types::{Port, WireRef},
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn wire(node: u64) -> WireRef {
        WireRef { node: NodeId(node), port: Port(0) }
    }

    fn graph(nodes: Vec<Node>, output: WireRef) -> Graph {
        Graph {
            name: "checks".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::from([("out".to_owned(), output)]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    fn integer(id: u64) -> Node {
        Node { id: NodeId(id), kind: NodeKind::ConstantInt(BigInt::from(id)), args: Vec::new() }
    }

    #[test]
    fn topological_check_rejects_duplicate_node_ids() {
        let error = check_topological(&graph(vec![integer(1), integer(1)], wire(1)))
            .expect_err("duplicate node");
        assert!(matches!(error, CheckError::DuplicateNode(NodeId(1))));
    }

    #[test]
    fn topological_check_rejects_forward_references() {
        let dependent = Node {
            id: NodeId(1),
            kind: NodeKind::IntBinary(crate::node::IntBinaryOp::Add),
            args: vec![wire(2), wire(2)],
        };
        let error = check_topological(&graph(vec![dependent, integer(2)], wire(1)))
            .expect_err("forward reference");
        assert!(matches!(
            error,
            CheckError::NotTopological { node: NodeId(1), dependency: NodeId(2) }
        ));
    }

    #[test]
    fn topological_check_rejects_missing_output_node() {
        let error =
            check_topological(&graph(vec![integer(1)], wire(2))).expect_err("missing output node");
        assert!(matches!(
            error,
            CheckError::InvalidOutput {
                name,
                node: NodeId(2),
            } if name == "out"
        ));
    }
}
