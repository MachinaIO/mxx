use crate::{
    graph::Graph,
    types::{ConcreteMatrixType, NodeId},
};
use serde::{Deserialize, Serialize};
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
    for scope in graph.scopes().values() {
        for (position, node) in scope.nodes().iter().enumerate() {
            let id = NodeId(position as u64);
            for argument in scope.arguments(node).expect("frozen same-scope arguments") {
                if argument.node.0 >= id.0 {
                    return Err(CheckError::NotTopological { node: id, dependency: argument.node });
                }
            }
        }
    }
    let root = graph.root_scope();
    for (name, output) in graph.outputs() {
        if root.node(output.value.node).is_none() {
            return Err(CheckError::InvalidOutput { name: name.clone(), node: output.value.node });
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
