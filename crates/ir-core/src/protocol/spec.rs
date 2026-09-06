//! Pure protocol specifications.

use crate::{Graph, WireType, node::NodeKind};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpecificationError {
    #[error("ideal and predicate specifications must be sampler-free")]
    NonPureSpecification,
    #[error("a pure predicate must have exactly one boolean output")]
    PredicateOutput,
}

#[derive(Clone)]
pub struct IdealSpec {
    pub graph: Graph,
}

#[derive(Clone)]
pub struct PurePredicateSpec {
    pub graph: Graph,
}

fn require_sampler_free(graph: &Graph) -> Result<(), SpecificationError> {
    let contains_sampler = graph.scopes().values().any(|scope| {
        scope.nodes().iter().any(|node| {
            matches!(
                node.kind(),
                NodeKind::UniformResidueSample { .. } |
                    NodeKind::UniformIntervalSample { .. } |
                    NodeKind::GaussianSample { .. } |
                    NodeKind::HashSample { .. } |
                    NodeKind::TrapdoorSample { .. } |
                    NodeKind::PreimageSample { .. }
            )
        })
    });
    if contains_sampler {
        return Err(SpecificationError::NonPureSpecification);
    }
    Ok(())
}

impl IdealSpec {
    pub fn new(graph: Graph) -> Result<Self, SpecificationError> {
        require_sampler_free(&graph)?;
        Ok(Self { graph })
    }
}

impl PurePredicateSpec {
    pub fn new(graph: Graph) -> Result<Self, SpecificationError> {
        require_sampler_free(&graph)?;
        if graph.outputs().len() != 1 {
            return Err(SpecificationError::PredicateOutput);
        }
        let output = graph.outputs().values().next().expect("one predicate output").value;
        let output_type = graph
            .root_scope()
            .node(output.node)
            .and_then(|node| node.output_types().get(output.port.0 as usize));
        if output_type != Some(&WireType::Bool) && output_type != Some(&WireType::ConstantBool) {
            return Err(SpecificationError::PredicateOutput);
        }
        Ok(Self { graph })
    }
}
