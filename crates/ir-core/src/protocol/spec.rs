//! Frozen graph annotations and pure protocol specifications.

use crate::{Graph, ScopedWireRef, WireType, node::NodeKind};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpecificationError {
    #[error("ideal and predicate specifications must be sampler-free")]
    NonPureSpecification,
    #[error("a pure predicate must have exactly one boolean output")]
    PredicateOutput,
}

/// Semantic wire sets retained by the DSL and resolved exactly once when the graph is frozen.
///
/// Labels are proof-facing names, not executable nodes. A label may name more than one wire so
/// callers can identify a typed tuple or family interface without reconstructing it by searching
/// the frozen graph.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FrozenSemanticAnchors {
    entries: BTreeMap<String, Vec<ScopedWireRef>>,
}

/// A frozen, owner-crate rule reference retained alongside an executable graph.
///
/// This is generator infrastructure: it identifies the exact wires to which an owning crate's
/// checked operational rule applies.  It contains neither a claimed equation nor a numeric bound.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Deserialize, serde::Serialize)]
#[doc(hidden)]
pub struct FrozenDerivationAttachment {
    pub namespace: String,
    pub rule: String,
    pub roles: Vec<(String, ScopedWireRef)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[doc(hidden)]
pub struct FrozenDerivationAttachments {
    entries: Vec<FrozenDerivationAttachment>,
}

impl FrozenDerivationAttachments {
    pub fn new(entries: Vec<FrozenDerivationAttachment>) -> Self {
        Self { entries }
    }

    #[doc(hidden)]
    pub fn iter(&self) -> impl Iterator<Item = &FrozenDerivationAttachment> {
        self.entries.iter()
    }

    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl FrozenSemanticAnchors {
    pub fn new(entries: BTreeMap<String, Vec<ScopedWireRef>>) -> Self {
        Self { entries }
    }

    pub fn get(&self, name: &str) -> Option<&[ScopedWireRef]> {
        self.entries.get(name).map(Vec::as_slice)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[ScopedWireRef])> {
        self.entries.iter().map(|(name, wires)| (name.as_str(), wires.as_slice()))
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
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
