//! Validation and occurrence-aware linking for multi-stage IR programs.
//!
//! This module is deliberately independent of any renderer.  It turns the
//! runtime artifact metadata on already validated graphs into typed links that
//! a later emitter can consume without looking up raw node numbers.

use crate::{
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, validate_manifest},
    encoding::{IR_VERSION, hash_canonical, spec_hash},
    expr::{IndexExpr, IntExpr, ParamEnv, Rational, RealExpr},
    graph::{FrozenGraphScopeId, FrozenValueRef, ScopedWireRef},
    node::{ArtifactInput, NodeKind},
    types::{ConcreteWireType, NodeId, Port, WireRef},
    validate::ValidatedGraph,
};
use num_traits::ToPrimitive;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// One exact crossing from a parent scope into the child input owned by a
/// structural node.  The owner and input index are retained together so a
/// certificate cannot substitute a sibling boundary or infer a path from
/// traversal order.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct ChildInputHop {
    pub parent_scope: FrozenGraphScopeId,
    pub owner: NodeId,
    pub input_index: usize,
}

/// One exact exit from a parallel-grid body to its parent grid output.
///
/// `output_index` is the body-output port and the corresponding parent-grid
/// output port.  The two ports are checked against the frozen graph and their
/// types must satisfy `Family(grid.shape, child_output_type)`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct ParallelOutputHop {
    pub parent_scope: FrozenGraphScopeId,
    pub owner: NodeId,
    pub output_index: usize,
}

/// A typed structural route between two wires. Exits are applied from the
/// source towards the least common ancestor; enters are applied from that
/// ancestor towards the target. Sequential-loop scopes are transparent;
/// explicit hops are required only for parallel-grid boundaries.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StructuralValueRoute {
    pub exits: Vec<ParallelOutputHop>,
    pub enters: Vec<ChildInputHop>,
}

/// Failure while following or deriving a structural child-input path.
///
/// Paths are deliberately limited to `ParallelGrid` boundaries.  A caller
/// must provide the exact owner and argument index for every hop; no graph
/// traversal order or node-name heuristic is accepted as a substitute.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ChildInputPathError {
    #[error("the source and target references belong to different frozen graphs")]
    DifferentFrozenGraph,
    #[error("path hop {hop} starts in scope {actual:?}, expected {expected:?}")]
    ParentScopeMismatch { hop: usize, actual: FrozenGraphScopeId, expected: FrozenGraphScopeId },
    #[error("scope {scope:?} is missing")]
    MissingScope { scope: FrozenGraphScopeId },
    #[error("owner {owner:?} is missing from scope {scope:?}")]
    MissingOwner { scope: FrozenGraphScopeId, owner: NodeId },
    #[error("owner {owner:?} in scope {scope:?} has no argument {input_index}")]
    MissingOwnerArgument { scope: FrozenGraphScopeId, owner: NodeId, input_index: usize },
    #[error("owner {owner:?} argument {input_index} does not equal the current wire")]
    OwnerArgumentMismatch { owner: NodeId, input_index: usize },
    #[error("owner {owner:?} is not a parallel-grid boundary")]
    UnsupportedBoundary { owner: NodeId },
    #[error("named subgraph boundary {scope:?} is unsupported")]
    NamedSubgraphBoundary { scope: FrozenGraphScopeId },
    #[error("owner {owner:?} has no exact parallel-grid child scope")]
    MissingChildScope { owner: NodeId },
    #[error("parallel-grid owner {owner:?} has no input mode {input_index}")]
    MissingInputMode { owner: NodeId, input_index: usize },
    #[error("parallel-grid child scope {child:?} has no input {input_index}")]
    MissingChildInput { child: FrozenGraphScopeId, input_index: usize },
    #[error("parallel-grid child input {input_index} is owned by the wrong scope")]
    ChildInputScopeMismatch { input_index: usize },
    #[error("wire scope is not the exact parallel-grid body for owner {owner:?}")]
    OutputScopeMismatch { owner: NodeId },
    #[error("parallel-grid body has no output {output_index}")]
    MissingChildOutput { output_index: usize },
    #[error("parallel-grid body output wire is exposed more than once at port {output_index}")]
    AmbiguousChildOutput { output_index: usize },
    #[error("parallel-grid parent has no output {output_index}")]
    MissingParentOutput { output_index: usize },
    #[error("parallel-grid output {output_index} has the wrong family type")]
    WrongParentOutputType { output_index: usize },
    #[error("parallel-grid input {input_index} is not an identity reindex")]
    UnsupportedRouteInputMode { input_index: usize },
    #[error("target scope {target_scope:?} is not a parallel-grid descendant of {source_scope:?}")]
    NotDescendant { source_scope: FrozenGraphScopeId, target_scope: FrozenGraphScopeId },
    #[error("the requested child-input path is ambiguous ({count} candidates)")]
    Ambiguous { count: usize },
    #[error("the requested child-input path does not exist")]
    NoPath,
    #[error("concrete scope index {scope} is missing")]
    MissingConcreteScope { scope: usize },
    #[error("concrete owner {owner:?} is missing from scope {scope}")]
    MissingConcreteOwner { scope: usize, owner: NodeId },
}

/// One stage supplied to [`ValidatedLinkedProgram::new`].
///
/// The stage key is intentionally not supplied by the caller: it is always
/// derived from `graph.source.name()`.  The production id is repeated beside
/// the manifest so a caller cannot accidentally associate a manifest with a
/// different runtime identity.
#[derive(Clone, Debug)]
pub struct LinkedProgramStage {
    pub production_id: ProductionId,
    pub graph: ValidatedGraph,
    pub manifest: Manifest,
}

impl LinkedProgramStage {
    pub fn new(production_id: ProductionId, graph: ValidatedGraph, manifest: Manifest) -> Self {
        Self { production_id, graph, manifest }
    }

    pub fn key(&self) -> &str {
        self.graph.source.name()
    }
}

/// A concrete wire reference whose type has been checked against its stage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TypedScopedWireRef {
    pub reference: ScopedWireRef,
    pub wire_type: ConcreteWireType,
}

/// A typed semantic role resolved into the nonce-erased linked program.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteSemanticWireRef {
    pub stage: usize,
    pub wire: ConcreteWireRef,
    pub wire_type: ConcreteWireType,
}

/// The exact named root output that produces an artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedProducerOutput {
    pub stage_key: String,
    pub production_id: ProductionId,
    pub name: String,
    pub root: ScopedWireRef,
    pub wire_type: ConcreteWireType,
}

/// One artifact input resolved to an earlier stage's named output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkedArtifactLink {
    pub consumer_stage: usize,
    pub consumer_stage_key: String,
    pub consumer_input: ArtifactInput,
    pub consumer: TypedScopedWireRef,
    pub producer_stage: usize,
    pub producer: LinkedProducerOutput,
}

/// A deterministic, typed, nonce-erased semantic AST for Lean emission.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteLinkedProgram {
    pub stages: Vec<ConcreteLinkedStage>,
    pub artifact_links: Vec<ConcreteArtifactLink>,
}

/// A closed integer expression whose only non-literal inputs are structural
/// binders introduced by an enclosing loop or grid. Parameter variables have
/// already been substituted before this representation is constructed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteStructuralIntExpr {
    Literal(#[serde(with = "crate::serde_support::bigint")] num_bigint::BigInt),
    StructuralSlot(u32),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    ExactDivide(Box<Self>, Box<Self>),
    RoundDivide(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum StructuralSlotKind {
    SequentialIteration,
    GridAxis { axis: usize },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StructuralSlotDecl {
    pub slot: u32,
    pub kind: StructuralSlotKind,
    #[serde(with = "crate::serde_support::bigint")]
    pub upper_bound: num_bigint::BigInt,
}

/// Closed coordinate expression used by family maps. Unlike the arithmetic
/// expression above, this retains coordinate/index semantics (including
/// Euclidean division, comparisons, and branch selection).
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteIndexMapExpr {
    Literal(#[serde(with = "crate::serde_support::bigint")] num_bigint::BigInt),
    Axis(usize),
    StructuralSlot(u32),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    EuclideanDivide(Box<Self>, Box<Self>),
    EuclideanRemainder(Box<Self>, Box<Self>),
    Equal(Box<Self>, Box<Self>),
    Less(Box<Self>, Box<Self>),
    LessEqual(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    Select { selector: Box<Self>, branches: Vec<Self> },
}

impl ConcreteStructuralIntExpr {
    /// Close a parameterized expression while retaining only an explicitly
    /// declared structural slot. Free variables and undeclared loop slots are
    /// rejected rather than silently evaluated.
    pub fn close(expr: &IntExpr, env: &ParamEnv, slots: &BTreeSet<u32>) -> Result<Self, String> {
        Ok(match expr {
            IntExpr::Const(v) => Self::Literal(v.clone()),
            IntExpr::Var(name) => Self::Literal(
                env.integers
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unbound parameter {name}"))?,
            ),
            IntExpr::LoopIndex(slot) if slots.contains(slot) => Self::StructuralSlot(*slot),
            IntExpr::LoopIndex(slot) => return Err(format!("out-of-scope structural slot {slot}")),
            IntExpr::Add(a, b) => Self::Add(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IntExpr::Sub(a, b) => Self::Sub(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IntExpr::Mul(a, b) => Self::Mul(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IntExpr::Div(a, b) => Self::ExactDivide(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IntExpr::RoundDiv(a, b) => Self::RoundDivide(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IntExpr::Log2Ceil(a) => Self::Log2Ceil(Box::new(Self::close(a, env, slots)?)),
        })
    }
}

impl ConcreteIndexMapExpr {
    pub fn close(expr: &IndexExpr, env: &ParamEnv, slots: &BTreeSet<u32>) -> Result<Self, String> {
        Ok(match expr {
            IndexExpr::Axis(a) => Self::Axis(*a),
            IndexExpr::Parameter(name) => Self::Literal(
                env.integers
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unbound parameter {name}"))?,
            ),
            IndexExpr::LoopIndex(slot) if slots.contains(slot) => Self::StructuralSlot(*slot),
            IndexExpr::LoopIndex(slot) => {
                return Err(format!("out-of-scope structural slot {slot}"))
            }
            IndexExpr::Constant(v) => Self::Literal(v.clone()),
            IndexExpr::Add(a, b) => Self::Add(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Subtract(a, b) => Self::Sub(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Multiply(a, b) => Self::Mul(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Divide(a, b) => Self::EuclideanDivide(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Remainder(a, b) => Self::EuclideanRemainder(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Equal(a, b) => Self::Equal(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Less(a, b) => Self::Less(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::LessEqual(a, b) => Self::LessEqual(
                Box::new(Self::close(a, env, slots)?),
                Box::new(Self::close(b, env, slots)?),
            ),
            IndexExpr::Log2Ceil(a) => Self::Log2Ceil(Box::new(Self::close(a, env, slots)?)),
            IndexExpr::Select { selector, branches } => Self::Select {
                selector: Box::new(Self::close(selector, env, slots)?),
                branches: branches
                    .iter()
                    .map(|x| Self::close(x, env, slots))
                    .collect::<Result<_, _>>()?,
            },
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteLinkedStage {
    pub key: String,
    pub bindings: crate::expr::ParamEnv,
    pub scope_ids: Vec<FrozenGraphScopeId>,
    pub scopes: Vec<ConcreteScope>,
    pub root_scope: usize,
    pub named_outputs: Vec<ConcreteNamedOutput>,
}

/// Follow one exact `ParallelGrid` input boundary in a frozen graph.
///
/// The returned wire is the corresponding input wire in the child scope.  The
/// operation checks the parent argument, child scope, child input, and grid
/// input-mode entry independently, so a caller cannot accidentally cross a
/// sibling boundary or use a sequential/named subgraph boundary.
pub fn follow_child_input_hop(
    graph: &crate::graph::Graph,
    current: &ScopedWireRef,
    hop: &ChildInputHop,
) -> Result<ScopedWireRef, ChildInputPathError> {
    if current.scope != hop.parent_scope {
        return Err(ChildInputPathError::ParentScopeMismatch {
            hop: 0,
            actual: current.scope.clone(),
            expected: hop.parent_scope.clone(),
        });
    }
    let parent = graph
        .scope(&hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    let owner = parent.node(hop.owner).ok_or_else(|| ChildInputPathError::MissingOwner {
        scope: hop.parent_scope.clone(),
        owner: hop.owner,
    })?;
    let arguments =
        parent.arguments(owner).ok_or_else(|| ChildInputPathError::MissingOwnerArgument {
            scope: hop.parent_scope.clone(),
            owner: hop.owner,
            input_index: hop.input_index,
        })?;
    let argument = arguments.get(hop.input_index).ok_or_else(|| {
        ChildInputPathError::MissingOwnerArgument {
            scope: hop.parent_scope.clone(),
            owner: hop.owner,
            input_index: hop.input_index,
        }
    })?;
    if argument != &current.wire {
        return Err(ChildInputPathError::OwnerArgumentMismatch {
            owner: hop.owner,
            input_index: hop.input_index,
        });
    }
    let crate::node::NodeKind::ParallelGrid(grid) = owner.kind() else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let expected_child = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(hop.parent_scope.clone()),
        owner: hop.owner,
    };
    let child = graph
        .child_scope_id(&hop.parent_scope, hop.owner)
        .ok_or(ChildInputPathError::MissingChildScope { owner: hop.owner })?;
    if child != expected_child {
        return Err(ChildInputPathError::MissingChildScope { owner: hop.owner });
    }
    if grid.input_modes.get(hop.input_index).is_none() {
        return Err(ChildInputPathError::MissingInputMode {
            owner: hop.owner,
            input_index: hop.input_index,
        });
    }
    let child_scope = graph
        .scope(&child)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: child.clone() })?;
    let wire = *child_scope.inputs().get(hop.input_index).ok_or_else(|| {
        ChildInputPathError::MissingChildInput {
            child: child.clone(),
            input_index: hop.input_index,
        }
    })?;
    Ok(ScopedWireRef { scope: child, wire })
}

/// Follow one exact `ParallelGrid` output boundary from its body to the
/// parent.  This is the inverse structural operation of
/// [`follow_child_input_hop`], with an explicit body-output port rather than
/// a search over all possible parent values.
pub fn follow_parallel_output_hop(
    graph: &crate::graph::Graph,
    current: &ScopedWireRef,
    hop: &ParallelOutputHop,
) -> Result<ScopedWireRef, ChildInputPathError> {
    let expected_child = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(hop.parent_scope.clone()),
        owner: hop.owner,
    };
    if current.scope != expected_child {
        return Err(ChildInputPathError::OutputScopeMismatch { owner: hop.owner });
    }
    let child = graph
        .scope(&expected_child)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: expected_child.clone() })?;
    let child_wire = *child
        .outputs()
        .get(hop.output_index)
        .ok_or(ChildInputPathError::MissingChildOutput { output_index: hop.output_index })?;
    if child_wire != current.wire {
        return Err(ChildInputPathError::OwnerArgumentMismatch {
            owner: hop.owner,
            input_index: hop.output_index,
        });
    }
    if child
        .outputs()
        .iter()
        .enumerate()
        .any(|(index, wire)| index != hop.output_index && *wire == current.wire)
    {
        return Err(ChildInputPathError::AmbiguousChildOutput { output_index: hop.output_index });
    }
    let child_type = graph
        .scope(&expected_child)
        .and_then(|scope| scope.node(child_wire.node))
        .and_then(|node| node.output_types().get(child_wire.port.0 as usize))
        .ok_or(ChildInputPathError::MissingChildOutput { output_index: hop.output_index })?;
    let parent = graph
        .scope(&hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    let owner = parent.node(hop.owner).ok_or_else(|| ChildInputPathError::MissingOwner {
        scope: hop.parent_scope.clone(),
        owner: hop.owner,
    })?;
    let crate::node::NodeKind::ParallelGrid(grid) = owner.kind() else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let parent_type = owner
        .output_types()
        .get(hop.output_index)
        .ok_or(ChildInputPathError::MissingParentOutput { output_index: hop.output_index })?;
    let expected_type = crate::types::WireType::Family {
        element: Box::new(child_type.clone()),
        shape: grid.shape.clone(),
    };
    if parent_type != &expected_type {
        return Err(ChildInputPathError::WrongParentOutputType { output_index: hop.output_index });
    }
    Ok(ScopedWireRef {
        scope: hop.parent_scope.clone(),
        wire: WireRef { node: hop.owner, port: Port(hop.output_index as u32) },
    })
}

/// Apply a structural route and return the transported wire.  Every boundary
/// is checked against the graph; no semantic equality between child wires and
/// family values is inferred here.
pub fn follow_structural_value_route(
    graph: &crate::graph::Graph,
    start: &ScopedWireRef,
    route: &StructuralValueRoute,
) -> Result<ScopedWireRef, ChildInputPathError> {
    let mut current = start.clone();
    for hop in &route.exits {
        current = follow_parallel_output_hop(graph, &current, hop)?;
    }
    for hop in &route.enters {
        current = follow_route_child_input_hop(graph, &current, hop)?;
    }
    Ok(current)
}

fn follow_route_child_input_hop(
    graph: &crate::graph::Graph,
    current: &ScopedWireRef,
    hop: &ChildInputHop,
) -> Result<ScopedWireRef, ChildInputPathError> {
    let parent = graph
        .scope(&hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    let owner = parent.node(hop.owner).ok_or_else(|| ChildInputPathError::MissingOwner {
        scope: hop.parent_scope.clone(),
        owner: hop.owner,
    })?;
    let crate::node::NodeKind::ParallelGrid(grid) = owner.kind() else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let crate::node::GridInputMode::Reindex { map } =
        grid.input_modes.get(hop.input_index).ok_or(ChildInputPathError::MissingInputMode {
            owner: hop.owner,
            input_index: hop.input_index,
        })?
    else {
        return Err(ChildInputPathError::UnsupportedRouteInputMode { input_index: hop.input_index });
    };
    let identity = map.input_indices.len() == grid.shape.len() &&
        map.input_indices
            .iter()
            .enumerate()
            .all(|(axis, expression)| *expression == crate::IndexExpr::Axis(axis));
    if !identity {
        return Err(ChildInputPathError::UnsupportedRouteInputMode { input_index: hop.input_index });
    }
    follow_child_input_hop(graph, current, hop)
}

/// Check that a structural route transports `start` exactly to `target`.
pub fn follows_structural_value_route(
    graph: &crate::graph::Graph,
    start: &ScopedWireRef,
    route: &StructuralValueRoute,
    target: &ScopedWireRef,
) -> Result<bool, ChildInputPathError> {
    Ok(follow_structural_value_route(graph, start, route)? == *target)
}

/// Return whether `hops` transports `start` exactly to `target`.
///
/// An empty path is valid precisely when the two references are equal.  Any
/// malformed hop is an error rather than a false positive.
pub fn follows_child_input_path(
    graph: &crate::graph::Graph,
    start: &ScopedWireRef,
    hops: &[ChildInputHop],
    target: &ScopedWireRef,
) -> Result<bool, ChildInputPathError> {
    let mut current = start.clone();
    for (index, hop) in hops.iter().enumerate() {
        current = follow_child_input_hop(graph, &current, hop).map_err(|error| match error {
            ChildInputPathError::ParentScopeMismatch { actual, expected, .. } => {
                ChildInputPathError::ParentScopeMismatch { hop: index, actual, expected }
            }
            other => other,
        })?;
    }
    Ok(current == *target)
}

/// Derive the unique safe path between two frozen values through parallel-grid
/// input boundaries.  The endpoint must be the exact child input reached by
/// the path; arbitrary descendants and unsupported structural boundaries are
/// rejected.
pub fn derive_child_input_path(
    graph: &crate::graph::Graph,
    start: &FrozenValueRef,
    target: &FrozenValueRef,
) -> Result<Vec<ChildInputHop>, ChildInputPathError> {
    if start.freeze_id() != target.freeze_id() {
        return Err(ChildInputPathError::DifferentFrozenGraph);
    }
    let start_ref = start.reference();
    let target_ref = target.reference();
    if start_ref == target_ref {
        return Ok(Vec::new());
    }
    let boundaries = parallel_boundaries(graph, &start_ref.scope, &target_ref.scope)?;
    if boundaries.is_empty() {
        return Err(ChildInputPathError::NoPath);
    }
    let mut candidates = Vec::new();
    derive_graph_candidates(
        graph,
        start_ref,
        target_ref,
        &boundaries,
        0,
        &mut Vec::new(),
        &mut candidates,
        false,
    )?;
    match candidates.len() {
        0 => Err(ChildInputPathError::NoPath),
        1 => Ok(candidates.remove(0)),
        count => Err(ChildInputPathError::Ambiguous { count }),
    }
}

/// Derive the canonical least-common-ancestor route between two frozen wires.
/// The source must cross body outputs on the way up, and the target must cross
/// declared grid inputs on the way down.  Named subgraphs, sequential bodies,
/// missing boundaries, and ambiguous crossings are rejected. Sequential-loop
/// scopes are transparent when both endpoints remain in the same iteration.
pub fn derive_structural_value_route(
    graph: &crate::graph::Graph,
    start: &FrozenValueRef,
    target: &FrozenValueRef,
) -> Result<StructuralValueRoute, ChildInputPathError> {
    if start.freeze_id() != target.freeze_id() {
        return Err(ChildInputPathError::DifferentFrozenGraph);
    }
    let start_ref = start.reference();
    let target_ref = target.reference();
    if start_ref == target_ref {
        return Ok(StructuralValueRoute { exits: Vec::new(), enters: Vec::new() });
    }
    let start_chain = parallel_scope_chain(graph, &start_ref.scope)?;
    let target_chain = parallel_scope_chain(graph, &target_ref.scope)?;
    let lca = target_chain
        .iter()
        .find(|scope| start_chain.iter().any(|candidate| candidate == *scope))
        .cloned()
        .ok_or(ChildInputPathError::NotDescendant {
            source_scope: start_ref.scope.clone(),
            target_scope: target_ref.scope.clone(),
        })?;

    let mut exits = Vec::new();
    let mut current = start_ref.clone();
    while current.scope != lca {
        let (parent, owner) = match current.scope.clone() {
            FrozenGraphScopeId::ParallelBody { parent, owner } => (parent, owner),
            FrozenGraphScopeId::Subgraph { canonical_name } => {
                return Err(ChildInputPathError::NamedSubgraphBoundary {
                    scope: FrozenGraphScopeId::Subgraph { canonical_name },
                })
            }
            FrozenGraphScopeId::SequentialBody { owner, .. } => {
                return Err(ChildInputPathError::UnsupportedBoundary { owner })
            }
            FrozenGraphScopeId::Root => {
                return Err(ChildInputPathError::NotDescendant {
                    source_scope: start_ref.scope.clone(),
                    target_scope: target_ref.scope.clone(),
                })
            }
        };
        let child = graph
            .scope(&current.scope)
            .ok_or_else(|| ChildInputPathError::MissingScope { scope: current.scope.clone() })?;
        let matching = child
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, wire)| (*wire == current.wire).then_some(index))
            .collect::<Vec<_>>();
        let output_index = match matching.as_slice() {
            [index] => *index,
            [] => return Err(ChildInputPathError::MissingChildOutput { output_index: 0 }),
            _ => return Err(ChildInputPathError::AmbiguousChildOutput { output_index: 0 }),
        };
        let hop = ParallelOutputHop { parent_scope: (*parent).clone(), owner, output_index };
        current = follow_parallel_output_hop(graph, &current, &hop)?;
        exits.push(hop);
    }

    let mut down_boundaries = Vec::new();
    let mut scope = target_ref.scope.clone();
    while scope != lca {
        let (parent, owner) = match scope.clone() {
            FrozenGraphScopeId::ParallelBody { parent, owner } => (parent, owner),
            FrozenGraphScopeId::Subgraph { canonical_name } => {
                return Err(ChildInputPathError::NamedSubgraphBoundary {
                    scope: FrozenGraphScopeId::Subgraph { canonical_name },
                })
            }
            FrozenGraphScopeId::SequentialBody { owner, .. } => {
                return Err(ChildInputPathError::UnsupportedBoundary { owner })
            }
            FrozenGraphScopeId::Root => {
                return Err(ChildInputPathError::NotDescendant {
                    source_scope: start_ref.scope.clone(),
                    target_scope: target_ref.scope.clone(),
                })
            }
        };
        down_boundaries.push(((*parent).clone(), owner));
        scope = *parent;
    }
    down_boundaries.reverse();
    let mut candidates = Vec::new();
    derive_graph_candidates(
        graph,
        &current,
        target_ref,
        &down_boundaries,
        0,
        &mut Vec::new(),
        &mut candidates,
        true,
    )?;
    let enters = match candidates.len() {
        0 => return Err(ChildInputPathError::NoPath),
        1 => candidates.remove(0),
        count => return Err(ChildInputPathError::Ambiguous { count }),
    };
    Ok(StructuralValueRoute { exits, enters })
}

fn parallel_scope_chain(
    graph: &crate::graph::Graph,
    start: &FrozenGraphScopeId,
) -> Result<Vec<FrozenGraphScopeId>, ChildInputPathError> {
    let mut chain = Vec::new();
    let mut current = start.clone();
    loop {
        graph
            .scope(&current)
            .ok_or_else(|| ChildInputPathError::MissingScope { scope: current.clone() })?;
        chain.push(current.clone());
        current = match current {
            FrozenGraphScopeId::Root => break,
            FrozenGraphScopeId::ParallelBody { parent, owner } => {
                if graph.child_scope_id(&parent, owner) !=
                    Some(FrozenGraphScopeId::ParallelBody { parent: parent.clone(), owner })
                {
                    return Err(ChildInputPathError::MissingChildScope { owner });
                }
                *parent
            }
            FrozenGraphScopeId::Subgraph { canonical_name } => {
                return Err(ChildInputPathError::NamedSubgraphBoundary {
                    scope: FrozenGraphScopeId::Subgraph { canonical_name },
                })
            }
            FrozenGraphScopeId::SequentialBody { parent, .. } => *parent,
        };
    }
    Ok(chain)
}

fn parallel_boundaries(
    graph: &crate::graph::Graph,
    source: &FrozenGraphScopeId,
    target: &FrozenGraphScopeId,
) -> Result<Vec<(FrozenGraphScopeId, NodeId)>, ChildInputPathError> {
    let mut reversed = Vec::new();
    let mut current = target.clone();
    while current != *source {
        let (parent, owner) = match current {
            FrozenGraphScopeId::ParallelBody { parent, owner } => (*parent, owner),
            FrozenGraphScopeId::Subgraph { .. } => {
                return Err(ChildInputPathError::NamedSubgraphBoundary { scope: current })
            }
            FrozenGraphScopeId::SequentialBody { owner, .. } => {
                return Err(ChildInputPathError::UnsupportedBoundary { owner })
            }
            FrozenGraphScopeId::Root => {
                return Err(ChildInputPathError::NotDescendant {
                    source_scope: source.clone(),
                    target_scope: target.clone(),
                })
            }
        };
        if graph.child_scope_id(&parent, owner).is_none() {
            return Err(ChildInputPathError::MissingChildScope { owner });
        }
        current = parent.clone();
        reversed.push((parent, owner));
    }
    reversed.reverse();
    Ok(reversed)
}

fn derive_graph_candidates(
    graph: &crate::graph::Graph,
    current: &ScopedWireRef,
    target: &ScopedWireRef,
    boundaries: &[(FrozenGraphScopeId, NodeId)],
    depth: usize,
    path: &mut Vec<ChildInputHop>,
    candidates: &mut Vec<Vec<ChildInputHop>>,
    strict_route: bool,
) -> Result<(), ChildInputPathError> {
    if candidates.len() > 1 {
        return Ok(());
    }
    if depth == boundaries.len() {
        if current == target {
            candidates.push(path.clone());
        }
        return Ok(());
    }
    let (parent_scope, owner) = &boundaries[depth];
    let parent = graph
        .scope(parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: parent_scope.clone() })?;
    let owner_node = parent.node(*owner).ok_or_else(|| ChildInputPathError::MissingOwner {
        scope: parent_scope.clone(),
        owner: *owner,
    })?;
    let arguments =
        parent.arguments(owner_node).ok_or_else(|| ChildInputPathError::MissingOwnerArgument {
            scope: parent_scope.clone(),
            owner: *owner,
            input_index: 0,
        })?;
    for input_index in 0..arguments.len() {
        if arguments[input_index] != current.wire {
            continue;
        }
        let hop = ChildInputHop { parent_scope: parent_scope.clone(), owner: *owner, input_index };
        let next = if strict_route {
            follow_route_child_input_hop(graph, current, &hop)?
        } else {
            follow_child_input_hop(graph, current, &hop)?
        };
        path.push(hop);
        derive_graph_candidates(
            graph,
            &next,
            target,
            boundaries,
            depth + 1,
            path,
            candidates,
            strict_route,
        )?;
        path.pop();
    }
    Ok(())
}

/// Follow one exact child-input hop in the closed semantic projection.
pub fn follow_concrete_child_input_hop(
    stage: &ConcreteLinkedStage,
    current: &ConcreteWireRef,
    hop: &ChildInputHop,
) -> Result<ConcreteWireRef, ChildInputPathError> {
    let parent_index = stage
        .scope_ids
        .iter()
        .position(|scope| scope == &hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    if current.scope >= stage.scopes.len() || current.scope >= stage.scope_ids.len() {
        return Err(ChildInputPathError::MissingConcreteScope { scope: current.scope });
    }
    if current.scope != parent_index {
        return Err(ChildInputPathError::ParentScopeMismatch {
            hop: 0,
            actual: stage.scope_ids[current.scope].clone(),
            expected: hop.parent_scope.clone(),
        });
    }
    let parent = stage
        .scopes
        .get(parent_index)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: parent_index })?;
    let owner = parent.nodes.get(hop.owner.0 as usize).ok_or_else(|| {
        ChildInputPathError::MissingConcreteOwner { scope: parent_index, owner: hop.owner }
    })?;
    let argument = owner.arguments.get(hop.input_index).ok_or_else(|| {
        ChildInputPathError::MissingOwnerArgument {
            scope: hop.parent_scope.clone(),
            owner: hop.owner,
            input_index: hop.input_index,
        }
    })?;
    if argument != current {
        return Err(ChildInputPathError::OwnerArgumentMismatch {
            owner: hop.owner,
            input_index: hop.input_index,
        });
    }
    let ConcreteNodePayload::ParallelGrid(grid) = &owner.kind else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let child_index =
        owner.child_scope.ok_or(ChildInputPathError::MissingChildScope { owner: hop.owner })?;
    let expected_child = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(hop.parent_scope.clone()),
        owner: hop.owner,
    };
    if stage.scope_ids.get(child_index) != Some(&expected_child) {
        return Err(ChildInputPathError::MissingChildScope { owner: hop.owner });
    }
    if grid.input_modes.get(hop.input_index).is_none() {
        return Err(ChildInputPathError::MissingInputMode {
            owner: hop.owner,
            input_index: hop.input_index,
        });
    }
    let child = stage
        .scopes
        .get(child_index)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: child_index })?;
    let input = child.inputs.get(hop.input_index).ok_or_else(|| {
        ChildInputPathError::MissingChildInput {
            child: expected_child,
            input_index: hop.input_index,
        }
    })?;
    if input.scope != child_index {
        return Err(ChildInputPathError::ChildInputScopeMismatch { input_index: hop.input_index });
    }
    Ok(input.clone())
}

/// Follow one exact parallel-grid output boundary in the closed linked
/// representation.  The concrete check mirrors
/// [`follow_parallel_output_hop`] and repeats the family element/shape check
/// after parameter closure.
pub fn follow_concrete_parallel_output_hop(
    stage: &ConcreteLinkedStage,
    current: &ConcreteWireRef,
    hop: &ParallelOutputHop,
) -> Result<ConcreteWireRef, ChildInputPathError> {
    let expected_child = FrozenGraphScopeId::ParallelBody {
        parent: Box::new(hop.parent_scope.clone()),
        owner: hop.owner,
    };
    let child_index = stage
        .scope_ids
        .iter()
        .position(|scope| scope == &expected_child)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: expected_child.clone() })?;
    if current.scope != child_index {
        return Err(ChildInputPathError::OutputScopeMismatch { owner: hop.owner });
    }
    let child = stage
        .scopes
        .get(child_index)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: child_index })?;
    let child_wire = child
        .outputs
        .get(hop.output_index)
        .ok_or(ChildInputPathError::MissingChildOutput { output_index: hop.output_index })?
        .clone();
    if child_wire != *current {
        return Err(ChildInputPathError::OwnerArgumentMismatch {
            owner: hop.owner,
            input_index: hop.output_index,
        });
    }
    if child
        .outputs
        .iter()
        .enumerate()
        .any(|(index, wire)| index != hop.output_index && *wire == *current)
    {
        return Err(ChildInputPathError::AmbiguousChildOutput { output_index: hop.output_index });
    }
    let child_type = child
        .nodes
        .get(child_wire.node.0 as usize)
        .and_then(|node| node.outputs.get(child_wire.port.0 as usize))
        .ok_or(ChildInputPathError::MissingChildOutput { output_index: hop.output_index })?;
    let parent_index = stage
        .scope_ids
        .iter()
        .position(|scope| scope == &hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    let parent = stage
        .scopes
        .get(parent_index)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: parent_index })?;
    let owner = parent.nodes.get(hop.owner.0 as usize).ok_or_else(|| {
        ChildInputPathError::MissingConcreteOwner { scope: parent_index, owner: hop.owner }
    })?;
    let ConcreteNodePayload::ParallelGrid(grid) = &owner.kind else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let parent_type = owner
        .outputs
        .get(hop.output_index)
        .ok_or(ChildInputPathError::MissingParentOutput { output_index: hop.output_index })?;
    let expected_type = ConcreteWireType::Family {
        element: Box::new(child_type.clone()),
        shape: grid
            .shape
            .iter()
            .map(|expression| {
                concrete_shape_extent(expression).ok_or(
                    ChildInputPathError::WrongParentOutputType { output_index: hop.output_index },
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    if parent_type != &expected_type {
        return Err(ChildInputPathError::WrongParentOutputType { output_index: hop.output_index });
    }
    Ok(ConcreteWireRef {
        scope: parent_index,
        node: hop.owner,
        port: Port(hop.output_index as u32),
    })
}

fn concrete_shape_extent(expression: &ConcreteStructuralIntExpr) -> Option<usize> {
    match expression {
        ConcreteStructuralIntExpr::Literal(value) => value.to_usize(),
        _ => None,
    }
}

/// Apply a structural route in the closed linked program.
pub fn follow_concrete_structural_value_route(
    stage: &ConcreteLinkedStage,
    start: &ConcreteWireRef,
    route: &StructuralValueRoute,
) -> Result<ConcreteWireRef, ChildInputPathError> {
    let mut current = start.clone();
    for hop in &route.exits {
        current = follow_concrete_parallel_output_hop(stage, &current, hop)?;
    }
    for hop in &route.enters {
        current = follow_concrete_route_child_input_hop(stage, &current, hop)?;
    }
    Ok(current)
}

fn follow_concrete_route_child_input_hop(
    stage: &ConcreteLinkedStage,
    current: &ConcreteWireRef,
    hop: &ChildInputHop,
) -> Result<ConcreteWireRef, ChildInputPathError> {
    let parent_index = stage
        .scope_ids
        .iter()
        .position(|scope| scope == &hop.parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: hop.parent_scope.clone() })?;
    let parent = stage
        .scopes
        .get(parent_index)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: parent_index })?;
    let owner = parent.nodes.get(hop.owner.0 as usize).ok_or_else(|| {
        ChildInputPathError::MissingConcreteOwner { scope: parent_index, owner: hop.owner }
    })?;
    let ConcreteNodePayload::ParallelGrid(grid) = &owner.kind else {
        return Err(ChildInputPathError::UnsupportedBoundary { owner: hop.owner });
    };
    let ConcreteGridInputMode::Reindex { map } =
        grid.input_modes.get(hop.input_index).ok_or(ChildInputPathError::MissingInputMode {
            owner: hop.owner,
            input_index: hop.input_index,
        })?
    else {
        return Err(ChildInputPathError::UnsupportedRouteInputMode { input_index: hop.input_index });
    };
    let identity = map.source_rank == grid.shape.len() &&
        map.output_rank == grid.shape.len() &&
        map.input_indices.len() == grid.shape.len() &&
        map.input_indices
            .iter()
            .enumerate()
            .all(|(axis, expression)| *expression == ConcreteIndexMapExpr::Axis(axis));
    if !identity {
        return Err(ChildInputPathError::UnsupportedRouteInputMode { input_index: hop.input_index });
    }
    follow_concrete_child_input_hop(stage, current, hop)
}

pub fn follows_concrete_structural_value_route(
    stage: &ConcreteLinkedStage,
    start: &ConcreteWireRef,
    route: &StructuralValueRoute,
    target: &ConcreteWireRef,
) -> Result<bool, ChildInputPathError> {
    Ok(follow_concrete_structural_value_route(stage, start, route)? == *target)
}

/// Check an exact path in a closed semantic projection.
pub fn follows_concrete_child_input_path(
    stage: &ConcreteLinkedStage,
    start: &ConcreteWireRef,
    hops: &[ChildInputHop],
    target: &ConcreteWireRef,
) -> Result<bool, ChildInputPathError> {
    let mut current = start.clone();
    for (index, hop) in hops.iter().enumerate() {
        current =
            follow_concrete_child_input_hop(stage, &current, hop).map_err(|error| match error {
                ChildInputPathError::ParentScopeMismatch { actual, expected, .. } => {
                    ChildInputPathError::ParentScopeMismatch { hop: index, actual, expected }
                }
                other => other,
            })?;
    }
    Ok(current == *target)
}

/// Derive the unique path between concrete wires through parallel-grid input
/// boundaries.  Scope ancestry comes from the frozen scope identifiers; the
/// closed node payload is then checked again at every hop.
pub fn derive_concrete_child_input_path(
    stage: &ConcreteLinkedStage,
    start: &ConcreteWireRef,
    target: &ConcreteWireRef,
) -> Result<Vec<ChildInputHop>, ChildInputPathError> {
    let source_scope = stage
        .scope_ids
        .get(start.scope)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: start.scope })?;
    let target_scope = stage
        .scope_ids
        .get(target.scope)
        .ok_or(ChildInputPathError::MissingConcreteScope { scope: target.scope })?;
    if start == target {
        return Ok(Vec::new());
    }
    let boundaries = parallel_boundaries_for_stage(stage, source_scope, target_scope)?;
    let mut candidates = Vec::new();
    derive_concrete_candidates(
        stage,
        start,
        target,
        &boundaries,
        0,
        &mut Vec::new(),
        &mut candidates,
    )?;
    match candidates.len() {
        0 => Err(ChildInputPathError::NoPath),
        1 => Ok(candidates.remove(0)),
        count => Err(ChildInputPathError::Ambiguous { count }),
    }
}

fn parallel_boundaries_for_stage(
    stage: &ConcreteLinkedStage,
    source: &FrozenGraphScopeId,
    target: &FrozenGraphScopeId,
) -> Result<Vec<(FrozenGraphScopeId, NodeId)>, ChildInputPathError> {
    let mut reversed = Vec::new();
    let mut current = target.clone();
    while current != *source {
        let (parent, owner) = match current {
            FrozenGraphScopeId::ParallelBody { parent, owner } => (*parent, owner),
            FrozenGraphScopeId::Subgraph { .. } => {
                return Err(ChildInputPathError::NamedSubgraphBoundary { scope: current })
            }
            FrozenGraphScopeId::SequentialBody { owner, .. } => {
                return Err(ChildInputPathError::UnsupportedBoundary { owner })
            }
            FrozenGraphScopeId::Root => {
                return Err(ChildInputPathError::NotDescendant {
                    source_scope: source.clone(),
                    target_scope: target.clone(),
                })
            }
        };
        if stage.scope_ids.iter().all(|candidate| candidate != &parent) {
            return Err(ChildInputPathError::MissingScope { scope: parent });
        }
        reversed.push((parent.clone(), owner));
        current = parent;
    }
    reversed.reverse();
    Ok(reversed)
}

fn derive_concrete_candidates(
    stage: &ConcreteLinkedStage,
    current: &ConcreteWireRef,
    target: &ConcreteWireRef,
    boundaries: &[(FrozenGraphScopeId, NodeId)],
    depth: usize,
    path: &mut Vec<ChildInputHop>,
    candidates: &mut Vec<Vec<ChildInputHop>>,
) -> Result<(), ChildInputPathError> {
    if candidates.len() > 1 {
        return Ok(());
    }
    if depth == boundaries.len() {
        if current == target {
            candidates.push(path.clone());
        }
        return Ok(());
    }
    let (parent_scope, owner) = &boundaries[depth];
    let parent_index = stage
        .scope_ids
        .iter()
        .position(|scope| scope == parent_scope)
        .ok_or_else(|| ChildInputPathError::MissingScope { scope: parent_scope.clone() })?;
    let parent = stage
        .scopes
        .get(parent_index)
        .ok_or_else(|| ChildInputPathError::MissingConcreteScope { scope: parent_index })?;
    let owner_node = parent.nodes.get(owner.0 as usize).ok_or_else(|| {
        ChildInputPathError::MissingConcreteOwner { scope: parent_index, owner: *owner }
    })?;
    for input_index in 0..owner_node.arguments.len() {
        if owner_node.arguments[input_index] != *current {
            continue;
        }
        let hop = ChildInputHop { parent_scope: parent_scope.clone(), owner: *owner, input_index };
        let next = follow_concrete_child_input_hop(stage, current, &hop)?;
        path.push(hop);
        derive_concrete_candidates(stage, &next, target, boundaries, depth + 1, path, candidates)?;
        path.pop();
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteScope {
    pub id: usize,
    pub structural_slots: Vec<StructuralSlotDecl>,
    pub nodes: Vec<ConcreteNode>,
    pub inputs: Vec<ConcreteWireRef>,
    pub outputs: Vec<ConcreteWireRef>,
}

/// A closed real expression.  Compile-time real variables have been replaced by
/// rational literals; integer subexpressions retain only explicitly scoped
/// structural slots.  In particular, no `ParamEnv` name can reach the emitted
/// semantic program.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteRealExpr {
    Rational(Rational),
    FromInt(ConcreteStructuralIntExpr),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Sqrt(Box<Self>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteSampleRange {
    pub minimum: ConcreteStructuralIntExpr,
    pub maximum: ConcreteStructuralIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteIndexRange {
    pub start: ConcreteStructuralIntExpr,
    pub end: ConcreteStructuralIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteIndexMap {
    pub source_rank: usize,
    pub output_rank: usize,
    pub input_indices: Vec<ConcreteIndexMapExpr>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteMatrixLiteral {
    Zero,
    Identity,
    UnitRow { index: ConcreteStructuralIntExpr },
    UnitColumn { index: ConcreteStructuralIntExpr },
    Gadget { base: ConcreteStructuralIntExpr, small: bool },
    PowerOfBase { base: ConcreteStructuralIntExpr, exponent: ConcreteStructuralIntExpr },
    Rotation { exponent: ConcreteStructuralIntExpr },
    Polynomial { coefficients: Vec<ConcreteStructuralIntExpr> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteSubgraphPayload {
    pub definition: String,
    pub bindings: Vec<(String, ConcreteStructuralIntExpr)>,
    pub canonical_input_exclusive_uppers: Vec<Option<num_bigint::BigUint>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteSequentialLoop {
    pub count: ConcreteStructuralIntExpr,
    pub index_slot: u32,
    pub bindings: Vec<(String, ConcreteStructuralIntExpr)>,
    pub carried_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteGridInputMode {
    Broadcast,
    Reindex { map: ConcreteIndexMap },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteParallelGrid {
    pub shape: Vec<ConcreteStructuralIntExpr>,
    pub index_slots: Vec<u32>,
    pub bindings: Vec<(String, ConcreteStructuralIntExpr)>,
    pub input_modes: Vec<ConcreteGridInputMode>,
}

/// Lossless closed payload for a frozen graph node.  This is intentionally a
/// separate enum from `NodeKind`: a renderer or cache consumer must not be able
/// to observe unresolved parameter expressions, production nonces, or an
/// application-defined opaque callback.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ConcreteNodePayload {
    Input {
        name: String,
        wire_type: ConcreteWireType,
        artifact: Option<ConcreteArtifactInput>,
    },
    ConstantInt(num_bigint::BigInt),
    EvaluateInt(ConcreteStructuralIntExpr),
    ConstantReal(ConcreteRealExpr),
    ConstantBool(bool),
    ConstantMatrix {
        matrix_type: crate::types::ConcreteMatrixType,
        value: ConcreteMatrixLiteral,
    },
    GadgetTrapdoor {
        matrix_type: crate::types::ConcreteMatrixType,
        base: ConcreteStructuralIntExpr,
    },
    TrapdoorPublic,
    IntBinary(crate::node::IntBinaryOp),
    IntCompare(crate::node::IntCompareOp),
    BitExtract {
        bit: ConcreteStructuralIntExpr,
    },
    IntToReal,
    BoolToInt,
    RealBinary(crate::node::RealBinaryOp),
    RealSqrt,
    MatrixBinary(crate::node::MatrixBinaryOp),
    MatrixMulAccumulate {
        coefficients: Vec<ConcreteStructuralIntExpr>,
        has_bias: bool,
    },
    MatrixNegate,
    MatrixScale {
        scalar: ConcreteStructuralIntExpr,
    },
    Transpose,
    Slice {
        rows: Option<ConcreteIndexRange>,
        columns: Option<ConcreteIndexRange>,
    },
    Tensor,
    Concat {
        axis: crate::node::ConcatAxis,
    },
    UniformResidueSample {
        matrix_type: crate::types::ConcreteMatrixType,
    },
    UniformIntervalSample {
        matrix_type: crate::types::ConcreteMatrixType,
        range: ConcreteSampleRange,
    },
    GaussianSample {
        matrix_type: crate::types::ConcreteMatrixType,
        sigma: ConcreteRealExpr,
        max_coefficient_bound: ConcreteStructuralIntExpr,
    },
    HashSample {
        matrix_type: crate::types::ConcreteMatrixType,
        tag_prefix: Vec<u8>,
        tag_expressions: Vec<ConcreteStructuralIntExpr>,
        tag_decimal_expressions: Vec<ConcreteStructuralIntExpr>,
        tag_u64_le_expressions: Vec<ConcreteStructuralIntExpr>,
    },
    TrapdoorSample {
        matrix_type: crate::types::ConcreteMatrixType,
        sigma: ConcreteRealExpr,
        gadget_base: ConcreteStructuralIntExpr,
        digit_count: ConcreteStructuralIntExpr,
        preimage_max_coefficient_bound: ConcreteStructuralIntExpr,
    },
    PreimageSample {
        matrix_type: crate::types::ConcreteMatrixType,
        max_coefficient_bound: ConcreteStructuralIntExpr,
    },
    ApplyPreimage,
    MaterializePreimageExact,
    PreimageBinary(crate::node::PreimageBinaryOp),
    PreimageConcatColumns,
    FamilyPreimageSample {
        matrix_type: crate::types::ConcreteMatrixType,
        max_coefficient_bound: ConcreteStructuralIntExpr,
    },
    GadgetDecompose {
        base: ConcreteStructuralIntExpr,
        small: bool,
        digit_count: ConcreteStructuralIntExpr,
    },
    DecompositionEntry {
        row: ConcreteStructuralIntExpr,
        column: ConcreteStructuralIntExpr,
    },
    ExtractCoefficient {
        position: ConcreteStructuralIntExpr,
        canonical_input_exclusive_upper: Option<num_bigint::BigUint>,
    },
    LiftIntegerToConstantPolynomial {
        matrix_type: crate::types::ConcreteMatrixType,
    },
    ThresholdDecode {
        plaintext_modulus: ConcreteStructuralIntExpr,
        length: ConcreteStructuralIntExpr,
        output_bool: bool,
    },
    CrtRecompose {
        plaintext_moduli: Vec<ConcreteStructuralIntExpr>,
        reconstruction_coefficients: Vec<ConcreteStructuralIntExpr>,
    },
    PackPolynomialCoefficients {
        matrix_type: crate::types::ConcreteMatrixType,
        coefficient_bits: ConcreteStructuralIntExpr,
    },
    SubgraphCall(ConcreteSubgraphPayload),
    SequentialLoop(ConcreteSequentialLoop),
    FamilyPack {
        shape: Vec<ConcreteStructuralIntExpr>,
    },
    FamilyGetStatic {
        indices: Vec<ConcreteIndexMapExpr>,
    },
    FamilyGetDynamic {
        rank: usize,
    },
    FamilySelectAxis {
        axis: usize,
    },
    FamilyReindex {
        output_shape: Vec<ConcreteStructuralIntExpr>,
        map: ConcreteIndexMap,
    },
    FamilyGather {
        output_shape: Vec<ConcreteStructuralIntExpr>,
        input_rank: usize,
    },
    ParallelGrid(ConcreteParallelGrid),
    Select {
        count: ConcreteStructuralIntExpr,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteNode {
    /// Closed typed IR payload; no unresolved `NodeKind` survives projection.
    pub kind: ConcreteNodePayload,
    pub arguments: Vec<ConcreteWireRef>,
    pub outputs: Vec<ConcreteWireType>,
    pub child_scope: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteArtifactInput {
    pub name: String,
    pub confidentiality: crate::artifact::ArtifactConfidentiality,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteWireRef {
    pub scope: usize,
    pub node: crate::types::NodeId,
    pub port: crate::types::Port,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteNamedOutput {
    pub name: String,
    pub wire: ConcreteWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ConcreteArtifactLink {
    pub consumer_stage: usize,
    pub consumer: ConcreteWireRef,
    pub argument: usize,
    pub consumer_artifact: String,
    pub consumer_confidentiality: crate::artifact::ArtifactConfidentiality,
    pub consumer_type: ConcreteWireType,
    pub producer_stage: usize,
    pub producer: ConcreteWireRef,
    pub producer_artifact: String,
    pub producer_confidentiality: crate::artifact::ArtifactConfidentiality,
    pub producer_type: ConcreteWireType,
}

pub type SemanticLinkedProgram = ConcreteLinkedProgram;

fn closed_error(error: impl std::fmt::Display) -> LinkedProgramError {
    LinkedProgramError::SemanticEncoding { message: error.to_string() }
}

fn close_matrix_type(
    matrix: &crate::types::MatrixType,
    env: &ParamEnv,
) -> Result<crate::types::ConcreteMatrixType, LinkedProgramError> {
    let value = |expr: &IntExpr| expr.evaluate(env).map_err(closed_error);
    let result = crate::types::ConcreteMatrixType {
        modulus: value(&matrix.modulus)?,
        ring_dimension: value(&matrix.ring_dimension)?
            .to_usize()
            .ok_or_else(|| closed_error("matrix ring dimension is not a natural usize"))?,
        rows: value(&matrix.rows)?
            .to_usize()
            .ok_or_else(|| closed_error("matrix rows are not a natural usize"))?,
        columns: value(&matrix.columns)?
            .to_usize()
            .ok_or_else(|| closed_error("matrix columns are not a natural usize"))?,
    };
    if result.modulus <= num_bigint::BigInt::from(1) ||
        result.ring_dimension == 0 ||
        result.rows == 0 ||
        result.columns == 0
    {
        return Err(closed_error("invalid concrete matrix type"));
    }
    Ok(result)
}

fn close_wire_type(
    wire_type: &crate::types::WireType,
    env: &ParamEnv,
) -> Result<ConcreteWireType, LinkedProgramError> {
    use crate::types::WireType;
    Ok(match wire_type {
        WireType::ConstantInt => ConcreteWireType::ConstantInt,
        WireType::ConstantReal => ConcreteWireType::ConstantReal,
        WireType::ConstantBool => ConcreteWireType::ConstantBool,
        WireType::Int => ConcreteWireType::Int,
        WireType::Real => ConcreteWireType::Real,
        WireType::Bool => ConcreteWireType::Bool,
        WireType::Bytes { length } => ConcreteWireType::Bytes {
            length: length
                .evaluate(env)
                .map_err(closed_error)?
                .to_usize()
                .ok_or_else(|| closed_error("byte length is not a natural usize"))?,
        },
        WireType::TypedBlob { type_name, schema_hash } => {
            ConcreteWireType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
        }
        WireType::Matrix(matrix) => ConcreteWireType::Matrix(close_matrix_type(matrix, env)?),
        WireType::Preimage(matrix) => ConcreteWireType::Preimage(close_matrix_type(matrix, env)?),
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => ConcreteWireType::Trapdoor {
            matrix: close_matrix_type(matrix, env)?,
            sigma: sigma.close(env).map_err(closed_error)?,
            gadget_base: gadget_base.evaluate(env).map_err(closed_error)?,
            digit_count: digit_count
                .evaluate(env)
                .map_err(closed_error)?
                .to_usize()
                .ok_or_else(|| closed_error("digit count is not a natural usize"))?,
            preimage_max_coefficient_bound: preimage_max_coefficient_bound
                .evaluate(env)
                .map_err(closed_error)?,
        },
        WireType::Family { element, shape } => ConcreteWireType::Family {
            element: Box::new(close_wire_type(element, env)?),
            shape: shape
                .iter()
                .map(|x| {
                    x.evaluate(env)
                        .map_err(closed_error)?
                        .to_usize()
                        .ok_or_else(|| closed_error("family extent is not a natural usize"))
                })
                .collect::<Result<_, _>>()?,
        },
    })
}

fn close_real(
    expression: &RealExpr,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<ConcreteRealExpr, LinkedProgramError> {
    Ok(match expression {
        RealExpr::Rational(value) => ConcreteRealExpr::Rational(value.clone()),
        RealExpr::Var(name) => ConcreteRealExpr::Rational(
            env.reals
                .get(name)
                .cloned()
                .ok_or_else(|| closed_error(format!("unbound real parameter {name}")))?,
        ),
        RealExpr::FromInt(value) => ConcreteRealExpr::FromInt(
            ConcreteStructuralIntExpr::close(value, env, slots).map_err(closed_error)?,
        ),
        RealExpr::Add(lhs, rhs) => ConcreteRealExpr::Add(
            Box::new(close_real(lhs, env, slots)?),
            Box::new(close_real(rhs, env, slots)?),
        ),
        RealExpr::Sub(lhs, rhs) => ConcreteRealExpr::Sub(
            Box::new(close_real(lhs, env, slots)?),
            Box::new(close_real(rhs, env, slots)?),
        ),
        RealExpr::Mul(lhs, rhs) => ConcreteRealExpr::Mul(
            Box::new(close_real(lhs, env, slots)?),
            Box::new(close_real(rhs, env, slots)?),
        ),
        RealExpr::Div(lhs, rhs) => ConcreteRealExpr::Div(
            Box::new(close_real(lhs, env, slots)?),
            Box::new(close_real(rhs, env, slots)?),
        ),
        RealExpr::Sqrt(value) => ConcreteRealExpr::Sqrt(Box::new(close_real(value, env, slots)?)),
    })
}

fn close_index(
    expression: &IndexExpr,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<ConcreteIndexMapExpr, LinkedProgramError> {
    ConcreteIndexMapExpr::close(expression, env, slots).map_err(closed_error)
}

fn close_index_map(
    map: &crate::expr::IndexMap,
    output_rank: usize,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<ConcreteIndexMap, LinkedProgramError> {
    Ok(ConcreteIndexMap {
        source_rank: map.input_indices.len(),
        output_rank,
        input_indices: map
            .input_indices
            .iter()
            .map(|x| close_index(x, env, slots))
            .collect::<Result<_, _>>()?,
    })
}

fn close_range(
    range: &Option<crate::node::IndexRange>,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<Option<ConcreteIndexRange>, LinkedProgramError> {
    range
        .as_ref()
        .map(|range| {
            Ok(ConcreteIndexRange {
                start: ConcreteStructuralIntExpr::close(&range.start, env, slots)
                    .map_err(closed_error)?,
                end: ConcreteStructuralIntExpr::close(&range.end, env, slots)
                    .map_err(closed_error)?,
            })
        })
        .transpose()
}

fn close_constant_matrix(
    value: &crate::node::ConstantMatrix,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<ConcreteMatrixLiteral, LinkedProgramError> {
    use crate::node::ConstantMatrix;
    let close = |x: &IntExpr| ConcreteStructuralIntExpr::close(x, env, slots).map_err(closed_error);
    Ok(match value {
        ConstantMatrix::Zero => ConcreteMatrixLiteral::Zero,
        ConstantMatrix::Identity => ConcreteMatrixLiteral::Identity,
        ConstantMatrix::UnitRow { index } => {
            ConcreteMatrixLiteral::UnitRow { index: close(index)? }
        }
        ConstantMatrix::UnitColumn { index } => {
            ConcreteMatrixLiteral::UnitColumn { index: close(index)? }
        }
        ConstantMatrix::Gadget { base, small } => {
            ConcreteMatrixLiteral::Gadget { base: close(base)?, small: *small }
        }
        ConstantMatrix::PowerOfBase { base, exponent } => {
            ConcreteMatrixLiteral::PowerOfBase { base: close(base)?, exponent: close(exponent)? }
        }
        ConstantMatrix::Rotation { exponent } => {
            ConcreteMatrixLiteral::Rotation { exponent: close(exponent)? }
        }
        ConstantMatrix::Polynomial { coefficients } => ConcreteMatrixLiteral::Polynomial {
            coefficients: coefficients.iter().map(close).collect::<Result<_, _>>()?,
        },
    })
}

fn close_bindings(
    bindings: &[(String, IntExpr)],
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
) -> Result<Vec<(String, ConcreteStructuralIntExpr)>, LinkedProgramError> {
    bindings
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                ConcreteStructuralIntExpr::close(value, env, slots).map_err(closed_error)?,
            ))
        })
        .collect()
}

fn close_payload(
    kind: &NodeKind,
    env: &ParamEnv,
    slots: &BTreeSet<u32>,
    input_type: Option<ConcreteWireType>,
) -> Result<ConcreteNodePayload, LinkedProgramError> {
    use crate::node::NodeKind;
    let close = |x: &IntExpr| ConcreteStructuralIntExpr::close(x, env, slots).map_err(closed_error);
    let matrix = |x: &crate::types::MatrixType| close_matrix_type(x, env);
    Ok(match kind {
        NodeKind::Input { name, artifact, .. } => ConcreteNodePayload::Input {
            name: name.clone(),
            wire_type: input_type
                .ok_or_else(|| closed_error("input node has no concrete output type"))?,
            artifact: artifact.as_ref().map(|input| ConcreteArtifactInput {
                name: input.artifact_name.clone(),
                confidentiality: input.confidentiality,
            }),
        },
        NodeKind::ConstantInt(value) => ConcreteNodePayload::ConstantInt(value.clone()),
        NodeKind::EvaluateInt(value) => ConcreteNodePayload::EvaluateInt(close(value)?),
        NodeKind::ConstantReal(value) => {
            ConcreteNodePayload::ConstantReal(close_real(value, env, slots)?)
        }
        NodeKind::ConstantBool(value) => ConcreteNodePayload::ConstantBool(*value),
        NodeKind::ConstantMatrix { matrix_type, value } => ConcreteNodePayload::ConstantMatrix {
            matrix_type: matrix(matrix_type)?,
            value: close_constant_matrix(value, env, slots)?,
        },
        NodeKind::GadgetTrapdoor { matrix_type, base } => ConcreteNodePayload::GadgetTrapdoor {
            matrix_type: matrix(matrix_type)?,
            base: close(base)?,
        },
        NodeKind::TrapdoorPublic => ConcreteNodePayload::TrapdoorPublic,
        NodeKind::IntBinary(op) => ConcreteNodePayload::IntBinary(*op),
        NodeKind::IntCompare(op) => ConcreteNodePayload::IntCompare(*op),
        NodeKind::BitExtract { bit } => ConcreteNodePayload::BitExtract { bit: close(bit)? },
        NodeKind::IntToReal => ConcreteNodePayload::IntToReal,
        NodeKind::BoolToInt => ConcreteNodePayload::BoolToInt,
        NodeKind::RealBinary(op) => ConcreteNodePayload::RealBinary(*op),
        NodeKind::RealSqrt => ConcreteNodePayload::RealSqrt,
        NodeKind::MatrixBinary(op) => ConcreteNodePayload::MatrixBinary(*op),
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
            ConcreteNodePayload::MatrixMulAccumulate {
                coefficients: coefficients.iter().map(close).collect::<Result<_, _>>()?,
                has_bias: *has_bias,
            }
        }
        NodeKind::MatrixNegate => ConcreteNodePayload::MatrixNegate,
        NodeKind::MatrixScale { scalar } => {
            ConcreteNodePayload::MatrixScale { scalar: close(scalar)? }
        }
        NodeKind::Transpose => ConcreteNodePayload::Transpose,
        NodeKind::Slice { rows, columns } => ConcreteNodePayload::Slice {
            rows: close_range(rows, env, slots)?,
            columns: close_range(columns, env, slots)?,
        },
        NodeKind::Tensor => ConcreteNodePayload::Tensor,
        NodeKind::Concat { axis } => ConcreteNodePayload::Concat { axis: *axis },
        NodeKind::UniformResidueSample { matrix_type } => {
            ConcreteNodePayload::UniformResidueSample { matrix_type: matrix(matrix_type)? }
        }
        NodeKind::UniformIntervalSample { matrix_type, range } => {
            ConcreteNodePayload::UniformIntervalSample {
                matrix_type: matrix(matrix_type)?,
                range: ConcreteSampleRange {
                    minimum: close(&range.minimum)?,
                    maximum: close(&range.maximum)?,
                },
            }
        }
        NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound } => {
            ConcreteNodePayload::GaussianSample {
                matrix_type: matrix(matrix_type)?,
                sigma: close_real(sigma, env, slots)?,
                max_coefficient_bound: close(max_coefficient_bound)?,
            }
        }
        NodeKind::HashSample {
            matrix_type,
            tag_prefix,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
        } => ConcreteNodePayload::HashSample {
            matrix_type: matrix(matrix_type)?,
            tag_prefix: tag_prefix.clone(),
            tag_expressions: tag_expressions.iter().map(close).collect::<Result<_, _>>()?,
            tag_decimal_expressions: tag_decimal_expressions
                .iter()
                .map(close)
                .collect::<Result<_, _>>()?,
            tag_u64_le_expressions: tag_u64_le_expressions
                .iter()
                .map(close)
                .collect::<Result<_, _>>()?,
        },
        NodeKind::TrapdoorSample {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => ConcreteNodePayload::TrapdoorSample {
            matrix_type: matrix(matrix_type)?,
            sigma: close_real(sigma, env, slots)?,
            gadget_base: close(gadget_base)?,
            digit_count: close(digit_count)?,
            preimage_max_coefficient_bound: close(preimage_max_coefficient_bound)?,
        },
        NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => {
            ConcreteNodePayload::PreimageSample {
                matrix_type: matrix(matrix_type)?,
                max_coefficient_bound: close(max_coefficient_bound)?,
            }
        }
        NodeKind::ApplyPreimage => ConcreteNodePayload::ApplyPreimage,
        NodeKind::MaterializePreimageExact => ConcreteNodePayload::MaterializePreimageExact,
        NodeKind::PreimageBinary(op) => ConcreteNodePayload::PreimageBinary(*op),
        NodeKind::PreimageConcatColumns => ConcreteNodePayload::PreimageConcatColumns,
        NodeKind::FamilyPreimageSample { matrix_type, max_coefficient_bound } => {
            ConcreteNodePayload::FamilyPreimageSample {
                matrix_type: matrix(matrix_type)?,
                max_coefficient_bound: close(max_coefficient_bound)?,
            }
        }
        NodeKind::GadgetDecompose { base, small, digit_count } => {
            ConcreteNodePayload::GadgetDecompose {
                base: close(base)?,
                small: *small,
                digit_count: close(digit_count)?,
            }
        }
        NodeKind::DecompositionEntry { row, column } => {
            ConcreteNodePayload::DecompositionEntry { row: close(row)?, column: close(column)? }
        }
        NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => {
            ConcreteNodePayload::ExtractCoefficient {
                position: close(position)?,
                canonical_input_exclusive_upper: canonical_input_exclusive_upper.clone(),
            }
        }
        NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
            ConcreteNodePayload::LiftIntegerToConstantPolynomial {
                matrix_type: matrix(matrix_type)?,
            }
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            ConcreteNodePayload::ThresholdDecode {
                plaintext_modulus: close(plaintext_modulus)?,
                length: close(length)?,
                output_bool: *output_bool,
            }
        }
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
            ConcreteNodePayload::CrtRecompose {
                plaintext_moduli: plaintext_moduli.iter().map(close).collect::<Result<_, _>>()?,
                reconstruction_coefficients: reconstruction_coefficients
                    .iter()
                    .map(close)
                    .collect::<Result<_, _>>()?,
            }
        }
        NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => {
            ConcreteNodePayload::PackPolynomialCoefficients {
                matrix_type: matrix(matrix_type)?,
                coefficient_bits: close(coefficient_bits)?,
            }
        }
        NodeKind::SubgraphCall(payload) => {
            ConcreteNodePayload::SubgraphCall(ConcreteSubgraphPayload {
                definition: payload.definition.clone(),
                bindings: close_bindings(&payload.bindings, env, slots)?,
                canonical_input_exclusive_uppers: payload.canonical_input_exclusive_uppers.clone(),
            })
        }
        NodeKind::SequentialLoop(payload) => {
            ConcreteNodePayload::SequentialLoop(ConcreteSequentialLoop {
                count: close(&payload.count)?,
                index_slot: payload.index_slot,
                bindings: close_bindings(&payload.bindings, env, slots)?,
                carried_count: payload.carried_count,
            })
        }
        NodeKind::FamilyPack { shape } => ConcreteNodePayload::FamilyPack {
            shape: shape.iter().map(close).collect::<Result<_, _>>()?,
        },
        NodeKind::FamilyGetStatic { indices } => ConcreteNodePayload::FamilyGetStatic {
            indices: indices
                .iter()
                .map(|x| close_index(x, env, slots))
                .collect::<Result<_, _>>()?,
        },
        NodeKind::FamilyGetDynamic { rank } => {
            ConcreteNodePayload::FamilyGetDynamic { rank: *rank }
        }
        NodeKind::FamilySelectAxis { axis } => {
            ConcreteNodePayload::FamilySelectAxis { axis: *axis }
        }
        NodeKind::FamilyReindex { output_shape, map } => ConcreteNodePayload::FamilyReindex {
            output_shape: output_shape.iter().map(close).collect::<Result<_, _>>()?,
            map: close_index_map(map, output_shape.len(), env, slots)?,
        },
        NodeKind::FamilyGather { output_shape, input_rank } => ConcreteNodePayload::FamilyGather {
            output_shape: output_shape.iter().map(close).collect::<Result<_, _>>()?,
            input_rank: *input_rank,
        },
        NodeKind::ParallelGrid(payload) => {
            ConcreteNodePayload::ParallelGrid(ConcreteParallelGrid {
                shape: payload.shape.iter().map(close).collect::<Result<_, _>>()?,
                index_slots: payload.index_slots.clone(),
                bindings: close_bindings(&payload.bindings, env, slots)?,
                input_modes: payload
                    .input_modes
                    .iter()
                    .map(|mode| match mode {
                        crate::node::GridInputMode::Broadcast => {
                            Ok(ConcreteGridInputMode::Broadcast)
                        }
                        crate::node::GridInputMode::Reindex { map } => {
                            Ok(ConcreteGridInputMode::Reindex {
                                map: close_index_map(map, payload.shape.len(), env, slots)?,
                            })
                        }
                    })
                    .collect::<Result<_, LinkedProgramError>>()?,
            })
        }
        NodeKind::Select { count } => ConcreteNodePayload::Select { count: close(count)? },
    })
}

fn register_structural_owner(
    graph: &crate::graph::Graph,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
    slots: &mut BTreeSet<u32>,
    declarations: &mut Vec<StructuralSlotDecl>,
) -> Result<(), LinkedProgramError> {
    let (parent, owner) = match scope_id {
        FrozenGraphScopeId::SequentialBody { parent, owner } |
        FrozenGraphScopeId::ParallelBody { parent, owner } => (parent.as_ref(), *owner),
        _ => return Ok(()),
    };
    register_structural_owner(graph, parent, env, slots, declarations)?;
    let parent_scope =
        graph.scope(parent).ok_or_else(|| closed_error("missing structural parent scope"))?;
    let node =
        parent_scope.node(owner).ok_or_else(|| closed_error("missing structural owner node"))?;
    match node.kind() {
        NodeKind::SequentialLoop(loop_spec) => {
            let upper_bound = loop_spec.count.evaluate(env).map_err(closed_error)?;
            if upper_bound <= num_bigint::BigInt::from(0) {
                return Err(closed_error("sequential loop count must be positive"));
            }
            slots.insert(loop_spec.index_slot);
            declarations.push(StructuralSlotDecl {
                slot: loop_spec.index_slot,
                kind: StructuralSlotKind::SequentialIteration,
                upper_bound,
            });
        }
        NodeKind::ParallelGrid(grid) => {
            for (axis, (slot, shape)) in
                grid.index_slots.iter().copied().zip(grid.shape.iter()).enumerate()
            {
                let upper_bound = shape.evaluate(env).map_err(closed_error)?;
                if upper_bound <= num_bigint::BigInt::from(0) {
                    return Err(closed_error("parallel-grid extent must be positive"));
                }
                slots.insert(slot);
                declarations.push(StructuralSlotDecl {
                    slot,
                    kind: StructuralSlotKind::GridAxis { axis },
                    upper_bound,
                });
            }
        }
        _ => return Err(closed_error("non-structural node owns a structural scope")),
    }
    Ok(())
}

/// A fully validated, occurrence-aware collection of stages and artifact links.
///
/// Construction is the only way to obtain this type.  In particular, callers
/// cannot provide a separate semantic stage name or a hand-written link list.
#[derive(Clone, Debug)]
pub struct ValidatedLinkedProgram {
    stages: Vec<LinkedProgramStage>,
    artifact_links: Vec<LinkedArtifactLink>,
}

impl ValidatedLinkedProgram {
    pub fn new(stages: Vec<LinkedProgramStage>) -> Result<Self, LinkedProgramError> {
        let mut stage_keys = BTreeMap::<String, usize>::new();
        let mut production_ids = BTreeMap::<ProductionId, usize>::new();
        for (index, stage) in stages.iter().enumerate() {
            let key = stage.key().to_owned();
            if let Some(first) = stage_keys.insert(key.clone(), index) {
                return Err(LinkedProgramError::DuplicateStageKey { key, first, second: index });
            }
            if let Some(first) = production_ids.insert(stage.production_id.clone(), index) {
                return Err(LinkedProgramError::DuplicateProductionId { first, second: index });
            }
            validate_stage(stage)?;
        }

        let mut artifact_links = Vec::new();
        for (consumer_stage, stage) in stages.iter().enumerate() {
            for (scope_id, scope) in &stage.graph.scopes {
                let mut scanned_artifact_wires = BTreeSet::new();
                for (node_index, node) in scope.execution_order.iter().enumerate() {
                    let NodeKind::Input { artifact: Some(input), .. } = node.kind() else {
                        continue;
                    };
                    let consumer_wire =
                        WireRef { node: crate::types::NodeId(node_index as u64), port: Port(0) };
                    scanned_artifact_wires.insert(consumer_wire);
                    let stored = scope.artifact_inputs.get(&consumer_wire).ok_or_else(|| {
                        LinkedProgramError::UnvalidatedArtifactInput {
                            stage: stage.key().to_owned(),
                            scope: scope_id.clone(),
                            wire: consumer_wire,
                        }
                    })?;
                    let consumer_type = scope
                        .wire_types
                        .get(&consumer_wire)
                        .ok_or_else(|| LinkedProgramError::MissingConsumerType {
                            stage: stage.key().to_owned(),
                            scope: scope_id.clone(),
                            wire: consumer_wire,
                        })?
                        .clone();
                    let producer_stage = match production_ids.get(&input.production_id) {
                        Some(&index) if index < consumer_stage => index,
                        Some(&index) => {
                            return Err(LinkedProgramError::LateArtifactLink {
                                consumer_stage: stage.key().to_owned(),
                                producer_stage: stages[index].key().to_owned(),
                                production_id: input.production_id.clone(),
                            });
                        }
                        None => {
                            return Err(LinkedProgramError::MissingProducer {
                                consumer_stage: stage.key().to_owned(),
                                production_id: input.production_id.clone(),
                            });
                        }
                    };
                    let producer =
                        resolve_producer(&stages[producer_stage], input, stored, &consumer_type)?;
                    artifact_links.push(LinkedArtifactLink {
                        consumer_stage,
                        consumer_stage_key: stage.key().to_owned(),
                        consumer_input: input.clone(),
                        consumer: TypedScopedWireRef {
                            reference: ScopedWireRef {
                                scope: scope_id.clone(),
                                wire: consumer_wire,
                            },
                            wire_type: consumer_type,
                        },
                        producer_stage,
                        producer,
                    });
                }
                if let Some((&wire, _)) = scope
                    .artifact_inputs
                    .iter()
                    .find(|(wire, _)| !scanned_artifact_wires.contains(wire))
                {
                    return Err(LinkedProgramError::UnscannedArtifactInput {
                        stage: stage.key().to_owned(),
                        scope: scope_id.clone(),
                        wire,
                    });
                }
            }
        }

        Ok(Self { stages, artifact_links })
    }

    pub fn stages(&self) -> &[LinkedProgramStage] {
        &self.stages
    }

    pub fn artifact_links(&self) -> &[LinkedArtifactLink] {
        &self.artifact_links
    }

    pub fn stage(&self, key: &str) -> Option<&LinkedProgramStage> {
        self.stages.iter().find(|stage| stage.key() == key)
    }

    /// Resolves a role captured by `FreezeMap` against the exact validated
    /// stage which was linked. No name or node-position search is performed.
    pub fn resolve_semantic_wire(
        &self,
        stage_key: &str,
        role: &FrozenValueRef,
    ) -> Result<ConcreteSemanticWireRef, LinkedProgramError> {
        let stage =
            self.stages.iter().position(|stage| stage.key() == stage_key).ok_or_else(|| {
                LinkedProgramError::SemanticEncoding {
                    message: format!("semantic role stage {stage_key:?} is not linked"),
                }
            })?;
        let linked_stage = &self.stages[stage];
        if linked_stage.graph.source.freeze_id() != role.freeze_id() {
            return Err(LinkedProgramError::SemanticEncoding {
                message: format!(
                    "semantic role for stage {stage_key:?} came from a different frozen graph"
                ),
            });
        }
        let validated_scope =
            linked_stage.graph.scope(&role.reference().scope).ok_or_else(|| {
                LinkedProgramError::SemanticEncoding {
                    message: format!(
                        "semantic role scope {:?} is not validated",
                        role.reference().scope
                    ),
                }
            })?;
        let wire_type = validated_scope
            .wire_types
            .get(&role.reference().wire)
            .cloned()
            .ok_or_else(|| LinkedProgramError::SemanticEncoding {
                message: format!("semantic role wire {:?} is not validated", role.reference().wire),
            })?;
        let declared_wire_type = linked_stage
            .graph
            .source
            .scope(&role.reference().scope)
            .and_then(|scope| scope.node(role.reference().wire.node))
            .and_then(|node| node.output_types().get(role.reference().wire.port.0 as usize))
            .ok_or_else(|| LinkedProgramError::SemanticEncoding {
                message: format!(
                    "semantic role wire {:?} is not present in its frozen source graph",
                    role.reference().wire
                ),
            })?;
        if declared_wire_type != role.wire_type() {
            return Err(LinkedProgramError::SemanticEncoding {
                message: format!(
                    "semantic role wire {:?} changed declared type from {:?} to {:?}",
                    role.reference().wire,
                    role.wire_type(),
                    declared_wire_type
                ),
            });
        }
        let projection = self.semantic_projection()?;
        let scope = scope_number(&projection.stages[stage], &role.reference().scope)?;
        Ok(ConcreteSemanticWireRef {
            stage,
            wire: ConcreteWireRef {
                scope,
                node: role.reference().wire.node,
                port: role.reference().wire.port,
            },
            wire_type,
        })
    }

    /// Closes a structural coordinate against the validated stage and the
    /// scope that owns the semantic wire.  Callers cannot provide a separate
    /// parameter environment or slot set: both are derived from the linked
    /// program, including every enclosing structural owner.
    pub fn close_frozen_structural_expr(
        &self,
        stage_key: &str,
        owner: &FrozenValueRef,
        expression: &crate::graph::FrozenStructuralIntExpr,
    ) -> Result<ConcreteStructuralIntExpr, LinkedProgramError> {
        let stage = self.stages.iter().find(|stage| stage.key() == stage_key).ok_or_else(|| {
            LinkedProgramError::SemanticEncoding {
                message: format!("semantic coordinate stage {stage_key:?} is not linked"),
            }
        })?;
        let freeze_id = stage.graph.source.freeze_id();
        if owner.freeze_id() != freeze_id || expression.freeze_id() != freeze_id {
            return Err(LinkedProgramError::SemanticEncoding {
                message: format!(
                    "semantic coordinate for stage {stage_key:?} came from a different frozen graph"
                ),
            });
        }
        // This validates that the owner is a typed, reached wire in this exact
        // stage before its scope is used to derive structural binders.
        self.resolve_semantic_wire(stage_key, owner)?;

        let mut slots = BTreeSet::new();
        let mut declarations = Vec::new();
        register_structural_owner(
            &stage.graph.source,
            &owner.reference().scope,
            &stage.graph.bindings,
            &mut slots,
            &mut declarations,
        )?;
        ConcreteStructuralIntExpr::close(expression.expression(), &stage.graph.bindings, &slots)
            .map_err(closed_error)
    }

    /// Projects this validated runtime link graph into one typed, nonce-free AST.
    pub fn semantic_projection(&self) -> Result<SemanticLinkedProgram, LinkedProgramError> {
        let stages = self.stages.iter().map(concrete_stage).collect::<Result<Vec<_>, _>>()?;
        let artifact_links = self
            .artifact_links
            .iter()
            .map(|link| {
                let consumer_scope =
                    scope_number(&stages[link.consumer_stage], &link.consumer.reference.scope)?;
                let producer_scope =
                    scope_number(&stages[link.producer_stage], &link.producer.root.scope)?;
                Ok(ConcreteArtifactLink {
                    consumer_stage: link.consumer_stage,
                    consumer: ConcreteWireRef {
                        scope: consumer_scope,
                        node: link.consumer.reference.wire.node,
                        port: link.consumer.reference.wire.port,
                    },
                    argument: link.consumer.reference.wire.port.0 as usize,
                    consumer_artifact: link.consumer_input.artifact_name.clone(),
                    consumer_confidentiality: link.consumer_input.confidentiality,
                    consumer_type: link.consumer.wire_type.clone(),
                    producer_stage: link.producer_stage,
                    producer: ConcreteWireRef {
                        scope: producer_scope,
                        node: link.producer.root.wire.node,
                        port: link.producer.root.wire.port,
                    },
                    producer_artifact: link.producer.name.clone(),
                    producer_confidentiality: link.consumer_input.confidentiality,
                    producer_type: link.producer.wire_type.clone(),
                })
            })
            .collect::<Result<Vec<_>, LinkedProgramError>>()?;
        Ok(ConcreteLinkedProgram { stages, artifact_links })
    }

    /// Hashes the nonce-independent semantic projection for cache identity.
    pub fn semantic_hash(&self) -> Result<[u8; 32], LinkedProgramError> {
        let projection = self.semantic_projection()?;
        hash_canonical(&projection)
            .map_err(|error| LinkedProgramError::SemanticEncoding { message: error.to_string() })
    }
}

fn concrete_stage(stage: &LinkedProgramStage) -> Result<ConcreteLinkedStage, LinkedProgramError> {
    let scope_ids = stage.graph.scopes.keys().cloned().collect::<Vec<_>>();
    let mut scopes = Vec::with_capacity(scope_ids.len());
    for (scope_number, scope_id) in scope_ids.iter().enumerate() {
        let validated = &stage.graph.scopes[scope_id];
        let raw = stage.graph.source.scope(scope_id).ok_or_else(|| {
            LinkedProgramError::SemanticEncoding {
                message: format!("missing source scope {scope_id:?}"),
            }
        })?;
        let mut slots = BTreeSet::new();
        let mut slot_declarations = Vec::new();
        register_structural_owner(
            &stage.graph.source,
            scope_id,
            &stage.graph.bindings,
            &mut slots,
            &mut slot_declarations,
        )?;
        for node in raw.nodes() {
            match node.kind() {
                NodeKind::SequentialLoop(loop_spec) => {
                    slots.insert(loop_spec.index_slot);
                    let upper_bound =
                        loop_spec.count.evaluate(&stage.graph.bindings).map_err(closed_error)?;
                    if upper_bound <= num_bigint::BigInt::from(0) {
                        return Err(closed_error("sequential loop count must be positive"));
                    }
                    slot_declarations.push(StructuralSlotDecl {
                        slot: loop_spec.index_slot,
                        kind: StructuralSlotKind::SequentialIteration,
                        upper_bound,
                    });
                }
                NodeKind::ParallelGrid(grid) => {
                    for (axis, (slot, shape)) in
                        grid.index_slots.iter().copied().zip(grid.shape.iter()).enumerate()
                    {
                        slots.insert(slot);
                        let upper_bound =
                            shape.evaluate(&stage.graph.bindings).map_err(closed_error)?;
                        if upper_bound <= num_bigint::BigInt::from(0) {
                            return Err(closed_error("parallel-grid extent must be positive"));
                        }
                        slot_declarations.push(StructuralSlotDecl {
                            slot,
                            kind: StructuralSlotKind::GridAxis { axis },
                            upper_bound,
                        });
                    }
                }
                _ => {}
            }
        }
        slot_declarations.sort_by_key(|declaration| declaration.slot);
        slot_declarations.dedup_by_key(|declaration| declaration.slot);
        let nodes = validated
            .execution_order
            .iter()
            .enumerate()
            .map(|(node_number, node)| {
                let arguments = raw.arguments(node).ok_or_else(|| LinkedProgramError::SemanticEncoding {
                    message: format!("missing frozen arguments in scope {scope_id:?}, node {node_number}"),
                })?.into_iter().map(|wire| ConcreteWireRef { scope: scope_number, node: wire.node, port: wire.port }).collect();
                let outputs = node.output_types().iter().enumerate().map(|(port, _)| {
                    validated.wire_types.get(&WireRef { node: crate::types::NodeId(node_number as u64), port: crate::types::Port(port as u32) }).cloned().ok_or_else(|| LinkedProgramError::SemanticEncoding { message: format!("missing concrete output in scope {scope_id:?}, node {node_number}, port {port}") })
                }).collect::<Result<Vec<_>, _>>()?;
                let child_scope = stage.graph.source.child_scope_id(scope_id, WireRef { node: crate::types::NodeId(node_number as u64), port: crate::types::Port(0) }.node).and_then(|child| scope_ids.iter().position(|candidate| *candidate == child));
                let input_type = match node.kind() {
                    NodeKind::Input { wire_type, .. } => Some(close_wire_type(wire_type, &stage.graph.bindings)?),
                    _ => None,
                };
                Ok(ConcreteNode {
                    kind: close_payload(node.kind(), &stage.graph.bindings, &slots, input_type)?,
                    arguments,
                    outputs,
                    child_scope,
                })
            })
            .collect::<Result<Vec<_>, LinkedProgramError>>()?;
        let inputs = raw
            .inputs()
            .iter()
            .map(|wire| ConcreteWireRef { scope: scope_number, node: wire.node, port: wire.port })
            .collect();
        let outputs = raw
            .outputs()
            .iter()
            .map(|wire| ConcreteWireRef { scope: scope_number, node: wire.node, port: wire.port })
            .collect();
        scopes.push(ConcreteScope {
            id: scope_number,
            structural_slots: slot_declarations,
            nodes,
            inputs,
            outputs,
        });
    }
    let root_scope =
        scope_ids.iter().position(|id| *id == FrozenGraphScopeId::Root).ok_or_else(|| {
            LinkedProgramError::SemanticEncoding {
                message: "validated stage has no root scope".to_owned(),
            }
        })?;
    let named_outputs = stage
        .graph
        .source
        .outputs()
        .iter()
        .map(|(name, output)| ConcreteNamedOutput {
            name: name.clone(),
            wire: ConcreteWireRef {
                scope: root_scope,
                node: output.value.node,
                port: output.value.port,
            },
        })
        .collect();
    Ok(ConcreteLinkedStage {
        key: stage.key().to_owned(),
        bindings: stage.graph.bindings.clone(),
        scope_ids,
        scopes,
        root_scope,
        named_outputs,
    })
}

fn scope_number(
    stage: &ConcreteLinkedStage,
    scope: &FrozenGraphScopeId,
) -> Result<usize, LinkedProgramError> {
    stage.scope_ids.iter().position(|candidate| candidate == scope).ok_or_else(|| {
        LinkedProgramError::SemanticEncoding {
            message: format!("scope {scope:?} is not present in semantic stage"),
        }
    })
}

#[derive(Debug, Error)]
pub enum LinkedProgramError {
    #[error("stage key {key:?} is duplicated by stages {first} and {second}")]
    DuplicateStageKey { key: String, first: usize, second: usize },
    #[error("production id is duplicated by stages {first} and {second}")]
    DuplicateProductionId { first: usize, second: usize },
    #[error(
        "stage {stage:?} declares production id {declared:?}, but its manifest uses {manifest:?}"
    )]
    ManifestProductionMismatch { stage: String, declared: ProductionId, manifest: ProductionId },
    #[error("stage {stage:?} manifest uses IR version {actual}, expected {expected}")]
    ManifestVersion { stage: String, expected: u32, actual: u32 },
    #[error("stage {stage:?} manifest is invalid: {message}")]
    InvalidManifest { stage: String, message: String },
    #[error(
        "stage {stage:?} production id does not match its frozen graph: expected {expected:?}, got {actual:?}"
    )]
    SpecHashMismatch { stage: String, expected: ProductionId, actual: ProductionId },
    #[error("consumer stage {consumer_stage:?} references missing producer {production_id:?}")]
    MissingProducer { consumer_stage: String, production_id: ProductionId },
    #[error(
        "consumer stage {consumer_stage:?} references producer stage {producer_stage:?} after it in supplied order"
    )]
    LateArtifactLink { consumer_stage: String, producer_stage: String, production_id: ProductionId },
    #[error(
        "artifact input {wire:?} in stage {stage:?}, scope {scope:?} was not retained by validation"
    )]
    UnvalidatedArtifactInput { stage: String, scope: FrozenGraphScopeId, wire: WireRef },
    #[error(
        "validated artifact input {wire:?} in stage {stage:?}, scope {scope:?} has no input node"
    )]
    UnscannedArtifactInput { stage: String, scope: FrozenGraphScopeId, wire: WireRef },
    #[error("consumer wire {wire:?} in stage {stage:?}, scope {scope:?} has no concrete type")]
    MissingConsumerType { stage: String, scope: FrozenGraphScopeId, wire: WireRef },
    #[error("producer stage {stage:?} has no artifact named {artifact:?}")]
    MissingProducerArtifact { stage: String, artifact: String },
    #[error("producer output {output:?} is not present in stage {stage:?}")]
    MissingProducerOutput { stage: String, output: String },
    #[error("producer output {output:?} in stage {stage:?} has no confidentiality declaration")]
    ProducerOutputConfidentiality { stage: String, output: String },
    #[error(
        "artifact {artifact:?} in stage {stage:?} does not match its producer manifest metadata"
    )]
    ManifestMetadataMismatch { stage: String, artifact: String },
    #[error(
        "artifact {artifact:?} in stage {stage:?} has a type or family shape inconsistent with its manifest"
    )]
    ArtifactTypeMetadataMismatch { stage: String, artifact: String },
    #[error(
        "artifact {artifact:?} links incompatible concrete types: producer {producer:?}, consumer {consumer:?}"
    )]
    ConcreteWireTypeMismatch {
        artifact: String,
        producer: ConcreteWireType,
        consumer: ConcreteWireType,
    },
    #[error("semantic projection encoding failed: {message}")]
    SemanticEncoding { message: String },
}

fn validate_stage(stage: &LinkedProgramStage) -> Result<(), LinkedProgramError> {
    let key = stage.key().to_owned();
    if stage.manifest.production_id != stage.production_id {
        return Err(LinkedProgramError::ManifestProductionMismatch {
            stage: key,
            declared: stage.production_id.clone(),
            manifest: stage.manifest.production_id.clone(),
        });
    }
    if stage.manifest.ir_version != IR_VERSION {
        return Err(LinkedProgramError::ManifestVersion {
            stage: key,
            expected: IR_VERSION,
            actual: stage.manifest.ir_version,
        });
    }
    validate_manifest(&stage.manifest).map_err(|error| LinkedProgramError::InvalidManifest {
        stage: stage.key().to_owned(),
        message: error.to_string(),
    })?;
    let expected = ProductionId {
        spec_hash: spec_hash(&stage.graph.source, &stage.graph.bindings).map_err(|error| {
            LinkedProgramError::InvalidManifest {
                stage: stage.key().to_owned(),
                message: error.to_string(),
            }
        })?,
        execution_nonce: stage.production_id.execution_nonce,
    };
    if expected != stage.production_id {
        return Err(LinkedProgramError::SpecHashMismatch {
            stage: stage.key().to_owned(),
            expected,
            actual: stage.production_id.clone(),
        });
    }
    Ok(())
}

fn resolve_producer(
    producer_stage: &LinkedProgramStage,
    input: &ArtifactInput,
    stored: &ManifestArtifact,
    consumer_type: &ConcreteWireType,
) -> Result<LinkedProducerOutput, LinkedProgramError> {
    let producer_manifest_artifact =
        producer_stage.manifest.artifacts.get(&input.artifact_name).ok_or_else(|| {
            LinkedProgramError::MissingProducerArtifact {
                stage: producer_stage.key().to_owned(),
                artifact: input.artifact_name.clone(),
            }
        })?;
    if stored != producer_manifest_artifact ||
        input.confidentiality != producer_manifest_artifact.confidentiality
    {
        return Err(LinkedProgramError::ManifestMetadataMismatch {
            stage: producer_stage.key().to_owned(),
            artifact: input.artifact_name.clone(),
        });
    }
    let output =
        producer_stage.graph.source.outputs().get(&input.artifact_name).ok_or_else(|| {
            LinkedProgramError::MissingProducerOutput {
                stage: producer_stage.key().to_owned(),
                output: input.artifact_name.clone(),
            }
        })?;
    if output.confidentiality != Some(producer_manifest_artifact.confidentiality) {
        return Err(LinkedProgramError::ProducerOutputConfidentiality {
            stage: producer_stage.key().to_owned(),
            output: input.artifact_name.clone(),
        });
    }
    let root = ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: output.value };
    let producer_type = producer_stage
        .graph
        .root_scope()
        .wire_types
        .get(&output.value)
        .ok_or_else(|| LinkedProgramError::MissingProducerOutput {
            stage: producer_stage.key().to_owned(),
            output: input.artifact_name.clone(),
        })?
        .clone();
    validate_artifact_metadata(
        producer_stage.key(),
        &input.artifact_name,
        &producer_type,
        producer_manifest_artifact,
    )?;
    validate_artifact_metadata(
        producer_stage.key(),
        &input.artifact_name,
        consumer_type,
        producer_manifest_artifact,
    )?;
    if producer_type != *consumer_type {
        return Err(LinkedProgramError::ConcreteWireTypeMismatch {
            artifact: input.artifact_name.clone(),
            producer: producer_type,
            consumer: consumer_type.clone(),
        });
    }
    Ok(LinkedProducerOutput {
        stage_key: producer_stage.key().to_owned(),
        production_id: producer_stage.production_id.clone(),
        name: input.artifact_name.clone(),
        root,
        wire_type: producer_type,
    })
}

fn validate_artifact_metadata(
    stage: &str,
    artifact: &str,
    wire_type: &ConcreteWireType,
    metadata: &ManifestArtifact,
) -> Result<(), LinkedProgramError> {
    let (element, shape) = match wire_type {
        ConcreteWireType::Family { element, shape } => (element.as_ref(), Some(shape.as_slice())),
        scalar => (scalar, None),
    };
    if ArtifactType::from_wire_type(element).as_ref() != Some(&metadata.artifact_type) ||
        shape != metadata.family_shape.as_deref()
    {
        return Err(LinkedProgramError::ArtifactTypeMetadataMismatch {
            stage: stage.to_owned(),
            artifact: artifact.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FreezeMap, Graph, GraphOutput, IntExpr, NodeHandle, ParamEnv, ValueHandle,
        artifact::{ArtifactConfidentiality, export_validated_manifest},
        encoding::spec_hash,
        graph::{FrozenStructuralIntExpr, SubgraphHandle, with_new_construction_scope},
        node::{GridInputMode, NodeKind, ParallelGrid, SequentialLoop},
        types::{ConcreteWireType, MatrixType, WireType},
        validate::{validate, validate_with_manifests},
    };
    use num_bigint::BigInt;

    #[test]
    fn frozen_coordinate_closure_is_scope_strict() {
        let env = ParamEnv::default();
        let enclosing = FrozenStructuralIntExpr::new(7, IntExpr::LoopIndex(3));
        let child_only = FrozenStructuralIntExpr::new(7, IntExpr::LoopIndex(9));

        assert!(
            ConcreteStructuralIntExpr::close(enclosing.expression(), &env, &BTreeSet::from([3]))
                .is_ok()
        );
        assert!(
            ConcreteStructuralIntExpr::close(child_only.expression(), &env, &BTreeSet::from([3]))
                .is_err()
        );
        assert!(
            ConcreteStructuralIntExpr::close(child_only.expression(), &env, &BTreeSet::from([9]))
                .is_ok()
        );
    }

    #[test]
    fn linked_coordinate_closure_checks_owner_and_freeze_identity() {
        let owner_value = NodeHandle::new(
            NodeKind::Input {
                name: "owner".to_owned(),
                wire_type: WireType::Matrix(matrix_type()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix_type())],
        )
        .output(0)
        .unwrap();
        let (frozen, map) = Graph::freeze(
            "coordinate-owner",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput {
                    value: owner_value.clone(),
                    confidentiality: Some(ArtifactConfidentiality::Public),
                },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        let validated = validate(&frozen, &ParamEnv::default()).unwrap();
        let linked =
            ValidatedLinkedProgram::new(vec![stage("coordinate-owner", validated, 71)]).unwrap();
        let owner = map.resolve_typed(&owner_value).unwrap();
        let coordinate = map.freeze_structural_expr(IntExpr::constant(4));
        assert!(
            linked.close_frozen_structural_expr("coordinate-owner", &owner, &coordinate).is_ok()
        );

        let other_value = NodeHandle::new(
            NodeKind::Input {
                name: "other".to_owned(),
                wire_type: WireType::Matrix(matrix_type()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Matrix(matrix_type())],
        )
        .output(0)
        .unwrap();
        let (other_frozen, other_map) = Graph::freeze(
            "coordinate-other",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput {
                    value: other_value,
                    confidentiality: Some(ArtifactConfidentiality::Public),
                },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        let foreign_coordinate = other_map.freeze_structural_expr(IntExpr::constant(4));
        assert!(
            linked
                .close_frozen_structural_expr("coordinate-owner", &owner, &foreign_coordinate)
                .is_err()
        );
        let _ = other_frozen;
    }

    #[test]
    fn linked_coordinate_closure_uses_only_enclosing_structural_slots() {
        let ty = WireType::Matrix(matrix_type());
        let outer = NodeHandle::new(
            NodeKind::Input { name: "outer".to_owned(), wire_type: ty.clone(), artifact: None },
            Vec::new(),
            vec![ty.clone()],
        )
        .output(0)
        .unwrap();
        let body = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input { name: "body".to_owned(), wire_type: ty.clone(), artifact: None },
                Vec::new(),
                vec![ty.clone()],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("coordinate-loop-body", scope, vec![input.clone()], vec![input])
                .unwrap()
        });
        let loop_node = NodeHandle::sequential_loop(
            body,
            vec![outer.clone()],
            vec![ty.clone()],
            SequentialLoop {
                count: IntExpr::constant(2),
                index_slot: 17,
                bindings: Vec::new(),
                carried_count: 1,
            },
        )
        .output(0)
        .unwrap();
        let (graph, map) = Graph::freeze(
            "coordinate-loop",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput {
                    value: loop_node,
                    confidentiality: Some(ArtifactConfidentiality::Public),
                },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        let owner = map.resolve_typed(&outer).unwrap();
        let coordinate = map.freeze_structural_expr(IntExpr::LoopIndex(17));
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let linked =
            ValidatedLinkedProgram::new(vec![stage("coordinate-loop", validated, 72)]).unwrap();
        assert!(
            linked.close_frozen_structural_expr("coordinate-loop", &owner, &coordinate).is_err()
        );
    }

    fn parallel_input_graph(
        duplicate_body_input: bool,
    ) -> (Graph, FreezeMap, ValueHandle, ValueHandle) {
        let matrix = WireType::Matrix(matrix_type());
        let family = WireType::Family {
            element: Box::new(matrix.clone()),
            shape: vec![IntExpr::constant(2)],
        };
        let outer = NodeHandle::new(
            NodeKind::Input {
                name: "family".to_owned(),
                wire_type: family.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family.clone()],
        )
        .output(0)
        .unwrap();
        let (body, body_input) = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "item".to_owned(),
                    wire_type: matrix.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![matrix.clone()],
            )
            .output(0)
            .unwrap();
            let inputs = if duplicate_body_input {
                vec![input.clone(), input.clone()]
            } else {
                vec![input.clone()]
            };
            let body =
                SubgraphHandle::new("child-input-path-body", scope, inputs, vec![input.clone()])
                    .unwrap();
            (body, input)
        });
        let arguments = if duplicate_body_input {
            vec![outer.clone(), outer.clone()]
        } else {
            vec![outer.clone()]
        };
        let modes = arguments
            .iter()
            .map(|_| GridInputMode::Reindex {
                map: crate::IndexMap::new([crate::IndexExpr::Axis(0)]),
            })
            .collect();
        let outputs = vec![family.clone()];
        let grid = NodeHandle::parallel_grid(
            body,
            arguments,
            outputs,
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![0],
                bindings: Vec::new(),
                input_modes: modes,
            },
        )
        .output(0)
        .unwrap();
        let (graph, map) = Graph::freeze(
            "child-input-path",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value: grid, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        (graph, map, outer, body_input)
    }

    fn nested_parallel_input_graph() -> (Graph, FreezeMap, ValueHandle, ValueHandle) {
        let matrix = WireType::Matrix(matrix_type());
        let matrix_family = WireType::Family {
            element: Box::new(matrix.clone()),
            shape: vec![IntExpr::constant(2)],
        };
        let nested_family = WireType::Family {
            element: Box::new(matrix_family.clone()),
            shape: vec![IntExpr::constant(2)],
        };
        let outer = NodeHandle::new(
            NodeKind::Input {
                name: "nestedFamily".to_owned(),
                wire_type: nested_family.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![nested_family.clone()],
        )
        .output(0)
        .unwrap();
        let (outer_body, deepest_input) = with_new_construction_scope(|outer_scope| {
            let outer_input = NodeHandle::new(
                NodeKind::Input {
                    name: "outerItem".to_owned(),
                    wire_type: matrix_family.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![matrix_family.clone()],
            )
            .output(0)
            .unwrap();
            let (inner_body, deepest_input) = with_new_construction_scope(|inner_scope| {
                let input = NodeHandle::new(
                    NodeKind::Input {
                        name: "innerItem".to_owned(),
                        wire_type: matrix.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![matrix.clone()],
                )
                .output(0)
                .unwrap();
                let body = SubgraphHandle::new(
                    "nested-inner-body",
                    inner_scope,
                    vec![input.clone()],
                    vec![input.clone()],
                )
                .unwrap();
                (body, input)
            });
            let inner_grid = NodeHandle::parallel_grid(
                inner_body,
                vec![outer_input.clone()],
                vec![matrix_family.clone()],
                ParallelGrid {
                    shape: vec![IntExpr::constant(2)],
                    index_slots: vec![1],
                    bindings: Vec::new(),
                    input_modes: vec![GridInputMode::Reindex {
                        map: crate::IndexMap::new([crate::IndexExpr::Axis(0)]),
                    }],
                },
            )
            .output(0)
            .unwrap();
            let body = SubgraphHandle::new(
                "nested-outer-body",
                outer_scope,
                vec![outer_input],
                vec![inner_grid],
            )
            .unwrap();
            (body, deepest_input)
        });
        let outer_grid = NodeHandle::parallel_grid(
            outer_body,
            vec![outer.clone()],
            vec![nested_family.clone()],
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![0],
                bindings: Vec::new(),
                input_modes: vec![GridInputMode::Reindex {
                    map: crate::IndexMap::new([crate::IndexExpr::Axis(0)]),
                }],
            },
        )
        .output(0)
        .unwrap();
        let (graph, map) = Graph::freeze(
            "nested-child-input-path",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value: outer_grid, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        (graph, map, outer, deepest_input)
    }

    #[test]
    fn child_input_path_follows_exact_parallel_grid_boundary() {
        let (graph, map, outer, body_input) = parallel_input_graph(false);
        let start = map.resolve_typed(&outer).unwrap();
        let target = map.resolve_typed(&body_input).unwrap();
        let owner = graph
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.kind(), NodeKind::ParallelGrid(_))
                    .then_some(crate::NodeId(index as u64))
            })
            .unwrap();
        let hop = ChildInputHop { parent_scope: FrozenGraphScopeId::Root, owner, input_index: 0 };
        assert_eq!(
            follow_child_input_hop(&graph, start.reference(), &hop).unwrap(),
            *target.reference()
        );
        assert!(
            follows_child_input_path(&graph, start.reference(), &[hop.clone()], target.reference())
                .unwrap()
        );
        assert!(
            follows_child_input_path(&graph, start.reference(), &[], start.reference()).unwrap()
        );
        assert_eq!(derive_child_input_path(&graph, &start, &target).unwrap(), vec![hop]);
    }

    #[test]
    fn child_input_path_rejects_wrong_missing_and_unsupported_hops() {
        let (graph, map, outer, body_input) = parallel_input_graph(false);
        let start = map.resolve_typed(&outer).unwrap();
        let target = map.resolve_typed(&body_input).unwrap();
        let owner = graph
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.kind(), NodeKind::ParallelGrid(_))
                    .then_some(crate::NodeId(index as u64))
            })
            .unwrap();
        let wrong_wire = ScopedWireRef {
            scope: FrozenGraphScopeId::Root,
            wire: WireRef { node: start.reference().wire.node, port: Port(99) },
        };
        let hop = ChildInputHop { parent_scope: FrozenGraphScopeId::Root, owner, input_index: 0 };
        assert!(matches!(
            follow_child_input_hop(&graph, &wrong_wire, &hop),
            Err(ChildInputPathError::OwnerArgumentMismatch { .. })
        ));
        let missing = ChildInputHop { input_index: 1, ..hop.clone() };
        assert!(matches!(
            follow_child_input_hop(&graph, start.reference(), &missing),
            Err(ChildInputPathError::MissingOwnerArgument { .. } |
                ChildInputPathError::MissingInputMode { .. } |
                ChildInputPathError::MissingChildInput { .. })
        ));
        assert!(
            matches!(derive_child_input_path(&graph, &start, &target), Ok(path) if path == vec![hop])
        );
    }

    #[test]
    fn child_input_path_rejects_ambiguous_duplicate_body_input() {
        let (graph, map, outer, body_input) = parallel_input_graph(true);
        let start = map.resolve_typed(&outer).unwrap();
        let target = map.resolve_typed(&body_input).unwrap();
        assert!(matches!(
            derive_child_input_path(&graph, &start, &target),
            Err(ChildInputPathError::Ambiguous { count: 2 })
        ));
    }

    #[test]
    fn child_input_path_derives_nested_parallel_grid_boundaries_in_order() {
        let (graph, map, outer, deepest_input) = nested_parallel_input_graph();
        let start = map.resolve_typed(&outer).unwrap();
        let target = map.resolve_typed(&deepest_input).unwrap();
        let path = derive_child_input_path(&graph, &start, &target).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].parent_scope, FrozenGraphScopeId::Root);
        assert!(matches!(path[1].parent_scope, FrozenGraphScopeId::ParallelBody { .. }));
        assert!(
            follows_child_input_path(&graph, start.reference(), &path, target.reference()).unwrap()
        );
    }

    #[test]
    fn structural_route_exports_body_output_and_reimports_it_exactly() {
        let (graph, map, outer, body_input) = parallel_input_graph(false);
        let body = map.resolve_typed(&body_input).unwrap();
        let outer_value = map.resolve_typed(&outer).unwrap();
        let owner = graph
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.kind(), NodeKind::ParallelGrid(_))
                    .then_some(crate::NodeId(index as u64))
            })
            .unwrap();
        let parent_wire = WireRef { node: owner, port: Port(0) };
        let parent_type =
            graph.root_scope().node(owner).unwrap().output_types().get(0).unwrap().clone();
        let parent = FrozenValueRef::new(
            body.freeze_id(),
            ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: parent_wire },
            parent_type,
        );
        let route = derive_structural_value_route(&graph, &body, &parent).unwrap();
        assert_eq!(route.exits.len(), 1);
        assert!(route.enters.is_empty());
        assert_eq!(
            follow_structural_value_route(&graph, body.reference(), &route).unwrap(),
            *parent.reference()
        );
        assert_ne!(parent.reference(), outer_value.reference());
    }

    #[test]
    fn structural_route_rejects_non_body_output_and_wrong_parent_type() {
        let (graph, map, _outer, body_input) = parallel_input_graph(false);
        let body = map.resolve_typed(&body_input).unwrap();
        let owner = graph
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.kind(), NodeKind::ParallelGrid(_))
                    .then_some(crate::NodeId(index as u64))
            })
            .unwrap();
        let wrong = ScopedWireRef {
            scope: body.reference().scope.clone(),
            wire: WireRef { node: body.reference().wire.node, port: Port(99) },
        };
        let hop =
            ParallelOutputHop { parent_scope: FrozenGraphScopeId::Root, owner, output_index: 0 };
        assert!(follow_parallel_output_hop(&graph, &wrong, &hop).is_err());
        assert!(follow_parallel_output_hop(&graph, body.reference(), &hop).is_ok());
        assert!(matches!(
            follow_parallel_output_hop(
                &graph,
                body.reference(),
                &ParallelOutputHop { output_index: 1, ..hop }
            ),
            Err(ChildInputPathError::MissingChildOutput { .. })
        ));
    }

    #[test]
    fn structural_route_accepts_parallel_sibling_exit_then_identity_entry() {
        let (graph, map, source, target) = sibling_parallel_graph();
        let source = map.resolve_typed(&source).unwrap();
        let target = map.resolve_typed(&target).unwrap();
        let route = derive_structural_value_route(&graph, &source, &target).unwrap();
        assert_eq!(route.exits.len(), 1);
        assert_eq!(route.enters.len(), 1);
        assert_eq!(route.enters[0].input_index, 0);
        assert!(
            follows_structural_value_route(&graph, source.reference(), &route, target.reference())
                .unwrap()
        );
    }

    fn sibling_parallel_graph() -> (Graph, FreezeMap, ValueHandle, ValueHandle) {
        let matrix = WireType::Matrix(matrix_type());
        let family = WireType::Family {
            element: Box::new(matrix.clone()),
            shape: vec![IntExpr::constant(2)],
        };
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "sibling-family".to_owned(),
                wire_type: family.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family.clone()],
        )
        .output(0)
        .unwrap();
        let (first_body, first_output) = with_new_construction_scope(|scope| {
            let value = NodeHandle::new(
                NodeKind::Input {
                    name: "first-item".to_owned(),
                    wire_type: matrix.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![matrix.clone()],
            )
            .output(0)
            .unwrap();
            (
                SubgraphHandle::new(
                    "sibling-first-body",
                    scope,
                    vec![value.clone()],
                    vec![value.clone()],
                )
                .unwrap(),
                value,
            )
        });
        let first_grid = NodeHandle::parallel_grid(
            first_body,
            vec![input.clone()],
            vec![family.clone()],
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![0],
                bindings: Vec::new(),
                input_modes: vec![GridInputMode::Reindex {
                    map: crate::IndexMap::new([crate::IndexExpr::Axis(0)]),
                }],
            },
        )
        .output(0)
        .unwrap();
        let (second_body, second_input) = with_new_construction_scope(|scope| {
            let value = NodeHandle::new(
                NodeKind::Input {
                    name: "second-item".to_owned(),
                    wire_type: matrix.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![matrix.clone()],
            )
            .output(0)
            .unwrap();
            (
                SubgraphHandle::new(
                    "sibling-second-body",
                    scope,
                    vec![value.clone()],
                    vec![value.clone()],
                )
                .unwrap(),
                value,
            )
        });
        let second_grid = NodeHandle::parallel_grid(
            second_body,
            vec![first_grid],
            vec![family],
            ParallelGrid {
                shape: vec![IntExpr::constant(2)],
                index_slots: vec![1],
                bindings: Vec::new(),
                input_modes: vec![GridInputMode::Reindex {
                    map: crate::IndexMap::new([crate::IndexExpr::Axis(0)]),
                }],
            },
        )
        .output(0)
        .unwrap();
        let (graph, map) = Graph::freeze(
            "sibling-parallel-route",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value: second_grid, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        (graph, map, first_output, second_input)
    }

    #[test]
    fn concrete_child_input_path_is_derived_and_rendered_from_scope_table() {
        let (graph, map, outer, body_input) = parallel_input_graph(false);
        let start = map.resolve_typed(&outer).unwrap();
        let target = map.resolve_typed(&body_input).unwrap();
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let linked =
            ValidatedLinkedProgram::new(vec![stage("child-input-path", validated, 73)]).unwrap();
        let projection = linked.semantic_projection().unwrap();
        let concrete = &projection.stages[0];
        let start_concrete = ConcreteWireRef {
            scope: concrete
                .scope_ids
                .iter()
                .position(|scope| scope == &start.reference().scope)
                .unwrap(),
            node: start.reference().wire.node,
            port: start.reference().wire.port,
        };
        let target_concrete = ConcreteWireRef {
            scope: concrete
                .scope_ids
                .iter()
                .position(|scope| scope == &target.reference().scope)
                .unwrap(),
            node: target.reference().wire.node,
            port: target.reference().wire.port,
        };
        let path =
            derive_concrete_child_input_path(concrete, &start_concrete, &target_concrete).unwrap();
        assert!(
            follows_concrete_child_input_path(concrete, &start_concrete, &path, &target_concrete)
                .unwrap()
        );
        let absent_target = ConcreteWireRef { port: Port(99), ..target_concrete.clone() };
        assert!(matches!(
            derive_concrete_child_input_path(concrete, &start_concrete, &absent_target),
            Err(ChildInputPathError::NoPath)
        ));
        assert!(
            crate::lean::render_child_input_path(concrete, &path)
                .unwrap()
                .contains("inputIndex := 0")
        );
        let owner = path[0].owner;
        let root_scope =
            concrete.scope_ids.iter().position(|scope| *scope == FrozenGraphScopeId::Root).unwrap();
        let parent_concrete = ConcreteWireRef { scope: root_scope, node: owner, port: Port(0) };
        let output_hop =
            ParallelOutputHop { parent_scope: FrozenGraphScopeId::Root, owner, output_index: 0 };
        assert_eq!(
            follow_concrete_parallel_output_hop(concrete, &target_concrete, &output_hop).unwrap(),
            parent_concrete
        );
        let route = StructuralValueRoute { exits: vec![output_hop], enters: Vec::new() };
        assert!(
            follows_concrete_structural_value_route(
                concrete,
                &target_concrete,
                &route,
                &parent_concrete
            )
            .unwrap()
        );
        assert!(
            crate::lean::render_structural_value_route(concrete, &route)
                .unwrap()
                .contains("outputIndex := 0")
        );
    }

    fn matrix_type() -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        }
    }

    fn stage_graph(name: &str, wire_type: WireType, artifact: Option<ArtifactInput>) -> Graph {
        let input = NodeHandle::new(
            NodeKind::Input { name: "input".to_owned(), wire_type: wire_type.clone(), artifact },
            Vec::new(),
            vec![wire_type],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            name,
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput {
                    value: input,
                    confidentiality: Some(ArtifactConfidentiality::Public),
                },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        graph
    }

    fn stage(_name: &str, graph: ValidatedGraph, nonce: u8) -> LinkedProgramStage {
        let production_id = ProductionId {
            spec_hash: spec_hash(&graph.source, &graph.bindings).unwrap(),
            execution_nonce: [nonce; 32],
        };
        let manifest = export_validated_manifest(production_id.clone(), &graph).unwrap();
        LinkedProgramStage::new(production_id, graph, manifest)
    }

    fn linked_pair(producer_nonce: u8) -> ValidatedLinkedProgram {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer = stage(
            "producer",
            validate(&producer_source, &ParamEnv::default()).unwrap(),
            producer_nonce,
        );
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_source = stage_graph("consumer", WireType::Matrix(matrix_type()), Some(input));
        let consumer_graph = validate_with_manifests(
            &consumer_source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("consumer", consumer_graph, producer_nonce.wrapping_add(1));
        ValidatedLinkedProgram::new(vec![producer, consumer]).unwrap()
    }

    #[test]
    fn links_artifact_to_earlier_named_root() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 1);
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_source =
            stage_graph("consumer", WireType::Matrix(matrix_type()), Some(input.clone()));
        let consumer = {
            let graph = validate_with_manifests(
                &consumer_source,
                &ParamEnv::default(),
                &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
            )
            .unwrap();
            stage("consumer", graph, 2)
        };
        let linked = ValidatedLinkedProgram::new(vec![producer, consumer]).unwrap();
        assert_eq!(linked.artifact_links().len(), 1);
        let link = &linked.artifact_links()[0];
        assert_eq!(link.producer.name, "out");
        assert_eq!(link.producer.root.scope, FrozenGraphScopeId::Root);
        assert_eq!(link.producer.wire_type, link.consumer.wire_type);
    }

    #[test]
    fn matrix_and_preimage_types_do_not_collapse() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 3);
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let source = stage_graph("consumer", WireType::Preimage(matrix_type()), Some(input));
        let graph = validate_with_manifests(
            &source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("consumer", graph, 4);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![producer, consumer]),
            Err(LinkedProgramError::ConcreteWireTypeMismatch { .. })
        ));
    }

    #[test]
    fn links_must_point_to_an_earlier_stage() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 5);
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_source = stage_graph("consumer", WireType::Matrix(matrix_type()), Some(input));
        let consumer_graph = validate_with_manifests(
            &consumer_source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("consumer", consumer_graph, 6);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![consumer, producer]),
            Err(LinkedProgramError::LateArtifactLink { .. })
        ));
    }

    #[test]
    fn production_id_is_bound_to_graph_and_parameters() {
        let source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let mut graph = validate(&source, &ParamEnv::default()).unwrap();
        let production_id = ProductionId {
            spec_hash: spec_hash(&graph.source, &graph.bindings).unwrap(),
            execution_nonce: [7; 32],
        };
        graph.bindings.integers.insert("unused".to_owned(), BigInt::from(1));
        let manifest = export_validated_manifest(production_id.clone(), &graph).unwrap();
        assert!(
            ValidatedLinkedProgram::new(vec![LinkedProgramStage::new(
                production_id,
                graph,
                manifest,
            )])
            .is_err()
        );
    }

    #[test]
    fn semantic_projection_erases_coordinated_execution_nonces() {
        let first = linked_pair(8);
        let second = linked_pair(19);
        assert_eq!(first.semantic_projection().unwrap(), second.semantic_projection().unwrap());
        assert_eq!(first.semantic_hash().unwrap(), second.semantic_hash().unwrap());
        let first_render =
            crate::lean::render_lean_program(&first, "MxxGenerated.TestProgram").unwrap();
        let second_render =
            crate::lean::render_lean_program(&second, "MxxGenerated.TestProgram").unwrap();
        assert_eq!(first_render.modules, second_render.modules);
        assert_eq!(first_render.linked_program_sha256, second_render.linked_program_sha256);
    }

    #[test]
    fn semantic_projection_contains_a_closed_lossless_payload() {
        let linked = linked_pair(33);
        let projection = linked.semantic_projection().unwrap();
        let payload = &projection.stages[0].scopes[0].nodes[0].kind;
        assert!(matches!(
            payload,
            ConcreteNodePayload::Input {
                name,
                wire_type: ConcreteWireType::Matrix(_),
                artifact: None,
            } if name == "input"
        ));
        let json = serde_json::to_string(&projection).unwrap();
        assert!(!json.contains("ParamEnv"));
        assert!(!json.contains("production_id"));
    }

    #[test]
    fn payload_mutation_changes_the_semantic_hash() {
        let linked = linked_pair(34);
        let mut projection = linked.semantic_projection().unwrap();
        let original = crate::encoding::hash_canonical(&projection).unwrap();
        projection.stages[0].scopes[0].nodes[0].kind = ConcreteNodePayload::Input {
            name: "changed".to_owned(),
            wire_type: ConcreteWireType::Matrix(crate::types::ConcreteMatrixType {
                modulus: BigInt::from(17),
                ring_dimension: 8,
                rows: 1,
                columns: 1,
            }),
            artifact: None,
        };
        assert_ne!(original, crate::encoding::hash_canonical(&projection).unwrap());
    }

    #[test]
    fn unresolved_structural_slots_are_rejected_when_not_declared() {
        let result = ConcreteStructuralIntExpr::close(
            &IntExpr::LoopIndex(91),
            &ParamEnv::default(),
            &BTreeSet::new(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn wrong_full_production_id_does_not_link_by_shape() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 21);
        let wrong_id = ProductionId {
            spec_hash: producer.production_id.spec_hash.clone(),
            execution_nonce: [22; 32],
        };
        let input = ArtifactInput {
            production_id: wrong_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let source = stage_graph("consumer", WireType::Matrix(matrix_type()), Some(input));
        let mut wrong_manifest = producer.manifest.clone();
        wrong_manifest.production_id = wrong_id.clone();
        let graph = validate_with_manifests(
            &source,
            &ParamEnv::default(),
            &BTreeMap::from([(wrong_id, wrong_manifest)]),
        )
        .unwrap();
        let consumer = stage("consumer", graph, 23);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![producer, consumer]),
            Err(LinkedProgramError::MissingProducer { .. })
        ));
    }

    #[test]
    fn artifact_links_in_nested_scopes_keep_their_scope() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 24);
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_source = nested_source(WireType::Matrix(matrix_type()), Some(input));
        let consumer_graph = validate_with_manifests(
            &consumer_source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("nested-consumer", consumer_graph, 25);
        let linked = ValidatedLinkedProgram::new(vec![producer, consumer]).unwrap();
        assert_eq!(linked.artifact_links().len(), 1);
        assert_ne!(linked.artifact_links()[0].consumer.reference.scope, FrozenGraphScopeId::Root);
    }

    #[test]
    fn nested_preimage_family_types_are_preserved_and_checked_fully() {
        let family_preimage = WireType::Family {
            element: Box::new(WireType::Preimage(matrix_type())),
            shape: vec![IntExpr::constant(2)],
        };
        let producer_source = stage_graph("family-producer", family_preimage.clone(), None);
        let producer =
            stage("family-producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 26);
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_source = stage_graph("family-consumer", family_preimage, Some(input));
        let consumer_graph = validate_with_manifests(
            &consumer_source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("family-consumer", consumer_graph, 27);
        let linked = ValidatedLinkedProgram::new(vec![producer, consumer]).unwrap();
        assert_eq!(
            linked.artifact_links()[0].consumer.wire_type,
            linked.artifact_links()[0].producer.wire_type
        );

        let producer_source = stage_graph(
            "family-mismatch-producer",
            WireType::Family {
                element: Box::new(WireType::Preimage(matrix_type())),
                shape: vec![IntExpr::constant(2)],
            },
            None,
        );
        let producer = stage(
            "family-mismatch-producer",
            validate(&producer_source, &ParamEnv::default()).unwrap(),
            28,
        );
        let input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let source = stage_graph(
            "family-mismatch-consumer",
            WireType::Family {
                element: Box::new(WireType::Matrix(matrix_type())),
                shape: vec![IntExpr::constant(2)],
            },
            Some(input),
        );
        let graph = validate_with_manifests(
            &source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), producer.manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("family-mismatch-consumer", graph, 29);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![producer, consumer]),
            Err(LinkedProgramError::ConcreteWireTypeMismatch { .. })
        ));
    }

    #[test]
    fn artifact_name_and_confidentiality_mismatches_are_rejected() {
        let producer_source = stage_graph("producer", WireType::Matrix(matrix_type()), None);
        let producer =
            stage("producer", validate(&producer_source, &ParamEnv::default()).unwrap(), 30);
        let mut altered_manifest = producer.manifest.clone();
        altered_manifest
            .artifacts
            .insert("other".to_owned(), altered_manifest.artifacts["out"].clone());
        let wrong_name = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "other".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let source = stage_graph("wrong-name", WireType::Matrix(matrix_type()), Some(wrong_name));
        let graph = validate_with_manifests(
            &source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), altered_manifest.clone())]),
        )
        .unwrap();
        let consumer = stage("wrong-name", graph, 31);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![producer.clone(), consumer]),
            Err(LinkedProgramError::MissingProducerArtifact { .. }) |
                Err(LinkedProgramError::MissingProducerOutput { .. })
        ));

        let mut private_manifest = producer.manifest.clone();
        private_manifest.artifacts.get_mut("out").unwrap().confidentiality =
            ArtifactConfidentiality::Private;
        let private_input = ArtifactInput {
            production_id: producer.production_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Private,
        };
        let source = stage_graph(
            "wrong-confidentiality",
            WireType::Matrix(matrix_type()),
            Some(private_input),
        );
        let graph = validate_with_manifests(
            &source,
            &ParamEnv::default(),
            &BTreeMap::from([(producer.production_id.clone(), private_manifest)]),
        )
        .unwrap();
        let consumer = stage("wrong-confidentiality", graph, 32);
        assert!(matches!(
            ValidatedLinkedProgram::new(vec![producer, consumer]),
            Err(LinkedProgramError::ManifestMetadataMismatch { .. })
        ));
    }

    fn nested_source(wire_type: WireType, artifact: Option<ArtifactInput>) -> Graph {
        let body = with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "body-input".to_owned(),
                    wire_type: wire_type.clone(),
                    artifact,
                },
                Vec::new(),
                vec![wire_type.clone()],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("artifact-body", scope, vec![input.clone()], vec![input]).unwrap()
        });
        let outer_input = NodeHandle::new(
            NodeKind::Input {
                name: "outer-input".to_owned(),
                wire_type: wire_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![wire_type.clone()],
        )
        .output(0)
        .unwrap();
        let call = NodeHandle::subgraph_call(body, vec![outer_input], Vec::new(), vec![None])
            .output(0)
            .unwrap();
        Graph::freeze(
            "nested-consumer",
            Vec::new(),
            BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value: call, confidentiality: Some(ArtifactConfidentiality::Public) },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0
    }
}
