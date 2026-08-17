//! Stable identities and job-local compact symbols for the operational checker.
//!
//! A frozen `NodeId` is unique only inside one scope definition.  The keys in
//! this module add the top-level program and the concrete call/loop occurrence
//! path, so they remain valid when definitions are reused.  `SymbolTables` is
//! the sole owner of compact job-local IDs; no lowering cache is an identity
//! authority.

use super::{
    normal_form::{FactorIdentity, ResolvedMatrixValueIdentity},
    scalar::ScalarSort,
};
use crate::{ProtocolInputId, StageId};
#[cfg(test)]
use mxx_ir_core::Port;
use mxx_ir_core::{FrozenGraphScopeId, IntExpr, NodeId, WireRef};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use std::{
    collections::{BTreeMap, HashMap, hash_map::Entry},
    hash::Hash,
};

/// The top-level executable program that owns an occurrence.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProgramKey {
    WorkflowStage(StageId),
    Ideal,
    Requirement(u32),
    Comparator,
}

/// One owner-resolved edge on a path from a top-level program into a scope.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum OccurrenceFrame {
    Call { parent: FrozenGraphScopeId, owner: NodeId },
    ParallelLoop { parent: FrozenGraphScopeId, owner: NodeId },
    SequentialLoop { parent: FrozenGraphScopeId, owner: NodeId },
}

/// One concrete use of a reusable frozen scope definition.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OccurrenceScope {
    pub program: ProgramKey,
    pub definition: FrozenGraphScopeId,
    pub path: Box<[OccurrenceFrame]>,
}

/// The producer and port of a value in one concrete occurrence scope.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WireSourceKey {
    pub scope: OccurrenceScope,
    pub wire: WireRef,
}

/// A runtime loop-index binder, identified by its introducing loop owner.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinderKey {
    pub loop_scope: OccurrenceScope,
    pub loop_node: NodeId,
    pub slot: u32,
}

/// The authoritative domain of one owner-resolved loop binder.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinderDescriptor {
    pub key: BinderKey,
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// A graph-generated source together with the binders that introduced its
/// active coordinates.  Coordinate *values* remain children of `Atom`; this
/// key intentionally records only the owners of those coordinates.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GraphWireSourceKey {
    pub wire: WireSourceKey,
    pub coordinate_binders: Box<[BinderKey]>,
}

/// The first graph occurrence that lowered a deterministic gadget
/// decomposition. This is audit-only metadata: it does not alter the matrix
/// value, so equality and hashing deliberately ignore the payload.
#[derive(Clone, Debug)]
pub struct GadgetDecompositionAuditOccurrence(pub GraphWireSourceKey);

impl From<GraphWireSourceKey> for GadgetDecompositionAuditOccurrence {
    fn from(source: GraphWireSourceKey) -> Self {
        Self(source)
    }
}

impl PartialEq for GadgetDecompositionAuditOccurrence {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for GadgetDecompositionAuditOccurrence {}

impl Hash for GadgetDecompositionAuditOccurrence {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
}

/// The source of an ordinary symbolic value.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AtomicSourceKey {
    ProtocolInput(ProtocolInputId),
    GraphWire(GraphWireSourceKey),
    /// A Graph-IR source operation whose semantics explicitly permit an
    /// unbounded public matrix.  This is distinct from an opaque computed
    /// wire, which the production bound bridge rejects.
    ExplicitLarge(GraphWireSourceKey),
    /// One symbolic carried value in a sequential-loop body.  This is never a
    /// runtime sampler: the recurrence descriptor below binds it to the
    /// previous iteration's state when the bound phase evaluates the loop.
    SequentialState(SequentialStateKey),
    Sampler(SamplerDescriptorId),
}

/// A carried-state placeholder is owned by the concrete loop occurrence and
/// its carried position, not by a local body node number.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SequentialStateKey {
    pub loop_scope: OccurrenceScope,
    pub loop_node: NodeId,
    pub carried_index: usize,
}

/// Backend convention used when a coefficient is extracted as an integer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CanonicalResidueConvention {
    Nonnegative,
    Centered,
}

/// An authoritative closed domain for an external/runtime integer atom.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IntegerSourceDomain {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

/// Closed sampler role used to construct relation provenance directly on an atomic source.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AtomicRelationRole {
    Preimage,
    GadgetDecomposition,
    DecomposedHash,
    SmallGadgetDecomposition { range_proved: bool },
    SmallDecomposedHash { range_proved: bool },
}

/// The complete descriptor for one atomic source.
///
/// Keeping the sort beside its stable key in this interner gives lowering and
/// normalization one typed source table without a second type cache.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AtomicSourceDescriptor {
    pub key: AtomicSourceKey,
    pub sort: ScalarSort,
    pub integer_domain: Option<IntegerSourceDomain>,
    pub canonical_residue_convention: Option<CanonicalResidueConvention>,
    pub relation_role: Option<AtomicRelationRole>,
}

/// The source of a trapdoor descriptor. Trapdoors are structural lowering
/// values and obey the same source identity rule as atoms.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TrapdoorSourceKey {
    ProtocolInput(ProtocolInputId),
    GraphWire(GraphWireSourceKey),
}

/// An integer expression after every loop-index slot has been resolved to its
/// owning binder.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResolvedIntExpr {
    Const(BigInt),
    Parameter(String),
    Binder(BinderKey),
    /// An owner-resolved runtime scalar source.  Unlike a parameter-shaped
    /// debug string, this retains the producer owner and ordered coordinates.
    Source {
        source: AtomicSourceKey,
        coordinates: Box<[Self]>,
    },
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    EuclideanDiv(Box<Self>, Box<Self>),
    EuclideanRemainder(Box<Self>, Box<Self>),
    RoundDiv(Box<Self>, Box<Self>),
    Log2Ceil(Box<Self>),
    ExtractMatrixCoefficient {
        matrix: Box<ResolvedMatrixValueIdentity>,
        position: Box<Self>,
    },
    /// Descriptor-local postorder form used for very deep scalar identities.
    /// Child indices are local to this immutable descriptor and never expose
    /// a scalar-store handle.
    Arena(Box<ResolvedIntExprArena>),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedIntExprArena {
    pub nodes: Box<[ResolvedIntExprArenaNode]>,
    pub root: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResolvedIntExprArenaNode {
    Const(BigInt),
    Parameter(String),
    Binder(BinderKey),
    Source { source: AtomicSourceKey, coordinates: Box<[u32]> },
    Binary { operation: ResolvedIntBinaryOperation, children: [u32; 2] },
    Log2Ceil(u32),
    ExtractMatrixCoefficient { matrix: Box<ResolvedMatrixValueIdentity>, position: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResolvedIntBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    EuclideanDiv,
    EuclideanRemainder,
    RoundDiv,
}

impl ResolvedIntExpr {
    /// Replaces only parameter and constant syntax.  A `LoopIndex` has no
    /// owner outside lowering and therefore cannot be converted here.
    pub fn from_closed_expr(value: &IntExpr) -> Option<Self> {
        match value {
            IntExpr::Const(value) => Some(Self::Const(value.clone())),
            IntExpr::Var(name) => Some(Self::Parameter(name.clone())),
            IntExpr::LoopIndex(_) => None,
            IntExpr::Add(left, right) => Some(Self::Add(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Sub(left, right) => Some(Self::Sub(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Mul(left, right) => Some(Self::Mul(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Div(left, right) => Some(Self::Div(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::RoundDiv(left, right) => Some(Self::RoundDiv(
                Box::new(Self::from_closed_expr(left)?),
                Box::new(Self::from_closed_expr(right)?),
            )),
            IntExpr::Log2Ceil(value) => {
                Some(Self::Log2Ceil(Box::new(Self::from_closed_expr(value)?)))
            }
        }
    }
}

/// Substitute one owner-resolved loop binder without consulting a global rewrite engine.
/// This is used for structural family and trapdoor instantiation; binder
/// ownership is preserved in every untouched coordinate and nested source.
pub(crate) fn substitute_resolved_int_expr(
    value: &ResolvedIntExpr,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> ResolvedIntExpr {
    match value {
        ResolvedIntExpr::Binder(candidate) if candidate == binder => replacement.clone(),
        ResolvedIntExpr::Source { source, coordinates } => ResolvedIntExpr::Source {
            source: source.clone(),
            coordinates: coordinates
                .iter()
                .map(|coordinate| substitute_resolved_int_expr(coordinate, binder, replacement))
                .collect(),
        },
        ResolvedIntExpr::Add(left, right) => ResolvedIntExpr::Add(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::Sub(left, right) => ResolvedIntExpr::Sub(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::Mul(left, right) => ResolvedIntExpr::Mul(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::Div(left, right) => ResolvedIntExpr::Div(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::EuclideanDiv(left, right) => ResolvedIntExpr::EuclideanDiv(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::EuclideanRemainder(left, right) => ResolvedIntExpr::EuclideanRemainder(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::RoundDiv(left, right) => ResolvedIntExpr::RoundDiv(
            Box::new(substitute_resolved_int_expr(left, binder, replacement)),
            Box::new(substitute_resolved_int_expr(right, binder, replacement)),
        ),
        ResolvedIntExpr::Log2Ceil(value) => ResolvedIntExpr::Log2Ceil(Box::new(
            substitute_resolved_int_expr(value, binder, replacement),
        )),
        ResolvedIntExpr::ExtractMatrixCoefficient { matrix, position } => {
            ResolvedIntExpr::ExtractMatrixCoefficient {
                matrix: Box::new(substitute_resolved_matrix_identity(matrix, binder, replacement)),
                position: Box::new(substitute_resolved_int_expr(position, binder, replacement)),
            }
        }
        ResolvedIntExpr::Const(_) | ResolvedIntExpr::Parameter(_) | ResolvedIntExpr::Binder(_) => {
            value.clone()
        }
        ResolvedIntExpr::Arena(arena) => substitute_resolved_int_arena(arena, binder, replacement)
            .expect("arena substitution root"),
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ArenaAppendKey {
    Expression(usize),
    ArenaNode(usize, u32),
}

struct ArenaAppendBuilder {
    nodes: Vec<ResolvedIntExprArenaNode>,
    completed: HashMap<ArenaAppendKey, u32>,
    interned: BTreeMap<ResolvedIntExprArenaNode, u32>,
}

/// Returns a descriptor-local arena for semantic comparison and substitution
/// helpers.  This is an internal normalization step, never a persistent
/// identity authority or a nested-expression export.
pub(crate) fn resolved_expr_as_arena(expression: &ResolvedIntExpr) -> Option<ResolvedIntExprArena> {
    let mut builder = ArenaAppendBuilder::new();
    let root = builder.append_expression(expression)?;
    Some(ResolvedIntExprArena { nodes: builder.nodes.into_boxed_slice(), root })
}

impl ArenaAppendBuilder {
    fn new() -> Self {
        Self { nodes: Vec::new(), completed: HashMap::new(), interned: BTreeMap::new() }
    }

    fn intern_node(&mut self, node: ResolvedIntExprArenaNode) -> u32 {
        if let Some(index) = self.interned.get(&node) {
            return *index;
        }
        let index = self.nodes.len() as u32;
        self.interned.insert(node.clone(), index);
        self.nodes.push(node);
        index
    }

    /// Imports an arbitrary resolved expression into one descriptor-local
    /// postorder arena.  The work list also flattens nested arenas, so a
    /// compound replacement never discards the source expression or falls
    /// back to a recursive value representation.
    fn append_expression(&mut self, expression: &ResolvedIntExpr) -> Option<u32> {
        enum Work<'a> {
            ExpressionEnter(&'a ResolvedIntExpr),
            ExpressionExit(&'a ResolvedIntExpr),
            ArenaEnter(&'a ResolvedIntExprArena, u32),
            ArenaExit(&'a ResolvedIntExprArena, u32),
            Alias(&'a ResolvedIntExpr, &'a ResolvedIntExprArena),
        }
        let expression_key = ArenaAppendKey::Expression(expression as *const _ as usize);
        if let Some(index) = self.completed.get(&expression_key) {
            return Some(*index);
        }
        let mut work = vec![Work::ExpressionEnter(expression)];
        while let Some(task) = work.pop() {
            match task {
                Work::ExpressionEnter(value) => {
                    let key = ArenaAppendKey::Expression(value as *const _ as usize);
                    if self.completed.contains_key(&key) {
                        continue;
                    }
                    if let ResolvedIntExpr::Arena(arena) = value {
                        work.push(Work::Alias(value, arena));
                        work.push(Work::ArenaEnter(arena, arena.root));
                        continue;
                    }
                    work.push(Work::ExpressionExit(value));
                    match value {
                        ResolvedIntExpr::Source { coordinates, .. } => {
                            work.extend(coordinates.iter().rev().map(Work::ExpressionEnter));
                        }
                        ResolvedIntExpr::Add(left, right) |
                        ResolvedIntExpr::Sub(left, right) |
                        ResolvedIntExpr::Mul(left, right) |
                        ResolvedIntExpr::Div(left, right) |
                        ResolvedIntExpr::EuclideanDiv(left, right) |
                        ResolvedIntExpr::EuclideanRemainder(left, right) |
                        ResolvedIntExpr::RoundDiv(left, right) => {
                            work.push(Work::ExpressionEnter(right));
                            work.push(Work::ExpressionEnter(left));
                        }
                        ResolvedIntExpr::Log2Ceil(input) => {
                            work.push(Work::ExpressionEnter(input));
                        }
                        ResolvedIntExpr::ExtractMatrixCoefficient { position, .. } => {
                            work.push(Work::ExpressionEnter(position));
                        }
                        ResolvedIntExpr::Const(_) |
                        ResolvedIntExpr::Parameter(_) |
                        ResolvedIntExpr::Binder(_) => {}
                        ResolvedIntExpr::Arena(_) => {}
                    }
                }
                Work::ExpressionExit(value) => {
                    let index = match value {
                        ResolvedIntExpr::Const(value) => self
                            .intern_node(ResolvedIntExprArenaNode::Const(value.clone()))
                            as usize,
                        ResolvedIntExpr::Parameter(value) => self
                            .intern_node(ResolvedIntExprArenaNode::Parameter(value.clone()))
                            as usize,
                        ResolvedIntExpr::Binder(value) => self
                            .intern_node(ResolvedIntExprArenaNode::Binder(value.clone()))
                            as usize,
                        ResolvedIntExpr::Source { source, coordinates } => {
                            let coordinates = coordinates
                                .iter()
                                .map(|coordinate| {
                                    self.completed
                                        .get(&ArenaAppendKey::Expression(
                                            coordinate as *const _ as usize,
                                        ))
                                        .copied()
                                })
                                .collect::<Option<Box<_>>>()
                                .expect("source expression children");
                            self.intern_node(ResolvedIntExprArenaNode::Source {
                                source: source.clone(),
                                coordinates,
                            }) as usize
                        }
                        ResolvedIntExpr::Add(left, right) |
                        ResolvedIntExpr::Sub(left, right) |
                        ResolvedIntExpr::Mul(left, right) |
                        ResolvedIntExpr::Div(left, right) |
                        ResolvedIntExpr::EuclideanDiv(left, right) |
                        ResolvedIntExpr::EuclideanRemainder(left, right) |
                        ResolvedIntExpr::RoundDiv(left, right) => {
                            let children = [
                                *self
                                    .completed
                                    .get(&ArenaAppendKey::Expression(
                                        left.as_ref() as *const _ as usize
                                    ))
                                    .expect("left expression child"),
                                *self
                                    .completed
                                    .get(&ArenaAppendKey::Expression(
                                        right.as_ref() as *const _ as usize
                                    ))
                                    .expect("right expression child"),
                            ];
                            let operation = match value {
                                ResolvedIntExpr::Add(_, _) => ResolvedIntBinaryOperation::Add,
                                ResolvedIntExpr::Sub(_, _) => ResolvedIntBinaryOperation::Sub,
                                ResolvedIntExpr::Mul(_, _) => ResolvedIntBinaryOperation::Mul,
                                ResolvedIntExpr::Div(_, _) => ResolvedIntBinaryOperation::Div,
                                ResolvedIntExpr::EuclideanDiv(_, _) => {
                                    ResolvedIntBinaryOperation::EuclideanDiv
                                }
                                ResolvedIntExpr::EuclideanRemainder(_, _) => {
                                    ResolvedIntBinaryOperation::EuclideanRemainder
                                }
                                ResolvedIntExpr::RoundDiv(_, _) => {
                                    ResolvedIntBinaryOperation::RoundDiv
                                }
                                _ => unreachable!(),
                            };
                            self.intern_node(ResolvedIntExprArenaNode::Binary {
                                operation,
                                children,
                            }) as usize
                        }
                        ResolvedIntExpr::Log2Ceil(input) => {
                            let input = *self
                                .completed
                                .get(&ArenaAppendKey::Expression(
                                    input.as_ref() as *const _ as usize
                                ))
                                .expect("log2 expression child");
                            self.intern_node(ResolvedIntExprArenaNode::Log2Ceil(input)) as usize
                        }
                        ResolvedIntExpr::ExtractMatrixCoefficient { matrix, position } => {
                            let position = *self
                                .completed
                                .get(&ArenaAppendKey::Expression(
                                    position.as_ref() as *const _ as usize
                                ))
                                .expect("extract expression child");
                            self.intern_node(ResolvedIntExprArenaNode::ExtractMatrixCoefficient {
                                matrix: matrix.clone(),
                                position,
                            }) as usize
                        }
                        ResolvedIntExpr::Arena(_) => return None,
                    } as u32;
                    self.completed
                        .insert(ArenaAppendKey::Expression(value as *const _ as usize), index);
                }
                Work::ArenaEnter(arena, index) => {
                    let key = ArenaAppendKey::ArenaNode(arena as *const _ as usize, index);
                    if self.completed.contains_key(&key) {
                        continue;
                    }
                    work.push(Work::ArenaExit(arena, index));
                    match arena.nodes.get(index as usize)? {
                        ResolvedIntExprArenaNode::Source { coordinates, .. } => work.extend(
                            coordinates.iter().rev().map(|child| Work::ArenaEnter(arena, *child)),
                        ),
                        ResolvedIntExprArenaNode::Binary { children, .. } => {
                            work.push(Work::ArenaEnter(arena, children[1]));
                            work.push(Work::ArenaEnter(arena, children[0]));
                        }
                        ResolvedIntExprArenaNode::Log2Ceil(child) => {
                            work.push(Work::ArenaEnter(arena, *child));
                        }
                        ResolvedIntExprArenaNode::ExtractMatrixCoefficient { position, .. } => {
                            work.push(Work::ArenaEnter(arena, *position));
                        }
                        ResolvedIntExprArenaNode::Const(_) |
                        ResolvedIntExprArenaNode::Parameter(_) |
                        ResolvedIntExprArenaNode::Binder(_) => {}
                    }
                }
                Work::ArenaExit(arena, index) => {
                    let source = arena.nodes.get(index as usize)?;
                    let node = match source {
                        ResolvedIntExprArenaNode::Const(value) => {
                            ResolvedIntExprArenaNode::Const(value.clone())
                        }
                        ResolvedIntExprArenaNode::Parameter(value) => {
                            ResolvedIntExprArenaNode::Parameter(value.clone())
                        }
                        ResolvedIntExprArenaNode::Binder(value) => {
                            ResolvedIntExprArenaNode::Binder(value.clone())
                        }
                        ResolvedIntExprArenaNode::Source { source, coordinates } => {
                            ResolvedIntExprArenaNode::Source {
                                source: source.clone(),
                                coordinates: coordinates
                                    .iter()
                                    .map(|child| {
                                        self.completed
                                            .get(&ArenaAppendKey::ArenaNode(
                                                arena as *const _ as usize,
                                                *child,
                                            ))
                                            .copied()
                                    })
                                    .collect::<Option<Box<_>>>()?,
                            }
                        }
                        ResolvedIntExprArenaNode::Binary { operation, children } => {
                            ResolvedIntExprArenaNode::Binary {
                                operation: *operation,
                                children: [
                                    *self.completed.get(&ArenaAppendKey::ArenaNode(
                                        arena as *const _ as usize,
                                        children[0],
                                    ))?,
                                    *self.completed.get(&ArenaAppendKey::ArenaNode(
                                        arena as *const _ as usize,
                                        children[1],
                                    ))?,
                                ],
                            }
                        }
                        ResolvedIntExprArenaNode::Log2Ceil(child) => {
                            ResolvedIntExprArenaNode::Log2Ceil(*self.completed.get(
                                &ArenaAppendKey::ArenaNode(arena as *const _ as usize, *child),
                            )?)
                        }
                        ResolvedIntExprArenaNode::ExtractMatrixCoefficient { matrix, position } => {
                            ResolvedIntExprArenaNode::ExtractMatrixCoefficient {
                                matrix: matrix.clone(),
                                position: *self.completed.get(&ArenaAppendKey::ArenaNode(
                                    arena as *const _ as usize,
                                    *position,
                                ))?,
                            }
                        }
                    };
                    let output = self.intern_node(node);
                    self.completed.insert(
                        ArenaAppendKey::ArenaNode(arena as *const _ as usize, index),
                        output,
                    );
                }
                Work::Alias(expression, arena) => {
                    let index = *self
                        .completed
                        .get(&ArenaAppendKey::ArenaNode(arena as *const _ as usize, arena.root))?;
                    self.completed
                        .insert(ArenaAppendKey::Expression(expression as *const _ as usize), index);
                }
            }
        }
        self.completed.get(&expression_key).copied()
    }
}

fn substitute_resolved_int_arena(
    arena: &ResolvedIntExprArena,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> Option<ResolvedIntExpr> {
    let mut builder = ArenaAppendBuilder::new();
    let mut mapped = vec![None; arena.nodes.len()];
    let mut work = vec![(arena.root, false)];
    while let Some((index, expanded)) = work.pop() {
        if mapped[index as usize].is_some() {
            continue;
        }
        let node = &arena.nodes[index as usize];
        if !expanded {
            work.push((index, true));
            match node {
                ResolvedIntExprArenaNode::Source { coordinates, .. } => {
                    work.extend(coordinates.iter().rev().map(|child| (*child, false)));
                }
                ResolvedIntExprArenaNode::Binary { children, .. } => {
                    work.push((children[1], false));
                    work.push((children[0], false));
                }
                ResolvedIntExprArenaNode::Log2Ceil(child) => work.push((*child, false)),
                ResolvedIntExprArenaNode::ExtractMatrixCoefficient { position, .. } => {
                    work.push((*position, false));
                }
                ResolvedIntExprArenaNode::Const(_) |
                ResolvedIntExprArenaNode::Parameter(_) |
                ResolvedIntExprArenaNode::Binder(_) => {}
            }
            continue;
        }
        let output = match node {
            ResolvedIntExprArenaNode::Binder(candidate) if candidate == binder => builder
                .append_expression(replacement)
                .expect("replacement expression can be represented"),
            ResolvedIntExprArenaNode::Const(value) => {
                builder.intern_node(ResolvedIntExprArenaNode::Const(value.clone()))
            }
            ResolvedIntExprArenaNode::Parameter(value) => {
                builder.intern_node(ResolvedIntExprArenaNode::Parameter(value.clone()))
            }
            ResolvedIntExprArenaNode::Binder(value) => {
                builder.intern_node(ResolvedIntExprArenaNode::Binder(value.clone()))
            }
            ResolvedIntExprArenaNode::Source { source, coordinates } => {
                builder.intern_node(ResolvedIntExprArenaNode::Source {
                    source: source.clone(),
                    coordinates: coordinates
                        .iter()
                        .map(|child| mapped[*child as usize])
                        .collect::<Option<Box<_>>>()?,
                })
            }
            ResolvedIntExprArenaNode::Binary { operation, children } => {
                builder.intern_node(ResolvedIntExprArenaNode::Binary {
                    operation: *operation,
                    children: [mapped[children[0] as usize]?, mapped[children[1] as usize]?],
                })
            }
            ResolvedIntExprArenaNode::Log2Ceil(child) => {
                builder.intern_node(ResolvedIntExprArenaNode::Log2Ceil(mapped[*child as usize]?))
            }
            ResolvedIntExprArenaNode::ExtractMatrixCoefficient { matrix, position } => builder
                .intern_node(ResolvedIntExprArenaNode::ExtractMatrixCoefficient {
                    matrix: Box::new(substitute_resolved_matrix_identity(
                        matrix,
                        binder,
                        replacement,
                    )),
                    position: mapped[*position as usize]?,
                }),
        };
        mapped[index as usize] = Some(output);
    }
    Some(ResolvedIntExpr::Arena(Box::new(ResolvedIntExprArena {
        nodes: builder.nodes.into_boxed_slice(),
        root: mapped[arena.root as usize]?,
    })))
}

pub(crate) fn substitute_resolved_matrix_identity(
    value: &ResolvedMatrixValueIdentity,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> ResolvedMatrixValueIdentity {
    use super::{normal_form::MatrixValueOperation, normal_form_ops::ScaleScalar};
    let substitute_range = |range: &ResolvedIndexRange| ResolvedIndexRange {
        start: substitute_resolved_int_expr(&range.start, binder, replacement),
        end: substitute_resolved_int_expr(&range.end, binder, replacement),
    };
    let substitute_slice = |spec: &SliceSpec| SliceSpec {
        rows: spec.rows.as_ref().map(substitute_range),
        columns: spec.columns.as_ref().map(substitute_range),
    };
    let substitute_factor =
        |factor: &FactorIdentity| substitute_factor_identity(factor, binder, replacement);
    let operation = |operation: &MatrixValueOperation| match operation {
        MatrixValueOperation::MatrixScale {
            scalar: ScaleScalar::Exact { key, value, matrix_type },
        } => MatrixValueOperation::MatrixScale {
            scalar: ScaleScalar::Exact {
                key: substitute_factor(key),
                value: value.clone(),
                matrix_type: matrix_type.clone(),
            },
        },
        MatrixValueOperation::Slice { spec } => {
            MatrixValueOperation::Slice { spec: substitute_slice(spec) }
        }
        MatrixValueOperation::CrtRecompose { spec, output_type } => {
            MatrixValueOperation::CrtRecompose {
                spec: CrtSpec {
                    plaintext_moduli: spec
                        .plaintext_moduli
                        .iter()
                        .map(|value| substitute_resolved_int_expr(value, binder, replacement))
                        .collect(),
                    reconstruction_coefficients: spec
                        .reconstruction_coefficients
                        .iter()
                        .map(|value| substitute_resolved_int_expr(value, binder, replacement))
                        .collect(),
                },
                output_type: output_type.clone(),
            }
        }
        MatrixValueOperation::View { view, output_type } => {
            let view = match view {
                super::normal_form_ops::ViewSpec::CoefficientPreserving {
                    view: super::normal_form_ops::CoefficientPreservingView::Slice(spec),
                } => super::normal_form_ops::ViewSpec::CoefficientPreserving {
                    view: super::normal_form_ops::CoefficientPreservingView::Slice(
                        substitute_slice(spec),
                    ),
                },
                other => other.clone(),
            };
            MatrixValueOperation::View { view, output_type: output_type.clone() }
        }
        other => other.clone(),
    };
    ResolvedMatrixValueIdentity {
        nodes: value
            .nodes
            .iter()
            .map(|node| super::normal_form::ResolvedMatrixValueIdentityNode {
                operation: operation(&node.operation),
                children: node.children.clone(),
                owner: node.owner.as_ref().map(substitute_factor),
                selector: node.selector.as_ref().map(substitute_factor),
            })
            .collect(),
        root: value.root,
    }
}

fn substitute_factor_identity(
    value: &FactorIdentity,
    binder: &BinderKey,
    replacement: &ResolvedIntExpr,
) -> FactorIdentity {
    use super::normal_form::{FactorLayoutIdentity, FactorOwner};
    let owner = match &value.owner {
        FactorOwner::Scalar(identity) => {
            FactorOwner::Scalar(substitute_resolved_int_expr(identity, binder, replacement))
        }
        FactorOwner::HashPlain { query, arguments } => FactorOwner::HashPlain {
            query: Box::new(substitute_factor_identity(query, binder, replacement)),
            arguments: arguments
                .iter()
                .map(|argument| match argument {
                    super::normal_form::HashPlainArgumentIdentity::Exact(factor) => {
                        super::normal_form::HashPlainArgumentIdentity::Exact(
                            substitute_factor_identity(factor, binder, replacement),
                        )
                    }
                    other => other.clone(),
                })
                .collect(),
        },
        FactorOwner::Derived { parent, tag } => FactorOwner::Derived {
            parent: Box::new(substitute_factor_identity(parent, binder, replacement)),
            tag: tag.clone(),
        },
        other => other.clone(),
    };
    FactorIdentity {
        owner,
        kind: value.kind.clone(),
        port: value.port,
        coordinates: value
            .coordinates
            .iter()
            .map(|(owner, identity)| {
                (owner.clone(), substitute_resolved_int_expr(identity, binder, replacement))
            })
            .collect(),
        public: value.public.clone(),
        layout: value.layout.as_ref().map(|layout| FactorLayoutIdentity {
            matrix: ResolvedMatrixType {
                modulus: substitute_resolved_int_expr(&layout.matrix.modulus, binder, replacement),
                ring_dimension: substitute_resolved_int_expr(
                    &layout.matrix.ring_dimension,
                    binder,
                    replacement,
                ),
                rows: substitute_resolved_int_expr(&layout.matrix.rows, binder, replacement),
                columns: substitute_resolved_int_expr(&layout.matrix.columns, binder, replacement),
            },
            view: layout.view.as_ref().map(|view| SliceSpec {
                rows: view.rows.as_ref().map(|range| ResolvedIndexRange {
                    start: substitute_resolved_int_expr(&range.start, binder, replacement),
                    end: substitute_resolved_int_expr(&range.end, binder, replacement),
                }),
                columns: view.columns.as_ref().map(|range| ResolvedIndexRange {
                    start: substitute_resolved_int_expr(&range.start, binder, replacement),
                    end: substitute_resolved_int_expr(&range.end, binder, replacement),
                }),
            }),
        }),
        selector: value
            .selector
            .as_ref()
            .map(|selector| Box::new(substitute_factor_identity(selector, binder, replacement))),
        trapdoor: value.trapdoor.clone(),
        selector_mapping: value.selector_mapping.clone(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedMatrixType {
    pub modulus: ResolvedIntExpr,
    pub ring_dimension: ResolvedIntExpr,
    pub rows: ResolvedIntExpr,
    pub columns: ResolvedIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResolvedIndexRange {
    pub start: ResolvedIntExpr,
    pub end: ResolvedIntExpr,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SliceSpec {
    pub rows: Option<ResolvedIndexRange>,
    pub columns: Option<ResolvedIndexRange>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Axis {
    Rows,
    Columns,
    Diagonal,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CrtSpec {
    pub plaintext_moduli: Box<[ResolvedIntExpr]>,
    pub reconstruction_coefficients: Box<[ResolvedIntExpr]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HashTagPart {
    Literal(Box<[u8]>),
    BinaryStatic(ResolvedIntExpr),
    DecimalStatic(ResolvedIntExpr),
    U64LeStatic(ResolvedIntExpr),
    BinaryArgument { argument: u16 },
    DecimalArgument { argument: u16 },
    U64LeArgument { argument: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct HashQuerySpec {
    pub matrix_type: ResolvedMatrixType,
    pub tag_program: Box<[HashTagPart]>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum MatrixConstantValue {
    Zero,
    Identity,
    UnitRow { index: ResolvedIntExpr },
    UnitColumn { index: ResolvedIntExpr },
    Gadget { base: ResolvedIntExpr, small: bool },
    PowerOfBase { base: ResolvedIntExpr, exponent: ResolvedIntExpr },
    Rotation { exponent: ResolvedIntExpr },
    Polynomial { coefficients: Box<[ResolvedIntExpr]> },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MatrixConstantSpec {
    pub matrix_type: ResolvedMatrixType,
    pub value: MatrixConstantValue,
}

impl MatrixConstantSpec {
    /// Computes the reviewed canonical maximum through the sole constant descriptor helper.
    /// Variants whose runtime layout is not encoded by this descriptor return no contract.
    pub fn canonical_coefficient_exclusive_upper(&self) -> Option<BigUint> {
        let ResolvedIntExpr::Const(modulus) = &self.matrix_type.modulus else { return None };
        let modulus = modulus.to_biguint()?;
        if modulus.is_zero() {
            return None;
        }
        let canonical = |value: &BigInt| canonical_nonnegative_residue(value, &modulus);
        let upper = |values: Vec<BigInt>| {
            values
                .iter()
                .map(canonical)
                .max()
                .and_then(|maximum| (maximum + BigInt::one()).to_biguint())
        };
        match &self.value {
            MatrixConstantValue::Zero => Some(BigUint::one()),
            MatrixConstantValue::Identity |
            MatrixConstantValue::UnitRow { .. } |
            MatrixConstantValue::UnitColumn { .. } => upper(vec![0.into(), 1.into()]),
            MatrixConstantValue::Gadget { .. } => None,
            MatrixConstantValue::PowerOfBase { base, exponent } => {
                let (ResolvedIntExpr::Const(base), ResolvedIntExpr::Const(exponent)) =
                    (base, exponent)
                else {
                    return None;
                };
                canonical_power_residue(base, exponent, &modulus)
                    .map(|residue| residue + BigUint::one())
            }
            MatrixConstantValue::Rotation { .. } => Some(modulus),
            MatrixConstantValue::Polynomial { coefficients } => coefficients
                .iter()
                .map(|coefficient| match coefficient {
                    ResolvedIntExpr::Const(value) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(upper),
        }
    }
}

/// Shared checker-private canonicalization used by constant analysis and, in
/// later stages, by the runtime bound evaluator. Keeping this arithmetic here
/// prevents descriptor analysis from inventing a second residue convention.
pub(crate) fn canonical_nonnegative_residue(value: &BigInt, modulus: &BigUint) -> BigInt {
    let modulus = BigInt::from(modulus.clone());
    ((value % &modulus) + &modulus) % &modulus
}

/// The canonical small-decomposition contract: every coefficient is below `upper`,
/// and the producer's closed small-digit limit is `limit`.
pub fn canonical_range_within_limit(upper: Option<&BigUint>, limit: &BigUint) -> bool {
    upper.is_some_and(|upper| upper <= limit)
}

/// Computes `base^exponent mod modulus` without first allocating `base^exponent`.
///
/// The exponent bit guard is a checker resource boundary. The Stage 7 bound
/// evaluator must use this same helper when it gains matrix-constant support.
pub(crate) fn canonical_power_residue(
    base: &BigInt,
    exponent: &BigInt,
    modulus: &BigUint,
) -> Option<BigUint> {
    const MAX_EXPONENT_BITS: u64 = 4_096;
    let exponent = exponent.to_biguint()?;
    if exponent.bits() > MAX_EXPONENT_BITS || modulus.is_zero() {
        return None;
    }
    let base = canonical_nonnegative_residue(base, modulus).to_biguint()?;
    Some(base.modpow(&exponent, modulus))
}

macro_rules! compact_id {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(pub u32);
    )+};
}

compact_id!(
    AtomicSourceId,
    BinderId,
    TrapdoorDescriptorId,
    MatrixConstantSpecId,
    SliceSpecId,
    HashQuerySpecId,
    CrtSpecId,
    SamplerDescriptorId,
);

/// A canonical reference to an operand carried by a sampler descriptor.
///
/// Sampler metadata must not retain a job-local scalar handle: such numbering is
/// local to one lowering run and is not an owner-aware identity.
/// Production descriptors use a [`FactorIdentity`] for an exact atom or a
/// stable graph source when the operand is only needed as provenance.  The
/// two cases are explicit so a caller cannot silently substitute one kind for
/// another or cross a scalar/DAG arena boundary.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CanonicalTermIdentity {
    Factor(FactorIdentity),
    Source(GraphWireSourceKey),
}

/// A source-level sampler record retains typed, owner-aware operand identities
/// for a later relation pass; lowering never guesses a relation from a matrix
/// shape or a job-local scalar handle.
///
/// A `GadgetDecomposition` is deterministic, so its graph-wire occurrence is
/// audit metadata rather than part of the produced value's identity.  The
/// interner retains the first occurrence only for diagnostics.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum SamplerIdentity {
    /// A non-relation sampler with an explicit, nonnegative coefficient cap.
    Gaussian {
        source: GraphWireSourceKey,
        indices: Box<[ResolvedIntExpr]>,
        max_coefficient_bound: ResolvedIntExpr,
    },
    /// A non-relation sampler with an explicit closed integer interval.
    UniformInterval {
        source: GraphWireSourceKey,
        indices: Box<[ResolvedIntExpr]>,
        minimum: ResolvedIntExpr,
        maximum: ResolvedIntExpr,
    },
    Preimage {
        source: GraphWireSourceKey,
        indices: Box<[ResolvedIntExpr]>,
        public: CanonicalTermIdentity,
        trapdoor: TrapdoorDescriptorId,
        target: CanonicalTermIdentity,
        cutoff: ResolvedIntExpr,
    },
    /// A decomposed hash sampler is registered against its exact gadget and
    /// plain-hash typed identities.  The ordered hash arguments include the key and
    /// every runtime tag integer, so equal shapes never substitute identities.
    DecomposedHash {
        source: GraphWireSourceKey,
        indices: Box<[ResolvedIntExpr]>,
        public: CanonicalTermIdentity,
        target: CanonicalTermIdentity,
        arguments: Box<[CanonicalTermIdentity]>,
        matrix_type: ResolvedMatrixType,
        base: ResolvedIntExpr,
        digit_count: ResolvedIntExpr,
        small: bool,
        range_proved: bool,
    },
    GadgetDecomposition {
        source: GadgetDecompositionAuditOccurrence,
        indices: Box<[ResolvedIntExpr]>,
        public: CanonicalTermIdentity,
        target: CanonicalTermIdentity,
        base: ResolvedIntExpr,
        digit_count: ResolvedIntExpr,
        small: bool,
        range_proved: bool,
    },
}

/// A trapdoor is a structural lowering value, not a scalar node. Its
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TrapdoorIdentity {
    pub source: TrapdoorSourceKey,
    pub indices: Box<[ResolvedIntExpr]>,
    pub matrix_type: ResolvedMatrixType,
    /// Stable identity of the public matrix input.  Trapdoor construction must
    /// not materialize a scalar node merely to populate metadata.
    pub public: CanonicalTermIdentity,
    pub sigma_bits: u64,
    pub gadget_base: ResolvedIntExpr,
    pub digit_count: ResolvedIntExpr,
    pub preimage_cutoff: ResolvedIntExpr,
}

/// A single job-local, amortized-O(1) stable-value-to-compact-ID map.
///
/// The public fields make the one owner inspectable, but callers should use
/// `intern` so an equal stable value always receives exactly one compact ID.
#[derive(Clone, Debug)]
pub struct Interner<T> {
    pub values: Vec<T>,
    pub by_value: HashMap<T, u32>,
}

impl<T> Default for Interner<T>
where
    T: Eq + Hash,
{
    fn default() -> Self {
        Self { values: Vec::new(), by_value: HashMap::new() }
    }
}

impl<T> Interner<T>
where
    T: Clone + Eq + Hash,
{
    pub fn intern(&mut self, value: T) -> u32 {
        let next = u32::try_from(self.values.len()).expect("too many operational symbols");
        match self.by_value.entry(value.clone()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                self.values.push(value);
                entry.insert(next);
                next
            }
        }
    }

    pub fn get(&self, id: u32) -> Option<&T> {
        self.values.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The sole owner of job-local identity descriptors for one lowering job.
///
/// Relation provenance and integer ranges deliberately do not appear here:
/// they are transfer facts owned by the scalar store, not a second identity cache.
#[derive(Clone, Debug, Default)]
pub struct SymbolTables {
    pub atomic_sources: Interner<AtomicSourceDescriptor>,
    pub binders: Interner<BinderDescriptor>,
    /// Closed parameter values used only if an intrinsic `IntParameter` node is constructed.
    /// Normal lowering emits `IntConst` after request closure.
    pub integer_parameters: BTreeMap<String, BigInt>,
    pub trapdoors: Interner<TrapdoorIdentity>,
    pub samplers: Interner<SamplerIdentity>,
    pub matrix_constants: Interner<MatrixConstantSpec>,
    pub slices: Interner<SliceSpec>,
    pub hash_queries: Interner<HashQuerySpec>,
    pub crts: Interner<CrtSpec>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_small_range_accepts_equality_and_strictly_smaller_only() {
        let limit = BigUint::from(8_u8);
        assert!(canonical_range_within_limit(Some(&BigUint::from(8_u8)), &limit));
        assert!(canonical_range_within_limit(Some(&BigUint::from(7_u8)), &limit));
        assert!(!canonical_range_within_limit(Some(&BigUint::from(9_u8)), &limit));
        assert!(!canonical_range_within_limit(None, &limit));
    }

    fn scope(program: ProgramKey) -> OccurrenceScope {
        OccurrenceScope { program, definition: FrozenGraphScopeId::Root, path: Box::new([]) }
    }

    #[test]
    fn program_key_keeps_equal_node_numbers_in_separate_stages_distinct() {
        let first = WireSourceKey {
            scope: scope(ProgramKey::WorkflowStage(StageId("encrypt".to_owned()))),
            wire: WireRef { node: NodeId(4), port: Port(0) },
        };
        let second = WireSourceKey {
            scope: scope(ProgramKey::WorkflowStage(StageId("decrypt".to_owned()))),
            wire: WireRef { node: NodeId(4), port: Port(0) },
        };

        assert_ne!(first, second);
    }

    #[test]
    fn call_and_loop_owners_make_occurrences_and_binders_distinct() {
        let definition = FrozenGraphScopeId::Subgraph { canonical_name: "shared".to_owned() };
        let first_scope = OccurrenceScope {
            program: ProgramKey::WorkflowStage(StageId("decode".to_owned())),
            definition: definition.clone(),
            path: Box::new([OccurrenceFrame::Call {
                parent: FrozenGraphScopeId::Root,
                owner: NodeId(20),
            }]),
        };
        let second_scope = OccurrenceScope {
            program: ProgramKey::WorkflowStage(StageId("decode".to_owned())),
            definition,
            path: Box::new([OccurrenceFrame::Call {
                parent: FrozenGraphScopeId::Root,
                owner: NodeId(50),
            }]),
        };
        assert_ne!(first_scope, second_scope);

        let first = BinderKey { loop_scope: first_scope, loop_node: NodeId(7), slot: 0 };
        let second = BinderKey { loop_scope: second_scope, loop_node: NodeId(7), slot: 0 };
        let nested_slot = BinderKey {
            loop_scope: scope(ProgramKey::WorkflowStage(StageId("decode".to_owned()))),
            loop_node: NodeId(7),
            slot: 1,
        };
        assert_ne!(first, second);
        assert_ne!(first, nested_slot);
    }

    #[test]
    fn same_scope_slot_zero_and_one_are_distinct_binders() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let slot_zero = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(7), slot: 0 };
        let slot_one = BinderKey { loop_scope, loop_node: NodeId(7), slot: 1 };

        assert_ne!(slot_zero, slot_one);
    }

    #[test]
    fn different_loop_owners_with_slot_zero_are_distinct_binders() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let outer = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(7), slot: 0 };
        let inner = BinderKey { loop_scope, loop_node: NodeId(19), slot: 0 };

        assert_ne!(outer, inner);
    }

    #[test]
    fn graph_wire_source_keeps_coordinate_owner_without_coordinate_value() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let binder = BinderKey { loop_scope: loop_scope.clone(), loop_node: NodeId(9), slot: 0 };
        let source = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: loop_scope,
                wire: WireRef { node: NodeId(11), port: Port(0) },
            },
            coordinate_binders: Box::new([binder]),
        };
        assert_eq!(source.coordinate_binders.len(), 1);
    }

    #[test]
    fn nonrelation_sampler_identity_includes_its_resolved_contract() {
        let source = GraphWireSourceKey {
            wire: WireSourceKey {
                scope: scope(ProgramKey::WorkflowStage(StageId("encrypt".to_owned()))),
                wire: WireRef { node: NodeId(11), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let first = SamplerIdentity::Gaussian {
            source: source.clone(),
            indices: Box::new([]),
            max_coefficient_bound: ResolvedIntExpr::Const(3.into()),
        };
        let different_bound = SamplerIdentity::Gaussian {
            source: source.clone(),
            indices: Box::new([]),
            max_coefficient_bound: ResolvedIntExpr::Const(4.into()),
        };
        let interval = SamplerIdentity::UniformInterval {
            source,
            indices: Box::new([]),
            minimum: ResolvedIntExpr::Const((-3).into()),
            maximum: ResolvedIntExpr::Const(2.into()),
        };
        let mut interner = Interner::default();
        let first_id = interner.intern(first);
        assert_ne!(first_id, interner.intern(different_bound));
        assert_ne!(first_id, interner.intern(interval));
    }

    #[test]
    fn deterministic_gadget_decomposition_interns_by_semantics_not_occurrence() {
        let occurrence = |node| GraphWireSourceKey {
            wire: WireSourceKey {
                scope: scope(ProgramKey::WorkflowStage(StageId("encrypt".to_owned()))),
                wire: WireRef { node: NodeId(node), port: Port(0) },
            },
            coordinate_binders: Box::new([]),
        };
        let decomposition = |source: GraphWireSourceKey,
                             public: usize,
                             target: usize,
                             base: i64,
                             digit_count: i64,
                             small: bool,
                             indices: Box<[usize]>,
                             range_proved: bool| {
            SamplerIdentity::GadgetDecomposition {
                source: source.into(),
                indices: indices
                    .iter()
                    .map(|index| ResolvedIntExpr::Const(BigInt::from(*index)))
                    .collect(),
                public: CanonicalTermIdentity::Source(occurrence(public as u64)),
                target: CanonicalTermIdentity::Source(occurrence(target as u64)),
                base: ResolvedIntExpr::Const(base.into()),
                digit_count: ResolvedIntExpr::Const(digit_count.into()),
                small,
                range_proved,
            }
        };
        let first = decomposition(occurrence(11), 7, 9, 32, 2, false, vec![3].into(), false);
        let same_value_other_occurrence =
            decomposition(occurrence(12), 7, 9, 32, 2, false, vec![3].into(), false);
        let different_target =
            decomposition(occurrence(12), 7, 10, 32, 2, false, vec![3].into(), false);
        let different_base =
            decomposition(occurrence(12), 7, 9, 64, 2, false, vec![3].into(), false);
        let different_digit_count =
            decomposition(occurrence(12), 7, 9, 32, 3, false, vec![3].into(), false);
        let different_small =
            decomposition(occurrence(12), 7, 9, 32, 2, true, vec![3].into(), false);
        let different_indices =
            decomposition(occurrence(12), 7, 9, 32, 2, false, vec![4].into(), false);
        let different_range_proved =
            decomposition(occurrence(12), 7, 9, 32, 2, false, vec![3].into(), true);

        let mut samplers = Interner::default();
        let first_id = samplers.intern(first);
        assert_eq!(first_id, samplers.intern(same_value_other_occurrence));
        assert_ne!(first_id, samplers.intern(different_target));
        assert_ne!(first_id, samplers.intern(different_base));
        assert_ne!(first_id, samplers.intern(different_digit_count));
        assert_ne!(first_id, samplers.intern(different_small));
        assert_ne!(first_id, samplers.intern(different_indices));
        assert_ne!(first_id, samplers.intern(different_range_proved));
        assert_eq!(samplers.len(), 7);

        let SamplerIdentity::GadgetDecomposition { source, .. } =
            &samplers.values[first_id as usize]
        else {
            panic!("first sampler is a gadget decomposition")
        };
        assert_eq!(
            source.0.wire.wire.node,
            NodeId(11),
            "the retained occurrence is diagnostic only"
        );
    }

    #[test]
    fn protocol_input_identity_is_shared_across_programs() {
        let input = ProtocolInputId::from("hash-key");
        assert_eq!(
            AtomicSourceKey::ProtocolInput(input.clone()),
            AtomicSourceKey::ProtocolInput(input)
        );
    }

    #[test]
    fn sequential_state_and_result_keep_loop_owner_and_carried_position() {
        let loop_scope = scope(ProgramKey::WorkflowStage(StageId("decode".to_owned())));
        let first_state = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope: loop_scope.clone(),
            loop_node: NodeId(17),
            carried_index: 0,
        });
        let second_state = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope: loop_scope.clone(),
            loop_node: NodeId(17),
            carried_index: 1,
        });
        let other_loop = AtomicSourceKey::SequentialState(SequentialStateKey {
            loop_scope,
            loop_node: NodeId(18),
            carried_index: 0,
        });
        assert_ne!(first_state, second_state);
        assert_ne!(first_state, other_loop);
    }

    #[test]
    fn interner_reuses_equal_stable_values() {
        let mut interner = Interner::default();
        let first = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("key")),
            sort: ScalarSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let second = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("key")),
            sort: ScalarSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });
        let third = interner.intern(AtomicSourceDescriptor {
            key: AtomicSourceKey::ProtocolInput(ProtocolInputId::from("other")),
            sort: ScalarSort::Int,
            integer_domain: Some(IntegerSourceDomain { minimum: 0.into(), maximum: 7.into() }),
            canonical_residue_convention: None,
            relation_role: None,
        });

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn unresolved_loop_slot_cannot_be_mistaken_for_an_owned_expression() {
        assert!(ResolvedIntExpr::from_closed_expr(&IntExpr::LoopIndex(0)).is_none());
        assert_eq!(
            ResolvedIntExpr::from_closed_expr(&IntExpr::constant(3)),
            Some(ResolvedIntExpr::Const(BigInt::from(3)))
        );
    }

    #[test]
    fn power_constant_uses_bounded_modular_exponentiation() {
        let spec = MatrixConstantSpec {
            matrix_type: ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(1.into()),
            },
            value: MatrixConstantValue::PowerOfBase {
                base: ResolvedIntExpr::Const(3.into()),
                exponent: ResolvedIntExpr::Const(1_000_000.into()),
            },
        };
        assert_eq!(spec.canonical_coefficient_exclusive_upper(), Some(BigUint::from(2_u8)));

        let oversized = MatrixConstantSpec {
            value: MatrixConstantValue::PowerOfBase {
                base: ResolvedIntExpr::Const(3.into()),
                exponent: ResolvedIntExpr::Const(BigInt::one() << 4_096_usize),
            },
            ..spec
        };
        assert_eq!(oversized.canonical_coefficient_exclusive_upper(), None);
    }

    #[test]
    fn gadget_without_closed_layout_has_no_canonical_upper_contract() {
        let spec = MatrixConstantSpec {
            matrix_type: ResolvedMatrixType {
                modulus: ResolvedIntExpr::Const(17.into()),
                ring_dimension: ResolvedIntExpr::Const(1.into()),
                rows: ResolvedIntExpr::Const(1.into()),
                columns: ResolvedIntExpr::Const(2.into()),
            },
            value: MatrixConstantValue::Gadget {
                base: ResolvedIntExpr::Const(2.into()),
                small: false,
            },
        };
        assert_eq!(spec.canonical_coefficient_exclusive_upper(), None);
    }
}
