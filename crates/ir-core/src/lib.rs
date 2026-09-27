//! Executable typed graph IR for lattice-cryptography computations.
//!
//! This crate owns executable graph structure, compile expressions, rings and CRT bases, concrete
//! type validation, canonical identities, artifact manifests, protocol declarations, and Lean
//! export. It depends on no other workspace crate: `mxx-dsl` builds its graphs and `mxx-backends`
//! executes them.
//!
//! A graph goes through three stages. Construction code creates immutable node handles
//! ([`graph`]); [`graph::Graph::freeze`] keeps the reachable nodes, one scope per body.
//! [`validate::validate`] then resolves parameters, rings, and concrete wire types under a
//! [`ParamEnv`] and returns a [`ValidatedGraph`], which executors and the Lean exporter consume.

pub mod artifact;
pub mod checks;
pub mod constraints;
pub mod encoding;
pub mod expr;
pub mod graph;
pub mod inventory;
pub mod lean;
pub mod node;
pub mod protocol;
pub mod ring;
mod serde_support;
pub mod types;
pub mod validate;

pub use constraints::{ParamConstraint, derive_param_constraints};
pub use expr::{IntExpr, ParamEnv, Rational, RealExpr};
pub use graph::{
    BenchmarkRole, CapturePolicy, CapturedValue, CompileParameter, CompileParameterKind,
    ConstructionScopeId, FreezeError, FreezeMap, FreezeResolveError, FrozenGraphScopeId, Graph,
    GraphOutput, GraphScope, NodeHandle, OutputRoot, ScopedWireRef, SealMap, SealedSubgraph,
    SourceLocation, SubgraphHandle, ValueHandle, current_construction_scope, with_benchmark_role,
    with_new_construction_scope,
};
pub use ring::{ConcreteRing, RingExpr, RingRef, generate_crt_basis};
pub use types::{NodeId, Port, WireRef, WireType};
pub use validate::{
    IntoValidatedGraph, LivenessSchedule, ValidatedGraph, ValidatedScope, ValidationError,
    concretize_wire_type, validate, validate_structure, validate_with_manifests,
};
