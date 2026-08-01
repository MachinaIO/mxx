//! Core typed graph IR for lattice-cryptography computations.
//!
//! This crate owns executable graph structure, compile expressions, concrete
//! type validation, canonical identities, and runtime artifact metadata.
//! Optional symbolic-term operations live in `mxx-ir-symbolic`.

pub mod artifact;
pub mod checks;
pub mod encoding;
pub mod expr;
pub mod graph;
pub mod node;
mod serde_support;
pub mod types;
pub mod validate;

pub use expr::{IntExpr, ParamEnv, Rational, RealExpr};
pub use graph::{
    CapturePolicy, CapturedValue, CompileParameter, CompileParameterKind, ConstructionScopeId,
    FreezeError, FreezeMap, FrozenGraphScopeId, Graph, GraphOutput, GraphScope, NodeHandle,
    OutputRoot, ScopedWireRef, SealMap, SealedSubgraph, SourceLocation, SubgraphHandle,
    ValueHandle, current_construction_scope, with_new_construction_scope,
};
pub use types::{NodeId, Port, WireRef, WireType};
pub use validate::{
    LivenessSchedule, ValidatedGraph, ValidatedScope, ValidationError, validate,
    validate_structure, validate_with_manifests,
};
