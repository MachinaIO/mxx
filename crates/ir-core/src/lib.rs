//! Core typed graph IR for lattice-cryptography computations.
//!
//! This crate owns executable graph structure, compile expressions, concrete
//! type validation, canonical identities, and runtime artifact metadata.
//! Optional symbolic-term operations live in `mxx-ir-symbolic`.

pub mod artifact;
pub mod builder;
pub mod checks;
pub mod encoding;
pub mod expr;
pub mod graph;
pub mod node;
mod serde_support;
pub mod types;
pub mod validate;

pub use builder::{
    GraphBuilder, MatrixFamilyWire, MatrixWire, OutputFamilyError, SubgraphBuildError,
    TrapdoorFamilyWire, TrapdoorWire, ValueFamilyWire,
};
pub use expr::{IntExpr, ParamEnv, Rational, RealExpr};
pub use graph::Graph;
pub use types::{NodeId, Port, WireRef, WireType};
pub use validate::{ValidatedGraph, ValidationError, validate, validate_with_manifests};
