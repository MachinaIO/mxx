//! Optional, scheme-agnostic symbolic operations for `mxx-ir-core`.
//!
//! The core IR and runtime do not depend on this crate. This crate deliberately
//! contains neither noise/residual analysis nor BGG-, Diamond-, or AKY-specific
//! invariant types.

pub use mxx_ir_core::{encoding, expr, graph, node, types};

pub mod atom;
mod bounds;
pub mod checks;
pub mod elaborate;
pub mod manifest;
pub mod overlay;
pub mod rewrite;
mod serde_support;
pub mod term;
pub mod ubound;

pub use elaborate::{
    ElaboratedGraph, ElaborationError, elaborate, elaborate_with_manifests, elaborate_with_overlay,
};
pub use overlay::SymbolicOverlay;
pub use ubound::UBound;
