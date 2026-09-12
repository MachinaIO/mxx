//! Runtime-owned concrete Lean layout adapters and certificate checking.
//!
//! The adapter consumes the same concrete DCRT parameters used by execution.  It does not
//! infer CRT moduli from an IR modulus. The checker consumes application-provided
//! packages and claims; neither component contains application-specific protocol logic.

pub mod check;
mod layout;

#[cfg(test)]
mod fixtures;

pub use layout::{
    LayoutError, LeanBackendArtifact, LeanGadgetMode, LeanRingLayout, export_dcrt_layouts,
    render_backend_context,
};
