//! Runtime-owned data adapters used when binding concrete Lean primitive layouts.
//!
//! The adapter consumes the same concrete DCRT parameters used by execution.  It does not
//! infer CRT moduli from an IR modulus and does not contain application-specific protocol logic.

mod layout;

#[cfg(test)]
mod fixtures;

pub use layout::{
    LayoutError, LeanBackendArtifact, LeanGadgetMode, LeanRingLayout, export_dcrt_layouts,
    render_backend_context,
};
