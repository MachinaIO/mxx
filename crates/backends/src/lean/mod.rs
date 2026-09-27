//! Runtime-owned data adapters used when binding concrete Lean primitive layouts.
//!
//! The adapter consumes the same concrete DCRT parameters used by execution.  It does not
//! infer CRT moduli from an IR modulus and does not contain application-specific protocol logic.
//!
//! `export_dcrt_layouts` and `render_backend_context` export concrete CRT gadget layouts taken from
//! the actual parameters. The handwritten Lean packages in `crates/backends/lean/` provide
//! `MxxPrimitives` (bounds, CRT decomposition, radix, negacyclic arithmetic, preimage and sampling
//! facts) and `MxxRuntime` (the relations generated scope relations use).

mod layout;

#[cfg(test)]
mod fixtures;

pub use layout::{
    LayoutError, LeanBackendArtifact, LeanGadgetMode, LeanRingLayout, export_dcrt_layouts,
    render_backend_context,
};
