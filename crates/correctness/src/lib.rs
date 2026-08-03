//! Perfect-correctness declarations and the Rust side of the Lean verification pipeline.
//!
//! This crate deliberately contains no probabilistic tail estimates. Correctness consumes only
//! integer sampler cutoffs that the concrete CPU runtime enforces.

pub mod check;
pub mod emit_lean;
pub mod protocol;
pub mod toy_example;

pub use check::{TheoremReport, VerifyError, verify_theorem_at};
pub use emit_lean::{EmitError, EmittedProtocol, emit_protocol_for};
pub use protocol::*;
