#![recursion_limit = "256"]

//! Perfect-correctness declarations and the Rust operational checker.
//!
//! This crate deliberately contains no probabilistic tail estimates. Correctness consumes only
//! integer sampler cutoffs that the concrete CPU runtime enforces.

pub mod bundle;
pub mod operational_noise;
pub mod operational_protocol;
pub mod protocol;
#[cfg(test)]
mod toy_example;

pub use bundle::*;
pub use operational_protocol::{
    ExactMatrixInputMetadata, ExactTrapdoorInputMetadata, OperationalProtocolError,
    operational_protocol_from_graphs,
};
pub use protocol::*;
