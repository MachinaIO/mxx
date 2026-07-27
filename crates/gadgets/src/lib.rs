//! Reusable lattice-cryptographic gadgets and protocol components.
//!
//! This crate sits above `mxx-primitives` and below complete functional
//! encryption, witness encryption, and indistinguishability obfuscation schemes.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bench_estimator;
pub mod bgg;
pub mod circuit;
pub mod circuit_gadgets;
pub mod commit;
pub mod decoder;
pub mod env;
pub mod input_injector;
pub mod lookup;
pub mod noise_refresh;
pub mod simulator;
pub mod slot_transfer;
pub mod storage;
pub mod utils;

#[cfg(test)]
pub(crate) use mxx_primitives::rlwe_enc;
pub(crate) use mxx_primitives::{element, matrix, poly, sampler};
pub use mxx_primitives::{impl_binop_with_refs, parallel_iter};
