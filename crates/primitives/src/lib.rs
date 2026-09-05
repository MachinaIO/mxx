//! Low-level lattice-cryptography primitives.
//!
//! This crate owns polynomial and matrix representations, samplers, RLWE
//! encryption helpers, OpenFHE integration, and the native CUDA runtime.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod element;
pub mod env;
pub mod matrix;
pub mod modulus;
pub(crate) mod openfhe_guard;
pub mod poly;
pub mod rlwe_enc;
pub mod sampler;
pub mod utils;
