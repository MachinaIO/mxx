//! Reusable lattice-cryptographic gadgets and protocol components.
//!
//! This crate sits above `mxx-primitives` and below complete functional
//! encryption, witness encryption, and indistinguishability obfuscation schemes.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod circuit;
pub mod circuit_gadgets;
pub mod decoder;
pub mod input_injector;
pub mod noise_refresh;
pub mod utils;

#[cfg(test)]
mod test_utils;
#[cfg(all(test, feature = "gpu"))]
mod test_utils_gpu;

// BGG-specific lookup evaluation lives in `mxx-bgg`. The WEE25
// commitment-backed lookup evaluator is not currently implemented.

#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use mxx_primitives::rlwe_enc;
pub use mxx_primitives::{element::PolyElem, impl_binop_with_refs, parallel_iter, poly::Poly};
pub(crate) use mxx_primitives::{matrix, poly, sampler};
