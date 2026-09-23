//! Reusable lattice-cryptographic gadgets and protocol components.
//!
//! This crate sits above `mxx-backends` and below complete functional
//! encryption, witness encryption, and indistinguishability obfuscation schemes.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod circuit;
pub mod circuit_gadgets;
pub mod decoder;
pub mod input_injector;
pub mod noise_refresh;
pub mod utils;

pub fn ring_from_params(
    parameters: &mxx_backends::poly::dcrt::params::DCRTPolyParams,
) -> mxx_dsl::Ring {
    use mxx_backends::poly::PolyParams;
    mxx_dsl::Ring::from_crt_moduli(
        parameters.to_crt().0.into_iter().map(Into::into).collect(),
        parameters.ring_dimension(),
    )
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_utils;
#[cfg(all(test, feature = "gpu"))]
mod test_utils_gpu;

// BGG-specific lookup evaluation lives in `mxx-bgg`. The WEE25
// commitment-backed lookup evaluator is not currently implemented.

pub use mxx_backends::{element::PolyElem, impl_binop_with_refs, parallel_iter, poly::Poly};
pub(crate) use mxx_backends::{matrix, poly, sampler};
