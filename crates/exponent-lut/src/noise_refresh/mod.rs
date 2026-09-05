//! Sparse-LWR PRF evaluation and native exponent-encoding noise refresh.
//!
//! This module contains the application-specific layers built on the generic
//! Exponent-LUT program, encoding, public-key, and RHS modules. Setup artifacts,
//! PBC routing, PRF evaluation, value readout, CRT refresh, and their dedicated
//! noise simulation live together because they form one refresh protocol.

pub mod artifacts;
pub mod pbc;
pub mod prf;
pub mod readout;
pub mod refresh;
pub mod setup;
#[cfg(all(test, feature = "gpu"))]
#[path = "../../tests/unit/noise_refresh_gpu.rs"]
mod setup_gpu;
pub mod simulation;
