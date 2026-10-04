//! Native exponent-encoding noise refresh.
//!
//! This module contains the refresh application built on the generic
//! Exponent-LUT program, encoding, public-key, and RHS modules. Setup
//! artifacts, value readout, CRT refresh, and their dedicated noise simulation
//! live together because they form one refresh protocol. The sparse-LWR PRF
//! Program and its PBC routing, which the refresh evaluates, are reusable
//! program gadgets in [`crate::program_gadgets`].

pub mod refresh;
pub mod setup;
pub mod simulation;

pub use refresh::{NoiseRefresh, RefreshDeclaration};
