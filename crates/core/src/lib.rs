//! `mxx-core`: domain-agnostic structured binaries.
//!
//! A structured binary ([`program::Program`]) is a tree of loops whose leaves carry code. The core
//! knows how many bytes each value has ([`types`]) and how leaves and loops compose
//! ([`program`]), never what the bytes mean. From a validated program ([`validate`]) it derives:
//!
//! - host execution ([`host`]);
//! - a single RV64IM *proof image* for the mathcal-A machine model ([`flatten`]), whose instruction
//!   set, loader, and environment calls [`rv64`] implements;
//! - counts of leaf executions and sizes for cost estimation ([`estimate`]).
//!
//! Everything specific to a domain, such as rings and moduli for lattices, belongs to the DSL
//! that builds the program and supplies its leaves ([`leaf`]). See `ARCHITECTURE-SPEC.md`.

pub mod claims;
pub mod compose;
pub mod env;
pub mod estimate;
pub mod expr;
pub mod flatten;
pub mod host;
pub mod leaf;
pub mod lean;
pub mod program;
pub mod rv64;
pub mod types;
pub mod validate;
