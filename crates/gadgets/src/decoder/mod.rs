//! Shared masked-decoder building blocks.
//!
//! This module exposes only BGG-independent PRG layout helpers. Sampling,
//! execution, persistence, and the supported masked-decoder graph belong to
//! Graph IR, `mxx-bgg`, and `mxx-runtime`.

pub mod prg;
