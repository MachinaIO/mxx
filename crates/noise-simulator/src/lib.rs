//! Noise simulation for elaborated mxx graph IR.
//!
//! The numerical kernel in this crate preserves the historical
//! `PolyNorm`/`PolyMatrixNorm` rules. Graph-specific analysis is layered on
//! top of the same kernel so symbolic elaboration does not own magnitude
//! arithmetic.

use bigdecimal::BigDecimal;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

pub mod dependency_set;
pub mod poly_matrix_norm;
pub mod poly_norm;
pub mod simulate;

pub use dependency_set::{DependencySet, SourceId};
pub use poly_matrix_norm::PolyMatrixNorm;
pub use poly_norm::PolyNorm;
pub use simulate::{
    DecodeNoiseReport, MatrixNoiseReport, NoiseReport, SimulationError, WireNoiseReport, simulate,
};

#[derive(Debug, Clone)]
pub struct SimulatorContext {
    pub ring_dim_sqrt: BigDecimal,
    pub base: BigDecimal,
    pub secret_size: usize,
    pub log_base_q: usize,
    pub m_g: usize,
    pub m_b: usize,
    pub log_base_q_small: usize,
}

impl SimulatorContext {
    pub fn new(
        ring_dim_sqrt: BigDecimal,
        base: BigDecimal,
        secret_size: usize,
        log_base_q: usize,
        log_base_q_small: usize,
    ) -> Self {
        let m_g = secret_size * log_base_q;
        let m_b = secret_size * (log_base_q + 2);
        Self { ring_dim_sqrt, base, secret_size, log_base_q, m_g, m_b, log_base_q_small }
    }

    pub fn fresh_source_id(&self) -> SourceId {
        static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-noise-simulator/transient-source/v1");
        hasher.update(counter.to_le_bytes());
        SourceId(hasher.finalize().into())
    }
}

impl PartialEq for SimulatorContext {
    fn eq(&self, other: &Self) -> bool {
        self.ring_dim_sqrt == other.ring_dim_sqrt &&
            self.base == other.base &&
            self.secret_size == other.secret_size &&
            self.log_base_q == other.log_base_q &&
            self.m_g == other.m_g &&
            self.m_b == other.m_b &&
            self.log_base_q_small == other.log_base_q_small
    }
}

impl Eq for SimulatorContext {}
