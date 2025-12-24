use bigdecimal::BigDecimal;

pub mod error_norm;
pub mod lattice_estimator;
pub mod poly_matrix_norm;
pub mod poly_norm;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatorContext {
    pub secpar_sqrt: BigDecimal,
    pub ring_dim_sqrt: BigDecimal,
    pub base: BigDecimal,
    pub secret_size: usize,
    pub log_base_q: usize,
    pub m_g: usize,
    pub m_b: usize,
}

impl SimulatorContext {
    pub fn new(
        secpar_sqrt: BigDecimal,
        ring_dim_sqrt: BigDecimal,
        base: BigDecimal,
        secret_size: usize,
        log_base_q: usize,
    ) -> Self {
        let m_g = secret_size * log_base_q;
        let m_b = secret_size * (log_base_q + 2);

        Self { secpar_sqrt, ring_dim_sqrt, base, secret_size, log_base_q, m_g, m_b }
    }
}
