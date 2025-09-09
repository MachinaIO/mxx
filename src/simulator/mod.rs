use bigdecimal::BigDecimal;

pub mod lattice_estimator;
pub mod poly_matrix_norm;
pub mod poly_norm;
pub mod wire_norm;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatorContext {
    pub secpar_sqrt: BigDecimal,
    pub ring_dim_sqrt: BigDecimal,
    pub base: BigDecimal,
    pub log_base_q: usize,
}

impl SimulatorContext {
    pub fn new(
        secpar_sqrt: BigDecimal,
        ring_dim_sqrt: BigDecimal,
        base: BigDecimal,
        log_base_q: usize,
    ) -> Self {
        Self { secpar_sqrt, ring_dim_sqrt, base, log_base_q }
    }
}
