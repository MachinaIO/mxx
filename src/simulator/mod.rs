pub mod lattice_estimator;
pub mod poly_matrix_norm;
pub mod poly_norm;
pub mod wire_norm;

#[derive(Debug, Clone, PartialEq)]
pub struct SimulatorContext {
    pub secpar_sqrt: f64,
    pub ring_dim_sqrt: f64,
    pub base: f64,
    pub log_base_q: usize,
}

impl SimulatorContext {
    pub fn new(
        secpar_sqrt: f64,
        ring_dim_sqrt: f64,
        base: f64,
        log_base_q: usize,
    ) -> Self {
        Self { secpar_sqrt, ring_dim_sqrt, base, log_base_q }
    }
}
