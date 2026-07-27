//! Analytical bounds shared by trapdoor samplers and higher-level simulators.

use bigdecimal::BigDecimal;
use num_traits::{FromPrimitive, Zero};

pub fn high_probability_envelope_from_sigma(sigma: &BigDecimal) -> BigDecimal {
    assert!(*sigma >= BigDecimal::zero(), "sigma must be nonnegative");
    sigma * BigDecimal::from(13u64) / BigDecimal::from(2u64)
}

pub fn maximum_coefficient_bound_from_sigma(sigma: &BigDecimal) -> BigDecimal {
    high_probability_envelope_from_sigma(sigma)
}

pub fn compute_preimage_sigma(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
    sigma: Option<f64>,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(sigma.unwrap_or(4.578)).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let b_nrow = b_nrow.unwrap_or(1);
    let term = BigDecimal::from(b_nrow as u64).sqrt().unwrap() * ring_dim_sqrt.clone() * m_g_sqrt +
        two_sqrt * ring_dim_sqrt +
        c_1;
    c_0 * sigma.clone() * ((base + 1) * sigma) * term
}

pub fn compute_preimage_norm(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
    sigma: Option<f64>,
) -> BigDecimal {
    high_probability_envelope_from_sigma(&compute_preimage_sigma(
        ring_dim_sqrt,
        m_g,
        base,
        b_nrow,
        sigma,
    ))
}
