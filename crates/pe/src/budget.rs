//! The statistical budget of the trapdoor: the regularity (leftover hash lemma) condition of
//! TrapGen and the truncation of the error contract `χ`.
//!
//! The scheme takes the relaxations of the manuscript's implementation section: there is no noise
//! flooding (every fresh error is a `χ` sample), the uniformity conditions on the matrices `R` and
//! `Z_t` that appear only in the proof's hybrids are dropped, so `m_B = m̄ + D` follows from TrapGen
//! alone, and keys are sampled at the trapdoor's minimum preimage width (the reduction's
//! key-simulation width is relaxed, as in the CP-ABE 1 of Datta et al., ACM CCS 2026).
//!
//! Widths here are the manuscript's widths `s` (density `exp(-π |x|² / s²)`) when named `s_*`,
//! and the samplers' standard deviations `σ = s / √(2π)` when named `sigma_*`; logarithms are
//! base 2. The budget bounds statistical distances under the assumption that every sampler returns
//! its ideal discrete Gaussian: the floating-point error of the samplers and the truncation and
//! rejection of preimages and key samples are heuristic and not budgeted.
//!
//! # The budget
//!
//! The target `ε = 2^-STATISTICAL_BITS` is split in half:
//!
//! - the regularity lemma, applied column by column to the `D` columns of `ā T`, gets `ε / 2`,
//!   `ε_LHL = ε / (2D)` per application;
//! - the truncation of `χ` at `c` standard deviations gets `ε / 2`: every one of its `K`
//!   coefficients (the anchor, the helpers, the message, the fresh encodings, and the errors `e_t`
//!   of the attribute ciphertexts) has tail mass at most `ε / (2K)`, so `c = √(2 ln(4K / ε))`.
//!
//! # Regularity
//!
//! With `ϱ = L_q log(2n) + 2 log(1 / ε_LHL)` (Lemma 8 of the manuscript),
//! `m̄ (log s_T - 1) ≥ log q + ϱ` and `m_B = m̄ + D`. The TrapGen width satisfies
//! `η_ε'(Z) ≤ s_T ≤ (√π / 2) √(q_min / ln(4n))`, and `χ` is at least the smoothing parameter
//! `η_ε'(Z) = √(ln(2 (1 + 1/ε')) / π)`, for `ε' = ε / (2K)`.
use crate::PeError;
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use num_bigint::{BigUint, ToBigInt};
use num_traits::Zero;
use serde::Serialize;

/// `log(1 / ε)` of the budget: `ε = 2^-30`.
pub const STATISTICAL_BITS: u32 = 30;

/// `√(2π)`: the manuscript's width of a standard deviation of one.
pub fn paper_scale() -> f64 {
    (2.0 * std::f64::consts::PI).sqrt()
}

/// The smoothing parameter `η_ε(Z) = √(ln(2 (1 + 1/ε)) / π)` (a manuscript width) for `ε =
/// 2^-log_inverse_epsilon`.
pub fn smoothing_parameter(log_inverse_epsilon: f64) -> f64 {
    // ln(2 (1 + 2^k)) = ln 2 + ln(1 + 2^k), evaluated stably for large k.
    let k = log_inverse_epsilon;
    let ln_one_plus = if k > 50.0 { k * std::f64::consts::LN_2 } else { (2f64.powf(k)).ln_1p() };
    ((std::f64::consts::LN_2 + ln_one_plus) / std::f64::consts::PI).sqrt()
}

/// The multiplier `c = √(2 ln(2 count / tail))` with `count 2 exp(-c² / 2) ≤ tail` for `tail =
/// 2^-log_inverse_tail`.
pub fn tail_multiplier(count: f64, log_inverse_tail: f64) -> f64 {
    (2.0 * ((2.0 * count).ln() + log_inverse_tail * std::f64::consts::LN_2)).sqrt()
}

/// The sizes the budget counts.
#[derive(Clone, Copy, Debug)]
pub struct BudgetShape {
    pub ring_dimension: usize,
    pub digits: usize,
    /// `ℓ`: the RHS packages of the ciphertext.
    pub attribute_length: usize,
    pub crt_depth: usize,
    /// `|K|`: the switching packages of the ciphertext.
    pub switching_packages: usize,
    /// `|J|`: the published helpers of the ciphertext.
    pub helpers: usize,
}

impl BudgetShape {
    /// `E = 1 + (ℓ + |K|) 2 D²`: `c_0` and the digit companions of every package, which cover
    /// the `D` private digit rows of its gadget ciphertext only: the top rows of the attribute
    /// ciphertexts and of the switching ciphertexts `C_k` are public.
    pub fn encodings(&self) -> f64 {
        1.0 + ((self.attribute_length + self.switching_packages) * 2 * self.digits * self.digits)
            as f64
    }

    /// The truncated coefficients `K` of `χ` for an anchor of `m_b` columns: the anchor, the
    /// helpers, and the message, `n (m_B + |J| 2D + 1)`, and the fresh encodings and the errors
    /// `e_t` of the attribute ciphertexts, `n 2D (E + ℓ)`.
    pub fn truncated_coefficients(&self, m_b: usize) -> f64 {
        let n = self.ring_dimension as f64;
        let m = (2 * self.digits) as f64;
        n * (m_b as f64 + self.helpers as f64 * m + 1.0) +
            n * m * (self.encodings() + self.attribute_length as f64)
    }
}

/// The budget of one candidate, with every derived multiplier and width.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct StatisticalBudget {
    /// `log(1 / ε_LHL)`.
    pub lhl_bits: f64,
    /// `log(1 / (ε / 2))`: the truncation's share.
    pub truncation_bits: f64,
    /// `ϱ`.
    pub rho: f64,
    /// `log q`.
    pub log_modulus: f64,
    /// The truncation multiplier `c`.
    pub cutoff_multiplier: f64,
    /// The smallest width the regularity and truncation arguments accept (a manuscript width).
    pub smoothing: f64,
    /// The largest TrapGen width (a manuscript width).
    pub max_width: f64,
    /// `m̄` and `m_B = m̄ + D`.
    pub trapdoor_columns: usize,
    pub anchor_columns: usize,
}

/// The inputs of [`StatisticalBudget::new`].
#[derive(Clone, Debug)]
pub struct BudgetRequest<'a> {
    pub shape: BudgetShape,
    pub modulus: &'a BigUint,
    pub smallest_prime: u64,
    /// Standard deviations of TrapGen and `χ`.
    pub trapdoor_sigma: f64,
    pub error_sigma: &'a BigDecimal,
}

fn log2_biguint(value: &BigUint) -> f64 {
    let bits = value.bits();
    let shift = bits.saturating_sub(64);
    (value >> shift).to_f64().unwrap_or(f64::INFINITY).log2() + shift as f64
}

fn to_f64(value: &BigDecimal) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn decimal_of(value: f64) -> Result<BigDecimal, PeError> {
    BigDecimal::from_f64(value).ok_or_else(|| PeError::Parameters(format!("{value} is not finite")))
}

impl StatisticalBudget {
    /// The budget and the anchor width of one candidate, or the first condition it violates.
    pub fn new(request: &BudgetRequest<'_>) -> Result<Self, PeError> {
        let reject = |message: String| Err(PeError::Parameters(message));
        let shape = &request.shape;
        let half = STATISTICAL_BITS as f64 + 1.0;
        let lhl_bits = half + (shape.digits as f64).log2();
        let n = shape.ring_dimension as f64;
        let rho = shape.crt_depth as f64 * (2.0 * n).log2() + 2.0 * lhl_bits;
        let log_modulus = log2_biguint(request.modulus);
        let scale = paper_scale();
        let trapdoor_width = request.trapdoor_sigma * scale;
        let denominator = trapdoor_width.log2() - 1.0;
        if denominator.is_nan() || denominator <= 0.0 {
            return reject(format!("the TrapGen width {trapdoor_width} is at most 2"));
        }
        let trapdoor_columns = ((log_modulus + rho) / denominator).ceil() as usize;
        let anchor_columns = trapdoor_columns + shape.digits;
        let count = shape.truncated_coefficients(anchor_columns);
        let cutoff_multiplier = tail_multiplier(count, half);
        let smoothing = smoothing_parameter(half + count.log2());
        let max_width = (std::f64::consts::PI.sqrt() / 2.0) *
            (request.smallest_prime as f64 / (4.0 * n).ln()).sqrt();
        if trapdoor_width < smoothing {
            return reject(format!("the TrapGen width {trapdoor_width} is below η = {smoothing}"));
        }
        if trapdoor_width > max_width {
            return reject(format!("the TrapGen width {trapdoor_width} exceeds {max_width}"));
        }
        let error_width = to_f64(request.error_sigma) * scale;
        if error_width.is_nan() || error_width < smoothing {
            return reject(format!("the χ width is below η = {smoothing}"));
        }
        Ok(Self {
            lhl_bits,
            truncation_bits: half,
            rho,
            log_modulus,
            cutoff_multiplier,
            smoothing,
            max_width,
            trapdoor_columns,
            anchor_columns,
        })
    }

    /// `m̄ = m_B - D`.
    pub fn m_bar(&self, digits: usize) -> usize {
        self.anchor_columns - digits
    }

    /// The cutoff `floor(c σ)` of a truncated contract of standard deviation `sigma`.
    pub fn cutoff(&self, sigma: &BigDecimal) -> Result<BigUint, PeError> {
        let cutoff = decimal_of(self.cutoff_multiplier)? * sigma;
        cutoff
            .with_scale(0)
            .to_bigint()
            .and_then(|value| value.to_biguint())
            .filter(|value| !value.is_zero() || sigma.is_zero())
            .ok_or_else(|| PeError::Parameters("a truncation cutoff is not representable".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded_shape() -> BudgetShape {
        BudgetShape {
            ring_dimension: 8192,
            digits: 8,
            attribute_length: 1,
            crt_depth: 4,
            switching_packages: 1,
            helpers: 1,
        }
    }

    fn request<'a>(
        modulus: &'a BigUint,
        error_sigma: &'a BigDecimal,
        trapdoor_sigma: f64,
    ) -> BudgetRequest<'a> {
        BudgetRequest {
            shape: recorded_shape(),
            modulus,
            smallest_prime: (1u64 << 42) - 1,
            trapdoor_sigma,
            error_sigma,
        }
    }

    /// The regularity lemma takes `ε / 2` over the `D` columns of `ā T`, the truncation the other
    /// half; `m̄` is the smallest column count of the entropy inequality.
    #[test]
    fn the_budget_splits_epsilon_between_regularity_and_truncation() {
        let shape = recorded_shape();
        assert_eq!(shape.encodings(), 257.0);
        let (modulus, error) = (BigUint::from(1u8) << 168u32, "3.19".parse().unwrap());
        let budget = StatisticalBudget::new(&request(&modulus, &error, 256.0)).unwrap();
        assert!((budget.lhl_bits - 34.0).abs() < 1e-9);
        assert!((budget.truncation_bits - 31.0).abs() < 1e-9);
        assert!((budget.rho - (4.0 * 14.0 + 68.0)).abs() < 1e-9);
        let log_width = (256.0 * paper_scale()).log2() - 1.0;
        let m_bar = budget.trapdoor_columns as f64;
        assert!(m_bar * log_width >= 168.0 + budget.rho);
        assert!((m_bar - 1.0) * log_width < 168.0 + budget.rho);
        assert_eq!(budget.m_bar(8), budget.trapdoor_columns);
        assert_eq!(budget.anchor_columns, budget.trapdoor_columns + 8);
        let count = shape.truncated_coefficients(budget.anchor_columns);
        let c = budget.cutoff_multiplier;
        assert!((count * 2.0 * (-c * c / 2.0).exp() - 2f64.powf(-31.0)).abs() < 1e-15);
        assert!((budget.smoothing - 3.55).abs() < 0.05, "η {}", budget.smoothing);
    }

    #[test]
    fn narrow_and_wide_trapdoor_widths_are_rejected() {
        let (modulus, error) = (BigUint::from(1u8) << 168u32, "3.19".parse().unwrap());
        // σ_T = 1 is below the smoothing parameter, and 2^30 above the largest width.
        assert!(StatisticalBudget::new(&request(&modulus, &error, 1.0)).is_err());
        assert!(StatisticalBudget::new(&request(&modulus, &error, (1u64 << 30) as f64)).is_err());
        // A zero `χ` is below the smoothing parameter.
        let zero = BigDecimal::from(0u8);
        assert!(StatisticalBudget::new(&request(&modulus, &zero, 256.0)).is_err());
    }
}
