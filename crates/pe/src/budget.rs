//! The statistical-distance budget, the regularity (leftover hash lemma) conditions, the
//! flooding width, and the SampleRight key width of the statistical instantiation.
//!
//! Widths here are the manuscript's widths `s` (density `exp(-π |x|² / s²)`) when named `s_*`,
//! and the samplers' standard deviations `σ = s / √(2π)` when named `sigma_*`; logarithms are
//! base 2. The budget bounds statistical distances under the assumption that every sampler returns
//! its ideal discrete Gaussian: the floating-point error of the samplers and the truncation and
//! rejection of preimages and key samples are heuristic and not budgeted.
//!
//! # The budget
//!
//! The target `ε = 2^-statistical_bits` is split between the items the selection uses: the
//! regularity lemma gets `ε / 2`, and the other `k` items share `ε / 2`, `ε_item = ε / (2 k)` each.
//!
//! - The regularity lemma, applied `N_LHL = D + 2 ℓ D + 2 E 2D` times column by column (TrapGen,
//!   the `Z_t` of hybrid `H_4`, and the encodings of `H_1` and `H_4`), with `E = 1 + (ℓ + |K|) 4
//!   D²` fresh encodings for the `|K|` switching packages of the ciphertext (`W - 1`, or `(B - 1)
//!   log_B W` under the LUT radix `B`); each application gets `ε_LHL = (ε / 2) / N_LHL`.
//! - The truncation of `χ`, `Z_t`, and `χ_fl` at `c` standard deviations: every one of their `K`
//!   coefficients has tail mass at most `ε_item / K`, so `c = √(2 ln(2 K / ε_item))`.
//! - Other statistical steps of the proof.
//! - With enforced flooding: the convolution of the wide flooding sampler.
//! - `Strict` only: the coefficients of the proof's `R` stay below `B_R = c_R σ_R` but for `ε_item`
//!   over their `K_R = E m_B 2D n` coefficients.
//! - `Strict` only: `Q_key` key queries each swap SampleLeft and SampleRight at distance `2 ε_ss`,
//!   so `ε_ss = ε_item / (2 Q_key)`.
//!
//! Flooding itself and the tail of the flooded quantity are bounded per coefficient (see below),
//! not taken from `ε`.
//!
//! # Regularity
//!
//! With `ϱ = L_q log(2n) + 2 log(1 / ε_LHL)` (Lemma 8 of the manuscript):
//!
//! ```text
//! m̄ (log s_T - 1) ≥ log q + ϱ,   m_B (log s_Z - 1) ≥ 2 log q + ϱ,   m_B (log s_R - 1) ≥ 3 log q + ϱ,
//! ```
//!
//! `s_T, s_Z, s_R ≤ (√π / 2) √(q_min / ln(4n))`, and every width at least the smoothing parameter
//! `η_ε'(Z) = √(ln(2 (1 + 1/ε')) / π)` for `ε' = ε_item / K`. `m_B = max(m̄_T + D, m_Z, m_R)`.
//!
//! # Flooding
//!
//! Hybrid `H_1` floods `σ_i(e_b) R` by `χ_fl`. For fixed `e_b`, every coefficient of `σ_i(e_b)
//! R` is Gaussian of standard deviation at most `σ_R √(m_B n) B_χ`, and all `K_fl = n 2D E` of
//! them stay below `B_coef = c_t σ_R √(m_B n) B_χ` but for `K_fl 2 exp(-c_t² / 2) ≤
//! 2^-flooding_bits`. Shifting a Gaussian of width `s` by `a` moves it by about `|a| / s` in
//! statistical distance, so `s_fl = 2^flooding_bits B_coef` floods one coefficient to
//! `2^-flooding_bits`. As in Noah's Ark, the distance is evaluated per coefficient: the number of
//! coefficients is not multiplied into the width but counts the samples a distinguisher needs,
//! separately from `ε`.
//!
//! # The key width
//!
//! `Practical` keys use the trapdoor's minimum preimage width `s_min`. `Strict` keys use
//! `max(s_min, s_SR)` for the width at which SampleRight, with the trapdoor `X` of `M_F X = I_W ⊗
//! g` built from the ledger witness `R̂_F` (`lem:keysim`), returns samples within `ε_ss` of the key
//! distribution:
//!
//! ```text
//! B_X^(0) = max(B_R̂, 1),   B_X^(t+1) = 2D n (b / 2) B_X^(t) + 1,   B_X = B_X^(r),
//! s_1(X) ≤ √(N W D) n B_X,   s_SR = η_ε_ss(Z) c_G √(1 + s_1(X)²),
//! ```
//!
//! where `B_R̂` bounds the coefficients of `R̂_F`, `b` is the gadget base, and `c_G` the gadget
//! sampler's width. Row `i` of `X` is reached from row 1 along a path of helper automorphisms,
//! each step `V_(j a) = V_j G̃^-1(σ_j(H_a)) - J_(η_(j,a))` one decomposition gain: with the LUT
//! radix `B` the helpers are `J_B` and a path has at most `r = log_B W` steps (one per nonzero
//! digit), and `r = 1` for the direct switching. The manuscript's condition `N n B_R̂` is the
//! binary-gadget form of this bound; the implementation's RNS gadget with base `b` and `D` digits
//! per row gives the factors above.
use crate::PeError;
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive};
use num_bigint::{BigUint, ToBigInt};
use num_traits::Zero;
use serde::Serialize;

/// The default `statistical_bits`: `ε = 2^-30`.
pub const DEFAULT_STATISTICAL_BITS: u32 = 30;
/// The default per-coefficient flooding distance `2^-30`.
pub const DEFAULT_FLOODING_BITS: u32 = 30;

/// How the key width of the statistical instantiation is chosen.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum KeyWidthPolicy {
    /// `max(s_min, s_SR)`: the manuscript's key-width condition, under which the reduction's
    /// SampleRight simulates the keys.
    Strict,
    /// `s_min`, the trapdoor's minimum preimage width only: the reduction's key-simulation
    /// condition is relaxed, as in the CP-ABE 1 of Datta et al. (ACM CCS 2026).
    Practical,
}

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

/// The sizes the budget counts, for an anchor of `m_b` columns.
#[derive(Clone, Copy, Debug)]
pub struct BudgetShape {
    pub ring_dimension: usize,
    pub digits: usize,
    pub attribute_length: usize,
    pub lookup_width: usize,
    pub crt_depth: usize,
    /// `|K|`: the switching packages of the ciphertext.
    pub switching_packages: usize,
    /// `|J|`: the published helpers of the ciphertext.
    pub helpers: usize,
}

impl BudgetShape {
    /// `E = 1 + (ℓ + |K|) 4 D²`: `c_0` and the digit companions of every package.
    pub fn encodings(&self) -> f64 {
        1.0 + ((self.attribute_length + self.switching_packages) * 4 * self.digits * self.digits)
            as f64
    }

    /// `N_LHL = D + 2 ℓ D + 2 E 2D`.
    pub fn lhl_applications(&self) -> f64 {
        (self.digits + 2 * self.attribute_length * self.digits) as f64 +
            2.0 * self.encodings() * (2 * self.digits) as f64
    }

    /// The truncated coefficients `K` of `χ` (the anchor, the helpers, and the message), `Z_t`,
    /// and `χ_fl` for an anchor of `m_b` columns.
    pub fn truncated_coefficients(&self, m_b: usize) -> f64 {
        let n = self.ring_dimension as f64;
        let m = (2 * self.digits) as f64;
        let chi = n * (m_b as f64 + self.helpers as f64 * m + 1.0);
        let programming = n * (self.attribute_length * m_b) as f64 * m;
        chi + programming + self.flooded_coefficients()
    }

    /// `K_fl = n 2D E`.
    pub fn flooded_coefficients(&self) -> f64 {
        self.ring_dimension as f64 * (2 * self.digits) as f64 * self.encodings()
    }

    /// `K_R = E m_B 2D n`.
    pub fn randomness_coefficients(&self, m_b: usize) -> f64 {
        self.encodings() * (m_b * 2 * self.digits * self.ring_dimension) as f64
    }
}

/// The budget of one candidate, with every derived multiplier and width.
#[derive(Clone, Debug, Serialize)]
pub struct StatisticalBudget {
    /// `log(1 / ε)`.
    pub statistical_bits: u32,
    /// The per-coefficient flooding exponent.
    pub flooding_bits: u32,
    /// `log(1 / ε_LHL)`.
    pub lhl_bits: f64,
    /// `log(1 / ε_item)`: the share of every other item of the budget.
    pub item_bits: f64,
    /// `ϱ`.
    pub rho: f64,
    /// `log q`.
    pub log_modulus: f64,
    /// The truncation multiplier `c`, fixed from the coefficient count at `m_B,max`.
    pub cutoff_multiplier: f64,
    /// The smallest width the regularity and flooding arguments accept (a manuscript width).
    pub smoothing: f64,
    /// The largest regularity width (a manuscript width).
    pub max_width: f64,
    /// `m̄_T`, `m_Z`, `m_R`, and `m_B`.
    pub trapdoor_columns: usize,
    pub programming_columns: usize,
    pub randomness_columns: usize,
    pub anchor_columns: usize,
    /// The flooding tail multiplier `c_t`.
    pub flooding_tail: f64,
    /// `B_coef`.
    #[serde(serialize_with = "decimal")]
    pub flooded_bound: BigDecimal,
    /// The standard deviation `σ_fl = s_fl / √(2π)` of `χ_fl`.
    #[serde(serialize_with = "decimal")]
    pub flooding_sigma: BigDecimal,
    /// `Strict` only: `c_R` and `log(1 / ε_ss)`.
    pub randomness_tail: Option<f64>,
    pub key_simulation_bits: Option<f64>,
}

fn decimal<S: serde::Serializer>(value: &BigDecimal, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&value.to_string())
}

/// The inputs of [`StatisticalBudget::new`].
#[derive(Clone, Debug)]
pub struct BudgetRequest<'a> {
    pub shape: BudgetShape,
    pub modulus: &'a BigUint,
    pub smallest_prime: u64,
    pub statistical_bits: u32,
    pub flooding_bits: u32,
    pub max_anchor_columns: usize,
    pub key_queries: Option<u64>,
    /// Whether the flooding width is enforced (its wide sampler takes a share of `ε`).
    pub flooding_enforced: bool,
    /// Standard deviations of TrapGen, `Z_t`, the proof's `R`, and `χ`.
    pub trapdoor_sigma: f64,
    pub programming_sigma: &'a BigDecimal,
    pub randomness_sigma: &'a BigDecimal,
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
    /// The budget, the regularity column counts, and the flooding width of one candidate, or the
    /// first condition it violates.
    pub fn new(request: &BudgetRequest<'_>) -> Result<Self, PeError> {
        let reject = |message: String| Err(PeError::Parameters(message));
        let shape = &request.shape;
        let items = 2 +
            usize::from(request.flooding_enforced) +
            2 * usize::from(request.key_queries.is_some());
        let item_bits = request.statistical_bits as f64 + 1.0 + (items as f64).log2();
        let lhl_bits = request.statistical_bits as f64 + 1.0 + shape.lhl_applications().log2();
        let n = shape.ring_dimension as f64;
        let rho = shape.crt_depth as f64 * (2.0 * n).log2() + 2.0 * lhl_bits;
        let log_modulus = log2_biguint(request.modulus);
        let count = shape.truncated_coefficients(request.max_anchor_columns);
        let cutoff_multiplier = tail_multiplier(count, item_bits);
        let smoothing = smoothing_parameter(item_bits + count.log2());
        let max_width = (std::f64::consts::PI.sqrt() / 2.0) *
            (request.smallest_prime as f64 / (4.0 * n).ln()).sqrt();
        let scale = paper_scale();
        let widths = [
            ("TrapGen", request.trapdoor_sigma * scale),
            ("Z_t", to_f64(request.programming_sigma) * scale),
            ("R", to_f64(request.randomness_sigma) * scale),
        ];
        for (name, width) in widths {
            if width.is_nan() || width < smoothing {
                return reject(format!("the {name} width {width} is below η = {smoothing}"));
            }
            if width > max_width {
                return reject(format!("the {name} width {width} exceeds {max_width}"));
            }
        }
        let error_width = to_f64(request.error_sigma) * scale;
        if error_width.is_nan() || error_width < smoothing {
            return reject(format!("the χ width is below η = {smoothing}"));
        }
        let columns = |rows: f64, width: f64| -> Result<usize, PeError> {
            let denominator = width.log2() - 1.0;
            if denominator <= 0.0 {
                return Err(PeError::Parameters(format!("the width {width} is at most 2")));
            }
            Ok(((rows * log_modulus + rho) / denominator).ceil() as usize)
        };
        let trapdoor_columns = columns(1.0, widths[0].1)?;
        let programming_columns = columns(2.0, widths[1].1)?;
        let randomness_columns = columns(3.0, widths[2].1)?;
        let anchor_columns =
            (trapdoor_columns + shape.digits).max(programming_columns).max(randomness_columns);
        if anchor_columns > request.max_anchor_columns {
            return reject(format!(
                "the anchor needs {anchor_columns} columns, above the maximum {}",
                request.max_anchor_columns
            ));
        }
        let error_cutoff = decimal_of(cutoff_multiplier)? * request.error_sigma;
        let flooding_tail =
            tail_multiplier(shape.flooded_coefficients(), request.flooding_bits as f64);
        let flooded_bound = decimal_of(flooding_tail)? *
            request.randomness_sigma *
            decimal_of(((anchor_columns * shape.ring_dimension) as f64).sqrt())? *
            BigDecimal::from(error_cutoff.with_scale(0).to_bigint().unwrap_or_default());
        let flooding_width = flooded_bound.clone() *
            BigDecimal::from(num_bigint::BigInt::from(1u8) << request.flooding_bits);
        let flooding_sigma = flooding_width / decimal_of(scale)?;
        if to_f64(&flooding_sigma) * scale < smoothing {
            return reject("the flooding width is below the smoothing parameter".into());
        }
        let (randomness_tail, key_simulation_bits) = match request.key_queries {
            None => (None, None),
            Some(queries) => (
                Some(tail_multiplier(shape.randomness_coefficients(anchor_columns), item_bits)),
                Some(item_bits + 1.0 + (queries.max(1) as f64).log2()),
            ),
        };
        Ok(Self {
            statistical_bits: request.statistical_bits,
            flooding_bits: request.flooding_bits,
            lhl_bits,
            item_bits,
            rho,
            log_modulus,
            cutoff_multiplier,
            smoothing,
            max_width,
            trapdoor_columns,
            programming_columns,
            randomness_columns,
            anchor_columns,
            flooding_tail,
            flooded_bound,
            flooding_sigma,
            randomness_tail,
            key_simulation_bits,
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

    /// Checks that the truncation multiplier covers the coefficient count at the selected
    /// `m_B` (it was fixed at `m_B,max`, which is at least `m_B`).
    pub fn verify(&self, shape: &BudgetShape) -> Result<(), PeError> {
        let needed =
            tail_multiplier(shape.truncated_coefficients(self.anchor_columns), self.item_bits);
        if needed > self.cutoff_multiplier {
            return Err(PeError::Parameters("the truncation budget is exceeded".into()));
        }
        Ok(())
    }

    /// The SampleRight width `σ_SR = η_ε_ss(Z) c_G √(1 + s_1(X)²)` (a standard deviation; the
    /// manuscript's width is `√(2π)` times larger) for the witness bound `B_R̂`, with the key length
    /// `N`, the lookup width `W`, the longest helper path `r` (`log_B W` for the LUT radix `B`),
    /// the gadget base `b`, and the gadget sampler's standard deviation `c_G`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_right_sigma(
        &self,
        shape: &BudgetShape,
        witness_bound: &BigUint,
        key_length: usize,
        path_length: usize,
        base: u32,
        gadget_sigma: f64,
    ) -> Result<BigDecimal, PeError> {
        let bits = self
            .key_simulation_bits
            .ok_or_else(|| PeError::Parameters("the key simulation budget is absent".into()))?;
        let n = BigUint::from(shape.ring_dimension);
        let gain = BigUint::from(2 * shape.digits) * &n * BigUint::from(base / 2);
        let b_x = (0..path_length.max(1))
            .fold(witness_bound.max(&BigUint::from(1u8)).clone(), |b_x, _| &gain * b_x + 1u8);
        let frobenius =
            BigDecimal::from(key_length as u64 * (shape.lookup_width * shape.digits) as u64)
                .sqrt()
                .ok_or_else(|| PeError::Parameters("sqrt".into()))?;
        let s1 = frobenius * BigDecimal::from(num_bigint::BigInt::from(n * b_x));
        let spread = (BigDecimal::from(1u8) + s1.square())
            .sqrt()
            .ok_or_else(|| PeError::Parameters("sqrt".into()))?;
        Ok(decimal_of(smoothing_parameter(bits))? * decimal_of(gadget_sigma)? * spread)
    }

    /// `B_R = floor(c_R σ_R)` (`Strict` only).
    pub fn randomness_bound(&self, randomness_sigma: &BigDecimal) -> Result<BigUint, PeError> {
        let tail = self
            .randomness_tail
            .ok_or_else(|| PeError::Parameters("the randomness budget is absent".into()))?;
        (decimal_of(tail)? * randomness_sigma)
            .with_scale(0)
            .to_bigint()
            .and_then(|value| value.to_biguint())
            .ok_or_else(|| PeError::Parameters("the randomness bound is not representable".into()))
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
            lookup_width: 2,
            crt_depth: 4,
            switching_packages: 1,
            helpers: 1,
        }
    }

    #[test]
    fn the_budget_reproduces_the_plan_estimates() {
        let shape = recorded_shape();
        assert_eq!(shape.encodings(), 513.0);
        assert_eq!(shape.lhl_applications(), 16_440.0);
        let modulus = BigUint::from(1u8) << 168u32;
        let request = BudgetRequest {
            shape,
            modulus: &modulus,
            smallest_prime: (1u64 << 42) - 1,
            statistical_bits: 30,
            flooding_bits: 30,
            max_anchor_columns: 400,
            key_queries: None,
            flooding_enforced: true,
            trapdoor_sigma: 256.0,
            programming_sigma: &BigDecimal::from(256),
            randomness_sigma: &BigDecimal::from(256),
            error_sigma: &"3.19".parse().unwrap(),
        };
        let budget = StatisticalBudget::new(&request).unwrap();
        // The regularity lemma takes `ε / 2`; truncation, the wide flooding sampler, and the other
        // steps share the other half.
        assert!((budget.lhl_bits - (31.0 + 16_440f64.log2())).abs() < 1e-9);
        assert!((budget.item_bits - (31.0 + 3f64.log2())).abs() < 1e-9);
        assert!((budget.rho - 146.0).abs() < 2.0, "rho {}", budget.rho);
        // K counts χ (anchor, helpers, message), Z_t, and χ_fl at m_B,max: about 2^27 here.
        let count = shape.truncated_coefficients(400);
        assert!((count.log2() - 26.9).abs() < 0.1, "log K {}", count.log2());
        let c = budget.cutoff_multiplier;
        assert!((count * 2.0 * (-c * c / 2.0).exp() - 2f64.powf(-budget.item_bits)).abs() < 1e-15);
        assert!((budget.flooding_tail - 8.89).abs() < 0.05, "c_t {}", budget.flooding_tail);
        assert!((budget.smoothing - 3.75).abs() < 0.1, "η {}", budget.smoothing);
        assert_eq!(budget.anchor_columns, 79);
        let log_flooding = (to_f64(&budget.flooding_sigma) * paper_scale()).log2();
        assert!((log_flooding - 55.7).abs() < 0.3, "log s_fl {log_flooding}");
    }

    #[test]
    fn narrow_widths_and_excess_columns_are_rejected() {
        let modulus = BigUint::from(1u8) << 168u32;
        let mut request = BudgetRequest {
            shape: recorded_shape(),
            modulus: &modulus,
            smallest_prime: (1u64 << 42) - 1,
            statistical_bits: 30,
            flooding_bits: 30,
            max_anchor_columns: 400,
            key_queries: Some(1 << 20),
            flooding_enforced: true,
            trapdoor_sigma: 1.0,
            programming_sigma: &BigDecimal::from(256),
            randomness_sigma: &BigDecimal::from(256),
            error_sigma: &"3.19".parse().unwrap(),
        };
        // σ_T = 1 satisfies the entropy inequality with enough columns, but not the smoothing
        // lower bound.
        assert!(StatisticalBudget::new(&request).is_err());
        request.trapdoor_sigma = 256.0;
        request.max_anchor_columns = 78;
        assert!(StatisticalBudget::new(&request).is_err());
        request.max_anchor_columns = 400;
        let budget = StatisticalBudget::new(&request).unwrap();
        assert!(budget.randomness_tail.unwrap() > budget.cutoff_multiplier);
        let wider = BudgetRequest { key_queries: Some(1 << 40), ..request.clone() };
        assert!(
            StatisticalBudget::new(&wider).unwrap().key_simulation_bits.unwrap() >
                budget.key_simulation_bits.unwrap()
        );
        let looser = BudgetRequest { statistical_bits: 20, ..request };
        assert!(StatisticalBudget::new(&looser).unwrap().anchor_columns < budget.anchor_columns);
    }

    /// `N_LHL` counts the ciphertext's switching packages: under the LUT radix 2, `W = 256` has 8
    /// instead of 255, so with one attribute the regularity lemma is applied about 28 times
    /// less, and the `K` of truncation counts the published helpers.
    #[test]
    fn the_regularity_count_follows_the_switching_packages() {
        let direct = BudgetShape {
            lookup_width: 256,
            switching_packages: 255,
            helpers: 255,
            ..recorded_shape()
        };
        let radix = BudgetShape { switching_packages: 8, helpers: 15, ..direct };
        let ratio = direct.lhl_applications() / radix.lhl_applications();
        assert!((ratio - 256.0 / 9.0).abs() < 0.1, "ratio {ratio}");
        assert!(radix.truncated_coefficients(400) < direct.truncated_coefficients(400));
    }
}
