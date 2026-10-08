//! Scheme parameters: the function class, the equal-basis RNS-BGG chain, the
//! sampling contracts, and the key width of the existing trapdoor sampler.
//!
//! Every logical level of the chain is the same ordered CRT basis, so every
//! Dynamic evaluation step has scale one and no rounding; the chain has
//! `2 T_max + 2` levels (see [`PeClass::levels`]).
//!
//! The sampling contracts are discrete Gaussians in the sampler's
//! standard-deviation convention with hard cutoffs:
//!
//! - `chi` (the error contract) for the anchor `b`, the helpers `h_k`, and the message `c_m`;
//! - `chi_fl` (the flooding contract) for the initial state `c_0` and every digit companion;
//! - the programming contract for the entries of every `Z_t`.
//!
//! [`PeParameterRequest::tail_cutoff`] derives each cutoff from the total
//! number of coefficients an encryption samples under it, so a rejection
//! sampler's truncation and the union of its tail events stay below
//! `2^-lambda`. The key is sampled at the width `s_key` the existing
//! sampler realizes for the one-row trapdoor ([`PeParameters::key_width`]):
//! both the anchor preimages and the extension coordinates use it, with one
//! cutoff.
use crate::{PeClass, PeError};
use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::{TrapdoorPreimageCutoffPolicy, compute_preimage_sigma},
};
use mxx_exponent_lut::rns_bgg::{
    RhsMode, RnsBggDynamicDistributions, RnsBggGaussian, RnsBggParameters,
};
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::ops::RangeInclusive;

/// The manuscript's Gaussian width `s` (density `exp(-pi |x|^2 / s^2)`) of
/// the sampler's standard deviation `sigma`: `s = sqrt(2 pi) sigma`.
pub fn paper_width(sigma: &BigDecimal) -> f64 {
    (2.0 * std::f64::consts::PI).sqrt() * sigma.to_f64().unwrap_or(f64::NAN)
}

/// A parameter search request.
#[derive(Clone, Debug)]
pub struct PeParameterRequest {
    pub class: PeClass,
    pub ring_dimension: u32,
    /// Bits of every CRT prime.
    pub crt_bits: usize,
    /// Bits of the gadget base.
    pub base_bits: u32,
    /// The CRT depths searched, in order.
    pub crt_depths: RangeInclusive<usize>,
    /// The trapdoor sigma of TrapGen.
    pub trapdoor_sigma: f64,
    /// Sigma of `chi`.
    pub error_sigma: BigDecimal,
    /// Sigma of `chi_fl`.
    pub flooding_sigma: BigDecimal,
    /// Sigma of the entries of `Z_t`.
    pub programming_sigma: BigDecimal,
    /// The statistical parameter `lambda` of the tail budget.
    pub statistical_bits: u32,
    /// The noise model of the correctness check: `None` for the worst case,
    /// `Some(k)` for the average case with failure target `2^-k` (see
    /// [`crate::noise::simulate_average`]).
    pub failure_exponent: Option<u32>,
    /// A Ring-LWE security target for both instances of
    /// [`PeParameters::security_bits`], estimated by `lattice-estimator-cli`;
    /// `None` estimates nothing (test parameters).
    pub lattice_security_bits: Option<u64>,
}

impl PeParameterRequest {
    /// The smallest cutoff `c` with `count * 2 exp(-c^2 / (2 sigma^2)) <=
    /// 2^-lambda`: every one of `count` samples of a Gaussian of standard
    /// deviation `sigma` is within `c` except with probability `2^-lambda`,
    /// so truncating them there (rejection) changes their joint distribution
    /// by at most that much.
    pub fn tail_cutoff(sigma: &BigDecimal, count: &BigUint, statistical_bits: u32) -> BigUint {
        let sigma = sigma.to_f64().unwrap_or(f64::NAN);
        let log_count = count.bits() as f64 * std::f64::consts::LN_2;
        let exponent =
            statistical_bits as f64 * std::f64::consts::LN_2 + log_count + std::f64::consts::LN_2;
        BigUint::from((sigma * (2.0 * exponent).sqrt()).ceil().max(1.0) as u64)
    }
}

/// Validated scheme parameters.
#[derive(Clone, Debug)]
pub struct PeParameters {
    class: PeClass,
    rns: RnsBggParameters,
    trapdoor_sigma: f64,
    error: RnsBggGaussian,
    flooding: RnsBggGaussian,
    programming: RnsBggGaussian,
    key_width: BigDecimal,
    key_cutoff: BigUint,
    anchor_columns: usize,
    identity: [u8; 32],
}

#[derive(Serialize)]
struct ParameterIdentity<'a> {
    class: &'a PeClass,
    rns: String,
    trapdoor_sigma: u64,
    error: &'a RnsBggGaussian,
    flooding: &'a RnsBggGaussian,
    programming: &'a RnsBggGaussian,
    key_width: String,
    key_cutoff: &'a BigUint,
    anchor_columns: usize,
}

impl PeParameters {
    /// Parameters of `class` over one ring of `level` and the given
    /// contracts. The key width is the existing sampler's preimage width for
    /// the one-row trapdoor of sigma `trapdoor_sigma`; `key_cutoff` must not
    /// be below the sampler's default cutoff for it.
    pub fn new(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        error: RnsBggGaussian,
        flooding: RnsBggGaussian,
        programming: RnsBggGaussian,
        key_cutoff: TrapdoorPreimageCutoffPolicy,
    ) -> Result<Self, PeError> {
        let invalid = |error: &dyn std::fmt::Display| PeError::Parameters(error.to_string());
        class.validate_ring_dimension(level.ring_dimension() as usize)?;
        if level.dropped_moduli() != 0 {
            return Err(PeError::Parameters("the ring must be a full CRT basis".into()));
        }
        let rns = RnsBggParameters::new(
            vec![level.clone(); class.levels()],
            flooding.sigma().clone(),
            class.levels(),
        )?
        .with_rhs_mode(RhsMode::Dynamic);
        rns.validate_equal_basis()?;
        let digits = level.modulus_digits();
        let base = 1u32
            .checked_shl(level.base_bits())
            .ok_or_else(|| PeError::Parameters("gadget base overflow".into()))?;
        let key_width = compute_preimage_sigma(
            &BigDecimal::from(level.ring_dimension()).sqrt().ok_or_else(|| invalid(&"sqrt"))?,
            digits as u64,
            &BigDecimal::from(base),
            None,
            Some(trapdoor_sigma),
        );
        let key_cutoff = key_cutoff
            .resolve(level.ring_dimension(), 1, digits, base, trapdoor_sigma)
            .map_err(|error| invalid(&error))?;
        let anchor_columns = anchor_columns(&rns, trapdoor_sigma, &key_cutoff)?;
        let mut parameters = Self {
            class,
            rns,
            trapdoor_sigma,
            error,
            flooding,
            programming,
            key_width,
            key_cutoff,
            anchor_columns,
            identity: [0; 32],
        };
        parameters.identity = parameters.digest();
        Ok(parameters)
    }

    /// The parameters of `request` at CRT depth `depth`, with every cutoff
    /// from the tail budget of one encryption and one key.
    pub fn at_depth(request: &PeParameterRequest, depth: usize) -> Result<Self, PeError> {
        let level = DCRTPolyParams::try_new(
            request.ring_dimension,
            depth,
            request.crt_bits,
            request.base_bits,
            None,
            None,
        )
        .map_err(|error| PeError::Parameters(format!("{error:?}")))?;
        let lambda = request.statistical_bits;
        let class = request.class;
        let n = BigUint::from(request.ring_dimension);
        let m = BigUint::from(2 * level.modulus_digits());
        let w = BigUint::from(class.lookup_width());
        let l = BigUint::from(class.attribute_length());
        // The anchor width before the cutoffs: the trapdoor shape does not
        // depend on them.
        let provisional = Self::new(
            class,
            level.clone(),
            request.trapdoor_sigma,
            RnsBggGaussian::with_default_cutoff(request.error_sigma.clone())?,
            RnsBggGaussian::with_default_cutoff(request.flooding_sigma.clone())?,
            RnsBggGaussian::with_default_cutoff(request.programming_sigma.clone())?,
            TrapdoorPreimageCutoffPolicy::Default,
        )?;
        let m_b = BigUint::from(provisional.anchor_columns);
        let one = BigUint::from(1u8);
        let errors = &n * (&m_b + (&w - &one) * &m + &one);
        let floods = &n * &m * (&one + (&l + &w - &one) * &m * &m);
        let programs = &n * &l * &m_b * &m;
        let keys = &n * provisional.key_length();
        let contract = |sigma: &BigDecimal, count: &BigUint| {
            RnsBggGaussian::new(
                sigma.clone(),
                PeParameterRequest::tail_cutoff(sigma, count, lambda),
            )
        };
        let key_width = provisional.key_width.clone();
        Self::new(
            class,
            level,
            request.trapdoor_sigma,
            contract(&request.error_sigma, &errors)?,
            contract(&request.flooding_sigma, &floods)?,
            contract(&request.programming_sigma, &programs)?,
            TrapdoorPreimageCutoffPolicy::Explicit(
                PeParameterRequest::tail_cutoff(&key_width, &keys, lambda)
                    .max(provisional.key_cutoff.clone()),
            ),
        )
    }

    fn digest(&self) -> [u8; 32] {
        let identity = ParameterIdentity {
            class: &self.class,
            rns: self
                .rns
                .levels()
                .iter()
                .map(|level| format!("{:?}/{}", level.to_crt().0, level.base_bits()))
                .collect::<Vec<_>>()
                .join(";"),
            trapdoor_sigma: self.trapdoor_sigma.to_bits(),
            error: &self.error,
            flooding: &self.flooding,
            programming: &self.programming,
            key_width: self.key_width.to_string(),
            key_cutoff: &self.key_cutoff,
            anchor_columns: self.anchor_columns,
        };
        let mut digest = Sha256::new();
        digest.update(b"mxx-pe/parameters/v1");
        digest.update(serde_json::to_vec(&identity).expect("parameter identity encoding"));
        digest.finalize().into()
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    pub const fn class(&self) -> &PeClass {
        &self.class
    }

    /// The equal-basis Dynamic RNS-BGG chain.
    pub const fn rns(&self) -> &RnsBggParameters {
        &self.rns
    }

    /// The one ring of the chain.
    pub fn level(&self) -> &DCRTPolyParams {
        &self.rns.levels()[0]
    }

    pub const fn trapdoor_sigma(&self) -> f64 {
        self.trapdoor_sigma
    }

    /// The trapdoor sigma as the exact graph expression every trapdoor node
    /// and import of the setup uses.
    pub(crate) fn trapdoor_sigma_expr(&self) -> Result<mxx_ir_core::RealExpr, PeError> {
        mxx_ir_core::RealExpr::from_f64_exact(self.trapdoor_sigma)
            .map_err(|_| PeError::Parameters("the trapdoor sigma is not finite".into()))
    }

    /// `chi`: the anchor, helper, and message errors.
    pub const fn error(&self) -> &RnsBggGaussian {
        &self.error
    }

    /// `chi_fl`: the initial-state and digit-companion errors.
    pub const fn flooding(&self) -> &RnsBggGaussian {
        &self.flooding
    }

    /// The entries of every `Z_t`.
    pub const fn programming(&self) -> &RnsBggGaussian {
        &self.programming
    }

    /// The key width `s_key = 1.8 (b + 1) sigma_T^2 (sqrt(n D) + sqrt(2 n) +
    /// 4.7)`, read from the existing sampler's width
    /// ([`compute_preimage_sigma`]) for the one-row trapdoor.
    pub const fn key_width(&self) -> &BigDecimal {
        &self.key_width
    }

    /// The common cutoff of every key coefficient.
    pub const fn key_cutoff(&self) -> &BigUint {
        &self.key_cutoff
    }

    /// `m_B`: the column count of the existing TrapGen's public row.
    pub const fn anchor_columns(&self) -> usize {
        self.anchor_columns
    }

    /// `D`, the gadget digits of one row.
    pub fn digits(&self) -> usize {
        self.level().modulus_digits()
    }

    /// `m = 2 D`.
    pub fn state_columns(&self) -> usize {
        2 * self.digits()
    }

    /// The key length `N = W m_B + W (W - 1) m + m` in ring elements.
    pub fn key_length(&self) -> usize {
        let w = self.class.lookup_width();
        w * self.anchor_columns + w * (w - 1) * self.state_columns() + self.state_columns()
    }

    /// The bound `B_C <= m_B n B_b B_Z` on the attribute ciphertext error
    /// `e_C = -e_b Z_t`.
    pub fn ciphertext_error_bound(&self) -> BigUint {
        BigUint::from(self.anchor_columns) *
            self.level().ring_dimension() *
            self.error.cutoff() *
            self.programming.cutoff()
    }

    /// The sampling contracts of the Dynamic packages: `chi_fl` for the
    /// initial state and every companion, `chi` for the helpers. Encryption
    /// supplies the attribute ciphertexts itself.
    pub fn distributions(&self) -> RnsBggDynamicDistributions {
        RnsBggDynamicDistributions {
            input: self.flooding.clone(),
            companion: self.flooding.clone(),
            helper: self.error.clone(),
            ciphertext: self.error.clone(),
        }
    }

    /// The ring modulus `q`.
    pub fn modulus(&self) -> BigUint {
        self.level().modulus().as_ref().clone()
    }

    /// The `lattice-estimator-cli` estimates, with unboundedly many samples,
    /// of the two Ring-LWE instances over the ring:
    ///
    /// - encryption: the uniform ternary secret `s_bar` with the smaller of the sigmas of `chi` and
    ///   `chi_fl`, since every published encoding under `s_bar` (the anchor, the helpers, the
    ///   message, the initial state, and the digit companions) is a sample with one of them;
    /// - the trapdoor: TrapGen's row `a_bar r + e` with `r` and `e` discrete Gaussians of the
    ///   trapdoor sigma.
    ///
    /// The anchor's identity column gives the sample `s_bar + B^(2)_1 + e_b` with the
    /// non-uniform coefficient `1`, which these estimates do not model (the existing TrapGen anchor
    /// convention, see the crate documentation).
    pub fn security_bits(&self) -> Result<(u64, u64), PeError> {
        let error = self.error.sigma().min(self.flooding.sigma()).clone();
        let encryption = mxx_exponent_lut::utils::lattice_security_bits(self.level(), &error)
            .map_err(PeError::Parameters)?;
        // The shortest decimal of the `f64`, as the sampler was configured.
        let sigma = self
            .trapdoor_sigma
            .to_string()
            .parse::<BigDecimal>()
            .map_err(|_| PeError::Parameters("the trapdoor sigma is not finite".into()))?;
        let trapdoor = mxx_exponent_lut::utils::gaussian_secret_lattice_security_bits(
            self.level(),
            &sigma,
            &sigma,
        )
        .map_err(PeError::Parameters)?;
        Ok((encryption, trapdoor))
    }

    /// Search `request`'s CRT depths, in order, for the first whose
    /// evaluation of the class under the request's noise model (see
    /// [`crate::noise::simulate`] and [`crate::noise::simulate_average`])
    /// decrypts and whose Ring-LWE estimates ([`Self::security_bits`]) both
    /// meet the target, if any. Each
    /// candidate's conditions are checked in full: correctness need not be
    /// monotone in the depth together with security.
    pub fn select(request: &PeParameterRequest) -> Result<(Self, crate::PeNoiseReport), PeError> {
        let mut rejected = Vec::new();
        for depth in request.crt_depths.clone() {
            let parameters = match Self::at_depth(request, depth) {
                Ok(parameters) => parameters,
                Err(error) => {
                    rejected.push(format!("depth {depth}: {error}"));
                    continue;
                }
            };
            let longest = crate::BranchingProgram::longest(request.class);
            let report = match request.failure_exponent {
                None => crate::noise::simulate(&parameters, &longest)?,
                Some(exponent) => crate::noise::simulate_average(&parameters, &longest, exponent)?,
            };
            if !report.decrypts() {
                rejected.push(format!("depth {depth}: decryption noise exceeds q/4"));
                continue;
            }
            if let Some(target) = request.lattice_security_bits {
                let (encryption, trapdoor) = parameters.security_bits()?;
                if encryption.min(trapdoor) < target {
                    rejected.push(format!(
                        "depth {depth}: {encryption} (encryption) and {trapdoor} (trapdoor) \
                         security bits"
                    ));
                    continue;
                }
            }
            return Ok((parameters, report));
        }
        Err(PeError::Parameters(format!("no candidate accepted: {}", rejected.join("; "))))
    }
}

/// `m_B`, read from the column metadata of the existing one-row TrapGen.
fn anchor_columns(
    rns: &RnsBggParameters,
    trapdoor_sigma: f64,
    cutoff: &BigUint,
) -> Result<usize, PeError> {
    let level = &rns.levels()[0];
    let ring = rns.ring(0)?;
    let trapdoor = ring.sample_trapdoor(
        1,
        mxx_ir_core::RealExpr::from_f64_exact(trapdoor_sigma)
            .map_err(|_| PeError::Parameters("the trapdoor sigma is not finite".into()))?,
        1u64 << level.base_bits(),
        level.modulus_digits(),
        num_bigint::BigInt::from(cutoff.clone()),
    );
    trapdoor
        .public_matrix()
        .matrix_type()
        .columns
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .ok()
        .and_then(|columns| columns.to_usize())
        .ok_or_else(|| PeError::Parameters("the anchor width is not concrete".into()))
}

/// The value of `key_width` as an exact rational for a graph sampler.
pub(crate) fn real_expr(value: &BigDecimal) -> Result<mxx_ir_core::RealExpr, PeError> {
    let (integer, exponent) = value.as_bigint_and_exponent();
    let power = num_bigint::BigInt::from(10u8).pow(exponent.unsigned_abs() as u32);
    let (numerator, denominator) = if exponent < 0 {
        (integer * power, num_bigint::BigInt::from(1u8))
    } else {
        (integer, power)
    };
    mxx_ir_core::Rational::new(numerator, denominator)
        .map(mxx_ir_core::RealExpr::Rational)
        .map_err(|_| PeError::Parameters("a Gaussian width is not a finite rational".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_cutoffs_grow_with_the_sample_count_and_the_security_parameter() {
        let sigma = BigDecimal::from(4u8);
        let few = PeParameterRequest::tail_cutoff(&sigma, &BigUint::from(10u8), 40);
        let many = PeParameterRequest::tail_cutoff(&sigma, &BigUint::from(1u64 << 40), 40);
        let stronger = PeParameterRequest::tail_cutoff(&sigma, &BigUint::from(10u8), 80);
        assert!(few < many && few < stronger);
        // 2 exp(-c^2 / 2 sigma^2) <= 2^-40 / 10 needs c >= 31.5 for sigma 4.
        assert_eq!(few, BigUint::from(32u8));
    }

    #[test]
    fn the_key_width_is_the_samplers_and_the_anchor_its_trapgen_width() {
        let class = PeClass::new(2, 2, 2, 1).unwrap();
        let level = DCRTPolyParams::new(16, 2, 17, 6, None, None);
        let contract = RnsBggGaussian::with_default_cutoff(1.into()).unwrap();
        let parameters = PeParameters::new(
            class,
            level.clone(),
            4.578,
            contract.clone(),
            contract.clone(),
            contract,
            TrapdoorPreimageCutoffPolicy::Default,
        )
        .unwrap();
        assert_eq!(parameters.anchor_columns(), level.modulus_digits() + 2);
        assert_eq!(
            parameters.key_cutoff(),
            &mxx_backends::sampler::bounds::default_preimage_cutoff(
                16,
                1,
                level.modulus_digits(),
                64,
                4.578
            )
            .unwrap()
        );
        assert_eq!(parameters.rns().levels().len(), class.levels());
        assert_eq!(
            parameters.key_length(),
            4 * parameters.anchor_columns() +
                12 * parameters.state_columns() +
                parameters.state_columns()
        );
    }
}
