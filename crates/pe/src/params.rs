//! Scheme parameters: the function class, the equal-basis RNS-BGG chain, the
//! sampling contracts, the trapdoor instantiation, and the key width.
//!
//! Every logical level of the chain is the same ordered CRT basis, so every
//! Dynamic evaluation step has scale one and no rounding; the chain has
//! `2 T_max + 2` levels (see [`PeClass::levels`]).
//!
//! Every fresh error is a sample of one contract `chi`, a discrete Gaussian in the sampler's
//! standard-deviation convention cut at the budget's multiplier: the anchor `b`, the helpers
//! `h_k`, the message `c_m`, the initial state `c_0`, every digit companion, and the error `e_t`
//! of every attribute ciphertext `C_t = [a_t; -s_bar a_t + e_t] + X^(Bz x_t) G`.
//!
//! The key is sampled at one width `s_key` with one cutoff `B_key = floor(6.5 s_key)` for both the
//! anchor preimages and the extension coordinates.
//!
//! # The trapdoor
//!
//! TrapGen is the manuscript's statistical instantiation `[ā | g - ā T]` with `m_B = m̄ + D`
//! columns, where `m̄` and the truncation multiplier of `chi` come from the statistical budget of
//! [`crate::budget`]. The selection takes the manuscript's implementation relaxations: no noise
//! flooding, no uniformity conditions on the hybrids' `R` and `Z_t`, and keys at the trapdoor's
//! minimum preimage width. The heuristics are stated in [`crate::budget`] (ideal samplers,
//! heuristic preimage widths, truncation, and rejection).
use crate::{
    PeClass, PeError,
    budget::{BudgetRequest, BudgetShape, StatisticalBudget, paper_scale},
};
use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::bounds::{
        TrapdoorPreimageCutoffPolicy, cutoff_is_representable, minimum_preimage_sigma,
        rational_from_decimal,
    },
};
use mxx_exponent_lut::rns_bgg::{
    RhsMode, RnsBggDynamicDistributions, RnsBggGaussian, RnsBggParameters,
};
use mxx_ir_core::types::TrapdoorInstantiation;
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::ops::RangeInclusive;

/// The manuscript's Gaussian width `s` (density `exp(-pi |x|^2 / s^2)`) of
/// the sampler's standard deviation `sigma`: `s = sqrt(2 pi) sigma`.
pub fn paper_width(sigma: &BigDecimal) -> f64 {
    paper_scale() * sigma.to_f64().unwrap_or(f64::NAN)
}

/// A parameter search request. Widths are the samplers' standard deviations.
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
    /// `σ_T`: the TrapGen sigma (of `R` and `E`, or of `T`), also the gadget sampler's sigma
    /// factor.
    pub trapdoor_sigma: f64,
    /// Sigma of `chi`, the contract of every fresh error.
    pub error_sigma: BigDecimal,
    /// The radix `B` of the LUT automorphism decomposition (see
    /// [`PeParameters::with_lut_radix`]); `None` switches every automorphism directly (`B = W`).
    pub lut_radix: Option<usize>,
    /// Publish the ciphertext's fresh encodings in two columns over `pq` (see
    /// [`PeParameters::with_compact_encodings`]).
    pub compact_encodings: bool,
    /// The noise model of the correctness check: `None` for the worst case,
    /// `Some(k)` for the average case with failure target `2^-k` (see
    /// [`crate::noise::simulate_average`]).
    pub failure_exponent: Option<u32>,
    /// A Ring-LWE security target for every instance of
    /// [`PeParameters::security_bits`], estimated by `lattice-estimator-cli`;
    /// `None` estimates nothing (test parameters).
    pub lattice_security_bits: Option<u64>,
}

/// Validated scheme parameters.
#[derive(Clone, Debug)]
pub struct PeParameters {
    class: PeClass,
    rns: RnsBggParameters,
    trapdoor_sigma: f64,
    instantiation: TrapdoorInstantiation,
    error: RnsBggGaussian,
    key_width: BigDecimal,
    key_cutoff: BigUint,
    anchor_columns: usize,
    budget: Option<StatisticalBudget>,
    lut_radix: Option<usize>,
    compact_extension: Option<Vec<u64>>,
    identity: [u8; 32],
}

#[derive(Serialize)]
struct ParameterIdentity<'a> {
    class: &'a PeClass,
    rns: String,
    trapdoor_sigma: u64,
    instantiation: TrapdoorInstantiation,
    error: &'a RnsBggGaussian,
    key_width: String,
    key_cutoff: &'a BigUint,
    anchor_columns: usize,
    budget: &'a Option<StatisticalBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lut_radix: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_extension: &'a Option<Vec<u64>>,
}

/// One ring level of `request` at CRT depth `depth`.
fn level_at(request: &PeParameterRequest, depth: usize) -> Result<DCRTPolyParams, PeError> {
    DCRTPolyParams::try_new(
        request.ring_dimension,
        depth,
        request.crt_bits,
        request.base_bits,
        None,
        None,
    )
    .map_err(|error| PeError::Parameters(format!("{error:?}")))
}

/// The equal-basis Dynamic chain of `class` over `level`.
fn chain(
    class: &PeClass,
    level: &DCRTPolyParams,
    error: &RnsBggGaussian,
) -> Result<RnsBggParameters, PeError> {
    class.validate_ring_dimension(level.ring_dimension() as usize)?;
    if level.dropped_moduli() != 0 {
        return Err(PeError::Parameters("the ring must be a full CRT basis".into()));
    }
    let rns = RnsBggParameters::new(
        vec![level.clone(); class.levels()],
        error.sigma().clone(),
        class.levels(),
    )?
    .with_rhs_mode(RhsMode::Dynamic);
    rns.validate_equal_basis()?;
    Ok(rns)
}

fn gadget_base(level: &DCRTPolyParams) -> Result<u32, PeError> {
    1u32.checked_shl(level.base_bits())
        .ok_or_else(|| PeError::Parameters("gadget base overflow".into()))
}

/// `|J_B|`: the helper automorphisms of `class` over rings of dimension `n` under the LUT radix
/// `radix` (`W - 1` without one).
fn helper_count(class: &PeClass, n: usize, radix: Option<usize>) -> Result<usize, PeError> {
    let automorphisms = class
        .layout()
        .automorphism_indices(n)
        .map_err(|error| PeError::Parameters(error.to_string()))?;
    Ok(mxx_exponent_lut::rns_bgg::helper_automorphisms(&automorphisms, n, radix).len())
}

impl PeParameters {
    /// Statistical parameters with an explicit `m̄` and the minimum key width, without the
    /// budget's conditions: for unit tests with noise-free (zero-width) contracts, which the
    /// selection of [`Self::at_depth`] rejects.
    #[cfg(test)]
    pub(crate) fn statistical_unchecked(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        m_bar: usize,
        error: RnsBggGaussian,
    ) -> Result<Self, PeError> {
        let instantiation = TrapdoorInstantiation::Statistical { m_bar };
        let key_width = minimum_preimage_sigma(
            level.ring_dimension(),
            1,
            level.modulus_digits(),
            gadget_base(&level)?,
            trapdoor_sigma,
            instantiation,
        )
        .ok_or_else(|| PeError::Parameters("invalid trapdoor sampler parameters".into()))?;
        Self::assemble(class, level, trapdoor_sigma, instantiation, error, key_width, None)
    }

    /// The parameters of a TrapGen of `instantiation` and keys of width `key_width`, cut at the
    /// default preimage cutoff.
    fn assemble(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        instantiation: TrapdoorInstantiation,
        error: RnsBggGaussian,
        key_width: BigDecimal,
        budget: Option<StatisticalBudget>,
    ) -> Result<Self, PeError> {
        let invalid = |error: &dyn std::fmt::Display| PeError::Parameters(error.to_string());
        let rns = chain(&class, &level, &error)?;
        let key_cutoff = TrapdoorPreimageCutoffPolicy::Default
            .resolve(&rational_from_decimal(&key_width))
            .map_err(|error| invalid(&error))?;
        let anchor_columns = instantiation
            .columns(1, level.modulus_digits())
            .ok_or_else(|| invalid(&"invalid trapdoor instantiation"))?;
        let mut parameters = Self {
            class,
            rns,
            trapdoor_sigma,
            instantiation,
            error,
            key_width,
            key_cutoff,
            anchor_columns,
            budget,
            lut_radix: None,
            compact_extension: None,
            identity: [0; 32],
        };
        parameters.identity = parameters.digest();
        Ok(parameters)
    }

    /// Decompose the LUT automorphism group `H = <g>` (of order `W = B^r`) in
    /// radix `B`: encryption publishes switching packages for the `(B - 1) r`
    /// automorphisms `K_B = {g^(a B^j)}` instead of `W - 1`, the key carries
    /// helper blocks for `J_B = K_B u K_B^-1`, and a branch is switched once
    /// per nonzero radix-`B` digit of its exponent, at most `r` times (see
    /// `mxx_exponent_lut::rns_bgg::RnsBggDynamicSetup::with_lut_radix`). `B`
    /// is a power of two with `B^r = W`; `B = W` keeps the direct switching.
    /// `B < W` needs the cyclic group of `W <= n / 2`.
    pub fn with_lut_radix(mut self, radix: usize) -> Result<Self, PeError> {
        let order = self.class.lookup_width();
        let mut power = 1usize;
        while radix >= 2 && power < order {
            power = power.saturating_mul(radix);
        }
        if radix < 2 || !radix.is_power_of_two() || power != order {
            return Err(PeError::Parameters(
                "the LUT radix must be a power of two whose power is the lookup width".into(),
            ));
        }
        if radix < order && 2 * order > self.level().ring_dimension() as usize {
            return Err(PeError::Parameters(
                "a LUT radix below the lookup width needs a cyclic group (W <= n / 2)".into(),
            ));
        }
        self.lut_radix = (radix < order).then_some(radix);
        self.identity = self.digest();
        Ok(self)
    }

    /// The radix of the LUT automorphism decomposition, `None` if every
    /// automorphism is switched directly.
    pub const fn lut_radix(&self) -> Option<usize> {
        self.lut_radix
    }

    /// Publish the ciphertext's fresh encodings (the digit companions, the
    /// automorphism helpers, and the initial state `c_0`) as
    /// `s (A~ - p X I_2) + e` mod `pq` instead of `s (A - X G) + e` mod `q`, for
    /// the auxiliary modulus `p`: the product of as many CRT primes as `q`'s,
    /// of the same bit size, `1 mod 2n`, and disjoint from `q`'s. Decryption
    /// expands each to `ModSwitch_(pq -> q)(c G~)` (see
    /// `mxx_exponent_lut::rns_bgg::RnsBggDynamicSetup::with_compact_encodings`).
    /// The packages shrink from `m^3` polynomials over `q` to `2 m^2` over
    /// `pq`; the attribute GSW ciphertexts `C_t` stay at `q`, and so does the
    /// number `m^2` of companions per package, which encrypt the digits of a
    /// GSW ciphertext over `q`. [`Self::security_bits`] then also estimates
    /// Ring-LWE at `pq`. The manuscript's reduction programs the package keys
    /// over `q`, so compact parameters are outside its theorem.
    pub fn with_compact_encodings(mut self) -> Result<Self, PeError> {
        let level = self.level().clone();
        let depth = level.crt_depth();
        let base = level.to_crt().0;
        let candidates = DCRTPolyParams::try_new(
            level.ring_dimension(),
            2 * depth,
            level.crt_bits(),
            level.base_bits(),
            None,
            None,
        )
        .map_err(|error| PeError::Parameters(error.to_string()))?
        .to_crt()
        .0;
        let extension = candidates
            .into_iter()
            .filter(|prime| !base.contains(prime))
            .take(depth)
            .collect::<Vec<_>>();
        if extension.len() != depth {
            return Err(PeError::Parameters("too few auxiliary CRT primes".into()));
        }
        self.compact_extension = Some(extension);
        self.identity = self.digest();
        Ok(self)
    }

    /// The CRT primes of the auxiliary modulus `p` of compact encodings.
    pub fn compact_extension(&self) -> Option<&[u64]> {
        self.compact_extension.as_deref()
    }

    /// The ring parameters of `q p` of compact encodings: `q`'s basis followed
    /// by `p`'s.
    pub fn compact_level(&self) -> Result<Option<DCRTPolyParams>, PeError> {
        let Some(extension) = &self.compact_extension else { return Ok(None) };
        let level = self.level();
        let moduli =
            level.to_crt().0.into_iter().chain(extension.iter().copied()).collect::<Vec<_>>();
        DCRTPolyParams::try_new(
            level.ring_dimension(),
            moduli.len(),
            level.crt_bits(),
            level.base_bits(),
            Some(moduli),
            None,
        )
        .map(Some)
        .map_err(|error| PeError::Parameters(error.to_string()))
    }

    /// The rings an execution of these parameters registers: the chain's ring,
    /// and the `q p` ring of compact encodings.
    pub fn rings(&self) -> Result<Vec<DCRTPolyParams>, PeError> {
        let mut rings = vec![self.level().clone()];
        rings.extend(self.compact_level()?);
        Ok(rings)
    }

    /// `|P| = |J_B|`: the automorphisms with a published helper (`W - 1` without a radix).
    pub fn helper_automorphism_count(&self) -> Result<usize, PeError> {
        helper_count(&self.class, self.level().ring_dimension() as usize, self.lut_radix)
    }

    /// The parameters of `request` at CRT depth `depth`: the budget's selection
    /// ([`crate::budget`]).
    pub fn at_depth(request: &PeParameterRequest, depth: usize) -> Result<Self, PeError> {
        let parameters = Self::statistical_at(request, level_at(request, depth)?)?;
        if request.compact_encodings { parameters.with_compact_encodings() } else { Ok(parameters) }
    }

    /// The selection at one ring level: the budget and `m_B`, `chi` with the budget's cutoff,
    /// the trapdoor's minimum preimage width for the key, and the numeric range of the key cutoff.
    fn statistical_at(
        request: &PeParameterRequest,
        level: DCRTPolyParams,
    ) -> Result<Self, PeError> {
        let class = request.class;
        let digits = level.modulus_digits();
        let modulus = level.modulus().as_ref().clone();
        let n = level.ring_dimension() as usize;
        let automorphisms = class
            .layout()
            .automorphism_indices(n)
            .map_err(|error| PeError::Parameters(error.to_string()))?;
        let shape = BudgetShape {
            ring_dimension: n,
            digits,
            attribute_length: class.rhs_packages(),
            crt_depth: level.crt_depth(),
            switching_packages: mxx_exponent_lut::rns_bgg::switching_automorphisms(
                &automorphisms,
                n,
                request.lut_radix,
            )
            .len(),
            helpers: helper_count(&class, n, request.lut_radix)?,
        };
        let budget = StatisticalBudget::new(&BudgetRequest {
            shape,
            modulus: &modulus,
            smallest_prime: level.to_crt().0.into_iter().min().unwrap_or(0),
            trapdoor_sigma: request.trapdoor_sigma,
            error_sigma: &request.error_sigma,
        })?;
        let error =
            RnsBggGaussian::new(request.error_sigma.clone(), budget.cutoff(&request.error_sigma)?)?;
        let instantiation = TrapdoorInstantiation::Statistical { m_bar: budget.m_bar(digits) };
        let key_width = minimum_preimage_sigma(
            level.ring_dimension(),
            1,
            digits,
            gadget_base(&level)?,
            request.trapdoor_sigma,
            instantiation,
        )
        .ok_or_else(|| PeError::Parameters("invalid trapdoor sampler parameters".into()))?;
        let parameters = Self::assemble(
            class,
            level,
            request.trapdoor_sigma,
            instantiation,
            error,
            key_width,
            Some(budget),
        )?;
        if !cutoff_is_representable(parameters.key_cutoff(), &modulus) {
            return Err(PeError::Parameters(format!(
                "the key cutoff 2^{} is not below 2^52 and q / 2",
                parameters.key_cutoff().bits()
            )));
        }
        match request.lut_radix {
            Some(radix) => parameters.with_lut_radix(radix),
            None => Ok(parameters),
        }
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
            instantiation: self.instantiation,
            error: &self.error,
            key_width: self.key_width.to_string(),
            key_cutoff: &self.key_cutoff,
            anchor_columns: self.anchor_columns,
            budget: &self.budget,
            lut_radix: self.lut_radix,
            compact_extension: &self.compact_extension,
        };
        let mut digest = Sha256::new();
        digest.update(b"mxx-pe/parameters/v2");
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

    /// The trapdoor instantiation of TrapGen and of every preimage.
    pub const fn instantiation(&self) -> TrapdoorInstantiation {
        self.instantiation
    }

    /// The statistical budget of the selection; `None` for the unit tests' unchecked parameters.
    pub const fn budget(&self) -> Option<&StatisticalBudget> {
        self.budget.as_ref()
    }

    /// The trapdoor sigma as the exact graph expression every trapdoor node
    /// and import of the setup uses.
    pub(crate) fn trapdoor_sigma_expr(&self) -> Result<mxx_ir_core::RealExpr, PeError> {
        mxx_ir_core::RealExpr::from_f64_exact(self.trapdoor_sigma)
            .map_err(|_| PeError::Parameters("the trapdoor sigma is not finite".into()))
    }

    /// The preimage width the trapdoor nodes request: the key width.
    pub(crate) fn preimage_sigma_expr(&self) -> Result<Option<mxx_ir_core::RealExpr>, PeError> {
        Ok(Some(real_expr(&self.key_width)?))
    }

    /// `chi`: every fresh error (the anchor, the helpers, the message, the initial state, the
    /// digit companions, and the attribute ciphertexts).
    pub const fn error(&self) -> &RnsBggGaussian {
        &self.error
    }

    /// The key width `σ_key` (a standard deviation): the trapdoor's minimum preimage width
    /// ([`minimum_preimage_sigma`], `√(c_G² (1 + ŝ_1²) + σ_T²)` with
    /// `ŝ_1 = σ_T √n (√m̄ + √D + √(ln(n / (2δ))))`).
    pub const fn key_width(&self) -> &BigDecimal {
        &self.key_width
    }

    /// The common cutoff of every key coefficient.
    pub const fn key_cutoff(&self) -> &BigUint {
        &self.key_cutoff
    }

    /// `m_B = m̄ + D`: the column count of TrapGen's public row.
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

    /// The key length `N = W m_B + W |P| m + m` in ring elements, for the `|P|`
    /// helper automorphisms (`W - 1` without a radix).
    pub fn key_length(&self) -> usize {
        let w = self.class.lookup_width();
        let helpers = self.helper_automorphism_count().unwrap_or(w - 1);
        w * self.anchor_columns + w * helpers * self.state_columns() + self.state_columns()
    }

    /// The bound `B_C` on the attribute ciphertext error `e_C = e_t`, a fresh `chi` sample: its
    /// cutoff.
    pub fn ciphertext_error_bound(&self) -> BigUint {
        self.error.cutoff().clone()
    }

    /// The sampling contracts of the Dynamic packages, all `chi`. Encryption supplies the
    /// attribute ciphertexts itself.
    pub fn distributions(&self) -> RnsBggDynamicDistributions {
        RnsBggDynamicDistributions {
            input: self.error.clone(),
            companion: self.error.clone(),
            helper: self.error.clone(),
            ciphertext: self.error.clone(),
        }
    }

    /// The ring modulus `q`.
    pub fn modulus(&self) -> BigUint {
        self.level().modulus().as_ref().clone()
    }

    /// The `lattice-estimator-cli` estimate, with unboundedly many samples, of the Ring-LWE
    /// instance of the encryption secret: the uniform ternary `s_bar` with the sigma of `chi`,
    /// since every published encoding under `s_bar` (the anchor, the helpers, the message, the
    /// initial state, and the digit companions) is a sample of it. TrapGen is uniform by the
    /// regularity conditions and has no estimate.
    ///
    /// With compact encodings ([`Self::with_compact_encodings`]), the companions, helpers, and
    /// initial state are samples over `pq` with the same secret and the same fresh sigmas, so
    /// the estimate is the smaller of the estimates over `q` and over `pq`.
    pub fn security_bits(&self) -> Result<u64, PeError> {
        let error = self.error.sigma().clone();
        let mut encryption = mxx_exponent_lut::utils::lattice_security_bits(self.level(), &error)
            .map_err(PeError::Parameters)?;
        if let Some(compact) = self.compact_level()? {
            encryption = encryption.min(
                mxx_exponent_lut::utils::lattice_security_bits(&compact, &error)
                    .map_err(PeError::Parameters)?,
            );
        }
        Ok(encryption)
    }

    /// Search `request`'s CRT depths, in order, for the first candidate that [`Self::at_depth`]
    /// accepts, whose evaluation of the class under the request's noise model (see
    /// [`crate::noise::simulate`] and [`crate::noise::simulate_average`]) decrypts, and whose
    /// Ring-LWE estimates ([`Self::security_bits`]) meet the target, if any. Each candidate's
    /// conditions are checked in full: correctness need not be monotone in the depth together
    /// with security.
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
                let security = parameters.security_bits()?;
                if security < target {
                    rejected.push(format!("depth {depth}: {security} security bits"));
                    continue;
                }
            }
            return Ok((parameters, report));
        }
        Err(PeError::Parameters(format!("no candidate accepted: {}", rejected.join("; "))))
    }

    /// [`Self::select`] over the ring dimensions `dimensions`, in order: the first accepted
    /// candidate of the first dimension that has one. Each candidate recomputes its auxiliary
    /// modulus, noise, and security estimates (over `pq` too with compact encodings), so a
    /// dimension too small for `pq` is rejected by its security estimate.
    pub fn select_dimension(
        request: &PeParameterRequest,
        dimensions: impl IntoIterator<Item = u32>,
    ) -> Result<(Self, crate::PeNoiseReport), PeError> {
        let mut rejected = Vec::new();
        for ring_dimension in dimensions {
            let request = PeParameterRequest { ring_dimension, ..request.clone() };
            match Self::select(&request) {
                Ok(accepted) => return Ok(accepted),
                Err(error) => rejected.push(format!("n = {ring_dimension}: {error}")),
            }
        }
        Err(PeError::Parameters(format!("no ring dimension accepted: {}", rejected.join("; "))))
    }
}

/// The value of `key_width` as an exact rational for a graph sampler.
pub(crate) fn real_expr(value: &BigDecimal) -> Result<mxx_ir_core::RealExpr, PeError> {
    Ok(mxx_ir_core::RealExpr::Rational(rational_from_decimal(value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_width_is_the_samplers_and_the_anchor_its_trapgen_width() {
        let class = PeClass::new(2, 2, 2, 1).unwrap();
        let level = DCRTPolyParams::new(16, 2, 17, 6, None, None);
        let contract = RnsBggGaussian::with_default_cutoff(1.into()).unwrap();
        let m_bar = 2 * level.modulus_digits();
        let parameters =
            PeParameters::statistical_unchecked(class, level.clone(), 4.578, m_bar, contract)
                .unwrap();
        assert_eq!(parameters.anchor_columns(), m_bar + level.modulus_digits());
        assert_eq!(
            parameters.key_cutoff(),
            &mxx_backends::sampler::bounds::default_preimage_cutoff(
                16,
                1,
                level.modulus_digits(),
                64,
                4.578,
                TrapdoorInstantiation::Statistical { m_bar },
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

    /// The selection's anchor and `chi` follow its budget, and the key takes the trapdoor's
    /// minimum preimage width with the default cutoff.
    #[test]
    fn the_statistical_selection_follows_its_budget() {
        let request = crate::test_support::statistical_request(PeClass::new(1, 2, 1, 1).unwrap());
        let parameters = PeParameters::at_depth(&request, 4).unwrap();
        let budget = parameters.budget().unwrap();
        let digits = parameters.digits();
        assert_eq!(parameters.anchor_columns(), budget.anchor_columns);
        assert_eq!(
            parameters.instantiation(),
            TrapdoorInstantiation::Statistical { m_bar: budget.anchor_columns - digits }
        );
        // `chi` has the requested width and the budget's cutoff `c σ`, above the default `6.5 σ`.
        assert!(budget.cutoff_multiplier > 6.5);
        assert_eq!(parameters.error().sigma(), &request.error_sigma);
        assert_eq!(parameters.error().cutoff(), &budget.cutoff(&request.error_sigma).unwrap());
        assert_eq!(
            parameters.key_width(),
            &minimum_preimage_sigma(
                16,
                1,
                digits,
                gadget_base(parameters.level()).unwrap(),
                request.trapdoor_sigma,
                parameters.instantiation(),
            )
            .unwrap()
        );
        assert_eq!(
            parameters.key_cutoff(),
            &mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound(parameters.key_width())
        );
        let longest = crate::BranchingProgram::longest(*parameters.class());
        assert!(crate::noise::simulate(&parameters, &longest).is_ok());
    }

    #[test]
    fn the_statistical_selection_rejects_a_narrow_chi() {
        let mut request =
            crate::test_support::statistical_request(PeClass::new(1, 2, 1, 1).unwrap());
        // A zero `chi` is below the smoothing parameter.
        request.error_sigma = BigDecimal::from(0u8);
        assert!(PeParameters::at_depth(&request, 4).is_err());
    }

    /// The LUT radix `B` is a power of two with `B^r = W`: over rings of dimension 16, `W = 8`
    /// takes `B = 2` (`|K_2| = 3`, so at most 6 helpers) and `B = 8` (the direct switching, 7
    /// helpers) but not `B = 4` or `B = 3`, and `W = 16 = n`, whose group is not cyclic, takes
    /// only the direct `B = 16`. The key length counts the `|J_B|` helpers.
    #[test]
    fn lut_radices_are_powers_of_two_of_the_lookup_width() {
        let width_8 = crate::test_support::parameters(PeClass::new(1, 4, 2, 1).unwrap());
        assert_eq!(width_8.class().lookup_width(), 8);
        let radix_2 = width_8.clone().with_lut_radix(2).unwrap();
        assert_eq!(radix_2.lut_radix(), Some(2));
        let helpers = radix_2.helper_automorphism_count().unwrap();
        assert!((3..=6).contains(&helpers), "{helpers}");
        let m = radix_2.state_columns();
        assert_eq!(radix_2.key_length(), 8 * radix_2.anchor_columns() + 8 * helpers * m + m);
        let direct = width_8.clone().with_lut_radix(8).unwrap();
        assert_eq!(direct.lut_radix(), None);
        assert_eq!(direct.identity(), width_8.identity());
        assert_eq!(direct.helper_automorphism_count().unwrap(), 7);
        assert_ne!(radix_2.identity(), direct.identity());
        for radix in [0, 1, 3, 4, 16] {
            assert!(width_8.clone().with_lut_radix(radix).is_err(), "radix {radix}");
        }
        let width_16 = crate::test_support::parameters(PeClass::new(1, 8, 2, 1).unwrap());
        assert_eq!(width_16.level().ring_dimension(), 16);
        assert!(width_16.clone().with_lut_radix(2).is_err());
        assert_eq!(width_16.with_lut_radix(16).unwrap().lut_radix(), None);
    }
}
