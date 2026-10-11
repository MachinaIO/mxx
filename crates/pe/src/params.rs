//! Scheme parameters: the function class, the equal-basis RNS-BGG chain, the
//! sampling contracts, the trapdoor instantiation, and the key width.
//!
//! Every logical level of the chain is the same ordered CRT basis, so every
//! Dynamic evaluation step has scale one and no rounding; the chain has
//! `2 T_max + 2` levels (see [`PeClass::levels`]).
//!
//! The sampling contracts are discrete Gaussians in the sampler's
//! standard-deviation convention with hard cutoffs:
//!
//! - `chi` (the error contract) for the anchor `b`, the helpers `h_k`, and the message `c_m`;
//! - `chi_fl` (the flooding contract) for the initial state `c_0`, every digit companion, and the
//!   error `e_t` of every attribute ciphertext `C_t = [a_t; -s_bar a_t + e_t] + X^(Bz x_t) G`.
//!
//! The key is sampled at one width `s_key` with one cutoff `B_key = floor(6.5 s_key)` for both the
//! anchor preimages and the extension coordinates.
//!
//! # Instantiations
//!
//! | | `Statistical` (the manuscript) | `Computational` (legacy) |
//! |---|---|---|
//! | TrapGen | `[ā | g - ā T]`, `m_B = m̄ + D` columns | `[ā | I | G - (ā R + E)]`, `m_B = D + 2` |
//! | `m_B` | the regularity conditions of [`crate::budget`] | `D + 2` |
//! | `chi_fl` | the flooding width of [`crate::budget`], or the requested width without `enforce_flooding` | the requested width |
//! | cutoffs of `chi`, `chi_fl` | `c` standard deviations from the budget | `6.5` |
//! | conditions | regularity, flooding, budget, key width, numeric range | none |
//! | Ring-LWE estimates | encryption | encryption and TrapGen |
//! | key width | [`KeyWidthPolicy`] | the trapdoor's minimum preimage width |
//!
//! `Statistical` with [`KeyWidthPolicy::Strict`] implements every construction and parameter
//! condition of the manuscript, under the heuristics stated in [`crate::budget`] (ideal samplers,
//! heuristic preimage widths, truncation, and rejection). `Statistical` with
//! [`KeyWidthPolicy::Practical`] relaxes the key-simulation width of the reduction.
//! `Statistical` without [`PeParameterRequest::enforce_flooding`] drops the flooding condition, so
//! the hybrid `H_1` of the reduction no longer applies; it measures the cost of flooding.
//! `Computational` keeps the legacy, heuristic parameters and is outside the manuscript's theorem.
use crate::{
    BranchingProgram, PeClass, PeError,
    budget::{BudgetRequest, BudgetShape, KeyWidthPolicy, StatisticalBudget, paper_scale},
};
use bigdecimal::{BigDecimal, ToPrimitive};
use mxx_backends::{
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{
        bounds::{
            TrapdoorPreimageCutoffPolicy, cutoff_is_representable, gadget_sampler_sigma,
            minimum_preimage_sigma, rational_from_decimal,
        },
        uniform::WIDE_GAUSSIAN_MAX,
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

/// The trapdoor instantiation a request selects; the statistical one derives `m̄` from the
/// regularity conditions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum TrapdoorMode {
    Computational,
    Statistical,
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
    /// The trapdoor instantiation.
    pub trapdoor_instantiation: TrapdoorMode,
    /// The key width of the statistical instantiation.
    pub key_width_policy: KeyWidthPolicy,
    /// `σ_T`: the TrapGen sigma (of `R` and `E`, or of `T`), also the gadget sampler's sigma
    /// factor.
    pub trapdoor_sigma: f64,
    /// Sigma of `chi`.
    pub error_sigma: BigDecimal,
    /// Sigma of `chi_fl` of the computational instantiation, and of the statistical one without
    /// `enforce_flooding`; the statistical instantiation with it derives it from the flooding
    /// condition.
    pub flooding_sigma: BigDecimal,
    /// Whether the statistical instantiation imposes the flooding condition: `chi_fl` of the
    /// budget's flooding width. Without it, `chi_fl` is `flooding_sigma` with the budget's cutoff,
    /// and the proof's hybrid `H_1` no longer applies.
    pub enforce_flooding: bool,
    /// `σ_R`: sigma of the matrix `R` of the proof's hybrids `H_1` and `H_4`, and of the `Z_t` by
    /// which the proof simulates the attribute ciphertexts from `(B, b)`. The scheme never samples
    /// either; it sizes `m_B` and the flooding width (`Z_t`'s regularity over two rows and its
    /// flooding bound follow from `R`'s over three).
    pub randomness_sigma: BigDecimal,
    /// `log(1 / ε_total)` of the statistical budget ([`crate::budget`]).
    pub statistical_security_bits: u32,
    /// The per-coefficient flooding distance exponent `s_flood`.
    pub flooding_statistical_bits: u32,
    /// `Q_key`: the key queries of the `Strict` budget.
    pub key_queries: u64,
    /// `m_B,max`: the largest anchor accepted, which fixes the truncation multiplier.
    pub max_anchor_columns: usize,
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

/// The selection record of the statistical instantiation: its budget and the key width's parts.
#[derive(Clone, Debug, Serialize)]
pub struct StatisticalSelection {
    pub key_width_policy: KeyWidthPolicy,
    #[serde(serialize_with = "decimal")]
    pub randomness_sigma: BigDecimal,
    pub key_queries: u64,
    pub max_anchor_columns: usize,
    /// Whether `chi_fl` is the budget's flooding width (see
    /// [`PeParameterRequest::enforce_flooding`]).
    pub flooding_enforced: bool,
    pub budget: StatisticalBudget,
    /// The trapdoor's minimum preimage sigma `σ_min`.
    #[serde(serialize_with = "decimal")]
    pub minimum_key_sigma: BigDecimal,
    /// `Strict` only: `B_R`, the witness bound `B_R̂`, and the SampleRight sigma `σ_SR`.
    pub randomness_bound: Option<BigUint>,
    pub witness_bound: Option<BigUint>,
    #[serde(serialize_with = "optional_decimal")]
    pub sample_right_sigma: Option<BigDecimal>,
}

fn decimal<S: serde::Serializer>(value: &BigDecimal, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&value.to_string())
}

fn optional_decimal<S: serde::Serializer>(
    value: &Option<BigDecimal>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&value.as_ref().map(ToString::to_string).unwrap_or_default())
}

/// Validated scheme parameters.
#[derive(Clone, Debug)]
pub struct PeParameters {
    class: PeClass,
    rns: RnsBggParameters,
    trapdoor_sigma: f64,
    instantiation: TrapdoorInstantiation,
    error: RnsBggGaussian,
    flooding: RnsBggGaussian,
    key_width: BigDecimal,
    key_cutoff: BigUint,
    anchor_columns: usize,
    statistical: Option<StatisticalSelection>,
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
    flooding: &'a RnsBggGaussian,
    key_width: String,
    key_cutoff: &'a BigUint,
    anchor_columns: usize,
    statistical: &'a Option<StatisticalSelection>,
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
    flooding: &RnsBggGaussian,
) -> Result<RnsBggParameters, PeError> {
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

/// `r = log_B W`: the most helper switches on a path from row 1 to any row of the key simulation
/// (`lem:keysim`) under the LUT radix `radix`, one per nonzero digit; one without a radix.
fn radix_path_length(lookup_width: usize, radix: Option<usize>) -> usize {
    match radix.filter(|radix| *radix >= 2 && *radix < lookup_width) {
        Some(radix) => lookup_width.ilog(radix) as usize,
        None => 1,
    }
}

impl PeParameters {
    /// Computational parameters of `class` over one ring of `level` and the given contracts. The
    /// key width is the existing sampler's preimage width for the one-row trapdoor of sigma
    /// `trapdoor_sigma`; `key_cutoff` must not be below its default cutoff.
    pub fn new(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        error: RnsBggGaussian,
        flooding: RnsBggGaussian,
        key_cutoff: TrapdoorPreimageCutoffPolicy,
    ) -> Result<Self, PeError> {
        Self::with_minimum_key_width(
            class,
            level,
            trapdoor_sigma,
            TrapdoorInstantiation::Computational,
            error,
            flooding,
            key_cutoff,
        )
    }

    /// Parameters whose key width is the trapdoor's minimum preimage width.
    #[allow(clippy::too_many_arguments)]
    fn with_minimum_key_width(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        instantiation: TrapdoorInstantiation,
        error: RnsBggGaussian,
        flooding: RnsBggGaussian,
        key_cutoff: TrapdoorPreimageCutoffPolicy,
    ) -> Result<Self, PeError> {
        let key_width = minimum_preimage_sigma(
            level.ring_dimension(),
            1,
            level.modulus_digits(),
            gadget_base(&level)?,
            trapdoor_sigma,
            instantiation,
        )
        .ok_or_else(|| PeError::Parameters("invalid trapdoor sampler parameters".into()))?;
        Self::assemble(
            class,
            level,
            trapdoor_sigma,
            instantiation,
            error,
            flooding,
            key_width,
            key_cutoff,
            None,
        )
    }

    /// Statistical parameters with an explicit `m̄` and the minimum key width, without the
    /// budget's conditions: for unit tests with noise-free (zero-width) contracts, which the
    /// selection of [`Self::at_depth`] rejects.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn statistical_unchecked(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        m_bar: usize,
        error: RnsBggGaussian,
        flooding: RnsBggGaussian,
    ) -> Result<Self, PeError> {
        Self::with_minimum_key_width(
            class,
            level,
            trapdoor_sigma,
            TrapdoorInstantiation::Statistical { m_bar },
            error,
            flooding,
            TrapdoorPreimageCutoffPolicy::Default,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        class: PeClass,
        level: DCRTPolyParams,
        trapdoor_sigma: f64,
        instantiation: TrapdoorInstantiation,
        error: RnsBggGaussian,
        flooding: RnsBggGaussian,
        key_width: BigDecimal,
        key_cutoff: TrapdoorPreimageCutoffPolicy,
        statistical: Option<StatisticalSelection>,
    ) -> Result<Self, PeError> {
        let invalid = |error: &dyn std::fmt::Display| PeError::Parameters(error.to_string());
        let rns = chain(&class, &level, &flooding)?;
        let key_cutoff = key_cutoff
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
            flooding,
            key_width,
            key_cutoff,
            anchor_columns,
            statistical,
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
    /// A `Strict` statistical selection bounds its key width with the radix,
    /// so it takes the radix from its request ([`PeParameterRequest::lut_radix`]).
    pub fn with_lut_radix(self, radix: usize) -> Result<Self, PeError> {
        if self.statistical.as_ref().is_some_and(|selection| selection.witness_bound.is_some()) {
            return Err(PeError::Parameters(
                "a Strict selection takes its LUT radix from the request".into(),
            ));
        }
        self.lut_radix_unchecked(radix)
    }

    /// [`Self::with_lut_radix`] without the `Strict` check.
    fn lut_radix_unchecked(mut self, radix: usize) -> Result<Self, PeError> {
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

    /// The parameters of `request` at CRT depth `depth`: the legacy contracts with the default
    /// cutoffs for `Computational`, the budget's selection ([`crate::budget`]) for `Statistical`.
    pub fn at_depth(request: &PeParameterRequest, depth: usize) -> Result<Self, PeError> {
        let level = level_at(request, depth)?;
        let parameters = match request.trapdoor_instantiation {
            TrapdoorMode::Computational => Self::new(
                request.class,
                level,
                request.trapdoor_sigma,
                RnsBggGaussian::with_default_cutoff(request.error_sigma.clone())?,
                RnsBggGaussian::with_default_cutoff(request.flooding_sigma.clone())?,
                TrapdoorPreimageCutoffPolicy::Default,
            )
            .and_then(|parameters| match request.lut_radix {
                Some(radix) => parameters.with_lut_radix(radix),
                None => Ok(parameters),
            }),
            // The statistical selection bounds the key width with the radix.
            TrapdoorMode::Statistical => Self::statistical_at(request, level),
        }?;
        if request.compact_encodings { parameters.with_compact_encodings() } else { Ok(parameters) }
    }

    /// Steps 1–9 of the statistical selection at one ring level: the budget and `m_B`, the
    /// contracts with the budget's cutoffs and the flooding width, the key width of the policy,
    /// and the numeric range of the key and flooding samples.
    fn statistical_at(
        request: &PeParameterRequest,
        level: DCRTPolyParams,
    ) -> Result<Self, PeError> {
        let class = request.class;
        let digits = level.modulus_digits();
        let base = gadget_base(&level)?;
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
            lookup_width: class.lookup_width(),
            crt_depth: level.crt_depth(),
            switching_packages: mxx_exponent_lut::rns_bgg::switching_automorphisms(
                &automorphisms,
                n,
                request.lut_radix,
            )
            .len(),
            helpers: helper_count(&class, n, request.lut_radix)?,
        };
        let strict = request.key_width_policy == KeyWidthPolicy::Strict;
        let budget = StatisticalBudget::new(&BudgetRequest {
            shape,
            modulus: &modulus,
            smallest_prime: level.to_crt().0.into_iter().min().unwrap_or(0),
            statistical_bits: request.statistical_security_bits,
            flooding_bits: request.flooding_statistical_bits,
            max_anchor_columns: request.max_anchor_columns,
            key_queries: strict.then_some(request.key_queries),
            flooding_enforced: request.enforce_flooding,
            trapdoor_sigma: request.trapdoor_sigma,
            randomness_sigma: &request.randomness_sigma,
            error_sigma: &request.error_sigma,
        })?;
        budget.verify(&shape)?;
        let contract = |sigma: &BigDecimal| -> Result<RnsBggGaussian, PeError> {
            Ok(RnsBggGaussian::new(sigma.clone(), budget.cutoff(sigma)?)?)
        };
        let error = contract(&request.error_sigma)?;
        let flooding_sigma =
            if request.enforce_flooding { &budget.flooding_sigma } else { &request.flooding_sigma };
        let flooding = contract(flooding_sigma)?;
        if flooding_sigma.to_f64().is_none_or(|sigma| sigma > WIDE_GAUSSIAN_MAX) ||
            flooding.cutoff().bits() > 62
        {
            return Err(PeError::Parameters(
                "the flooding width exceeds the wide Gaussian sampler".into(),
            ));
        }
        let instantiation = TrapdoorInstantiation::Statistical { m_bar: budget.m_bar(digits) };
        let minimum_key_sigma = minimum_preimage_sigma(
            level.ring_dimension(),
            1,
            digits,
            base,
            request.trapdoor_sigma,
            instantiation,
        )
        .ok_or_else(|| PeError::Parameters("invalid trapdoor sampler parameters".into()))?;
        let (key_width, randomness_bound, witness_bound, sample_right_sigma) = if strict {
            let rns = chain(&class, &level, &flooding)?;
            let randomness_bound = budget.randomness_bound(&request.randomness_sigma)?;
            let witness_bound = crate::noise::witness_bound(
                &rns,
                &class,
                &BranchingProgram::longest(class),
                &request.randomness_sigma,
                &randomness_bound,
                request.lut_radix,
            )?;
            let m = 2 * digits;
            let w = class.lookup_width();
            let helpers = helper_count(&class, level.ring_dimension() as usize, request.lut_radix)?;
            let key_length = w * budget.anchor_columns + w * helpers * m + m;
            let sample_right_sigma = budget.sample_right_sigma(
                &shape,
                &witness_bound,
                key_length,
                radix_path_length(w, request.lut_radix),
                base,
                gadget_sampler_sigma(base, request.trapdoor_sigma),
            )?;
            (
                minimum_key_sigma.clone().max(sample_right_sigma.clone()),
                Some(randomness_bound),
                Some(witness_bound),
                Some(sample_right_sigma),
            )
        } else {
            (minimum_key_sigma.clone(), None, None, None)
        };
        let parameters = Self::assemble(
            class,
            level,
            request.trapdoor_sigma,
            instantiation,
            error,
            flooding,
            key_width,
            TrapdoorPreimageCutoffPolicy::Default,
            Some(StatisticalSelection {
                key_width_policy: request.key_width_policy,
                randomness_sigma: request.randomness_sigma.clone(),
                key_queries: request.key_queries,
                max_anchor_columns: request.max_anchor_columns,
                flooding_enforced: request.enforce_flooding,
                budget,
                minimum_key_sigma,
                randomness_bound,
                witness_bound,
                sample_right_sigma,
            }),
        )?;
        if !cutoff_is_representable(parameters.key_cutoff(), &modulus) {
            return Err(PeError::Parameters(format!(
                "the key cutoff 2^{} is not below 2^52 and q / 2",
                parameters.key_cutoff().bits()
            )));
        }
        match request.lut_radix {
            Some(radix) => parameters.lut_radix_unchecked(radix),
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
            flooding: &self.flooding,
            key_width: self.key_width.to_string(),
            key_cutoff: &self.key_cutoff,
            anchor_columns: self.anchor_columns,
            statistical: &self.statistical,
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

    /// The statistical selection record; `None` for computational parameters.
    pub const fn statistical(&self) -> Option<&StatisticalSelection> {
        self.statistical.as_ref()
    }

    /// The trapdoor sigma as the exact graph expression every trapdoor node
    /// and import of the setup uses.
    pub(crate) fn trapdoor_sigma_expr(&self) -> Result<mxx_ir_core::RealExpr, PeError> {
        mxx_ir_core::RealExpr::from_f64_exact(self.trapdoor_sigma)
            .map_err(|_| PeError::Parameters("the trapdoor sigma is not finite".into()))
    }

    /// The preimage width the trapdoor nodes request: `None` (the trapdoor's minimum, the legacy
    /// graph) for computational parameters, the key width otherwise.
    pub(crate) fn preimage_sigma_expr(&self) -> Result<Option<mxx_ir_core::RealExpr>, PeError> {
        Ok(match self.instantiation {
            TrapdoorInstantiation::Computational => None,
            TrapdoorInstantiation::Statistical { .. } => Some(real_expr(&self.key_width)?),
        })
    }

    /// `chi`: the anchor, helper, and message errors.
    pub const fn error(&self) -> &RnsBggGaussian {
        &self.error
    }

    /// `chi_fl`: the initial-state, digit-companion, and attribute-ciphertext errors.
    pub const fn flooding(&self) -> &RnsBggGaussian {
        &self.flooding
    }

    /// The key width `σ_key` (a standard deviation): the trapdoor's minimum preimage width
    /// ([`minimum_preimage_sigma`]; for `Computational`, `1.8 (b + 1) σ_T² (√(n D) + √(2 n) +
    /// 4.7)`; for `Statistical`, `√(c_G² (1 + ŝ_1²) + σ_T²)` with
    /// `ŝ_1 = σ_T √n (√m̄ + √D + √(ln(n / (2δ))))`), or for `Strict` the larger of it and the
    /// SampleRight width.
    pub const fn key_width(&self) -> &BigDecimal {
        &self.key_width
    }

    /// The common cutoff of every key coefficient.
    pub const fn key_cutoff(&self) -> &BigUint {
        &self.key_cutoff
    }

    /// `m_B`: the column count of TrapGen's public row (`D + 2` or `m̄ + D`).
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

    /// The bound `B_C` on the attribute ciphertext error `e_C = e_t`, a fresh `chi_fl` sample:
    /// the flooding contract's cutoff.
    pub fn ciphertext_error_bound(&self) -> BigUint {
        self.flooding.cutoff().clone()
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
    /// of the Ring-LWE instances over the ring:
    ///
    /// - encryption: the uniform ternary secret `s_bar` with the smaller of the sigmas of `chi` and
    ///   `chi_fl`, since every published encoding under `s_bar` (the anchor, the helpers, the
    ///   message, the initial state, and the digit companions) is a sample with one of them;
    /// - `Computational` only: TrapGen's row `a_bar r + e` with `r` and `e` discrete Gaussians of
    ///   the trapdoor sigma. The statistical TrapGen is uniform by the regularity conditions and
    ///   has no estimate (`None`).
    ///
    /// With compact encodings ([`Self::with_compact_encodings`]), the companions, helpers, and
    /// initial state are samples over `pq` with the same secret and the same fresh sigmas, so
    /// the encryption estimate is the smaller of the estimates over `q` and over `pq`.
    ///
    /// The computational anchor's identity column gives the sample `s_bar + B^(2)_1 + e_b` with
    /// the non-uniform coefficient `1`, which these estimates do not model.
    pub fn security_bits(&self) -> Result<(u64, Option<u64>), PeError> {
        let error = self.error.sigma().min(self.flooding.sigma()).clone();
        let mut encryption = mxx_exponent_lut::utils::lattice_security_bits(self.level(), &error)
            .map_err(PeError::Parameters)?;
        if let Some(compact) = self.compact_level()? {
            encryption = encryption.min(
                mxx_exponent_lut::utils::lattice_security_bits(&compact, &error)
                    .map_err(PeError::Parameters)?,
            );
        }
        if self.statistical.is_some() || !self.instantiation.is_computational() {
            return Ok((encryption, None));
        }
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
        Ok((encryption, Some(trapdoor)))
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
                let (encryption, trapdoor) = parameters.security_bits()?;
                if encryption.min(trapdoor.unwrap_or(u64::MAX)) < target {
                    rejected.push(format!(
                        "depth {depth}: {encryption} (encryption) and {trapdoor:?} (trapdoor) \
                         security bits"
                    ));
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
        let parameters = PeParameters::new(
            class,
            level.clone(),
            4.578,
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
                4.578,
                mxx_ir_core::types::TrapdoorInstantiation::Computational,
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

    fn statistical_at(depth: usize, policy: KeyWidthPolicy) -> Result<PeParameters, PeError> {
        let mut request = crate::test_support::statistical_request(PeClass::new(1, 2, 1, 1)?);
        request.key_width_policy = policy;
        PeParameters::at_depth(&request, depth)
    }

    #[test]
    fn the_statistical_selection_follows_its_budget() {
        let parameters = statistical_at(4, KeyWidthPolicy::Practical).unwrap();
        let selection = parameters.statistical().unwrap();
        let budget = &selection.budget;
        let digits = parameters.digits();
        assert_eq!(parameters.anchor_columns(), budget.anchor_columns);
        assert_eq!(
            parameters.instantiation(),
            TrapdoorInstantiation::Statistical { m_bar: budget.anchor_columns - digits }
        );
        // Every truncated contract has the budget's cutoff `c σ`, above the default `6.5 σ`.
        assert!(budget.cutoff_multiplier > 6.5);
        for contract in [parameters.error(), parameters.flooding()] {
            assert_eq!(contract.cutoff(), &budget.cutoff(contract.sigma()).unwrap());
        }
        assert_eq!(parameters.flooding().sigma(), &budget.flooding_sigma);
        // Practical keys take the trapdoor's minimum width, with the default cutoff.
        assert_eq!(parameters.key_width(), &selection.minimum_key_sigma);
        assert_eq!(
            parameters.key_cutoff(),
            &mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound(parameters.key_width())
        );
        assert!(selection.sample_right_sigma.is_none());
        let longest = BranchingProgram::longest(*parameters.class());
        assert!(crate::noise::simulate(&parameters, &longest).is_ok());
    }

    #[test]
    fn statistical_parameters_without_flooding_take_the_requested_width() {
        let mut request =
            crate::test_support::statistical_request(PeClass::new(1, 2, 1, 1).unwrap());
        request.enforce_flooding = false;
        let parameters = PeParameters::at_depth(&request, 4).unwrap();
        let selection = parameters.statistical().unwrap();
        assert!(!selection.flooding_enforced);
        assert_eq!(parameters.flooding().sigma(), &request.flooding_sigma);
        assert_eq!(
            parameters.flooding().cutoff(),
            &selection.budget.cutoff(&request.flooding_sigma).unwrap()
        );
        let enforced = statistical_at(4, KeyWidthPolicy::Practical).unwrap();
        assert_eq!(parameters.anchor_columns(), enforced.anchor_columns());
        assert_ne!(parameters.identity(), enforced.identity());
    }

    #[test]
    fn strict_keys_add_the_sample_right_width_or_are_rejected_as_unrepresentable() {
        let practical = statistical_at(4, KeyWidthPolicy::Practical).unwrap();
        match statistical_at(4, KeyWidthPolicy::Strict) {
            Ok(strict) => {
                let selection = strict.statistical().unwrap();
                assert!(selection.witness_bound.is_some());
                assert!(strict.key_width() >= practical.key_width());
                assert!(
                    strict.key_width() == selection.sample_right_sigma.as_ref().unwrap() ||
                        strict.key_width() == &selection.minimum_key_sigma
                );
            }
            Err(PeError::Parameters(message)) => {
                assert!(message.contains("key cutoff"), "{message}")
            }
            Err(error) => panic!("{error}"),
        }
    }

    #[test]
    fn the_statistical_selection_rejects_what_the_computational_one_accepts() {
        let class = PeClass::new(1, 2, 1, 1).unwrap();
        let mut request = crate::test_support::statistical_request(class);
        // A zero `chi` is below the smoothing parameter.
        request.error_sigma = BigDecimal::from(0u8);
        assert!(PeParameters::at_depth(&request, 4).is_err());
        request.trapdoor_instantiation = TrapdoorMode::Computational;
        let computational = PeParameters::at_depth(&request, 4).unwrap();
        assert!(computational.statistical().is_none());
        assert_eq!(computational.anchor_columns(), computational.digits() + 2);
        // An anchor wider than `m_B,max` is rejected.
        let mut request = crate::test_support::statistical_request(class);
        request.max_anchor_columns = 8;
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

    /// The ledger witness `B_R̂` follows the radix evaluation, whose branch of two nonzero digits
    /// is switched twice, so it exceeds the direct one; a `Strict` selection takes the radix
    /// from its request, bounds its witness and `s_SR` with it (`s_SR` over a helper path of
    /// `log_B W = 2` steps), and rejects a later radix.
    #[test]
    fn the_witness_and_the_sample_right_width_follow_the_radix() {
        let class = PeClass::new(1, 2, 2, 1).unwrap();
        let request = crate::test_support::statistical_request(class);
        let practical = PeParameters::at_depth(&request, 4).unwrap();
        let rns = chain(&class, practical.level(), practical.flooding()).unwrap();
        let longest = BranchingProgram::longest(class);
        let witness = |radix| {
            crate::noise::witness_bound(
                &rns,
                &class,
                &longest,
                &request.randomness_sigma,
                &BigUint::from(64u8),
                radix,
            )
            .unwrap()
        };
        assert!(witness(Some(2)) > witness(None));
        let mut strict = request.clone();
        strict.key_width_policy = KeyWidthPolicy::Strict;
        let mut radix = strict.clone();
        radix.lut_radix = Some(2);
        match (PeParameters::at_depth(&strict, 4), PeParameters::at_depth(&radix, 4)) {
            (Ok(direct), Ok(split)) => {
                assert_eq!(split.lut_radix(), Some(2));
                let [direct, split] = [&direct, &split].map(|p| p.statistical().unwrap());
                assert!(split.witness_bound > direct.witness_bound);
                assert!(split.sample_right_sigma > direct.sample_right_sigma);
                assert!(
                    PeParameters::at_depth(&strict, 4).unwrap().with_lut_radix(2).is_err(),
                    "a Strict selection fixes its radix"
                );
            }
            (Err(PeError::Parameters(message)), _) | (_, Err(PeError::Parameters(message))) => {
                assert!(message.contains("key cutoff"), "{message}")
            }
            (Err(error), _) | (_, Err(error)) => panic!("{error}"),
        }
    }
}
