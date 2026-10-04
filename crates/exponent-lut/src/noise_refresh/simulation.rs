//! Noise-refresh accounting built on the generic exponent-program simulator.
//!
//! Shared reports, digit/event budgets, exact threshold arithmetic and storage
//! accounting. Backend noise adapters own concrete Program snapshots and
//! cryptographic transfer equations.

use crate::noise_utils::{NoiseMoments, NonnegativeRational, RightActionNoiseGain};
use mxx_ir_core::{
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId},
    types::ConcreteMatrixType,
    validate::ValidatedGraph,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::collections::BTreeMap;
use thiserror::Error;

pub const AVERAGE_CASE_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseModelKind {
    WorstCase,
    AverageCase,
}

impl NoiseModelKind {
    /// Application-wide simulation selection. Average acceptance remains
    /// conditional on the explicitly selected coefficient and tail models.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL")
            .as_deref()
            .unwrap_or("average-case")
        {
            "worst-case" => Ok(Self::WorstCase),
            "average-case" => Ok(Self::AverageCase),
            _ => {
                Err("MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL must be worst-case or average-case"
                    .into())
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageCaseConfig {
    pub failure_exponent: u32,
    pub tail_correction_bits: u32,
    pub allow_average_acceptance: bool,
}

impl Default for AverageCaseConfig {
    fn default() -> Self {
        Self { failure_exponent: 100, tail_correction_bits: 0, allow_average_acceptance: false }
    }
}

impl AverageCaseConfig {
    /// Derive the joint event count from the setup-owned transcript topology.
    /// No caller-provided event or input-domain multiplier is accepted.
    pub fn joint_event_count(
        &self,
        topology: &RefreshTranscriptTopology,
    ) -> Result<BigUint, NoiseSimulationError> {
        Ok(BigUint::from(topology.joint_event_count()?))
    }

    /// Exact `z^2 = 2 * (693148 / 1000000) * (lambda + ceil(log2(D)) + 1)`.
    pub fn z_squared(
        &self,
        topology: &RefreshTranscriptTopology,
    ) -> Result<NonnegativeRational, NoiseSimulationError> {
        self.z_squared_for_events(&self.joint_event_count(topology)?)
    }

    /// Arithmetic shared with Program audits. Callers derive the event count
    /// from the executed workload, including its structural multiplicities.
    pub fn z_squared_for_events(
        &self,
        event_count: &BigUint,
    ) -> Result<NonnegativeRational, NoiseSimulationError> {
        let events = self
            .failure_exponent
            .checked_add(ceil_log2(event_count)?)
            .and_then(|value| value.checked_add(1))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        Ok(NonnegativeRational::new(
            BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(events),
            BigUint::from(1_000_000u32),
        )?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LutOutputNoiseGains {
    pub target: RightActionNoiseGain,
    pub identity_trace: RightActionNoiseGain,
    pub nonidentity_trace: Vec<TraceNoiseGains>,
    pub normalization: RightActionNoiseGain,
    pub inherited_bound: BigUint,
    pub output_bound: BigUint,
    pub output_variance: NonnegativeRational,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TraceNoiseGains {
    pub action: RightActionNoiseGain,
    pub helper_action: RightActionNoiseGain,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshNoiseParameters {
    pub slot_count: usize,
    /// Full ciphertext modulus q, used to derive exact J and kappa ratios.
    pub global_modulus: Option<BigUint>,
    pub plaintext_moduli: Vec<BigUint>,
    /// Destination modulus P_t for each slot, in the same order as q_t.
    pub switch_moduli: Vec<BigUint>,
    pub mask_digit_counts: Vec<usize>,
    pub fresh_digit_count: usize,
    pub base_p: BigUint,
    pub transcript_topology: Option<RefreshTranscriptTopology>,
    /// Setup-derived route actions.  `None` is an incomplete snapshot and is
    /// rejected by the backend snapshot constructor; it is never
    /// replaced by a scalar or an identity estimate.
    pub route_gains: Option<RefreshRouteNoiseGains>,
    pub smudging: Option<MaskSmudgingParameters>,
    /// Deterministic native half-payload error from public even-corrected
    /// mask/fresh route targets, before coefficient modulus switching.
    pub route_bias_bounds: Vec<BigUint>,
}

/// Setup-owned refresh transcript shape. `component_columns` is the concrete
/// encoding polynomial count selected by the scheme. The joint tail event count includes every mask
/// slot/P-adic digit and the shared fresh P-adic digit group. The masking
/// dimension excludes both digit axes and the fresh group.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshTranscriptTopology {
    pub crt_slot_count: usize,
    pub component_columns: usize,
    /// Ring coefficient/rotation labels (`N`).
    pub coefficient_count: usize,
    /// Number of independent P-adic mask labels per slot.
    pub mask_digit_counts: Vec<usize>,
    /// Number of independent P-adic fresh labels in the shared group.
    pub fresh_digit_count: usize,
}

impl RefreshTranscriptTopology {
    /// Size PRF labels from the selected concrete encoding carrier.
    /// Mask digits are slot-specific; fresh-error digits form one shared group.
    pub fn new(
        component_columns: usize,
        coefficient_count: usize,
        mask_digit_counts: Vec<usize>,
        fresh_digit_count: usize,
    ) -> Result<Self, NoiseSimulationError> {
        if component_columns == 0 ||
            coefficient_count == 0 ||
            mask_digit_counts.is_empty() ||
            mask_digit_counts.iter().any(|count| *count == 0) ||
            fresh_digit_count == 0
        {
            return Err(NoiseSimulationError::InvalidRefresh(
                "encoding topology dimensions and digit counts must be positive",
            ));
        }
        Ok(Self {
            crt_slot_count: mask_digit_counts.len(),
            component_columns,
            coefficient_count,
            mask_digit_counts,
            fresh_digit_count,
        })
    }

    pub(crate) fn joint_event_count(&self) -> Result<usize, NoiseSimulationError> {
        if self.crt_slot_count == 0 ||
            self.component_columns == 0 ||
            self.coefficient_count == 0 ||
            self.mask_digit_counts.len() != self.crt_slot_count ||
            self.mask_digit_counts.iter().any(|count| *count == 0) ||
            self.fresh_digit_count == 0
        {
            return Err(NoiseSimulationError::InvalidRefresh("invalid transcript topology"));
        }
        let mask_digits = self.mask_digit_counts.iter().try_fold(0usize, |sum, count| {
            sum.checked_add(*count).ok_or(NoiseSimulationError::IntegerConversion)
        })?;
        let mask_events = mask_digits
            .checked_mul(self.component_columns)
            .and_then(|value| value.checked_mul(self.coefficient_count))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        let fresh_events = self
            .component_columns
            .checked_mul(self.coefficient_count)
            .and_then(|value| value.checked_mul(self.fresh_digit_count))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        mask_events.checked_add(fresh_events).ok_or(NoiseSimulationError::IntegerConversion)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshRouteNoiseGains {
    /// Per-slot gain for the state action `G^-1(kappa_t G)`.
    pub state_scaling: Vec<RightActionNoiseGain>,
    /// Sum of L1 and squared-L2 gains over all mask P-adic digit/coordinate routes.
    pub mask: Vec<RightActionNoiseGain>,
    /// Per-slot sums over all fresh P-adic digit/coordinate routes.
    pub fresh: Vec<RightActionNoiseGain>,
    /// Per-slot gain for the directly applied decoder preimage.
    pub decoder: Vec<RightActionNoiseGain>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MaskSmudgingParameters {
    pub lambda_mask: usize,
    /// Mask-only dimension `D`; this excludes P-adic digits and fresh labels.
    pub transcript_coordinates: usize,
}

impl RefreshNoiseParameters {
    pub(crate) fn validate(&self) -> Result<(), NoiseSimulationError> {
        if self.slot_count == 0 ||
            self.slot_count != self.plaintext_moduli.len() ||
            self.plaintext_moduli.iter().any(|modulus| modulus <= &BigUint::one()) ||
            self.base_p < BigUint::from(2u8) ||
            !is_power_of_two(&self.base_p) ||
            self.mask_digit_counts.len() != self.slot_count ||
            self.mask_digit_counts.iter().any(|count| *count == 0) ||
            self.switch_moduli.len() != self.slot_count ||
            self.fresh_digit_count == 0 ||
            (!self.route_bias_bounds.is_empty() &&
                self.route_bias_bounds.len() != self.slot_count)
        {
            return Err(NoiseSimulationError::InvalidRefresh("invalid refresh schema"));
        }
        if let Some(global) = &self.global_modulus {
            if global <= &BigUint::one() ||
                self.plaintext_moduli.iter().any(|modulus| global % modulus != BigUint::zero())
            {
                return Err(NoiseSimulationError::InvalidRefresh("invalid global modulus"));
            }
            for (modulus, destination) in self.plaintext_moduli.iter().zip(&self.switch_moduli) {
                if destination <= &BigUint::one() ||
                    destination % 2u8 == BigUint::zero() ||
                    global % destination != BigUint::zero() ||
                    destination % modulus != BigUint::zero() ||
                    (global / destination) % 2u8 == BigUint::zero()
                {
                    return Err(NoiseSimulationError::InvalidRefresh("invalid switching modulus"));
                }
            }
        }
        if let Some(topology) = &self.transcript_topology {
            if topology.crt_slot_count != self.slot_count ||
                topology.mask_digit_counts != self.mask_digit_counts ||
                topology.fresh_digit_count != self.fresh_digit_count
            {
                return Err(NoiseSimulationError::InvalidRefresh(
                    "transcript topology does not match digit schema",
                ));
            }
        }
        Ok(())
    }

    pub fn new(
        plaintext_moduli: Vec<BigUint>,
        switch_moduli: Vec<BigUint>,
        base_p: BigUint,
        mask_digit_counts: Vec<usize>,
        fresh_digit_count: usize,
    ) -> Result<Self, NoiseSimulationError> {
        if plaintext_moduli.is_empty() ||
            plaintext_moduli.iter().any(|modulus| modulus <= &BigUint::one()) ||
            base_p < BigUint::from(2u8) ||
            mask_digit_counts.iter().any(|count| *count == 0) ||
            fresh_digit_count == 0
        {
            return Err(NoiseSimulationError::InvalidRefresh("empty refresh noise parameters"));
        }
        let parameters = Self {
            slot_count: plaintext_moduli.len(),
            global_modulus: None,
            plaintext_moduli,
            switch_moduli,
            base_p,
            transcript_topology: None,
            mask_digit_counts,
            fresh_digit_count,
            route_gains: None,
            smudging: None,
            route_bias_bounds: Vec::new(),
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub fn with_route_gains(
        mut self,
        gains: RefreshRouteNoiseGains,
    ) -> Result<Self, NoiseSimulationError> {
        if gains.state_scaling.len() != self.slot_count ||
            gains.mask.len() != self.slot_count ||
            gains.fresh.len() != self.slot_count ||
            gains.decoder.len() != self.slot_count ||
            gains
                .state_scaling
                .iter()
                .chain(gains.mask.iter())
                .any(|gain| gain.l1.is_zero() != gain.l2_squared.is_zero()) ||
            gains
                .fresh
                .iter()
                .chain(gains.decoder.iter())
                .any(|gain| gain.l1.is_zero() != gain.l2_squared.is_zero())
        {
            return Err(NoiseSimulationError::InvalidRefresh("incomplete route gains"));
        }
        self.route_gains = Some(gains);
        self.validate()?;
        Ok(self)
    }

    pub fn with_global_modulus(
        mut self,
        global_modulus: BigUint,
    ) -> Result<Self, NoiseSimulationError> {
        if global_modulus <= BigUint::one() ||
            self.plaintext_moduli.iter().any(|modulus| {
                &global_modulus < modulus || !(&global_modulus % modulus).is_zero()
            })
        {
            return Err(NoiseSimulationError::InvalidRefresh("invalid global modulus"));
        }
        self.global_modulus = Some(global_modulus);
        self.validate()?;
        Ok(self)
    }

    pub fn with_transcript_topology(
        mut self,
        topology: RefreshTranscriptTopology,
    ) -> Result<Self, NoiseSimulationError> {
        topology.joint_event_count()?;
        if topology.crt_slot_count != self.slot_count {
            return Err(NoiseSimulationError::InvalidRefresh("transcript slot mismatch"));
        }
        if topology.mask_digit_counts != self.mask_digit_counts ||
            topology.fresh_digit_count != self.fresh_digit_count
        {
            return Err(NoiseSimulationError::InvalidRefresh("transcript digit mismatch"));
        }
        self.transcript_topology = Some(topology);
        self.validate()?;
        Ok(self)
    }

    pub fn with_mask_smudging_parameters(
        mut self,
        parameters: MaskSmudgingParameters,
    ) -> Result<Self, NoiseSimulationError> {
        if parameters.transcript_coordinates == 0 {
            return Err(NoiseSimulationError::InvalidRefresh("empty smudging transcript"));
        }
        self.smudging = Some(parameters);
        self.validate()?;
        Ok(self)
    }

    /// Nominal physical payload moments, conditional on independent uniform
    /// PRF digits. These are not ciphertext errors or a tail assumption.
    pub fn mask_uniform_moments(&self, slot: usize) -> NoiseMoments {
        super::refresh::native_digit_uniform_moments(
            &self.base_p,
            self.mask_digit_counts[slot],
            true,
        )
    }

    pub fn fresh_uniform_moments(&self) -> NoiseMoments {
        super::refresh::native_digit_uniform_moments(&self.base_p, self.fresh_digit_count, false)
    }
}

/// Application noise evaluated through the program and readout, before the
/// d_m-dependent refresh equations.  Parameter search reuses this exact
/// value while scanning mask digit counts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreRefreshNoiseEvaluation<P> {
    pub program: P,
    pub readout_bound: BigUint,
    pub readout_variance: NonnegativeRational,
    /// Deterministic readout error already included in `readout_bound` but
    /// excluded from `readout_variance`. Average refresh transports this with
    /// l1 gains, since fixed bias must not receive independent-sum discounts.
    pub readout_bias_bound: BigUint,
}

impl<P> PreRefreshNoiseEvaluation<P> {
    /// Transports readout noise through a modeled scalar bridge. The residual
    /// bias is supplied already weighted by the plaintext and secret bounds;
    /// it remains deterministic in every downstream refresh route.
    pub fn apply_gadget_bridge(&mut self, gain: &RightActionNoiseGain, bias: &BigUint) {
        self.readout_bound = &self.readout_bound * &gain.l1 + bias;
        self.readout_variance = self.readout_variance.scaled(&gain.l2_squared);
        self.readout_bias_bound = &self.readout_bias_bound * &gain.l1 + bias;
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshNoiseReport {
    pub slots: Vec<RefreshSlotNoiseReport>,
    pub mask_coverage_accepted: bool,
    pub rounding_accepted: bool,
    pub fresh_error_accepted: bool,
    pub domain_accepted: bool,
    pub mask_smudging_accepted: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshSlotNoiseReport {
    pub slot: usize,
    pub bound: BigUint,
    pub margin: BigInt,
    pub kappa: BigUint,
    pub switch_modulus: BigUint,
    pub lift: BigUint,
    pub spacing: BigUint,
    pub mask_modulus: BigUint,
    pub mask_bound: BigUint,
    pub pre_switch_bound: BigUint,
    /// Exact numerator over 2J for the deterministic paired-rounding bound.
    pub rounding_numerator: BigUint,
    pub rounding_denominator: BigUint,
    pub decoder_bound: BigUint,
    pub residual_bound: BigUint,
    pub smudging_margin: BigInt,
    pub rounding_lhs: BigUint,
    pub rounding_margin: BigInt,
    pub fresh_error_accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageRefreshNoiseReport {
    pub slots: Vec<AverageRefreshSlotNoiseReport>,
    pub joint_event_count: BigUint,
    pub z_squared: NonnegativeRational,
    pub max_f_avg: BigUint,
    pub mask_coverage_accepted: bool,
    pub rounding_accepted: bool,
    pub fresh_error_accepted: bool,
    pub domain_accepted: bool,
    pub mask_smudging_accepted: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageRefreshSlotNoiseReport {
    pub slot: usize,
    pub variance: NonnegativeRational,
    pub f_avg: BigUint,
    pub kappa: BigUint,
    pub switch_modulus: BigUint,
    pub lift: BigUint,
    pub spacing: BigUint,
    pub rounding_numerator: BigUint,
    pub rounding_denominator: BigUint,
    pub deterministic_mask_bound: BigUint,
    pub rounding_lhs: BigUint,
    pub rounding_margin: BigInt,
    pub rounding_deficit: BigUint,
    pub smudging_margin: BigInt,
    pub fresh_error_accepted: bool,
    pub variance_breakdown: AverageVarianceBreakdown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageVarianceBreakdown {
    pub readout_mask: NonnegativeRational,
    pub readout_fresh: NonnegativeRational,
    pub decoder: NonnegativeRational,
    pub state: NonnegativeRational,
    /// Conditional nominal payload moments in physical coordinates. Other
    /// fields above are operational variances in the selected adapter's
    /// coordinate convention.
    pub mask_uniform_moments: NoiseMoments,
    pub fresh_uniform_moments: NoiseMoments,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutNoiseReport<P> {
    pub program: P,
    pub readout_bound: BigUint,
    pub refresh: RefreshNoiseReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutAverageNoiseReport<P> {
    pub snapshot_identity: [u8; 32],
    pub program: P,
    pub refresh: AverageRefreshNoiseReport,
    pub security_authority: NoiseModelKind,
    pub correctness_authority: NoiseModelKind,
    pub hard_authority_accepted: bool,
    pub correctness_accepted: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NoiseSimulationError {
    #[error(transparent)]
    Arithmetic(#[from] crate::noise_utils::NoiseArithmeticError),
    #[error("{0}")]
    Program(String),
    #[error("program output is missing")]
    MissingOutput,
    #[error("invalid refresh parameters: {0}")]
    InvalidRefresh(&'static str),
    #[error("integer conversion overflow")]
    IntegerConversion,
    #[error("average-case event budget is not typed completely")]
    AverageTypedCompleteness,
    #[error("average-case acceptance is disabled")]
    AverageAcceptanceDisabled,
    #[error("decomposition schema is invalid or incomplete")]
    InvalidDecomposition,
    #[error("program LUT declaration is missing")]
    MissingLut,
    #[error("CRT modulus factorization could not be established: {0}")]
    CrtModulusFactorization(String),
}

// The refresh contract is stated for the prime factors of each CRT modulus,
// rather than for the modulus value itself.  OpenFHE's native CRT towers fit
// in u64, so use deterministic 64-bit Miller--Rabin followed by deterministic
// Pollard-rho recursion here.  This keeps the production path exact for both
// the usual prime towers and composite validation fixtures; a failure to
// factor is reported instead of weakening the acceptance predicate.
fn modular_multiply_u64(left: u64, right: u64, modulus: u64) -> u64 {
    ((left as u128 * right as u128) % modulus as u128) as u64
}

fn modular_power_u64(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = modular_multiply_u64(result, base, modulus);
        }
        base = modular_multiply_u64(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn is_prime_u64(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    for small_prime in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value == small_prime {
            return true;
        }
        if value % small_prime == 0 {
            return false;
        }
    }
    let mut odd_part = value - 1;
    let mut powers_of_two = 0;
    while odd_part & 1 == 0 {
        odd_part >>= 1;
        powers_of_two += 1;
    }
    // This is the published deterministic seven-base test for all u64
    // inputs (the intermediate products use u128 above).
    for base in [2u64, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022] {
        if base % value == 0 {
            continue;
        }
        let mut witness = modular_power_u64(base % value, odd_part, value);
        if witness == 1 || witness == value - 1 {
            continue;
        }
        let mut probably_prime = false;
        for _ in 1..powers_of_two {
            witness = modular_multiply_u64(witness, witness, value);
            if witness == value - 1 {
                probably_prime = true;
                break;
            }
        }
        if !probably_prime {
            return false;
        }
    }
    true
}

fn pollard_rho_u64(value: u64) -> Option<u64> {
    if value % 2 == 0 {
        return Some(2);
    }
    if value % 3 == 0 {
        return Some(3);
    }
    for constant in 1..=128u64 {
        let constant = constant % value;
        let mut tortoise = 2u64;
        let mut hare = 2u64;
        for _ in 0..200_000 {
            let step = |point: u64| {
                ((modular_multiply_u64(point, point, value) as u128 + constant as u128) %
                    value as u128) as u64
            };
            tortoise = step(tortoise);
            hare = step(step(hare));
            let difference = tortoise.abs_diff(hare);
            let divisor = gcd_u64(difference, value);
            if divisor > 1 && divisor < value {
                return Some(divisor);
            }
            if divisor == value {
                break;
            }
        }
    }
    None
}

fn factor_u64(value: u64, factors: &mut Vec<u64>) -> Result<(), NoiseSimulationError> {
    if value == 1 {
        return Ok(());
    }
    if is_prime_u64(value) {
        factors.push(value);
        return Ok(());
    }
    let divisor = pollard_rho_u64(value)
        .ok_or_else(|| NoiseSimulationError::CrtModulusFactorization(value.to_string()))?;
    factor_u64(divisor, factors)?;
    factor_u64(value / divisor, factors)
}

/// Returns the prime factors of a CRT modulus, including multiplicity.
///
/// The result is sorted so callers can use the first factor as the exact
/// smallest-prime-factor boundary.  Moduli outside the native u64 CRT range
/// are rejected because this routine must never silently substitute a weaker
/// modulus-level comparison.
pub fn prime_factors_of_crt_modulus(
    modulus: &BigUint,
) -> Result<Vec<BigUint>, NoiseSimulationError> {
    let value = modulus.to_u64().ok_or_else(|| {
        NoiseSimulationError::CrtModulusFactorization("modulus does not fit u64".to_owned())
    })?;
    if value <= 1 {
        return Err(NoiseSimulationError::CrtModulusFactorization(value.to_string()));
    }
    let mut factors = Vec::new();
    factor_u64(value, &mut factors)?;
    factors.sort_unstable();
    Ok(factors.into_iter().map(BigUint::from).collect())
}

/// Checks the exact fresh-error condition `B_e < r` for every prime factor
/// `r` of one CRT modulus.  A factorization failure is an error, not an
/// acceptance.
pub fn fresh_error_below_every_prime_factor(
    fresh_bound: &BigUint,
    crt_modulus: &BigUint,
) -> Result<bool, NoiseSimulationError> {
    Ok(prime_factors_of_crt_modulus(crt_modulus)?.into_iter().all(|factor| fresh_bound < &factor))
}

fn ceil_log2(value: &BigUint) -> Result<u32, NoiseSimulationError> {
    if value.is_zero() {
        return Err(NoiseSimulationError::InvalidRefresh("empty transcript topology"));
    }
    let bits = value.bits();
    let exact_power = value == &(BigUint::one() << (bits - 1));
    u32::try_from(if exact_power { bits - 1 } else { bits })
        .map_err(|_| NoiseSimulationError::IntegerConversion)
}

fn is_power_of_two(value: &BigUint) -> bool {
    !value.is_zero() && (value & (value - 1u8)).is_zero()
}

fn integer_sqrt_floor(value: &BigUint) -> BigUint {
    if value.is_zero() {
        return BigUint::zero();
    }
    let mut low = BigUint::zero();
    let mut high = BigUint::one() << ((value.bits() as usize + 1) / 2 + 1);
    while &low + 1u8 < high {
        let mid = (&low + &high) >> 1usize;
        if &mid * &mid <= *value {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

pub(crate) fn ceil_div(
    numerator: &BigUint,
    denominator: &BigUint,
) -> Result<BigUint, NoiseSimulationError> {
    if denominator.is_zero() {
        return Err(NoiseSimulationError::IntegerConversion);
    }
    if numerator.is_zero() {
        Ok(BigUint::zero())
    } else {
        Ok((numerator + denominator - 1u8) / denominator)
    }
}

fn ceil_sqrt(value: &BigUint) -> BigUint {
    if value.is_zero() {
        return BigUint::zero();
    }
    let floor = integer_sqrt_floor(value);
    if &floor * &floor == *value { floor } else { floor + 1u8 }
}

/// Computes the exact integer average-case transfer bound
/// `ceil(sqrt(ceil(z^2 * V * 4^b / 4)))`.
pub fn average_favg(
    z_squared: &NonnegativeRational,
    variance: &NonnegativeRational,
    tail_correction_bits: u32,
) -> Result<BigUint, NoiseSimulationError> {
    let scale_bits =
        tail_correction_bits.checked_mul(2).ok_or(NoiseSimulationError::IntegerConversion)?;
    let numerator = &z_squared.numerator * &variance.numerator * (BigUint::one() << scale_bits);
    let denominator = BigUint::from(4u8) * &z_squared.denominator * &variance.denominator;
    Ok(ceil_sqrt(&ceil_div(&numerator, &denominator)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crt_prime_factorization_handles_prime_and_composite_towers() {
        assert_eq!(
            prime_factors_of_crt_modulus(&BigUint::from(37u8)).unwrap(),
            vec![BigUint::from(37u8)]
        );
        assert_eq!(
            prime_factors_of_crt_modulus(&BigUint::from(45u8)).unwrap(),
            vec![BigUint::from(3u8), BigUint::from(3u8), BigUint::from(5u8)]
        );
    }

    #[test]
    fn fresh_error_bound_is_strict_against_each_prime_factor() {
        assert!(
            fresh_error_below_every_prime_factor(&BigUint::from(2u8), &BigUint::from(15u8))
                .unwrap()
        );
        assert!(
            !fresh_error_below_every_prime_factor(&BigUint::from(3u8), &BigUint::from(15u8))
                .unwrap()
        );
        assert!(
            fresh_error_below_every_prime_factor(&BigUint::from(4u8), &BigUint::from(37u8))
                .unwrap()
        );
        assert!(
            !fresh_error_below_every_prime_factor(&BigUint::from(37u8), &BigUint::from(37u8))
                .unwrap()
        );
    }

    #[test]
    fn refresh_route_and_smudging_artifacts_are_required() {
        let refresh = RefreshNoiseParameters::new(
            vec![BigUint::from(257u16), BigUint::from(263u16)],
            vec![BigUint::from(257u16) * 263u16; 2],
            BigUint::from(16u8),
            vec![2; 2],
            2,
        )
        .unwrap();
        let unit = RightActionNoiseGain { l1: BigUint::one(), l2_squared: BigUint::one() };
        let routes = RefreshRouteNoiseGains {
            state_scaling: vec![unit.clone(), unit.clone()],
            mask: vec![unit.clone(); 2],
            fresh: vec![unit.clone(), unit.clone()],
            decoder: vec![unit.clone(), unit.clone()],
        };
        assert!(refresh.clone().with_route_gains(routes).unwrap().smudging.is_none());
        assert!(
            refresh
                .with_route_gains(RefreshRouteNoiseGains {
                    state_scaling: vec![
                        RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::one() },
                        RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::one() }
                    ],
                    mask: vec![
                        RightActionNoiseGain {
                            l1: BigUint::one(),
                            l2_squared: BigUint::one()
                        };
                        2
                    ],
                    fresh: Vec::new(),
                    decoder: Vec::new(),
                })
                .is_err()
        );
    }

    #[test]
    fn mask_rounding_boundary_is_strict_and_uses_single_mask_bound() {
        let mask_bound = BigUint::from(7u8); // M_m - 1
        let f_avg = BigUint::from(2u8);
        let boundary = &mask_bound + &f_avg * 2u8;
        assert!(boundary < BigUint::from(12u8));
        assert!(!(boundary < BigUint::from(11u8)));
        assert_eq!(BigInt::from(12u8) - BigInt::from(boundary.clone()), BigInt::from(1u8));
        assert_eq!(BigInt::from(10u8) - BigInt::from(boundary), BigInt::from(-1i8));
    }

    #[test]
    fn test_refresh_moments_follow_native_digit_counts() {
        let mut refresh = RefreshNoiseParameters::new(
            vec![3u8.into(), 5u8.into()],
            vec![15u8.into(); 2],
            16u8.into(),
            vec![1, 2],
            2,
        )
        .unwrap();
        let mask = refresh.mask_uniform_moments(0);
        assert_eq!(mask.mean_numerator, (-1).into());
        assert_eq!(mask.mean_denominator, 2u8.into());
        assert_eq!(
            mask.variance.scaled(&4u8.into()),
            NonnegativeRational::new(255u16.into(), 3u8.into()).unwrap()
        );
        let fresh = refresh.fresh_uniform_moments();
        assert_eq!(fresh.mean_numerator, 17.into());
        assert_eq!(fresh.mean_denominator, 2u8.into());
        assert_eq!(fresh.variance, refresh.mask_uniform_moments(1).variance);
        refresh.mask_digit_counts[0] = 2;
        assert_eq!(refresh.mask_uniform_moments(0), refresh.mask_uniform_moments(1));
        assert_eq!(refresh.fresh_uniform_moments(), fresh);
        for base in [2u8, 16, 32] {
            refresh.base_p = base.into();
            refresh.fresh_digit_count = 1;
            let native = refresh.fresh_uniform_moments();
            assert_eq!(native.mean_numerator, 1.into());
            assert_eq!(native.mean_denominator, 2u8.into());
            assert_eq!(
                native.variance.scaled(&12u8.into()),
                NonnegativeRational::new((u32::from(base).pow(2) - 1).into(), 1u8.into()).unwrap()
            );
        }
    }

    #[test]
    fn average_tail_factor_is_derived_from_transcript_topology() {
        let topology = RefreshTranscriptTopology {
            crt_slot_count: 2,
            component_columns: 4,
            coefficient_count: 3,
            mask_digit_counts: vec![1; 2],
            fresh_digit_count: 1,
        };
        let config = AverageCaseConfig {
            failure_exponent: 10,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        // Mask routes are slot-specific; fresh routes are one shared group.
        assert_eq!(config.joint_event_count(&topology).unwrap(), BigUint::from(36u8));
        assert_eq!(
            config.z_squared(&topology).unwrap(),
            NonnegativeRational::new(
                BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(17u8),
                BigUint::from(1_000_000u32),
            )
            .unwrap()
        );
    }

    #[test]
    fn transcript_topology_rejects_missing_digit_groups() {
        for (mask_digits, fresh_digits) in [(Vec::new(), 1), (vec![0], 1), (vec![1], 0)] {
            assert!(RefreshTranscriptTopology::new(1, 1, mask_digits, fresh_digits).is_err());
        }
    }

    #[test]
    fn transcript_topology_keeps_only_concrete_encoding_dimensions() {
        let topology = RefreshTranscriptTopology::new(2, 8, vec![3, 4], 2).unwrap();
        assert_eq!(topology.crt_slot_count, 2);
        assert_eq!(topology.component_columns, 2);
        assert_eq!(topology.coefficient_count, 8);
        assert_eq!(topology.mask_digit_counts, vec![3, 4]);
        assert_eq!(topology.fresh_digit_count, 2);
    }

    #[test]
    fn average_favg_uses_exact_ceil_sqrt_after_division_by_four() {
        let z_squared = NonnegativeRational::new(BigUint::one(), BigUint::one()).unwrap();
        let variance = NonnegativeRational::new(BigUint::from(16u8), BigUint::one()).unwrap();
        assert_eq!(average_favg(&z_squared, &variance, 0).unwrap(), BigUint::from(2u8));
        assert_eq!(average_favg(&z_squared, &variance, 1).unwrap(), BigUint::from(4u8));
    }
}

/// Storage accounting separates canonical protocol material from decoder
/// preimages and from bounded resident waves.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutStorageReport {
    pub canonical_artifact_bytes: u128,
    pub artifact_input_bytes: u128,
    pub decoder_preimage_bytes: u128,
    pub peak_loaded_host_bytes: u64,
    pub peak_loaded_device_bytes_per_gpu: u64,
    pub peak_workspace_bytes_per_gpu: u64,
}

/// Storage accounting separates canonical protocol artifacts from decoder
/// preimages without introducing an application-specific store format. Compact
/// matrices remain the persisted representation; gadget decompositions are
/// derived by consumers and never counted as stored artifacts.
impl ExponentLutStorageReport {
    /// Computes storage accounting from the producer manifests and validated
    /// graph metadata.  Manifest entries are deduplicated by production and
    /// artifact name, so passing preprocessing and online evaluation manifests together
    /// does not double count the shared setup production.
    ///
    /// A family contributes its complete logical artifact bytes to persistent
    /// storage, while only one member contributes to the host-residency bound.
    /// This models the runtime's bounded member loading contract and avoids
    /// pretending that an entire family is resident at once.
    pub fn from_metadata<'manifest, 'graph, M, G>(manifests: M, graphs: G) -> Self
    where
        M: IntoIterator<Item = &'manifest Manifest>,
        G: IntoIterator<Item = &'graph ValidatedGraph>,
    {
        let mut report = Self::default();
        let mut seen_artifacts = std::collections::BTreeSet::<(ProductionId, String)>::new();

        for manifest in manifests {
            for (name, artifact) in &manifest.artifacts {
                if !seen_artifacts.insert((manifest.production_id.clone(), name.clone())) {
                    continue;
                }
                let persistent = artifact_persistent_bytes(artifact);
                let member = artifact_member_bytes(artifact);
                report.peak_loaded_host_bytes =
                    report.peak_loaded_host_bytes.saturating_add(member);
                match artifact_storage_purpose(artifact) {
                    ArtifactStoragePurpose::DecoderPreimage => {
                        report.decoder_preimage_bytes += persistent;
                    }
                    ArtifactStoragePurpose::Canonical => {
                        report.canonical_artifact_bytes += persistent;
                    }
                }
            }
        }

        // Non-relation-bearing artifact inputs are online evaluation label/input material.
        // They are not manifest outputs and therefore must be collected from
        // the validated graph's artifact metadata.
        for graph in graphs {
            for scope in graph.scopes.values() {
                for artifact in scope.artifact_inputs.values() {
                    if !matches!(artifact.artifact_type, ArtifactType::Preimage { .. }) {
                        report.artifact_input_bytes =
                            report.artifact_input_bytes + artifact_persistent_bytes(artifact);
                    }
                }
            }
        }
        report
    }

    /// Adds modeled operation residency to manifest-derived storage facts.
    /// The estimator supplies these values because artifact manifests do not
    /// contain device allocator observations.
    pub fn with_residency(
        mut self,
        peak_loaded_host_bytes: u64,
        peak_loaded_device_bytes_per_gpu: u64,
        peak_workspace_bytes_per_gpu: u64,
    ) -> Self {
        self.peak_loaded_host_bytes = self.peak_loaded_host_bytes.max(peak_loaded_host_bytes);
        self.peak_loaded_device_bytes_per_gpu = peak_loaded_device_bytes_per_gpu;
        self.peak_workspace_bytes_per_gpu = peak_workspace_bytes_per_gpu;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArtifactStoragePurpose {
    Canonical,
    DecoderPreimage,
}

/// Derives the accounting purpose from frozen artifact metadata. Relation
/// bearing preimages are decoder material; compact matrix artifacts remain
/// canonical regardless of their layout token.
fn artifact_storage_purpose(artifact: &ManifestArtifact) -> ArtifactStoragePurpose {
    if !matches!(artifact.artifact_type, ArtifactType::Preimage { .. }) {
        return ArtifactStoragePurpose::Canonical;
    }
    ArtifactStoragePurpose::DecoderPreimage
}

fn matrix_coefficient_bytes(matrix: &ConcreteMatrixType) -> u64 {
    let modulus_bytes = matrix.ring.modulus().bits().div_ceil(8).max(1);
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(matrix.ring.ring_dimension()))
        .saturating_mul(modulus_bytes)
}

fn compact_coefficient_bytes(matrix: &ConcreteMatrixType, bound: &BigInt) -> u64 {
    let magnitude_bytes =
        bound.to_biguint().map(|value| value.bits().div_ceil(8).max(1)).unwrap_or(u64::MAX);
    u64::try_from(matrix.rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::try_from(matrix.columns).unwrap_or(u64::MAX))
        .saturating_mul(u64::from(matrix.ring.ring_dimension()))
        .saturating_mul(1u64.saturating_add(magnitude_bytes))
}

fn artifact_member_bytes(artifact: &ManifestArtifact) -> u64 {
    match &artifact.artifact_type {
        ArtifactType::Matrix(matrix) => matrix_coefficient_bytes(matrix),
        ArtifactType::SmallMatrix { matrix, max_coefficient_bound, .. } |
        ArtifactType::Preimage { matrix, max_coefficient_bound, .. } => {
            let bound_bytes = max_coefficient_bound
                .to_biguint()
                .map(|value| value.to_bytes_le().len().max(1))
                .unwrap_or(0);
            let framing = 4u64
                .saturating_add(1)
                .saturating_add(8)
                .saturating_add(8)
                .saturating_add(8)
                .saturating_add(4)
                .saturating_add(u64::try_from(bound_bytes).unwrap_or(u64::MAX))
                .saturating_add(4)
                .saturating_add(8);
            compact_coefficient_bytes(matrix, max_coefficient_bound).saturating_add(framing)
        }
        ArtifactType::Trapdoor { matrix, .. } => matrix_coefficient_bytes(matrix),
        ArtifactType::Bytes { length } => u64::try_from(*length).unwrap_or(u64::MAX),
        // Variable-size payload lengths are not available in the manifest type.
        ArtifactType::Int | ArtifactType::TypedBlob { .. } => 0,
    }
}

fn artifact_persistent_bytes(artifact: &ManifestArtifact) -> u128 {
    u128::from(artifact_member_bytes(artifact)) * artifact.family_count.unwrap_or(1) as u128
}

#[cfg(test)]
mod storage_report_tests {
    use super::*;
    use mxx_ir_core::artifact::ArtifactAvailability;

    #[test]
    fn storage_report_separates_canonical_decoder_and_member_peak() {
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([21; 32]),
            execution_nonce: [22; 32],
        };
        let matrix = ConcreteMatrixType {
            ring: crate::test_support::concrete_ring(&[257], 8),
            rows: 2,
            columns: 3,
        };
        let compact = |layout, family_count| ManifestArtifact {
            artifact_type: ArtifactType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: BigInt::from(7),
                bound_domain: mxx_ir_core::types::CoefficientBoundDomain::Global,
            },
            family_count,
            availability: ArtifactAvailability::Transferred,
            layout,
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id,
            artifacts: BTreeMap::from([
                (
                    "public-matrix".to_owned(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(matrix.clone()),
                        family_count: None,
                        availability: ArtifactAvailability::Transferred,
                        layout: None,
                    },
                ),
                ("decoder".to_owned(), compact(Some("decoder-preimage".to_owned()), Some(4))),
            ]),
        };
        let report = ExponentLutStorageReport::from_metadata([&manifest], std::iter::empty());
        assert_eq!(
            report.canonical_artifact_bytes,
            u128::from(artifact_member_bytes(manifest.artifacts.get("public-matrix").unwrap()))
        );
        assert_eq!(
            report.decoder_preimage_bytes,
            artifact_persistent_bytes(manifest.artifacts.get("decoder").unwrap())
        );
        assert_eq!(
            report.peak_loaded_host_bytes,
            manifest.artifacts.values().map(artifact_member_bytes).sum::<u64>()
        );
        assert_eq!(report.artifact_input_bytes, 0);
    }

    #[test]
    fn storage_report_preserves_logical_family_bytes_above_u64() {
        let production_id = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([31; 32]),
            execution_nonce: [32; 32],
        };
        let artifact = ManifestArtifact {
            artifact_type: ArtifactType::Preimage {
                matrix: ConcreteMatrixType::scalar(crate::test_support::concrete_ring(&[257], 8)),
                max_coefficient_bound: BigInt::from(7),
                bound_domain: mxx_ir_core::types::CoefficientBoundDomain::Global,
            },
            family_count: Some(usize::MAX),
            availability: ArtifactAvailability::Transferred,
            layout: None,
        };
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id,
            artifacts: BTreeMap::from([("huge-family".to_owned(), artifact)]),
        };
        let report = ExponentLutStorageReport::from_metadata([&manifest], std::iter::empty());
        assert!(report.decoder_preimage_bytes > u128::from(u64::MAX));
        assert_eq!(
            report.decoder_preimage_bytes,
            artifact_persistent_bytes(manifest.artifacts.get("huge-family").unwrap())
        );
    }
}
