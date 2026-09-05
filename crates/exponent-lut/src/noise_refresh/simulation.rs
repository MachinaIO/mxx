//! Noise-refresh accounting built on the generic exponent-program simulator.
//!
//! The primitive program transfers live in [`crate::noise`]. This module adds
//! the sparse-LWR inputs, value readout, refresh routes, digit errors, and the
//! final correctness thresholds.

use crate::{
    layout::ExpLayout,
    noise::{
        AverageVariance, GateNoiseDecomposition, GateNoiseGains, NoiseDecompositionPlan,
        ProgramNoiseError, ProgramNoiseGainPlan, ProgramNoiseInputs, ProgramNoiseReport,
        RightActionNoiseGain, simulate_program,
    },
    noise_refresh::prf::SparseLwrPrfProgram,
    program::{
        ExponentLutProgram, FamilyWireSource, ProgramInputId, ProgramStep, WireFamilyId, WireId,
    },
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AVERAGE_CASE_REPORT_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseModelKind {
    WorstCase,
    AverageCase,
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
    ) -> Result<AverageVariance, NoiseSimulationError> {
        let events = self
            .failure_exponent
            .checked_add(ceil_log2(&self.joint_event_count(topology)?)?)
            .and_then(|value| value.checked_add(1))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        Ok(AverageVariance::new(
            BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(events),
            BigUint::from(1_000_000u32),
        )?)
    }
}

/// Setup-produced gain data for the ordered program and value readout.
/// Every scalar in this structure is produced by [`NoiseGainPlanBuilder`]
/// from its corresponding concrete decomposition plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseGainPlan {
    program: ProgramNoiseGainPlan,
    readout: ValueReadoutNoiseGains,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NoiseGainPlanBuilder {
    pub gate_decompositions: Vec<GateNoiseDecomposition>,
    /// Concrete decomposition of `D_T = G^-1(T' G)`, not a direct
    /// convolution-coefficient or modulus-derived proxy.
    pub target: NoiseDecompositionPlan,
    pub nonidentity_trace: Vec<(NoiseDecompositionPlan, NoiseDecompositionPlan)>,
    pub normalization: NoiseDecompositionPlan,
}

impl NoiseGainPlanBuilder {
    pub fn new(
        gate_decompositions: Vec<GateNoiseDecomposition>,
        target: NoiseDecompositionPlan,
        nonidentity_trace: Vec<(NoiseDecompositionPlan, NoiseDecompositionPlan)>,
        normalization: NoiseDecompositionPlan,
    ) -> Result<Self, NoiseSimulationError> {
        let builder = Self { gate_decompositions, target, nonidentity_trace, normalization };
        builder.validate_plans()?;
        Ok(builder)
    }

    fn validate_plans(&self) -> Result<(), NoiseSimulationError> {
        self.target.validate()?;
        self.normalization.validate()?;
        for decomposition in &self.gate_decompositions {
            decomposition.action.validate()?;
            if let Some(helper) = &decomposition.helper_action {
                helper.validate()?;
            }
        }
        for (action, helper) in &self.nonidentity_trace {
            action.validate()?;
            helper.validate()?;
        }
        Ok(())
    }

    /// Derive all action gains for a route from its explicit or compact
    /// balanced plans.
    pub fn derive_route_gains(
        plans: &[NoiseDecompositionPlan],
    ) -> Result<Vec<RightActionNoiseGain>, NoiseSimulationError> {
        plans
            .iter()
            .map(|plan| {
                plan.validate()?;
                let gain = plan.action_gain();
                if gain.l1.is_zero() || gain.l2_squared.is_zero() {
                    return Err(NoiseSimulationError::InvalidDecomposition);
                }
                Ok(gain)
            })
            .collect()
    }

    /// Attach the derived per-action gains to the setup-produced row metadata.
    /// The metadata length and LUT helper presence must match the decomposition
    /// input exactly; no singleton fallback is available.
    pub fn build(
        &self,
        mut gates: Vec<GateNoiseGains>,
        mut readout: ValueReadoutNoiseGains,
    ) -> Result<NoiseGainPlan, NoiseSimulationError> {
        self.validate_plans()?;
        if gates.len() != self.gate_decompositions.len() ||
            readout.nonidentity_trace.len() != self.nonidentity_trace.len()
        {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        for (gate, decomposition) in gates.iter_mut().zip(&self.gate_decompositions) {
            gate.action = decomposition.action.action_gain();
            gate.helper_action =
                decomposition.helper_action.as_ref().map(|plan| plan.action_gain());
            if gate.action.l1.is_zero() ||
                gate.action.l2_squared.is_zero() ||
                gate.helper_action
                    .as_ref()
                    .is_some_and(|gain| gain.l1.is_zero() || gain.l2_squared.is_zero())
            {
                return Err(NoiseSimulationError::InvalidDecomposition);
            }
        }
        readout.target = self.target.action_gain();
        readout.identity_trace =
            RightActionNoiseGain { l1: BigUint::one(), l2_squared: BigUint::one() };
        readout.normalization = self.normalization.action_gain();
        for (trace, (action, helper)) in
            readout.nonidentity_trace.iter_mut().zip(&self.nonidentity_trace)
        {
            trace.action = action.action_gain();
            trace.helper_action = helper.action_gain();
        }
        Ok(NoiseGainPlan { program: ProgramNoiseGainPlan::new(gates), readout })
    }
}

impl NoiseGainPlan {
    pub fn validate(
        &self,
        program: &ExponentLutProgram,
        family_active_counts: &[usize],
    ) -> Result<(), NoiseSimulationError> {
        self.program.validate(program, family_active_counts)?;
        let readout_gains =
            [&self.readout.target, &self.readout.identity_trace, &self.readout.normalization];
        if readout_gains.iter().any(|gain| gain.l1.is_zero() || gain.l2_squared.is_zero()) ||
            self.readout.nonidentity_trace.iter().any(|gain| {
                gain.action.l1.is_zero() ||
                    gain.action.l2_squared.is_zero() ||
                    gain.helper_action.l1.is_zero() ||
                    gain.helper_action.l2_squared.is_zero()
            })
        {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        // The identity automorphism is the exact unit action in the trace.
        if self.readout.identity_trace !=
            (RightActionNoiseGain { l1: BigUint::one(), l2_squared: BigUint::one() })
        {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValueReadoutNoiseGains {
    pub target: RightActionNoiseGain,
    pub identity_trace: RightActionNoiseGain,
    pub nonidentity_trace: Vec<TraceNoiseGains>,
    pub normalization: RightActionNoiseGain,
    pub inherited_bound: BigUint,
    pub output_bound: BigUint,
    pub output_variance: AverageVariance,
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
    /// rejected by `ExponentLutNoiseSnapshot::from_setup`; it is never
    /// replaced by a scalar or an identity estimate.
    pub route_gains: Option<RefreshRouteNoiseGains>,
    pub smudging: Option<MaskSmudgingParameters>,
    pub mask_digit_variances: Vec<AverageVariance>,
    pub fresh_digit_variances: Vec<AverageVariance>,
}

/// Setup-owned refresh transcript shape. `component_columns` is the concrete
/// `2 * ell_beta` column count. The joint tail event count includes every mask
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
    pub(crate) fn joint_event_count(&self) -> Result<usize, NoiseSimulationError> {
        if self.crt_slot_count == 0 ||
            self.component_columns == 0 ||
            self.component_columns % 2 != 0 ||
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

    /// Number `D` of mask transcript coordinates used by the smudging bound:
    /// `mask_slot_count * (2 * ell_beta) * coefficient_count`.
    pub(crate) fn mask_smudging_coordinates(&self) -> Result<usize, NoiseSimulationError> {
        if self.crt_slot_count == 0 ||
            self.component_columns == 0 ||
            self.component_columns % 2 != 0 ||
            self.coefficient_count == 0
        {
            return Err(NoiseSimulationError::InvalidRefresh("invalid transcript topology"));
        }
        self.crt_slot_count
            .checked_mul(self.component_columns)
            .and_then(|value| value.checked_mul(self.coefficient_count))
            .ok_or(NoiseSimulationError::IntegerConversion)
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

/// Exact variance of a centered complete base-`P` digit.  Routing artifacts
/// may scale this value with their concrete decomposition before attaching it
/// through `with_digit_variances`.
pub fn centered_base_p_digit_variance(
    base_p: &BigUint,
) -> Result<AverageVariance, NoiseSimulationError> {
    if *base_p < BigUint::from(2u8) || base_p % 2u8 != BigUint::zero() {
        return Err(NoiseSimulationError::InvalidRefresh("base P must be even and at least two"));
    }
    // The state tracked by this module is the doubled error `2e`.  A
    // centered complete digit has Var(e) = (P^2 - 1) / 12, hence the exact
    // doubled-coordinate variance is four times that value.
    Ok(AverageVariance::new(base_p * base_p - 1u8, BigUint::from(3u8))?)
}

impl RefreshNoiseParameters {
    fn validate(&self) -> Result<(), NoiseSimulationError> {
        if self.slot_count == 0 ||
            self.slot_count != self.plaintext_moduli.len() ||
            self.plaintext_moduli.iter().any(|modulus| modulus <= &BigUint::one()) ||
            self.base_p < BigUint::from(2u8) ||
            !is_power_of_two(&self.base_p) ||
            self.mask_digit_counts.len() != self.slot_count ||
            self.mask_digit_counts.iter().any(|count| *count == 0) ||
            self.switch_moduli.len() != self.slot_count ||
            self.fresh_digit_count == 0
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
            mask_digit_variances: Vec::new(),
            fresh_digit_variances: Vec::new(),
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
                .any(|gain| gain.l1.is_zero() || gain.l2_squared.is_zero()) ||
            gains
                .fresh
                .iter()
                .chain(gains.decoder.iter())
                .any(|gain| gain.l1.is_zero() || gain.l2_squared.is_zero())
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

    pub fn with_digit_variances(
        mut self,
        mask_digit_variances: Vec<AverageVariance>,
        fresh_digit_variances: Vec<AverageVariance>,
    ) -> Result<Self, NoiseSimulationError> {
        if mask_digit_variances.len() != self.slot_count ||
            fresh_digit_variances.len() != self.slot_count
        {
            return Err(NoiseSimulationError::InvalidRefresh("digit variance slot mismatch"));
        }
        self.mask_digit_variances = mask_digit_variances;
        self.fresh_digit_variances = fresh_digit_variances;
        self.validate()?;
        Ok(self)
    }
}

/// Application noise evaluated through the program and readout, before the
/// d_m-dependent refresh equations.  Parameter search reuses this exact
/// value while scanning mask digit counts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreRefreshNoiseEvaluation {
    pub program: ProgramNoiseReport,
    pub readout_bound: BigUint,
    pub readout_variance: AverageVariance,
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
    pub z_squared: AverageVariance,
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
    pub variance: AverageVariance,
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
    pub readout_mask: AverageVariance,
    pub readout_fresh: AverageVariance,
    pub decoder: AverageVariance,
    pub state: AverageVariance,
    pub mask_digit: AverageVariance,
    pub fresh_digit: AverageVariance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutNoiseReport {
    pub program: ProgramNoiseReport,
    pub readout_bound: BigUint,
    pub refresh: RefreshNoiseReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExponentLutAverageNoiseReport {
    pub snapshot_identity: [u8; 32],
    pub program: ProgramNoiseReport,
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
    Program(#[from] ProgramNoiseError),
    #[error("program input bound is missing: {0:?}")]
    MissingInputBound(ProgramInputId),
    #[error("program input variance is missing: {0:?}")]
    MissingInputVariance(ProgramInputId),
    #[error("public family member error state is missing: family {family:?}, member {member}")]
    MissingPublicFamilyInput { family: WireFamilyId, member: usize },
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
    #[error("program wire bound is missing: {0:?}")]
    MissingWireBound(WireId),
    #[error("program wire variance is missing: {0:?}")]
    MissingWireVariance(WireId),
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

#[derive(Clone)]
pub struct ExponentLutNoiseSnapshot {
    pub authority: NoiseModelKind,
    pub layout: ExpLayout,
    pub program: ExponentLutProgram,
    pub initial_state_bound: BigUint,
    pub initial_state_doubled_variance: AverageVariance,
    pub helper_error_cutoff: BigUint,
    pub helper_doubled_variance: AverageVariance,
    /// Fresh error carried by each online public-a BGG input.  This is kept
    /// separate from helper error state so the average model cannot
    /// accidentally charge a public monomial twice (or treat it as cached).
    pub public_input_error_cutoff: BigUint,
    pub public_input_doubled_variance: AverageVariance,
    pub gate_gains: Vec<GateNoiseGains>,
    pub readout_gains: ValueReadoutNoiseGains,
    pub refresh: RefreshNoiseParameters,
    pub gain_plan: NoiseGainPlan,
    pub family_active_counts: Vec<usize>,
    target_plaintext_gain: RightActionNoiseGain,
    native_modulus: BigUint,
    identity: [u8; 32],
}

impl ExponentLutNoiseSnapshot {
    pub fn from_setup(
        identity: [u8; 32],
        program: SparseLwrPrfProgram,
        initial_state_bound: BigUint,
        initial_variance: AverageVariance,
        helper_error_cutoff: BigUint,
        helper_variance: AverageVariance,
        refresh: RefreshNoiseParameters,
        gain_plan: NoiseGainPlan,
        family_active_counts: Vec<usize>,
    ) -> Result<Self, NoiseSimulationError> {
        let layout = program.profile().layout;
        if let Some(program_layout) = program.accumulation.layout() {
            if program_layout.exponent != layout {
                return Err(NoiseSimulationError::InvalidDecomposition);
            }
        } else {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        gain_plan.validate(&program.accumulation, &family_active_counts)?;
        refresh.validate()?;
        let ring_dimension = program
            .profile()
            .ring_dimension()
            .map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        layout.validate(ring_dimension).map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        let target_plaintext_gain = program
            .readout
            .target_plaintext_gain(ring_dimension)
            .map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        let native_modulus = BigUint::from(layout.effective_modulus(ring_dimension));
        if refresh.route_gains.is_none() {
            return Err(NoiseSimulationError::InvalidRefresh("route gains are required"));
        }
        if refresh.smudging.is_none() {
            return Err(NoiseSimulationError::InvalidRefresh("smudging parameters are required"));
        }
        if refresh.global_modulus.is_none() {
            return Err(NoiseSimulationError::InvalidRefresh("global modulus is required"));
        }
        if refresh.mask_digit_variances.len() != refresh.slot_count ||
            refresh.fresh_digit_variances.len() != refresh.slot_count
        {
            return Err(NoiseSimulationError::InvalidRefresh("digit variances are required"));
        }
        if refresh.fresh_digit_variances.iter().any(|variance| variance.denominator.is_zero()) {
            return Err(NoiseSimulationError::InvalidRefresh("invalid digit variance"));
        }
        let topology = refresh
            .transcript_topology
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("transcript topology is required"))?;
        if topology.coefficient_count != ring_dimension {
            return Err(NoiseSimulationError::InvalidRefresh("transcript ring dimension mismatch"));
        }
        let expected_coordinates = topology.mask_smudging_coordinates()?;
        if refresh
            .smudging
            .as_ref()
            .map(|parameters| parameters.transcript_coordinates != expected_coordinates)
            .unwrap_or(true)
        {
            return Err(NoiseSimulationError::InvalidRefresh(
                "smudging coordinates do not match mask topology",
            ));
        }
        if refresh.mask_digit_variances.iter().any(|variance| variance.denominator.is_zero()) {
            return Err(NoiseSimulationError::InvalidRefresh("invalid mask digit variance"));
        }
        let trace = program
            .readout
            .trace_plan(ring_dimension)
            .map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        if gain_plan.readout.nonidentity_trace.len() != trace.helper_count() {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        Ok(Self {
            authority: NoiseModelKind::WorstCase,
            layout,
            program: program.accumulation,
            initial_state_bound,
            initial_state_doubled_variance: initial_variance,
            helper_error_cutoff: helper_error_cutoff.clone(),
            helper_doubled_variance: helper_variance.clone(),
            public_input_error_cutoff: helper_error_cutoff.clone(),
            public_input_doubled_variance: helper_variance.clone(),
            gate_gains: gain_plan.program.gates().to_vec(),
            readout_gains: gain_plan.readout.clone(),
            refresh,
            gain_plan,
            family_active_counts,
            target_plaintext_gain,
            native_modulus,
            identity,
        })
    }

    pub fn setup_identity(&self) -> &[u8; 32] {
        &self.identity
    }

    pub fn simulate(&self) -> Result<ExponentLutNoiseReport, NoiseSimulationError> {
        // Use the same topology-derived public-family map as AverageCase so
        // Family body MulPublic never silently drops online input errors.
        self.simulate_with_inputs(self.setup_average_inputs())
    }

    /// Evaluates the fixed program/readout portion using the snapshot's
    /// topology-derived inputs.  Refresh digit counts do not affect this
    /// result and callers may safely reuse it across a `d_m` scan.
    pub fn evaluate_pre_refresh(&self) -> Result<PreRefreshNoiseEvaluation, NoiseSimulationError> {
        self.evaluate_pre_refresh_with_inputs(self.setup_average_inputs())
    }

    pub fn evaluate_pre_refresh_with_inputs(
        &self,
        inputs: ProgramNoiseInputs,
    ) -> Result<PreRefreshNoiseEvaluation, NoiseSimulationError> {
        let program = simulate_program(
            &self.program,
            &inputs,
            &self.helper_error_cutoff,
            &self.helper_doubled_variance,
            self.gain_plan.program.gates(),
            &self.family_active_counts,
        )?;
        let target_bound = &self.readout_gains.target.l1 * &program.output_bound;
        let mut trace_bound = &self.readout_gains.identity_trace.l1 * &target_bound;
        for gain in &self.readout_gains.nonidentity_trace {
            trace_bound += &gain.action.l1 * &target_bound;
            // The switch error is multiplied by sigma(mu*T'), where the
            // native accumulator mu is a signed monomial. The online mask
            // error separately acts through its cached public decomposition.
            trace_bound += &(&self.target_plaintext_gain.l1 + &gain.helper_action.l1) *
                &self.helper_error_cutoff;
        }
        let readout_bound = &self.readout_gains.normalization.l1 * trace_bound;
        let output_wire =
            *self.program.outputs().last().ok_or(NoiseSimulationError::MissingOutput)?;
        let target_variance = program
            .wire_variances
            .get(&output_wire)
            .cloned()
            .ok_or(NoiseSimulationError::MissingWireVariance(output_wire))?
            .scaled(&self.readout_gains.target.l2_squared);
        let mut trace_variance = target_variance.clone();
        for gain in &self.readout_gains.nonidentity_trace {
            trace_variance = trace_variance.add(&target_variance.scaled(&gain.action.l2_squared));
            trace_variance = trace_variance.add(&self.helper_doubled_variance.scaled(
                &(&self.target_plaintext_gain.l2_squared + &gain.helper_action.l2_squared),
            ));
        }
        let readout_variance = trace_variance.scaled(&self.readout_gains.normalization.l2_squared);
        Ok(PreRefreshNoiseEvaluation { program, readout_bound, readout_variance })
    }

    pub fn simulate_with_inputs(
        &self,
        inputs: ProgramNoiseInputs,
    ) -> Result<ExponentLutNoiseReport, NoiseSimulationError> {
        let pre_refresh = self.evaluate_pre_refresh_with_inputs(inputs)?;
        self.simulate_worst_from_pre_refresh(pre_refresh)
    }

    pub fn simulate_worst_from_pre_refresh(
        &self,
        pre_refresh: PreRefreshNoiseEvaluation,
    ) -> Result<ExponentLutNoiseReport, NoiseSimulationError> {
        let PreRefreshNoiseEvaluation { program, readout_bound: output_bound, .. } = pre_refresh;
        let refresh = self.simulate_worst_refresh(&output_bound)?;
        Ok(ExponentLutNoiseReport { program, readout_bound: output_bound, refresh })
    }

    /// Evaluates only the d_m-dependent worst-case refresh equations.  This
    /// is the search fast path: it does not clone or materialize the program
    /// report for every mask-digit candidate.
    pub fn worst_refresh_accepted_from_pre_refresh(
        &self,
        pre_refresh: &PreRefreshNoiseEvaluation,
    ) -> Result<bool, NoiseSimulationError> {
        Ok(self.simulate_worst_refresh(&pre_refresh.readout_bound)?.accepted)
    }

    fn simulate_worst_refresh(
        &self,
        output_bound: &BigUint,
    ) -> Result<RefreshNoiseReport, NoiseSimulationError> {
        let routes = self
            .refresh
            .route_gains
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("route gains are required"))?;
        self.refresh.validate()?;
        let fresh_digits = u32::try_from(self.refresh.fresh_digit_count)
            .map_err(|_| NoiseSimulationError::IntegerConversion)?;
        let fresh_modulus = self.refresh.base_p.pow(fresh_digits);
        // The unchanged fresh digit construction has this conservative support.
        let fresh_bound = &fresh_modulus - 1u8;
        let mut mask_coverage_accepted = true;
        let mut rounding_accepted = true;
        let mut fresh_error_accepted = true;
        let mut mask_smudging_accepted = true;
        let global_modulus = self
            .refresh
            .global_modulus
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("global modulus is required"))?;
        let topology = self
            .refresh
            .transcript_topology
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("transcript topology is required"))?;
        let smudging = self
            .refresh
            .smudging
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("smudging parameters are required"))?;
        let smudging_shift =
            smudging.lambda_mask.checked_add(1).ok_or(NoiseSimulationError::IntegerConversion)?;
        let smudging_factor =
            (BigUint::one() << smudging_shift) * BigUint::from(smudging.transcript_coordinates);
        let slots = self
            .refresh
            .plaintext_moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let digits = u32::try_from(self.refresh.mask_digit_counts[slot])
                    .map_err(|_| NoiseSimulationError::IntegerConversion)?;
                let mask_modulus = self.refresh.base_p.pow(digits);
                let mask_bound = &mask_modulus / 2u8;
                mask_coverage_accepted &= mask_modulus >= self.native_modulus;
                let pre_switch_bound = &routes.state_scaling[slot].l1 * &self.initial_state_bound +
                    (&routes.mask[slot].l1 + &routes.fresh[slot].l1) * output_bound;
                let decoder_bound = &routes.decoder[slot].l1 * &self.helper_error_cutoff;
                let destination = &self.refresh.switch_moduli[slot];
                let lift = global_modulus / destination;
                let spacing = destination / modulus;
                let rounding_numerator =
                    (&lift - 1u8) * (BigUint::from(topology.coefficient_count) + 2u8);
                let rounding_denominator = &lift * 2u8;
                let route_bound = switched_error_bound(
                    &pre_switch_bound,
                    &lift,
                    topology.coefficient_count,
                    &decoder_bound,
                )?;
                let total_bound = &mask_bound + &route_bound;
                let kappa = global_modulus / modulus;
                let rounding_lhs = &total_bound * 2u8;
                let rounding_margin =
                    BigInt::from(spacing.clone()) - BigInt::from(rounding_lhs.clone());
                let smudging_margin = BigInt::from(mask_modulus.clone()) -
                    BigInt::from(&smudging_factor * &route_bound);
                mask_smudging_accepted &= smudging_margin.sign() != num_bigint::Sign::Minus;
                let fresh_ok = fresh_error_below_every_prime_factor(&fresh_bound, modulus)?;
                rounding_accepted &= rounding_margin.sign() == num_bigint::Sign::Plus;
                fresh_error_accepted &= fresh_ok;
                Ok(RefreshSlotNoiseReport {
                    slot,
                    bound: total_bound.clone(),
                    margin: rounding_margin.clone(),
                    kappa,
                    switch_modulus: destination.clone(),
                    lift,
                    spacing,
                    mask_modulus,
                    mask_bound,
                    pre_switch_bound,
                    rounding_numerator,
                    rounding_denominator,
                    decoder_bound,
                    residual_bound: route_bound,
                    smudging_margin,
                    rounding_lhs,
                    rounding_margin,
                    fresh_error_accepted: fresh_ok,
                })
            })
            .collect::<Result<Vec<_>, NoiseSimulationError>>()?;
        let domain_accepted = mask_coverage_accepted;
        let accepted =
            rounding_accepted && fresh_error_accepted && domain_accepted && mask_smudging_accepted;
        Ok(RefreshNoiseReport {
            slots,
            mask_coverage_accepted,
            rounding_accepted,
            fresh_error_accepted,
            domain_accepted,
            mask_smudging_accepted,
            accepted,
        })
    }

    pub fn simulate_average(
        &self,
        config: &AverageCaseConfig,
    ) -> Result<ExponentLutAverageNoiseReport, NoiseSimulationError> {
        self.simulate_average_with_inputs(config, self.setup_average_inputs())
    }

    /// Build the complete candidate input state from setup-owned topology.
    /// Public-family error inputs are explicit here; an AverageCase run never
    /// silently substitutes an empty family map for a declared family.
    fn setup_average_inputs(&self) -> ProgramNoiseInputs {
        let mut inputs = ProgramNoiseInputs {
            input_bounds: self
                .program
                .inputs()
                .keys()
                .map(|id| (*id, self.initial_state_bound.clone()))
                .collect(),
            input_variances: self
                .program
                .inputs()
                .keys()
                .map(|id| (*id, self.initial_state_doubled_variance.clone()))
                .collect(),
            ..ProgramNoiseInputs::default()
        };
        let mut family_index = 0usize;
        for step in self.program.steps() {
            if let ProgramStep::Family { body, .. } = step {
                let wire_family = body.wire_inputs().values().find_map(|source| match source {
                    FamilyWireSource::Member(family) => Some(family),
                    FamilyWireSource::Broadcast(_) => None,
                });
                let Some(wire_family) = wire_family else { continue };
                if self.family_active_counts.get(family_index).is_some() {
                    // Every online `a_j[i]` is a fresh BGG input.  Its
                    // monomial action has norm one; only this input error
                    // is charged here, exactly once.  The family is uniform
                    // in this application, so retain one value instead of a
                    // vector with one clone per active member.
                    inputs
                        .public_family_uniform_bounds
                        .insert(*wire_family, self.public_input_error_cutoff.clone());
                    inputs
                        .public_family_uniform_variances
                        .insert(*wire_family, self.public_input_doubled_variance.clone());
                }
                family_index += 1;
            }
        }
        inputs
    }

    pub fn simulate_average_with_inputs(
        &self,
        config: &AverageCaseConfig,
        inputs: ProgramNoiseInputs,
    ) -> Result<ExponentLutAverageNoiseReport, NoiseSimulationError> {
        if !config.allow_average_acceptance {
            return Err(NoiseSimulationError::AverageAcceptanceDisabled);
        }
        let pre_refresh = self.evaluate_pre_refresh_with_inputs(inputs)?;
        self.simulate_average_with_pre_refresh(config, pre_refresh)
    }

    pub fn simulate_average_with_pre_refresh(
        &self,
        config: &AverageCaseConfig,
        pre_refresh: PreRefreshNoiseEvaluation,
    ) -> Result<ExponentLutAverageNoiseReport, NoiseSimulationError> {
        if !config.allow_average_acceptance {
            return Err(NoiseSimulationError::AverageAcceptanceDisabled);
        }
        let (_, report) = self.simulate_average_internal(config, &pre_refresh, true)?;
        report.ok_or(NoiseSimulationError::InvalidRefresh("average report was not materialized"))
    }

    /// Evaluates only average-case refresh equations for search.  The shared
    /// pre-refresh evaluation is borrowed, so no program/report clone is made
    /// for rejected d_m candidates.
    pub fn average_refresh_accepted_from_pre_refresh(
        &self,
        config: &AverageCaseConfig,
        pre_refresh: &PreRefreshNoiseEvaluation,
    ) -> Result<bool, NoiseSimulationError> {
        if !config.allow_average_acceptance {
            return Err(NoiseSimulationError::AverageAcceptanceDisabled);
        }
        let (accepted, _) = self.simulate_average_internal(config, pre_refresh, false)?;
        Ok(accepted)
    }

    fn simulate_average_internal(
        &self,
        config: &AverageCaseConfig,
        pre_refresh: &PreRefreshNoiseEvaluation,
        materialize_program: bool,
    ) -> Result<(bool, Option<ExponentLutAverageNoiseReport>), NoiseSimulationError> {
        let hard_refresh = self.simulate_worst_refresh(&pre_refresh.readout_bound)?;
        // The pre-refresh evaluator already records the exact terminal
        // variance. Reuse it instead of traversing the program again.
        let output_variance = pre_refresh.readout_variance.clone();
        let routes = self
            .refresh
            .route_gains
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("route gains are required"))?;
        let fresh_digits = u32::try_from(self.refresh.fresh_digit_count)
            .map_err(|_| NoiseSimulationError::IntegerConversion)?;
        let fresh_modulus = self.refresh.base_p.pow(fresh_digits);
        let fresh_bound = &fresh_modulus - 1u8;
        let mask_coverage_accepted = hard_refresh.mask_coverage_accepted;
        let topology = self
            .refresh
            .transcript_topology
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("transcript topology is required"))?;
        let joint_event_count = config.joint_event_count(topology)?;
        let z_squared = config.z_squared(topology)?;
        let route_variance = |slot: usize| {
            // Mask and fresh PRF labels are distinct iid routed outputs;
            // their variances add rather than forming a coherent combined
            // state action.
            let lift = &hard_refresh.slots[slot].lift;
            let switched = |variance: AverageVariance| {
                AverageVariance::new(variance.numerator, variance.denominator * lift * lift)
                    .map_err(NoiseSimulationError::from)
            };
            let breakdown = AverageVarianceBreakdown {
                readout_mask: switched(output_variance.scaled(&routes.mask[slot].l2_squared))?,
                readout_fresh: switched(output_variance.scaled(&routes.fresh[slot].l2_squared))?,
                decoder: self.helper_doubled_variance.scaled(&routes.decoder[slot].l2_squared),
                state: switched(
                    self.initial_state_doubled_variance
                        .scaled(&routes.state_scaling[slot].l2_squared),
                )?,
                // Nominal payload distribution is reported separately; it is
                // not an operational error and is not added to the tail.
                mask_digit: self.refresh.mask_digit_variances[slot].clone(),
                fresh_digit: self.refresh.fresh_digit_variances[slot].clone(),
            };
            let variance = breakdown
                .readout_mask
                .add(&breakdown.readout_fresh)
                .add(&breakdown.decoder)
                .add(&breakdown.state);
            Ok::<(AverageVariance, AverageVarianceBreakdown), NoiseSimulationError>((
                variance, breakdown,
            ))
        };
        let mut rounding_accepted = true;
        let mut fresh_error_accepted = true;
        let mut domain_accepted = mask_coverage_accepted;
        let mut mask_smudging_accepted = true;
        let mut max_f_avg = BigUint::zero();
        let smudging = self
            .refresh
            .smudging
            .as_ref()
            .ok_or(NoiseSimulationError::InvalidRefresh("smudging parameters are required"))?;
        let smudging_shift =
            smudging.lambda_mask.checked_add(1).ok_or(NoiseSimulationError::IntegerConversion)?;
        let smudging_factor =
            (BigUint::one() << smudging_shift) * BigUint::from(smudging.transcript_coordinates);
        let slots = self
            .refresh
            .plaintext_moduli
            .iter()
            .enumerate()
            .map(|(slot, modulus)| {
                let exact = &hard_refresh.slots[slot];
                let mask_bound = &exact.mask_bound;
                let (variance, variance_breakdown) = route_variance(slot)?;
                let f_avg = average_favg(&z_squared, &variance, config.tail_correction_bits)? +
                    ceil_div(&exact.rounding_numerator, &exact.rounding_denominator)?;
                if f_avg > max_f_avg {
                    max_f_avg = f_avg.clone();
                }
                let global_modulus =
                    self.refresh.global_modulus.as_ref().ok_or(
                        NoiseSimulationError::InvalidRefresh("global modulus is required"),
                    )?;
                let kappa = global_modulus / modulus;
                if kappa.is_zero() {
                    return Err(NoiseSimulationError::InvalidRefresh("invalid CRT modulus ratio"));
                }
                let rounding_lhs = (mask_bound + &f_avg) * 2u8;
                let rounding_margin =
                    BigInt::from(exact.spacing.clone()) - BigInt::from(rounding_lhs.clone());
                let rounding = rounding_margin.sign() == num_bigint::Sign::Plus;
                let smudging_margin = BigInt::from(exact.mask_modulus.clone()) -
                    BigInt::from(&smudging_factor * &f_avg);
                mask_smudging_accepted &= smudging_margin.sign() != num_bigint::Sign::Minus;
                let fresh = fresh_error_below_every_prime_factor(&fresh_bound, modulus)?;
                let domain = true;
                rounding_accepted &= rounding;
                fresh_error_accepted &= fresh;
                domain_accepted &= domain;
                let rounding_deficit = if rounding_margin.sign() == num_bigint::Sign::Minus {
                    rounding_margin.magnitude().clone()
                } else {
                    BigUint::zero()
                };
                Ok(AverageRefreshSlotNoiseReport {
                    slot,
                    variance,
                    f_avg,
                    kappa: kappa.clone(),
                    switch_modulus: exact.switch_modulus.clone(),
                    lift: exact.lift.clone(),
                    spacing: exact.spacing.clone(),
                    rounding_numerator: exact.rounding_numerator.clone(),
                    rounding_denominator: exact.rounding_denominator.clone(),
                    deterministic_mask_bound: mask_bound.clone(),
                    rounding_lhs,
                    rounding_margin,
                    rounding_deficit,
                    smudging_margin,
                    fresh_error_accepted: fresh,
                    variance_breakdown,
                })
            })
            .collect::<Result<Vec<_>, NoiseSimulationError>>()?;
        let correctness_accepted =
            rounding_accepted && fresh_error_accepted && domain_accepted && mask_smudging_accepted;
        // The selected correctness authority is average-case.  The worst-case
        // report contributes only independent mask/fresh/domain obligations;
        // its rounding and smudging result must not veto average correctness.
        let hard_authority_accepted = hard_refresh.mask_coverage_accepted &&
            hard_refresh.fresh_error_accepted &&
            hard_refresh.domain_accepted;
        let accepted = hard_authority_accepted && correctness_accepted;
        let refresh = AverageRefreshNoiseReport {
            slots,
            joint_event_count,
            z_squared,
            max_f_avg,
            mask_coverage_accepted,
            rounding_accepted,
            fresh_error_accepted,
            domain_accepted,
            mask_smudging_accepted,
            accepted,
        };
        let report = materialize_program.then(|| ExponentLutAverageNoiseReport {
            snapshot_identity: self.identity,
            program: pre_refresh.program.clone(),
            refresh,
            security_authority: NoiseModelKind::WorstCase,
            correctness_authority: NoiseModelKind::AverageCase,
            hard_authority_accepted,
            correctness_accepted,
            accepted,
        });
        Ok((accepted, report))
    }
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

/// Exact hard operational budget shared by reports and numeric diagnostics.
pub(crate) fn switched_error_bound(
    pre_switch: &BigUint,
    lift: &BigUint,
    ring_dimension: usize,
    decoder: &BigUint,
) -> Result<BigUint, NoiseSimulationError> {
    if lift.is_zero() {
        return Err(NoiseSimulationError::InvalidRefresh("zero switching lift"));
    }
    let rounding = (lift - 1u8) * (BigUint::from(ring_dimension) + 2u8);
    Ok(ceil_div(&(pre_switch * 2u8 + rounding), &(lift * 2u8))? + decoder)
}

fn ceil_div(numerator: &BigUint, denominator: &BigUint) -> Result<BigUint, NoiseSimulationError> {
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
fn average_favg(
    z_squared: &AverageVariance,
    variance: &AverageVariance,
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
    use crate::noise::{
        GateNoiseOperation, NoiseDecompositionSource, balanced_digit_step,
        balanced_interval_next_bound, balanced_interval_representable, balanced_profile,
        centered_residue,
    };
    use std::collections::BTreeMap;

    struct TestTransfer {
        output_bound: BigUint,
        fresh_bound: BigUint,
        output_variance: AverageVariance,
    }

    fn transfer_mul_private(
        input_bound: &BigUint,
        input_variance: &AverageVariance,
        action: &RightActionNoiseGain,
        helper_bound: &BigUint,
        helper_variance: &AverageVariance,
    ) -> TestTransfer {
        TestTransfer {
            output_bound: &action.l1 * input_bound + helper_bound,
            fresh_bound: helper_bound.clone(),
            output_variance: input_variance.scaled(&action.l2_squared).add(helper_variance),
        }
    }

    fn transfer_mul_public(
        private_bound: &BigUint,
        private_variance: &AverageVariance,
        public_bound: &BigUint,
        public_variance: &AverageVariance,
        action: &RightActionNoiseGain,
    ) -> TestTransfer {
        TestTransfer {
            output_bound: private_bound + &action.l1 * public_bound,
            fresh_bound: &action.l1 * public_bound,
            output_variance: private_variance.add(&public_variance.scaled(&action.l2_squared)),
        }
    }

    fn transfer_lut(
        input_bound: &BigUint,
        input_variance: &AverageVariance,
        direct: &RightActionNoiseGain,
        helper: &RightActionNoiseGain,
        helper_bound: &BigUint,
        helper_variance: &AverageVariance,
    ) -> TestTransfer {
        let helper_l1 = BigUint::one() + &helper.l1;
        let helper_l2 = BigUint::one() + &helper.l2_squared;
        TestTransfer {
            output_bound: &direct.l1 * input_bound + helper_bound * &helper_l1,
            fresh_bound: helper_bound * helper_l1,
            output_variance: input_variance
                .scaled(&direct.l2_squared)
                .add(&helper_variance.scaled(&helper_l2)),
        }
    }

    struct AddInputSummary {
        inherited_bound: BigUint,
        fresh_bound: BigUint,
        inherited_variance: AverageVariance,
        fresh_variance: AverageVariance,
        operand_count: usize,
    }

    fn transfer_add(input: AddInputSummary) -> TestTransfer {
        let variance = input.inherited_variance.add(&input.fresh_variance);
        TestTransfer {
            output_bound: input.inherited_bound + input.fresh_bound,
            fresh_bound: BigUint::zero(),
            output_variance: variance.scaled(&BigUint::from(input.operand_count)),
        }
    }

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
    fn balanced_decomposition_gain_uses_actual_coefficients() {
        let plan = NoiseDecompositionPlan::new(
            BigUint::from(16u8),
            1,
            vec![BigInt::from(-3), BigInt::from(1), BigInt::from(2), BigInt::from(-1)],
            2,
            2,
            1,
        )
        .unwrap();
        let gain = plan.action_gain();
        assert_eq!(gain.l1, BigUint::from(5u8));
        assert_eq!(gain.l2_squared, BigUint::from(13u8));
    }

    #[test]
    fn decomposition_shape_validation() {
        assert!(NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![], 1, 1, 1).is_err());
        assert!(
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(0)], 1, 1, 1)
                .is_ok()
        );
        assert!(
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(0)], 1, 1, 0)
                .is_err()
        );
        // One target row, two digits, two ring coefficients, and one output
        // column require four concrete coefficients.
        assert!(
            NoiseDecompositionPlan::new(BigUint::from(2u8), 2, vec![BigInt::from(0); 3], 1, 1, 2,)
                .is_err()
        );
    }

    #[test]
    fn decomposition_gain_sums_digit_and_ring_axes_in_canonical_order() {
        // [decomposed row][output column][ring coefficient], with one target
        // row expanded over two digits and two ring coefficients.
        let plan = NoiseDecompositionPlan::new(
            BigUint::from(2u8),
            2,
            vec![
                1.into(),
                (-2).into(),
                3.into(),
                4.into(),
                1.into(),
                (-6).into(),
                0.into(),
                1.into(),
            ],
            1,
            2,
            2,
        )
        .unwrap();
        let gain = plan.action_gain();
        // A right action is bounded per output column.  In this tensor the
        // first column contains [1, -2, 1, -6], so its exact norms are
        // l1=10 and l2^2=42; the second column has l1=8 and l2^2=26.
        // `action_gain` takes the maximum over output columns, as required by
        // gamma_1(D) and gamma_2^2(D), rather than summing both columns.
        assert_eq!(gain.l1, BigUint::from(10u8));
        assert_eq!(gain.l2_squared, BigUint::from(42u8));
    }

    #[test]
    fn compact_sparse_profile_matches_explicit_balanced_tensor() {
        // 6 in balanced base four is (-2) + 2*4.  Put that monomial in
        // output column one and ring coefficient one; the right action must
        // not accidentally sum across output columns as if this were a left
        // action.
        let explicit = NoiseDecompositionPlan::new(
            BigUint::from(4u8),
            2,
            vec![0.into(), 0.into(), 0.into(), (-2).into(), 0.into(), 0.into(), 0.into(), 2.into()],
            1,
            2,
            2,
        )
        .unwrap();
        let compact = NoiseDecompositionPlan::sparse_monomial(
            BigUint::from(4u8),
            2,
            1,
            2,
            2,
            0,
            1,
            1,
            6.into(),
        )
        .unwrap();
        assert_eq!(compact.action_gain(), explicit.action_gain());
        assert_eq!(compact.action_gain().l1, BigUint::from(4u8));
        assert_eq!(compact.action_gain().l2_squared, BigUint::from(8u8));
    }

    #[test]
    fn compact_unit_diagonal_and_power_match_explicit_tensors() {
        let diagonal = NoiseDecompositionPlan::diagonal_unit(BigUint::from(4u8), 2, 2, 2).unwrap();
        let explicit_diagonal = NoiseDecompositionPlan::new(
            BigUint::from(4u8),
            2,
            vec![
                1.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                1.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
            ],
            2,
            2,
            2,
        )
        .unwrap();
        assert_eq!(diagonal.action_gain(), explicit_diagonal.action_gain());

        let power =
            NoiseDecompositionPlan::power_of_base(BigUint::from(4u8), 3, 1, 1, 1, 0, 0, 0, 2)
                .unwrap();
        let explicit_power = NoiseDecompositionPlan::new(
            BigUint::from(4u8),
            3,
            vec![0.into(), 0.into(), 1.into()],
            1,
            1,
            1,
        )
        .unwrap();
        assert_eq!(power.action_gain(), explicit_power.action_gain());

        // GadgetScalar repeats each local CRT digit over all target columns.
        // The optimized implementation must agree with the representative
        // digit-class calculation, independent of that repetition count.
        let gadget =
            NoiseDecompositionPlan::gadget_scalar(BigUint::from(4u8), 3, 4, 12, 2, 5.into())
                .unwrap()
                .bind_crt_layout(vec![64u8.into()], 3)
                .unwrap();
        let expected = (0..3)
            .map(|digit| {
                let coefficient = BigInt::from(5u8) * BigInt::from(4u8).pow(digit as u32);
                let coefficient = centered_residue(&coefficient, &BigUint::from(64u8));
                balanced_profile(&coefficient, &BigUint::from(4u8), 3).unwrap()
            })
            .collect::<Vec<_>>();
        let gain = gadget.action_gain();
        assert_eq!(gain.l1, expected.iter().map(|value| value.0.clone()).max().unwrap());
        assert_eq!(gain.l2_squared, expected.iter().map(|value| value.1.clone()).max().unwrap());
    }

    #[test]
    fn dense_bound_is_compact_at_n_65536_and_depends_on_base() {
        let plan = NoiseDecompositionPlan::dense_bound(
            BigUint::from(1u8) << 25,
            2,
            4,
            3,
            1usize << 16,
            (BigUint::from(1u8) << 49) - 1u8,
        )
        .unwrap();
        assert!(matches!(plan.source, NoiseDecompositionSource::DenseBound { .. }));
        assert!(std::mem::size_of_val(&plan) < 512);
        let gain = plan.action_gain();
        let per_column = BigUint::from(2u8) * (BigUint::from(1u8) << 24);
        assert_eq!(gain.l1, per_column * 4u8 * (1usize << 16));
        assert_eq!(
            gain.l2_squared,
            BigUint::from(8u8) * (BigUint::from(1u8) << 48) * (1usize << 16)
        );

        let binary = NoiseDecompositionPlan::dense_bound(
            BigUint::from(2u8),
            50,
            4,
            3,
            1usize << 16,
            (BigUint::from(1u8) << 49) - 1u8,
        )
        .unwrap();
        assert_ne!(binary.action_gain().l1, gain.l1);
    }

    #[test]
    fn dense_bound_uses_exact_balanced_interval_recurrence() {
        let mut bound = BigUint::from(5u8);
        let mut bounds = Vec::new();
        for _ in 0..3 {
            bounds.push(bound.clone());
            bound = balanced_interval_next_bound(&bound, &BigUint::from(8u8));
        }
        assert_eq!(bounds, vec![5u8.into(), 1u8.into(), 0u8.into()]);

        let mut bound = BigUint::from(22u8);
        let mut bounds = Vec::new();
        for _ in 0..4 {
            bounds.push(bound.clone());
            bound = balanced_interval_next_bound(&bound, &BigUint::from(4u8));
        }
        assert_eq!(bounds, vec![22u8.into(), 6u8.into(), 2u8.into(), 0u8.into()]);

        for base in [2u8, 4, 8, 16] {
            for coefficient_bound in 0u8..=64 {
                let bound = BigUint::from(coefficient_bound);
                let base_big = BigUint::from(base);
                let expected = (-(coefficient_bound as i32)..=(coefficient_bound as i32))
                    .map(|value| {
                        balanced_digit_step(&BigInt::from(value), &BigInt::from(base))
                            .1
                            .magnitude()
                            .clone()
                    })
                    .max()
                    .unwrap_or_default();
                assert_eq!(
                    balanced_interval_next_bound(&bound, &base_big),
                    expected,
                    "base={base}, coefficient_bound={coefficient_bound}"
                );
            }
        }
    }

    #[test]
    fn compact_gains_sum_independent_crt_tower_decompositions() {
        let unit = NoiseDecompositionPlan::unit(BigUint::from(4u8), 4, 1, 1, 1, 0, 0, 0)
            .unwrap()
            .bind_crt_layout(vec![17u8.into(), 19u8.into()], 2)
            .unwrap();
        assert_eq!(unit.action_gain().l1, BigUint::from(2u8));
        assert_eq!(unit.action_gain().l2_squared, BigUint::from(2u8));

        let sparse = NoiseDecompositionPlan::sparse_monomial(
            BigUint::from(4u8),
            4,
            1,
            1,
            1,
            0,
            0,
            0,
            18.into(),
        )
        .unwrap()
        .bind_crt_layout(vec![17u8.into(), 19u8.into()], 2)
        .unwrap();
        // 18 is 1 modulo 17 and -1 modulo 19, so both tower profiles have
        // unit gain despite the distinct full-modulus source value.
        assert_eq!(sparse.action_gain().l1, BigUint::from(2u8));
        assert_eq!(sparse.action_gain().l2_squared, BigUint::from(2u8));
    }

    #[test]
    fn bounded_matrix_gain_uses_direct_k_shape_and_bound() {
        let plan =
            NoiseDecompositionPlan::bounded_matrix(BigUint::from(4u8), 3, 2, 8, BigUint::from(5u8))
                .unwrap();
        assert_eq!(plan.target_rows, 3);
        assert_eq!(plan.target_columns, 2);
        assert_eq!(plan.ring_dimension, 8);
        assert_eq!(plan.action_gain().l1, BigUint::from(120u8));
        assert_eq!(plan.action_gain().l2_squared, BigUint::from(600u16));
    }

    #[test]
    fn compact_refresh_routes_keep_prf_base_distinct_from_gadget_and_bind_each_kappa() {
        // The PRF digit base is two while the BGG gadget base is four.  Route
        // scalars are therefore sparse target coefficients decomposed by the
        // gadget schema, not `PowerOfBase` digits in the PRF base.
        let crt = vec![BigUint::from(3u8), BigUint::from(5u8)];
        let route = |value: BigInt| {
            NoiseDecompositionPlan::sparse_monomial(BigUint::from(4u8), 2, 2, 4, 8, 1, 0, 0, value)
                .unwrap()
                .bind_crt_layout(crt.clone(), 1)
                .unwrap()
        };
        let mask = route(BigInt::from(2u8));
        let fresh_slot_0 = route(BigInt::from(5u8)); // distinct kappa_0-scaled route
        let fresh_slot_1 = route(BigInt::from(3u8)); // distinct kappa_1-scaled route
        assert_ne!(mask.action_gain(), fresh_slot_0.action_gain());
        assert_ne!(fresh_slot_0.action_gain(), fresh_slot_1.action_gain());

        let state_0 = NoiseDecompositionPlan::gadget_scalar(
            BigUint::from(4u8),
            2,
            2,
            4,
            8,
            BigInt::from(5u8),
        )
        .unwrap()
        .bind_crt_layout(crt.clone(), 1)
        .unwrap();
        let state_1 = NoiseDecompositionPlan::gadget_scalar(
            BigUint::from(4u8),
            2,
            2,
            4,
            8,
            BigInt::from(3u8),
        )
        .unwrap()
        .bind_crt_layout(crt, 1)
        .unwrap();
        assert_ne!(state_0.action_gain(), state_1.action_gain());
    }

    #[test]
    fn dense_bound_handles_even_modulus_tie_and_rejects_unrepresentable_interval() {
        let even_tie =
            NoiseDecompositionPlan::dense_bound(BigUint::from(8u8), 1, 1, 1, 1, BigUint::from(4u8))
                .unwrap()
                .bind_crt_layout(vec![8u8.into()], 1)
                .unwrap();
        assert_eq!(even_tie.action_gain().l1, BigUint::from(4u8));
        assert_eq!(even_tie.action_gain().l2_squared, BigUint::from(16u8));

        let q7_centered_bound = BigUint::from(7u8) / 2u8;
        assert_eq!(q7_centered_bound, BigUint::from(3u8));
        assert!(!balanced_interval_representable(&q7_centered_bound, &BigUint::from(2u8), 1,));
        assert!(NoiseDecompositionPlan::dense_bound(
            BigUint::from(2u8),
            1,
            1,
            1,
            1,
            BigUint::from(3u8),
        )
        .is_err());
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
    fn lut_branch_includes_direct_and_acted_helper_terms() {
        let direct =
            RightActionNoiseGain { l1: BigUint::from(2u8), l2_squared: BigUint::from(9u8) };
        let helper =
            RightActionNoiseGain { l1: BigUint::from(4u8), l2_squared: BigUint::from(16u8) };
        let branch = transfer_lut(
            &BigUint::from(5u8),
            &AverageVariance::new(BigUint::from(7u8), BigUint::one()).unwrap(),
            &direct,
            &helper,
            &BigUint::from(2u8),
            &AverageVariance::new(BigUint::from(2u8), BigUint::one()).unwrap(),
        );
        assert_eq!(branch.output_bound, BigUint::from(20u8));
        assert_eq!(branch.fresh_bound, BigUint::from(10u8));
        // gamma2_squared(D_C) V + (1 + gamma2_squared(D_A)) V_chi = 9*7 + 17*2.
        assert_eq!(branch.output_variance.numerator, BigUint::from(97u8));
    }

    #[test]
    fn public_family_transfer_uses_public_member_state() {
        let action =
            RightActionNoiseGain { l1: BigUint::from(3u8), l2_squared: BigUint::from(9u8) };
        let transfer = transfer_mul_public(
            &BigUint::from(7u8),
            &AverageVariance::new(BigUint::from(2u8), BigUint::one()).unwrap(),
            &BigUint::from(5u8),
            &AverageVariance::new(BigUint::from(5u8), BigUint::one()).unwrap(),
            &action,
        );
        assert_eq!(transfer.output_bound, BigUint::from(22u8));
        assert_eq!(transfer.output_variance.numerator, BigUint::from(47u8));
    }

    #[test]
    fn centered_digit_variances_cover_nonbinary_bases() {
        let p2 = centered_base_p_digit_variance(&BigUint::from(2u8)).unwrap();
        let p16 = centered_base_p_digit_variance(&BigUint::from(16u8)).unwrap();
        let p32 = centered_base_p_digit_variance(&BigUint::from(32u8)).unwrap();
        assert_eq!(p2, AverageVariance::new(BigUint::one(), BigUint::one()).unwrap());
        assert_eq!(p16, AverageVariance::new(BigUint::from(255u16), BigUint::from(3u8)).unwrap());
        assert_eq!(p32, AverageVariance::new(BigUint::from(1023u16), BigUint::from(3u8)).unwrap());
        assert!(centered_base_p_digit_variance(&BigUint::from(3u8)).is_err());
    }

    #[test]
    fn flat_branches_use_l2_squared_not_l1_squared() {
        let variance = AverageVariance::new(BigUint::from(7u8), BigUint::one()).unwrap();
        let direct = RightActionNoiseGain { l1: 2u8.into(), l2_squared: 9u8.into() };
        let helper = RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() };
        let helper_variance = AverageVariance::zero();
        let branch = transfer_lut(
            &BigUint::from(5u8),
            &variance,
            &direct,
            &helper,
            &BigUint::zero(),
            &helper_variance,
        );
        assert_eq!(
            branch.output_variance,
            AverageVariance::new(BigUint::from(63u8), BigUint::one()).unwrap()
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
    fn refresh_artifacts_require_per_slot_variance() {
        let refresh = RefreshNoiseParameters::new(
            vec![BigUint::from(3u8), BigUint::from(5u8)],
            vec![BigUint::from(15u8); 2],
            BigUint::from(16u8),
            vec![2; 2],
            2,
        )
        .unwrap();
        let variance = centered_base_p_digit_variance(&BigUint::from(16u8)).unwrap();
        assert!(
            refresh
                .clone()
                .with_digit_variances(
                    vec![variance.clone()],
                    vec![variance.clone(), variance.clone()]
                )
                .is_err()
        );
        assert!(
            refresh
                .clone()
                .with_digit_variances(
                    vec![variance.clone(), variance.clone()],
                    vec![variance.clone(), variance.clone()],
                )
                .is_ok()
        );
    }

    #[test]
    fn gain_plan_requires_one_nonzero_gain_per_ordered_step() {
        let decomposition =
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(1)], 1, 1, 1)
                .unwrap();
        let action = decomposition.action_gain();
        let readout = ValueReadoutNoiseGains {
            target: action.clone(),
            identity_trace: action.clone(),
            nonidentity_trace: Vec::new(),
            normalization: action.clone(),
            inherited_bound: BigUint::zero(),
            output_bound: BigUint::zero(),
            output_variance: AverageVariance::zero(),
        };
        let gate = GateNoiseGains {
            step_index: 0,
            operation_index: 0,
            branch_index: 0,
            operation: GateNoiseOperation::Add,
            gate: "Add".to_owned(),
            action,
            helper_action: None,
            inherited_bound: BigUint::zero(),
            fresh_bound: BigUint::zero(),
            output_bound: BigUint::zero(),
            inherited_variance: AverageVariance::zero(),
            fresh_variance: AverageVariance::zero(),
            output_variance: AverageVariance::zero(),
        };
        let builder = NoiseGainPlanBuilder::new(
            vec![GateNoiseDecomposition { action: decomposition.clone(), helper_action: None }],
            decomposition.clone(),
            Vec::new(),
            decomposition,
        )
        .unwrap();
        let plan = builder.build(vec![gate], readout).unwrap();
        assert_eq!(plan.program.gates().len(), 1);
        assert!(plan.readout.nonidentity_trace.is_empty());
        assert_eq!(plan.program.gates()[0].action.l1, BigUint::one());
        assert_eq!(plan.readout.identity_trace.l2_squared, BigUint::one());
    }

    #[test]
    fn family_visitor_records_scaled_public_variance_and_reduction() {
        use crate::program::{AttributeKnowledge, ExponentLutProgramBuilder, ProgramWireType};

        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, 16).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selector_family = builder.rhs_family(2).unwrap();
        let public_family = builder.wire_family(2).unwrap();
        let output = builder
            .select(builder.input_wire(input).unwrap(), selector_family, public_family, 2)
            .unwrap();
        // Leave an unrelated operation after the declared output.  The
        // report must use the declared output wire rather than `steps.last()`
        // when deriving readout variance.
        let trailing = builder
            .add(builder.input_wire(input).unwrap(), builder.input_wire(input).unwrap())
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        let unit = AverageVariance::new(BigUint::one(), BigUint::one()).unwrap();
        let action = |operation, operation_index, branch_index, l1, l2_squared| GateNoiseGains {
            step_index: if operation == GateNoiseOperation::Add { 1 } else { 0 },
            operation_index,
            branch_index,
            operation,
            gate: "family".to_owned(),
            action: RightActionNoiseGain { l1, l2_squared },
            helper_action: None,
            inherited_bound: BigUint::zero(),
            fresh_bound: BigUint::zero(),
            output_bound: BigUint::zero(),
            inherited_variance: AverageVariance::zero(),
            fresh_variance: AverageVariance::zero(),
            output_variance: AverageVariance::zero(),
        };
        let actions = vec![
            action(GateNoiseOperation::MulPrivate, 0, 0, 2u8.into(), 4u8.into()),
            action(GateNoiseOperation::MulPublic, 1, 1, 3u8.into(), 9u8.into()),
            action(GateNoiseOperation::Add, 2, 0, 1u8.into(), 1u8.into()),
            action(GateNoiseOperation::Add, 3, 0, 1u8.into(), 1u8.into()),
        ];
        let inputs = ProgramNoiseInputs {
            input_bounds: BTreeMap::from([(input, BigUint::from(3u8))]),
            input_variances: BTreeMap::from([(input, unit.clone())]),
            public_family_uniform_bounds: BTreeMap::new(),
            public_family_uniform_variances: BTreeMap::new(),
            public_family_bounds: BTreeMap::from([(public_family, vec![BigUint::from(5u8); 2])]),
            public_family_variances: BTreeMap::from([(
                public_family,
                vec![AverageVariance::new(BigUint::from(3u8), BigUint::one()).unwrap(); 2],
            )]),
        };
        let report =
            simulate_program(&program, &inputs, &BigUint::one(), &unit, &actions, &[2]).unwrap();
        assert_eq!(report.steps[0].family_body.len(), 2);
        assert_eq!(report.steps[0].family_member_count, 2);
        assert_eq!(report.steps[0].family_body[0].output_bound, BigUint::from(7u8));
        assert_eq!(report.steps[0].family_body[1].fresh_variance.numerator, BigUint::from(27u8));
        // A family member is not a separate noise semantic: the recorded
        // rows must equal the same primitive transfers used by ordinary
        // MulPrivate/MulPublic gates.
        let expected_private = transfer_mul_private(
            &BigUint::from(3u8),
            &unit,
            &actions[0].action,
            &BigUint::one(),
            &unit,
        );
        let expected_public = transfer_mul_public(
            &expected_private.output_bound,
            &expected_private.output_variance,
            &BigUint::from(5u8),
            &AverageVariance::new(BigUint::from(3u8), BigUint::one()).unwrap(),
            &actions[1].action,
        );
        assert_eq!(report.steps[0].family_body[0].output_bound, expected_private.output_bound);
        assert_eq!(
            report.steps[0].family_body[0].output_variance,
            expected_private.output_variance
        );
        assert_eq!(report.steps[0].family_body[1].output_bound, expected_public.output_bound);
        assert_eq!(report.steps[0].family_body[1].output_variance, expected_public.output_variance);
        assert_eq!(report.steps[0].output_bound, BigUint::from(44u8));
        assert_eq!(report.steps[0].output_variance.numerator, BigUint::from(128u8));
        let expected_reduction = transfer_add(AddInputSummary {
            inherited_bound: BigUint::from(44u8),
            fresh_bound: BigUint::zero(),
            inherited_variance: AverageVariance::new(BigUint::from(64u8), BigUint::one()).unwrap(),
            fresh_variance: AverageVariance::zero(),
            operand_count: 2,
        });
        assert_eq!(report.steps[0].output_bound, expected_reduction.output_bound);
        assert_eq!(report.steps[0].output_variance, expected_reduction.output_variance);
        assert_ne!(report.steps.last().unwrap().output_variance, report.wire_variances[&output]);
        assert_eq!(report.output_bound, report.wire_bounds[&output]);
        let _ = trailing;

        // A mixed representation (uniform bounds, member-specific
        // variances) must not take the closed-form shortcut: the two member
        // variances are distinct and therefore cannot be multiplied from a
        // single representative.
        let mixed_inputs = ProgramNoiseInputs {
            input_bounds: BTreeMap::from([(input, BigUint::from(3u8))]),
            input_variances: BTreeMap::from([(input, unit.clone())]),
            public_family_uniform_bounds: BTreeMap::from([(public_family, BigUint::from(5u8))]),
            public_family_uniform_variances: BTreeMap::new(),
            public_family_bounds: BTreeMap::new(),
            public_family_variances: BTreeMap::from([(
                public_family,
                vec![
                    AverageVariance::new(BigUint::from(3u8), BigUint::one()).unwrap(),
                    AverageVariance::new(BigUint::from(7u8), BigUint::one()).unwrap(),
                ],
            )]),
        };
        let mixed_report =
            simulate_program(&program, &mixed_inputs, &BigUint::one(), &unit, &actions, &[2])
                .unwrap();
        assert_eq!(mixed_report.steps[0].output_variance.numerator, BigUint::from(200u8));
        assert_eq!(report.steps[0].family_body.len(), 2);
        assert_eq!(mixed_report.steps[0].family_body.len(), 2);
        assert_ne!(report.steps[0].output_variance, mixed_report.steps[0].output_variance);
    }

    #[test]
    fn three_member_primitive_aggregation_is_explicit_and_exact() {
        use crate::program::{AttributeKnowledge, ExponentLutProgramBuilder, ProgramWireType};

        // Exercise the actual structured Family body and Add aggregation lowering,
        // rather than only calling the primitive helpers directly.
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 2 }, 16).unwrap();
        let input = builder
            .input(3, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selector_family = builder.rhs_family(3).unwrap();
        let public_family = builder.wire_family(3).unwrap();
        let output = builder
            .select(builder.input_wire(input).unwrap(), selector_family, public_family, 3)
            .unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();

        let unit = AverageVariance::new(BigUint::one(), BigUint::one()).unwrap();
        let private_action = RightActionNoiseGain { l1: 2u8.into(), l2_squared: 4u8.into() };
        let public_action = RightActionNoiseGain { l1: 3u8.into(), l2_squared: 9u8.into() };
        let public_bound = BigUint::from(5u8);
        let public_variance = AverageVariance::new(BigUint::from(3u8), BigUint::one()).unwrap();
        let actions = vec![
            GateNoiseGains {
                step_index: 0,
                operation_index: 0,
                branch_index: 0,
                operation: GateNoiseOperation::MulPrivate,
                gate: "family".to_owned(),
                action: private_action.clone(),
                helper_action: None,
                inherited_bound: BigUint::zero(),
                fresh_bound: BigUint::zero(),
                output_bound: BigUint::zero(),
                inherited_variance: AverageVariance::zero(),
                fresh_variance: AverageVariance::zero(),
                output_variance: AverageVariance::zero(),
            },
            GateNoiseGains {
                step_index: 0,
                operation_index: 1,
                branch_index: 1,
                operation: GateNoiseOperation::MulPublic,
                gate: "family".to_owned(),
                action: public_action.clone(),
                helper_action: None,
                inherited_bound: BigUint::zero(),
                fresh_bound: BigUint::zero(),
                output_bound: BigUint::zero(),
                inherited_variance: AverageVariance::zero(),
                fresh_variance: AverageVariance::zero(),
                output_variance: AverageVariance::zero(),
            },
            GateNoiseGains {
                step_index: 1,
                operation_index: 2,
                branch_index: 0,
                operation: GateNoiseOperation::Add,
                gate: "family".to_owned(),
                action: RightActionNoiseGain { l1: BigUint::one(), l2_squared: BigUint::one() },
                helper_action: None,
                inherited_bound: BigUint::zero(),
                fresh_bound: BigUint::zero(),
                output_bound: BigUint::zero(),
                inherited_variance: AverageVariance::zero(),
                fresh_variance: AverageVariance::zero(),
                output_variance: AverageVariance::zero(),
            },
        ];
        let base_inputs = ProgramNoiseInputs {
            input_bounds: BTreeMap::from([(input, BigUint::from(3u8))]),
            input_variances: BTreeMap::from([(input, unit.clone())]),
            public_family_uniform_bounds: BTreeMap::from([(public_family, public_bound.clone())]),
            public_family_uniform_variances: BTreeMap::from([(
                public_family,
                public_variance.clone(),
            )]),
            ..ProgramNoiseInputs::default()
        };
        let explicit_inputs = ProgramNoiseInputs {
            input_bounds: BTreeMap::from([(input, BigUint::from(3u8))]),
            input_variances: BTreeMap::from([(input, unit.clone())]),
            public_family_bounds: BTreeMap::from([(public_family, vec![public_bound.clone(); 3])]),
            public_family_variances: BTreeMap::from([(
                public_family,
                vec![public_variance.clone(); 3],
            )]),
            ..ProgramNoiseInputs::default()
        };
        let uniform =
            simulate_program(&program, &base_inputs, &BigUint::one(), &unit, &actions, &[3])
                .unwrap();
        let explicit =
            simulate_program(&program, &explicit_inputs, &BigUint::one(), &unit, &actions, &[3])
                .unwrap();
        assert_eq!(uniform.wire_bounds, explicit.wire_bounds);
        assert_eq!(uniform.wire_variances, explicit.wire_variances);
        assert_eq!(uniform.output_bound, explicit.output_bound);
        assert_eq!(uniform.steps[0].output_bound, explicit.steps[0].output_bound);
        assert_eq!(uniform.steps[0].output_variance, explicit.steps[0].output_variance);
        // Uniform families remain one compact representative; explicit
        // heterogeneous/member inputs retain one diagnostic constituent per
        // member without changing the simulated result.
        assert_eq!(uniform.steps[0].constituent_bounds.len(), 1);
        assert_eq!(explicit.steps[0].constituent_bounds.len(), 3);

        let public_bounds = [5u8, 6u8, 7u8];
        let public_variances = [3u8, 4u8, 5u8];
        let mut aggregate_bound = BigUint::zero();
        let mut aggregate_variance = AverageVariance::zero();
        for (bound, variance) in public_bounds.iter().zip(public_variances) {
            let private = transfer_mul_private(
                &BigUint::from(3u8),
                &unit,
                &private_action,
                &BigUint::one(),
                &unit,
            );
            let public = transfer_mul_public(
                &private.output_bound,
                &private.output_variance,
                &BigUint::from(*bound),
                &AverageVariance::new(BigUint::from(variance), BigUint::one()).unwrap(),
                &public_action,
            );
            aggregate_bound += public.output_bound;
            aggregate_variance = aggregate_variance.add(&public.output_variance);
        }
        let reduced = transfer_add(AddInputSummary {
            inherited_bound: aggregate_bound,
            fresh_bound: BigUint::zero(),
            inherited_variance: aggregate_variance.clone(),
            fresh_variance: AverageVariance::zero(),
            operand_count: public_bounds.len(),
        });
        assert_eq!(reduced.output_bound, BigUint::from(75u8));
        assert_eq!(reduced.output_variance.numerator, BigUint::from(369u32));

        let heterogeneous_inputs = ProgramNoiseInputs {
            input_bounds: BTreeMap::from([(input, BigUint::from(3u8))]),
            input_variances: BTreeMap::from([(input, unit.clone())]),
            public_family_bounds: BTreeMap::from([(
                public_family,
                public_bounds.iter().copied().map(BigUint::from).collect(),
            )]),
            public_family_variances: BTreeMap::from([(
                public_family,
                public_variances
                    .iter()
                    .copied()
                    .map(|value| {
                        AverageVariance::new(BigUint::from(value), BigUint::one()).unwrap()
                    })
                    .collect(),
            )]),
            ..ProgramNoiseInputs::default()
        };
        let heterogeneous = simulate_program(
            &program,
            &heterogeneous_inputs,
            &BigUint::one(),
            &unit,
            &actions,
            &[3],
        )
        .unwrap();
        assert_eq!(heterogeneous.steps[0].output_bound, BigUint::from(75u8));
        assert_eq!(heterogeneous.steps[0].output_variance, reduced.output_variance);
        assert_eq!(heterogeneous.wire_bounds[&output], reduced.output_bound);
        assert_eq!(heterogeneous.wire_variances[&output], reduced.output_variance);
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
        // Smudging uses only mask coordinates: 2 slots * 4 columns * 3 coefficients.
        assert_eq!(topology.mask_smudging_coordinates().unwrap(), 24);
        assert_eq!(
            config.z_squared(&topology).unwrap(),
            AverageVariance::new(
                BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(17u8),
                BigUint::from(1_000_000u32),
            )
            .unwrap()
        );
    }

    #[test]
    fn average_favg_uses_exact_ceil_sqrt_after_division_by_four() {
        let z_squared = AverageVariance::new(BigUint::one(), BigUint::one()).unwrap();
        let variance = AverageVariance::new(BigUint::from(16u8), BigUint::one()).unwrap();
        assert_eq!(average_favg(&z_squared, &variance, 0).unwrap(), BigUint::from(2u8));
        assert_eq!(average_favg(&z_squared, &variance, 1).unwrap(), BigUint::from(4u8));
    }
}
