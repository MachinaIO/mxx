//! Exact application-specific noise bounds for Power-LUT programs.
//!
//! This module deliberately models the public Power-LUT lowering rather than
//! re-interpreting the executable IR.  Every operation is represented by an
//! affine transfer `E -> gain * E + additive`; all arithmetic is exact
//! `BigUint` arithmetic and no centered-residue cap is applied.

use std::collections::BTreeMap;

use mxx_noise_simulator::{ProductGeometry, right_action_gain};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{CheckedMul, Euclid, One, Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const AVERAGE_CASE_REPORT_SCHEMA_VERSION: u32 = 2;

/// Noise channel selected by an application-specific simulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseModelKind {
    WorstCase,
    AverageCase,
}

/// Exact squared-domain average magnitude. The denominator is strictly
/// positive and callers must reduce the pair before serializing it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageVariance {
    pub numerator: BigUint,
    pub denominator: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseMagnitude {
    Worst { hard_bound: BigUint },
    Average { variance: AverageVariance },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HeuristicId {
    H1StateUncorrelated,
    H2DigitUniformFallback,
    H3BranchSetupIndependence,
    H4SlotRhsIndependence,
    H5PrfLabelIndependence,
    H6GaussianTailClosure,
    ExactUnderGaussian,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageCaseConfig {
    pub failure_exponent: u32,
    pub input_domain_log2: u32,
    pub extra_event_log2: u32,
    pub tail_correction_bits: u32,
    pub allow_average_acceptance: bool,
}

impl Default for AverageCaseConfig {
    fn default() -> Self {
        Self {
            failure_exponent: 100,
            input_domain_log2: 0,
            extra_event_log2: 0,
            tail_correction_bits: 0,
            allow_average_acceptance: false,
        }
    }
}

/// The acceptance authority is deliberately explicit in every average-case
/// report. Lattice security is harness-owned; setup hard gates and AverageCase
/// smudging/rounding authorities are reported separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AcceptedUnder {
    WorstCase,
    AverageCase,
    Diagnostic,
}

/// Exact event factors used by the Gaussian tail ledger.  The values are
/// logarithmic upper bounds, so `log2_events` is an integer and no floating
/// point logarithm enters the acceptance decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageEventBudget {
    pub input_domain_log2: u32,
    pub coefficient_log2: u32,
    pub slot_log2: u32,
    pub inspection_event_log2: u32,
}

impl AverageEventBudget {
    pub fn log2_events(&self) -> Result<u32, NoiseSimulationError> {
        self.input_domain_log2
            .checked_add(self.coefficient_log2)
            .and_then(|value| value.checked_add(self.slot_log2))
            .and_then(|value| value.checked_add(self.inspection_event_log2))
            .ok_or(NoiseSimulationError::IntegerConversion)
    }
}

impl AverageCaseConfig {
    pub fn event_budget(&self, coefficient_count: usize, slot_count: usize) -> AverageEventBudget {
        AverageEventBudget {
            input_domain_log2: self.input_domain_log2,
            coefficient_log2: coefficient_count.next_power_of_two().trailing_zeros(),
            slot_log2: slot_count.next_power_of_two().trailing_zeros(),
            inspection_event_log2: self.extra_event_log2,
        }
    }

    /// Exact `z^2 = 2*(693148/1000000)*(lambda + log2(N_events) + 1)`.
    pub fn z_squared(
        &self,
        events: AverageEventBudget,
    ) -> Result<AverageVariance, NoiseSimulationError> {
        let events = BigUint::from(
            self.failure_exponent
                .checked_add(events.log2_events()?)
                .and_then(|value| value.checked_add(1))
                .ok_or(NoiseSimulationError::IntegerConversion)?,
        );
        AverageVariance::new(
            BigUint::from(2u8) * BigUint::from(693_148u32) * events,
            BigUint::from(1_000_000u32),
        )
    }

    pub fn z_squared_for_log2(
        &self,
        event_log2: u32,
    ) -> Result<AverageVariance, NoiseSimulationError> {
        let events = self
            .failure_exponent
            .checked_add(event_log2)
            .and_then(|value| value.checked_add(1))
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        AverageVariance::new(
            BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(events),
            BigUint::from(1_000_000u32),
        )
    }

    /// Returns the one event count used by both AverageCase rounding and
    /// smudging. Topology counts are included as evidence, but never receive
    /// independent tail multipliers.
    pub fn joint_event_count(
        &self,
        mask_event_count: &BigUint,
        fresh_event_count: &BigUint,
    ) -> Result<BigUint, NoiseSimulationError> {
        let topology = mask_event_count + fresh_event_count;
        let input_factor = BigUint::one() << self.input_domain_log2;
        let extra_factor = BigUint::one() << self.extra_event_log2;
        topology
            .checked_mul(&input_factor)
            .and_then(|value| value.checked_mul(&extra_factor))
            .ok_or(NoiseSimulationError::IntegerConversion)
    }
}

fn ceil_log2(value: &BigUint) -> Result<u32, NoiseSimulationError> {
    if value.is_zero() {
        return Err(NoiseSimulationError::InvalidAverageEvents);
    }
    let bits = value.bits();
    if value == &(BigUint::one() << (bits - 1)) {
        u32::try_from(bits - 1).map_err(|_| NoiseSimulationError::IntegerConversion)
    } else {
        u32::try_from(bits).map_err(|_| NoiseSimulationError::IntegerConversion)
    }
}

fn ceil_div(numerator: &BigUint, denominator: &BigUint) -> Result<BigUint, NoiseSimulationError> {
    if denominator.is_zero() {
        return Err(NoiseSimulationError::IntegerConversion);
    }
    if numerator.is_zero() {
        return Ok(BigUint::zero());
    }
    Ok((numerator + denominator - BigUint::one()) / denominator)
}

fn ceil_sqrt(value: &BigUint) -> Result<BigUint, NoiseSimulationError> {
    if value.is_zero() {
        return Ok(BigUint::zero());
    }
    let mut low = BigUint::zero();
    let mut high = BigUint::one() << value.bits().div_ceil(2);
    while low < high {
        let mid = (&low + &high) >> 1usize;
        if &mid * &mid < *value {
            low = mid + BigUint::one();
        } else {
            high = mid;
        }
    }
    Ok(low)
}

fn average_favg(
    z_sq: &AverageVariance,
    stochastic_variance: &AverageVariance,
    tail_correction_bits: u32,
) -> Result<BigUint, NoiseSimulationError> {
    let tail_bits =
        tail_correction_bits.checked_mul(2).ok_or(NoiseSimulationError::IntegerConversion)?;
    let tail_factor = BigUint::one() << tail_bits;
    let numerator = &z_sq.numerator * tail_factor * &stochastic_variance.numerator;
    let denominator = BigUint::from(4u8) * &z_sq.denominator * &stochastic_variance.denominator;
    ceil_sqrt(&ceil_div(&numerator, &denominator)?)
}

fn average_smudging_required(
    lambda: u64,
    dimension: &BigUint,
    max_favg: &BigUint,
) -> Result<BigUint, NoiseSimulationError> {
    let exponent = lambda.checked_add(1).ok_or(NoiseSimulationError::IntegerConversion)?;
    BigUint::one()
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(max_favg))
        .map(|value| (BigUint::one() << exponent) * value)
        .ok_or(NoiseSimulationError::IntegerConversion)
}

impl AverageVariance {
    pub fn new(numerator: BigUint, denominator: BigUint) -> Result<Self, NoiseSimulationError> {
        if denominator.is_zero() {
            return Err(NoiseSimulationError::IntegerConversion);
        }
        let common = gcd(&numerator, &denominator);
        Ok(Self { numerator: numerator / &common, denominator: denominator / common })
    }

    pub fn zero() -> Self {
        Self { numerator: BigUint::zero(), denominator: BigUint::one() }
    }

    pub fn scaled(&self, gain_sq: &BigUint) -> Self {
        Self::new(&self.numerator * gain_sq, self.denominator.clone())
            .expect("positive denominator")
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            &self.denominator * &other.denominator,
        )
        .expect("positive denominator")
    }
}

/// Exact doubled-coordinate variance after an independent additive Gaussian
/// helper contribution. This is a small arithmetic primitive used by the
/// opt-in AverageCase path; WorstCase never calls it.
pub fn average_variance_transfer(
    input: &AverageVariance,
    gain_sq: &BigUint,
    additive: &AverageVariance,
) -> Result<AverageVariance, NoiseSimulationError> {
    let numerator = &input.numerator * gain_sq * &additive.denominator +
        &additive.numerator * &input.denominator;
    AverageVariance::new(numerator, &input.denominator * &additive.denominator)
}

/// One average-case affine transfer in doubled-coordinate squared units.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageNoiseTransfer {
    pub gain_sq: BigUint,
    pub additive_variance: AverageVariance,
    pub heuristics: Vec<HeuristicId>,
}

impl AverageNoiseTransfer {
    pub fn apply(&self, input: &AverageVariance) -> AverageVariance {
        average_variance_transfer(input, &self.gain_sq, &self.additive_variance)
            .expect("average transfer denominators are positive")
    }

    pub fn compose(first: &Self, second: &Self) -> Self {
        Self {
            gain_sq: &first.gain_sq * &second.gain_sq,
            additive_variance: second
                .additive_variance
                .add(&first.additive_variance.scaled(&second.gain_sq)),
            heuristics: first.heuristics.iter().chain(second.heuristics.iter()).copied().collect(),
        }
    }
}

/// Strict squared-domain AverageCase refresh threshold. `spacing` and `d2`
/// use the same doubled coordinate unit; equality is rejected.
pub fn average_refresh_accepts(
    spacing: &BigUint,
    d2: &BigUint,
    z_sq_num: &BigUint,
    z_sq_den: &BigUint,
    tail_correction_bits: u32,
    variance: &AverageVariance,
) -> Result<bool, NoiseSimulationError> {
    if z_sq_den.is_zero() || spacing * 2u8 <= d2 * 2u8 {
        return Ok(false);
    }
    let margin = spacing * 2u8 - d2 * 2u8;
    let tail_bits =
        tail_correction_bits.checked_mul(2).ok_or(NoiseSimulationError::IntegerConversion)?;
    let tail = BigUint::one() << tail_bits;
    let lhs = &margin * &margin * z_sq_den * &variance.denominator;
    let rhs = BigUint::from(4u8) * z_sq_num * tail * &variance.numerator;
    Ok(lhs > rhs)
}

use crate::{
    pbc::{PbcActiveCellIndex, PbcPublicLayout},
    prf::SparseLwrPrfProgram,
    program::{LutId, PowerLutProgram, ProgramGate, ProgramInputId, ProgramWireId, RhsInputId},
};

/// Errors returned by the application-specific bound evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NoiseSimulationError {
    #[error("ring dimension must be positive")]
    ZeroRingDimension,
    #[error("secret dimension must be exactly two")]
    InvalidSecretDimension,
    #[error("gadget base must be a power of two greater than one")]
    InvalidGadgetBase,
    #[error("gadget digit count must be positive")]
    InvalidGadgetDigitCount,
    #[error("program input bound is missing: {0:?}")]
    MissingInputBound(ProgramInputId),
    #[error("program wire bound is missing: {0:?}")]
    MissingWireBound(ProgramWireId),
    #[error("LUT is missing: {0:?}")]
    MissingLut(LutId),
    #[error("LUT has no entries")]
    InvalidLutWidth,
    #[error("one-hot active count is missing for gate {0}")]
    MissingActiveCount(usize),
    #[error("one-hot active count {active} exceeds family width {width}")]
    ActiveCountExceedsWidth { active: usize, width: usize },
    #[error("one-hot active count is not representable for gate {0}")]
    InvalidActiveCount(usize),
    #[error("program RHS input is missing: {0:?}")]
    MissingRhsInput(RhsInputId),
    #[error("program has no output wire")]
    MissingOutput,
    #[error("generic program simulation does not support OneHot; use the sparse-PRF path")]
    UnsupportedOneHot,
    #[error("invalid refresh parameters: {0}")]
    InvalidRefresh(&'static str),
    #[error("CRT modulus does not divide the full modulus")]
    InvalidCrtDivision,
    #[error("CRT moduli are not a pairwise-coprime complete factorization")]
    InvalidCrtFactorization,
    #[error("integer conversion overflow")]
    IntegerConversion,
    #[error("PBC layout is invalid: {0}")]
    InvalidPbcLayout(String),
    #[error("noise-simulator bound helper failed: {0}")]
    BoundHelper(String),
    #[error("AverageCase acceptance is disabled")]
    AverageAcceptanceDisabled,
    #[error("AverageCase addition is not classified as independent or coherent")]
    UnsupportedAverageAddition,
    #[error("AverageCase event budget is invalid")]
    InvalidAverageEvents,
    #[error("AverageCase setup identity or model snapshot mismatch")]
    AverageIdentityMismatch,
    #[error("AverageCase requires a p=2 refresh with complete decoder inputs")]
    UnsupportedAverageRefresh,
}

/// One exact affine noise transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffineNoiseTransfer {
    pub gain: BigUint,
    pub additive: BigUint,
}

impl AffineNoiseTransfer {
    pub fn apply(&self, input: &BigUint) -> BigUint {
        &self.gain * input + &self.additive
    }
}

/// Common structural parameters for the Power-LUT lowering.
///
/// The four gains intentionally remain separate named quantities.  The dense
/// constructor initializes all of them to the same regular-decomposition
/// value; keeping them separate preserves the operation-specific report
/// schema without accepting detached caller-supplied gains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PowerLutNoiseParameters {
    ring_dimension: usize,
    secret_dimension: usize,
    gadget_base: BigUint,
    gadget_digit_count: usize,
    helper_error_bound: BigUint,
    gamma_fixed_rhs: BigUint,
    gamma_c: BigUint,
    gamma_a: BigUint,
    gamma_selector: BigUint,
    gamma_fixed_rhs_l2_sq: BigUint,
    gamma_c_l2_sq: BigUint,
    gamma_a_l2_sq: BigUint,
    gamma_selector_l2_sq: BigUint,
}

impl PowerLutNoiseParameters {
    /// Creates the regular power-of-two dense fallback.
    ///
    /// For `Delta = beta / 2`, the fallback is
    /// `F_beta = 2 * ell_beta * n * Delta`.
    pub fn dense(
        ring_dimension: usize,
        gadget_base: BigUint,
        gadget_digit_count: usize,
        helper_error_bound: BigUint,
    ) -> Result<Self, NoiseSimulationError> {
        if ring_dimension == 0 {
            return Err(NoiseSimulationError::ZeroRingDimension);
        }
        if gadget_base <= BigUint::one() ||
            (&gadget_base & (&gadget_base - BigUint::one())) != BigUint::zero()
        {
            return Err(NoiseSimulationError::InvalidGadgetBase);
        }
        if gadget_digit_count == 0 {
            return Err(NoiseSimulationError::InvalidGadgetDigitCount);
        }
        let delta = &gadget_base >> 1;
        let magnitude: BigUint = delta * BigUint::from(gadget_digit_count);
        let state =
            mxx_noise_simulator::MatrixState::new(BigUint::zero(), magnitude.clone(), false)
                .map_err(|error| NoiseSimulationError::BoundHelper(error.to_string()))?;
        let dense =
            right_action_gain(&state, ProductGeometry { inner_dimension: 2, ring_dimension })
                .map_err(|error| NoiseSimulationError::BoundHelper(error.to_string()))?;
        let dense_l2_sq =
            BigUint::from(2u8) * BigUint::from(ring_dimension) * (&magnitude * &magnitude);
        Ok(Self {
            ring_dimension,
            secret_dimension: 2,
            gadget_base,
            gadget_digit_count,
            helper_error_bound,
            gamma_fixed_rhs: dense.clone(),
            gamma_c: dense.clone(),
            gamma_a: dense.clone(),
            gamma_selector: dense,
            gamma_fixed_rhs_l2_sq: dense_l2_sq.clone(),
            gamma_c_l2_sq: dense_l2_sq.clone(),
            gamma_a_l2_sq: dense_l2_sq.clone(),
            gamma_selector_l2_sq: dense_l2_sq,
        })
    }

    pub fn delta(&self) -> BigUint {
        &self.gadget_base >> 1
    }

    pub fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    pub fn gadget_digit_count(&self) -> usize {
        self.gadget_digit_count
    }

    pub fn helper_error_bound(&self) -> &BigUint {
        &self.helper_error_bound
    }

    pub fn gamma_fixed_rhs(&self) -> &BigUint {
        &self.gamma_fixed_rhs
    }

    pub fn gamma_c(&self) -> &BigUint {
        &self.gamma_c
    }

    pub fn gamma_a(&self) -> &BigUint {
        &self.gamma_a
    }

    pub fn gamma_selector(&self) -> &BigUint {
        &self.gamma_selector
    }

    pub fn gamma_fixed_rhs_l2_sq(&self) -> &BigUint {
        &self.gamma_fixed_rhs_l2_sq
    }

    pub fn gamma_c_l2_sq(&self) -> &BigUint {
        &self.gamma_c_l2_sq
    }

    pub fn gamma_a_l2_sq(&self) -> &BigUint {
        &self.gamma_a_l2_sq
    }

    pub fn gamma_selector_l2_sq(&self) -> &BigUint {
        &self.gamma_selector_l2_sq
    }

    pub fn regular_dense_gain(&self) -> BigUint {
        BigUint::from(2u8) *
            BigUint::from(self.ring_dimension) *
            BigUint::from(self.gadget_digit_count) *
            self.delta()
    }

    /// Conservative exact doubled variance for the balanced digit fallback.
    /// Each digit is uniform on the centered support and independent across
    /// gadget positions, hence `V2 = ell * (beta^2 - 1) / 3`.
    pub fn helper_doubled_variance(&self) -> AverageVariance {
        AverageVariance::new(
            BigUint::from(self.gadget_digit_count) *
                (&self.gadget_base * &self.gadget_base - BigUint::one()),
            BigUint::from(3u8),
        )
        .expect("constant denominator")
    }
}

/// Inputs needed to evaluate one public program.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProgramNoiseInputs {
    pub input_bounds: BTreeMap<ProgramInputId, BigUint>,
    /// Exact doubled-coordinate variances for the opt-in AverageCase path.
    pub input_variances: BTreeMap<ProgramInputId, AverageVariance>,
    /// Actual active selector count for the internal PRF OneHot path, keyed by
    /// gate index. Generic [`simulate_program`] rejects OneHot gates.
    pub one_hot_active_counts: BTreeMap<usize, usize>,
}

/// One gate's exact affine step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GateNoiseStep {
    pub gate_index: usize,
    pub input_bound: BigUint,
    pub transfer: AffineNoiseTransfer,
    pub output_bound: BigUint,
    pub lut_width: usize,
    pub active_count: Option<usize>,
}

/// Full deterministic report for a Power-LUT program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramNoiseReport {
    pub wire_bounds: BTreeMap<ProgramWireId, BigUint>,
    pub steps: Vec<GateNoiseStep>,
    pub output_bound: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AverageAdditionClass {
    IndependentSum,
    CoherentScale(BigUint),
    SharedStateBranchSum(usize),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageGateNoiseStep {
    pub gate_index: usize,
    pub input_variance: AverageVariance,
    pub transfer: AverageNoiseTransfer,
    pub output_variance: AverageVariance,
    pub addition: AverageAdditionClass,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageProgramNoiseReport {
    pub wire_variances: BTreeMap<ProgramWireId, AverageVariance>,
    pub steps: Vec<AverageGateNoiseStep>,
    pub output_variance: AverageVariance,
    pub output_addition: AverageAdditionClass,
    pub heuristics: Vec<HeuristicId>,
    pub accepted_under: AcceptedUnder,
}

fn append_heuristics(into: &mut Vec<HeuristicId>, values: &[HeuristicId]) {
    for value in values {
        if !into.contains(value) {
            into.push(*value);
        }
    }
}

fn variance_ge(left: &AverageVariance, right: &AverageVariance) -> bool {
    &left.numerator * &right.denominator >= &right.numerator * &left.denominator
}

/// Simulates the generic public program in the exact doubled-coordinate
/// squared domain.  OneHot remains intentionally unavailable here because
/// its monomial support is a PRF-specific structural property, not a generic
/// program property.
pub fn simulate_average_program(
    program: &PowerLutProgram,
    parameters: &PowerLutNoiseParameters,
    inputs: &ProgramNoiseInputs,
) -> Result<AverageProgramNoiseReport, NoiseSimulationError> {
    if parameters.secret_dimension != 2 {
        return Err(NoiseSimulationError::InvalidSecretDimension);
    }
    let mut wire_variances = BTreeMap::new();
    for (input, wire) in program.input_wires() {
        let variance = inputs
            .input_variances
            .get(input)
            .ok_or(NoiseSimulationError::MissingInputBound(*input))?;
        wire_variances.insert(*wire, variance.clone());
    }
    let mut steps = Vec::with_capacity(program.gates().len());
    let mut heuristics = vec![HeuristicId::H1StateUncorrelated];
    for (gate_index, gate) in program.gates().iter().enumerate() {
        let (input_wire, output_wire, transfer, addition) = match gate {
            ProgramGate::Unary { input, lut, output } => {
                let table = program.lut(*lut).ok_or(NoiseSimulationError::MissingLut(*lut))?;
                (
                    *input,
                    *output,
                    average_fixed_lut_transfer(parameters, table.values().len())?,
                    AverageAdditionClass::IndependentSum,
                )
            }
            ProgramGate::Binary { lhs, rhs, lut, output } => {
                if !program.rhs_inputs().contains_key(rhs) {
                    return Err(NoiseSimulationError::MissingRhsInput(*rhs));
                }
                let table = program.lut(*lut).ok_or(NoiseSimulationError::MissingLut(*lut))?;
                (
                    *lhs,
                    *output,
                    average_two_input_lut_transfer(
                        parameters,
                        table.values().len(),
                        BigUint::one(),
                    )?,
                    AverageAdditionClass::CoherentScale(BigUint::one()),
                )
            }
            ProgramGate::OneHot { .. } => return Err(NoiseSimulationError::UnsupportedOneHot),
        };
        let input_variance = wire_variances
            .get(&input_wire)
            .ok_or(NoiseSimulationError::MissingWireBound(input_wire))?
            .clone();
        let output_variance = transfer.apply(&input_variance);
        append_heuristics(&mut heuristics, &transfer.heuristics);
        wire_variances.insert(output_wire, output_variance.clone());
        steps.push(AverageGateNoiseStep {
            gate_index,
            input_variance,
            transfer,
            output_variance,
            addition,
        });
    }
    let output_variances = program
        .outputs()
        .iter()
        .map(|wire| wire_variances.get(wire).ok_or(NoiseSimulationError::MissingWireBound(*wire)))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(Clone::clone)
        .collect::<Vec<_>>();
    let (output_variance, output_addition) = match output_variances.as_slice() {
        [] => return Err(NoiseSimulationError::MissingOutput),
        [single] => (single.clone(), AverageAdditionClass::IndependentSum),
        many => {
            // Outputs from one public program share the input state.  Treat
            // their fan-out coherently unless a circuit-specific covariance
            // model is supplied; this is intentionally conservative.
            let maximum = many.iter().skip(1).fold(many[0].clone(), |current, candidate| {
                if variance_ge(candidate, &current) { candidate.clone() } else { current }
            });
            let width = BigUint::from(many.len());
            append_heuristics(&mut heuristics, &[HeuristicId::H3BranchSetupIndependence]);
            (
                maximum.scaled(&(&width * &width)),
                AverageAdditionClass::SharedStateBranchSum(many.len()),
            )
        }
    };
    Ok(AverageProgramNoiseReport {
        wire_variances,
        steps,
        output_variance,
        output_addition,
        heuristics,
        accepted_under: AcceptedUnder::Diagnostic,
    })
}

/// Exact transfer for setup-fixed Fuse with a semantic multiplier action.
///
/// The semantic action gain is supplied explicitly. For the Power-LUT
/// `Binary` path, RHS values are validated monomials and the caller passes
/// `1`; no public-value norm is inferred here.
pub fn fixed_fuse_transfer(
    parameters: &PowerLutNoiseParameters,
    semantic_multiplier_action_gain: BigUint,
) -> AffineNoiseTransfer {
    AffineNoiseTransfer {
        gain: parameters.gamma_fixed_rhs().clone(),
        additive: semantic_multiplier_action_gain * parameters.helper_error_bound(),
    }
}

/// Average-case transfer for setup-fixed Fuse. The squared action gain is
/// derived from the same dense public structural model as the worst channel.
pub fn average_fixed_fuse_transfer(
    parameters: &PowerLutNoiseParameters,
    semantic_multiplier_action_gain: BigUint,
) -> Result<AverageNoiseTransfer, NoiseSimulationError> {
    let helper = parameters.helper_doubled_variance();
    let semantic_sq = &semantic_multiplier_action_gain * &semantic_multiplier_action_gain;
    let additive = helper.scaled(&(BigUint::from(4u8) * semantic_sq));
    Ok(AverageNoiseTransfer {
        gain_sq: parameters.gamma_fixed_rhs_l2_sq().clone(),
        additive_variance: additive,
        heuristics: vec![HeuristicId::H1StateUncorrelated, HeuristicId::H2DigitUniformFallback],
    })
}

/// Exact transfer for a fixed-LUT branch of the public Power-LUT lowering.
pub fn fixed_lut_transfer(
    parameters: &PowerLutNoiseParameters,
    lut_width: usize,
) -> Result<AffineNoiseTransfer, NoiseSimulationError> {
    if lut_width == 0 {
        return Err(NoiseSimulationError::InvalidLutWidth);
    }
    let width = BigUint::from(lut_width);
    Ok(AffineNoiseTransfer {
        gain: &width * parameters.gamma_c(),
        additive: &width *
            (BigUint::one() + parameters.gamma_a()) *
            parameters.helper_error_bound(),
    })
}

/// Average-case fixed LUT transfer.  The branch count is linear in variance;
/// it is not replaced by a square-rooted worst-case bound.
pub fn average_fixed_lut_transfer(
    parameters: &PowerLutNoiseParameters,
    lut_width: usize,
) -> Result<AverageNoiseTransfer, NoiseSimulationError> {
    if lut_width == 0 {
        return Err(NoiseSimulationError::InvalidLutWidth);
    }
    let width = BigUint::from(lut_width);
    let helper = parameters.helper_doubled_variance();
    let additive = helper
        .scaled(&(BigUint::from(4u8) * &width * (BigUint::one() + parameters.gamma_a_l2_sq())));
    Ok(AverageNoiseTransfer {
        gain_sq: width * parameters.gamma_c_l2_sq(),
        additive_variance: additive,
        heuristics: vec![HeuristicId::H1StateUncorrelated, HeuristicId::H3BranchSetupIndependence],
    })
}

pub fn average_two_input_lut_transfer(
    parameters: &PowerLutNoiseParameters,
    lut_width: usize,
    semantic_multiplier_action_gain: BigUint,
) -> Result<AverageNoiseTransfer, NoiseSimulationError> {
    let fuse = average_fixed_fuse_transfer(parameters, semantic_multiplier_action_gain)?;
    let lut = average_fixed_lut_transfer(parameters, lut_width)?;
    Ok(AverageNoiseTransfer::compose(&fuse, &lut))
}

/// Exact transfer for a fixed-Fuse followed by fixed-LUT two-input branch.
///
/// `semantic_multiplier_action_gain` is explicit because it is a property of
/// the public RHS family. Power-LUT monomial RHS values use `1`.
pub fn two_input_lut_transfer(
    parameters: &PowerLutNoiseParameters,
    lut_width: usize,
    semantic_multiplier_action_gain: BigUint,
) -> Result<AffineNoiseTransfer, NoiseSimulationError> {
    Ok(compose(
        fixed_fuse_transfer(parameters, semantic_multiplier_action_gain),
        fixed_lut_transfer(parameters, lut_width)?,
    ))
}

/// Exact PRF-specific OneHot transfer for active public monomial values.
///
/// The public values must be the validated `X^a` monomials of the sparse-LWR
/// program, whose coefficient norm is one. `active_count` is the number of
/// actual non-padding cells supplied by [`PbcActiveCellIndex`], not the
/// rectangular family width. This contract is intentionally not part of the
/// generic [`simulate_program`] API.
pub fn monomial_one_hot_transfer(
    parameters: &PowerLutNoiseParameters,
    active_count: usize,
    lut_width: usize,
) -> Result<AffineNoiseTransfer, NoiseSimulationError> {
    let active = BigUint::from(active_count);
    let select = AffineNoiseTransfer {
        gain: &active * parameters.gamma_selector(),
        additive: &active * parameters.helper_error_bound(),
    };
    Ok(compose(select, fixed_lut_transfer(parameters, lut_width)?))
}

/// Average-case grouped OneHot transfer.  Exactly `active_count` monomial
/// cells contribute independent variance, while the inherited state is
/// scaled once by the aggregate selector gain.
pub fn average_monomial_one_hot_transfer(
    parameters: &PowerLutNoiseParameters,
    active_count: usize,
    lut_width: usize,
) -> Result<AverageNoiseTransfer, NoiseSimulationError> {
    if active_count == 0 {
        return Err(NoiseSimulationError::InvalidActiveCount(0));
    }
    let select = average_monomial_selection_transfer(parameters, active_count)?;
    let lut = average_fixed_lut_transfer(parameters, lut_width)?;
    Ok(AverageNoiseTransfer::compose(&select, &lut))
}

fn average_monomial_selection_transfer(
    parameters: &PowerLutNoiseParameters,
    active_count: usize,
) -> Result<AverageNoiseTransfer, NoiseSimulationError> {
    if active_count == 0 {
        return Err(NoiseSimulationError::InvalidActiveCount(0));
    }
    let active = BigUint::from(active_count);
    let helper = parameters.helper_doubled_variance();
    Ok(AverageNoiseTransfer {
        gain_sq: active.clone() * parameters.gamma_selector_l2_sq(),
        additive_variance: helper.scaled(&(BigUint::from(4u8) * active)),
        heuristics: vec![HeuristicId::H1StateUncorrelated, HeuristicId::H5PrfLabelIndependence],
    })
}

fn compose(first: AffineNoiseTransfer, second: AffineNoiseTransfer) -> AffineNoiseTransfer {
    AffineNoiseTransfer {
        gain: &second.gain * &first.gain,
        additive: &second.gain * &first.additive + &second.additive,
    }
}

/// Simulates public `Unary` and `Binary` program gates.
///
/// Generic OneHot gates are rejected because their public-value norm is not a
/// property of [`PowerLutProgram`]. Use [`simulate_sparse_prf`] for the
/// validated grouped sparse-LWR path.
pub fn simulate_program(
    program: &PowerLutProgram,
    parameters: &PowerLutNoiseParameters,
    inputs: &ProgramNoiseInputs,
) -> Result<ProgramNoiseReport, NoiseSimulationError> {
    simulate_program_inner(program, parameters, inputs, false)
}

fn simulate_program_inner(
    program: &PowerLutProgram,
    parameters: &PowerLutNoiseParameters,
    inputs: &ProgramNoiseInputs,
    _allow_monomial_one_hot: bool,
) -> Result<ProgramNoiseReport, NoiseSimulationError> {
    if parameters.secret_dimension != 2 {
        return Err(NoiseSimulationError::InvalidSecretDimension);
    }
    let mut wire_bounds = BTreeMap::new();
    for (input, wire) in program.input_wires() {
        let bound = inputs
            .input_bounds
            .get(input)
            .ok_or(NoiseSimulationError::MissingInputBound(*input))?;
        wire_bounds.insert(*wire, bound.clone());
    }
    let mut steps = Vec::with_capacity(program.gates().len());
    for (gate_index, gate) in program.gates().iter().enumerate() {
        let (input_wire, output_wire, lut_id, transfer, active_count) = match gate {
            ProgramGate::Unary { input, lut, output } => {
                let lut_table = program.lut(*lut).ok_or(NoiseSimulationError::MissingLut(*lut))?;
                (
                    *input,
                    *output,
                    *lut,
                    fixed_lut_transfer(parameters, lut_table.values().len())?,
                    None,
                )
            }
            ProgramGate::Binary { lhs, rhs, lut, output } => {
                if !program.rhs_inputs().contains_key(rhs) {
                    return Err(NoiseSimulationError::MissingRhsInput(*rhs));
                }
                let lut_table = program.lut(*lut).ok_or(NoiseSimulationError::MissingLut(*lut))?;
                (
                    *lhs,
                    *output,
                    *lut,
                    two_input_lut_transfer(parameters, lut_table.values().len(), BigUint::one())?,
                    None,
                )
            }
            ProgramGate::OneHot { .. } => {
                return Err(NoiseSimulationError::UnsupportedOneHot);
            }
        };
        let input_bound = wire_bounds
            .get(&input_wire)
            .ok_or(NoiseSimulationError::MissingWireBound(input_wire))?
            .clone();
        let output_bound = transfer.apply(&input_bound);
        wire_bounds.insert(output_wire, output_bound.clone());
        let lut_width =
            program.lut(lut_id).ok_or(NoiseSimulationError::MissingLut(lut_id))?.values().len();
        steps.push(GateNoiseStep {
            gate_index,
            input_bound: input_bound.clone(),
            transfer,
            output_bound,
            lut_width,
            active_count,
        });
    }
    let output_bound = program
        .outputs()
        .iter()
        .map(|wire| wire_bounds.get(wire).ok_or(NoiseSimulationError::MissingWireBound(*wire)))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .cloned()
        .ok_or(NoiseSimulationError::MissingOutput)?;
    Ok(ProgramNoiseReport { wire_bounds, steps, output_bound })
}

/// Report for the sequential sparse-LWR bucket recurrence and final rounding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparsePrfBucketNoiseReport {
    pub bucket: usize,
    pub active_count: usize,
    pub input_bound: BigUint,
    pub gamma_selector: BigUint,
    pub one_hot_output_bound: BigUint,
    pub one_hot_additive_bound: BigUint,
    pub one_hot_bit_growth: isize,
    pub selection_inherited_bound: BigUint,
    pub selection_additive_bound: BigUint,
    pub selection_inherited_bits: usize,
    pub selection_additive_bits: usize,
    pub input_bits: usize,
    pub output_bits: usize,
    pub gamma_c: BigUint,
    pub gamma_a: BigUint,
    pub lut_output_bound: BigUint,
    pub lut_additive_bound: BigUint,
    pub lut_bit_growth: isize,
}

/// Diagnostic for one grouped intermediate reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparsePrfGroupNoiseReport {
    pub group: usize,
    pub start_bucket: usize,
    pub bucket_len: usize,
    pub input_bound: BigUint,
    pub unreduced_bound: BigUint,
    pub output_bound: BigUint,
    pub lut_width: usize,
    pub gamma_c: BigUint,
    pub gamma_a: BigUint,
    pub inherited_bound: BigUint,
    pub base_helper_additive: BigUint,
    pub gamma_a_additive: BigUint,
    pub additive_bound: BigUint,
    pub output_bits: usize,
    pub bit_growth: isize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparsePrfNoiseReport {
    pub q_l: usize,
    pub p: usize,
    pub bucket_count: usize,
    pub k: usize,
    pub intermediate_groups: usize,
    pub terminal_start_bucket: usize,
    pub terminal_bucket_len: usize,
    pub lut_width: usize,
    pub bucket_bounds: Vec<BigUint>,
    pub bucket_stages: Vec<SparsePrfBucketNoiseReport>,
    pub group_stages: Vec<SparsePrfGroupNoiseReport>,
    pub terminal_lut_width: usize,
    pub terminal_input_bound: BigUint,
    pub terminal_output_bound: BigUint,
    pub terminal_additive_bound: BigUint,
    pub terminal_bit_growth: isize,
    pub terminal_inherited_bound: BigUint,
    pub terminal_base_helper_additive: BigUint,
    pub terminal_gamma_a_additive: BigUint,
    pub terminal_gamma_c: BigUint,
    pub terminal_gamma_a: BigUint,
    pub terminal_inherited_bits: usize,
    pub terminal_base_helper_additive_bits: usize,
    pub terminal_gamma_a_additive_bits: usize,
    pub terminal_additive_bits: usize,
    pub terminal_output_bits: usize,
    pub rounding: ProgramNoiseReport,
    pub output_bound: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageSparsePrfNoiseReport {
    pub schema_version: u32,
    pub q_l: usize,
    pub p: usize,
    pub bucket_count: usize,
    pub k: usize,
    pub intermediate_groups: usize,
    pub terminal_start_bucket: usize,
    pub terminal_bucket_len: usize,
    pub lut_width: usize,
    pub bucket_variances: Vec<AverageVariance>,
    pub group_output_variances: Vec<AverageVariance>,
    pub terminal_input_variance: AverageVariance,
    pub output_variance: AverageVariance,
    pub heuristics: Vec<HeuristicId>,
    pub accepted_under: AcceptedUnder,
}

/// Simulates the public grouped sparse-LWR recurrence.  The plan itself is
/// authoritative: each complete group receives one reduction LUT and the
/// terminal receives exactly one rounding LUT.
pub fn simulate_average_sparse_prf(
    program: &SparseLwrPrfProgram,
    parameters: &PowerLutNoiseParameters,
    layout: &PbcPublicLayout,
    initial_variance: AverageVariance,
) -> Result<AverageSparsePrfNoiseReport, NoiseSimulationError> {
    if program.plan.bucket_count() != layout.parameters.bucket_count ||
        program.plan.lut_width() != program.lut_width()
    {
        return Err(NoiseSimulationError::InvalidPbcLayout("PRF/layout plan mismatch".into()));
    }
    let active_counts = PbcActiveCellIndex::build(layout)
        .map_err(|error| NoiseSimulationError::InvalidPbcLayout(error.to_string()))?
        .bucket_active_widths()
        .collect::<Vec<_>>();
    if active_counts.len() != program.plan.bucket_count() {
        return Err(NoiseSimulationError::InvalidPbcLayout("bucket count mismatch".into()));
    }
    let mut current = initial_variance;
    let mut bucket_variances = Vec::with_capacity(active_counts.len());
    let mut group_output_variances = Vec::with_capacity(program.plan.intermediate_groups());
    let mut heuristics =
        vec![HeuristicId::H1StateUncorrelated, HeuristicId::H5PrfLabelIndependence];
    let reduction = average_fixed_lut_transfer(parameters, program.plan.lut_width())?;
    for (bucket, active) in active_counts.into_iter().enumerate() {
        let selection = average_monomial_selection_transfer(parameters, active)?;
        current = selection.apply(&current);
        append_heuristics(&mut heuristics, &selection.heuristics);
        bucket_variances.push(current.clone());
        let end_of_group =
            bucket + 1 <= program.plan.terminal_start() && (bucket + 1) % program.plan.k() == 0;
        if end_of_group {
            current = reduction.apply(&current);
            append_heuristics(&mut heuristics, &reduction.heuristics);
            group_output_variances.push(current.clone());
        }
    }
    let terminal_input_variance = current.clone();
    let terminal = average_fixed_lut_transfer(parameters, program.lut_width())?;
    current = terminal.apply(&current);
    append_heuristics(&mut heuristics, &terminal.heuristics);
    heuristics.push(HeuristicId::H6GaussianTailClosure);
    Ok(AverageSparsePrfNoiseReport {
        schema_version: AVERAGE_CASE_REPORT_SCHEMA_VERSION,
        q_l: program.profile().q_l(),
        p: program.profile().p(),
        bucket_count: program.plan.bucket_count(),
        k: program.plan.k(),
        intermediate_groups: program.plan.intermediate_groups(),
        terminal_start_bucket: program.plan.terminal_start(),
        terminal_bucket_len: program.plan.terminal_len(),
        lut_width: program.plan.lut_width(),
        bucket_variances,
        group_output_variances,
        terminal_input_variance,
        output_variance: current,
        heuristics,
        accepted_under: AcceptedUnder::Diagnostic,
    })
}

/// Simulates one shared sparse-LWR bucket body for each actual active width.
pub fn simulate_sparse_prf(
    program: &SparseLwrPrfProgram,
    parameters: &PowerLutNoiseParameters,
    layout: &PbcPublicLayout,
    initial_bound: BigUint,
) -> Result<SparsePrfNoiseReport, NoiseSimulationError> {
    if program.plan.bucket_count() != layout.parameters.bucket_count ||
        program.plan.lut_width() != program.lut_width()
    {
        return Err(NoiseSimulationError::InvalidPbcLayout("PRF/layout plan mismatch".into()));
    }
    let bucket_active_counts = PbcActiveCellIndex::build(layout)
        .map_err(|error| NoiseSimulationError::InvalidPbcLayout(error.to_string()))?
        .bucket_active_widths()
        .collect::<Vec<_>>();
    let mut current = initial_bound;
    if bucket_active_counts.len() != program.plan.bucket_count() {
        return Err(NoiseSimulationError::InvalidPbcLayout("bucket count mismatch".into()));
    }
    let mut bucket_bounds = Vec::with_capacity(bucket_active_counts.len());
    let mut bucket_stages = Vec::with_capacity(bucket_active_counts.len());
    let mut group_stages = Vec::with_capacity(program.plan.intermediate_groups());
    let reduction_transfer = fixed_lut_transfer(parameters, program.plan.lut_width())?;
    let mut group_input = current.clone();
    for (bucket, active) in bucket_active_counts.iter().copied().enumerate() {
        let input_bound = current.clone();
        let one_hot_transfer = AffineNoiseTransfer {
            gain: BigUint::from(active) * parameters.gamma_selector().clone(),
            additive: BigUint::from(active) * parameters.helper_error_bound().clone(),
        };
        let selection_inherited_bound = &one_hot_transfer.gain * &input_bound;
        let selection_additive_bound = one_hot_transfer.additive.clone();
        let one_hot_output = one_hot_transfer.apply(&input_bound);
        let one_hot_bit_growth = one_hot_output.bits() as isize - input_bound.bits() as isize;
        bucket_stages.push(SparsePrfBucketNoiseReport {
            bucket,
            active_count: active,
            input_bound: input_bound.clone(),
            gamma_selector: parameters.gamma_selector().clone(),
            one_hot_output_bound: one_hot_output.clone(),
            one_hot_additive_bound: one_hot_transfer.additive,
            one_hot_bit_growth,
            selection_inherited_bound: selection_inherited_bound.clone(),
            selection_additive_bound: selection_additive_bound.clone(),
            selection_inherited_bits: selection_inherited_bound.bits() as usize,
            selection_additive_bits: selection_additive_bound.bits() as usize,
            input_bits: input_bound.bits() as usize,
            output_bits: one_hot_output.bits() as usize,
            gamma_c: parameters.gamma_c().clone(),
            gamma_a: parameters.gamma_a().clone(),
            lut_output_bound: one_hot_output.clone(),
            lut_additive_bound: BigUint::zero(),
            lut_bit_growth: 0,
        });
        current = one_hot_output;
        bucket_bounds.push(current.clone());
        let is_intermediate_end =
            bucket + 1 <= program.plan.terminal_start() && (bucket + 1) % program.plan.k() == 0;
        if is_intermediate_end {
            let unreduced_bound = current.clone();
            let inherited_bound = &reduction_transfer.gain * &unreduced_bound;
            let base_helper_additive =
                BigUint::from(program.plan.lut_width()) * parameters.helper_error_bound().clone();
            let gamma_a_additive = BigUint::from(program.plan.lut_width()) *
                parameters.gamma_a().clone() *
                parameters.helper_error_bound().clone();
            let additive_bound = &base_helper_additive + &gamma_a_additive;
            current = reduction_transfer.apply(&current);
            group_stages.push(SparsePrfGroupNoiseReport {
                group: group_stages.len(),
                start_bucket: bucket + 1 - program.plan.k(),
                bucket_len: program.plan.k(),
                input_bound: group_input,
                unreduced_bound: unreduced_bound.clone(),
                output_bound: current.clone(),
                lut_width: program.plan.lut_width(),
                gamma_c: parameters.gamma_c().clone(),
                gamma_a: parameters.gamma_a().clone(),
                inherited_bound,
                base_helper_additive,
                gamma_a_additive,
                additive_bound,
                output_bits: current.bits() as usize,
                bit_growth: current.bits() as isize - unreduced_bound.bits() as isize,
            });
            group_input = current.clone();
        }
    }
    // The profile's declared LUT width is the committed minimal power-of-two
    // terminal domain for the accumulated exponent; the public constructor
    // validates this domain against its rounding table.
    let terminal_lut_width = program.lut_width();
    let terminal_input_bound = current.clone();
    let mut rounding_inputs = ProgramNoiseInputs::default();
    rounding_inputs.input_bounds.insert(program.rounding_input(), current);
    let rounding = simulate_program(&program.rounding_program, parameters, &rounding_inputs)?;
    let output_bound = rounding.output_bound.clone();
    let terminal_output_bound = rounding.output_bound.clone();
    let terminal_additive_bound = rounding
        .steps
        .first()
        .map(|step| step.transfer.additive.clone())
        .ok_or(NoiseSimulationError::MissingOutput)?;
    let terminal_transfer =
        rounding.steps.first().ok_or(NoiseSimulationError::MissingOutput)?.transfer.clone();
    let terminal_inherited_bound = &terminal_transfer.gain * &terminal_input_bound;
    let terminal_base_helper_additive =
        BigUint::from(terminal_lut_width) * parameters.helper_error_bound().clone();
    let terminal_gamma_a_additive = BigUint::from(terminal_lut_width) *
        parameters.gamma_a().clone() *
        parameters.helper_error_bound().clone();
    if terminal_additive_bound != terminal_base_helper_additive.clone() + &terminal_gamma_a_additive
    {
        return Err(NoiseSimulationError::BoundHelper(
            "terminal additive components do not match LUT transfer".into(),
        ));
    }
    let terminal_bit_growth =
        terminal_output_bound.bits() as isize - terminal_input_bound.bits() as isize;
    Ok(SparsePrfNoiseReport {
        q_l: program.profile().q_l(),
        p: program.profile().p(),
        bucket_count: program.plan.bucket_count(),
        k: program.plan.k(),
        intermediate_groups: program.plan.intermediate_groups(),
        terminal_start_bucket: program.plan.terminal_start(),
        terminal_bucket_len: program.plan.terminal_len(),
        lut_width: program.plan.lut_width(),
        bucket_bounds,
        bucket_stages,
        group_stages,
        terminal_lut_width,
        terminal_input_bound,
        terminal_output_bound: terminal_output_bound.clone(),
        terminal_additive_bound: terminal_additive_bound.clone(),
        terminal_bit_growth,
        terminal_inherited_bound: terminal_inherited_bound.clone(),
        terminal_base_helper_additive: terminal_base_helper_additive.clone(),
        terminal_gamma_a_additive: terminal_gamma_a_additive.clone(),
        terminal_gamma_c: parameters.gamma_c().clone(),
        terminal_gamma_a: parameters.gamma_a().clone(),
        terminal_inherited_bits: terminal_inherited_bound.bits() as usize,
        terminal_base_helper_additive_bits: terminal_base_helper_additive.bits() as usize,
        terminal_gamma_a_additive_bits: terminal_gamma_a_additive.bits() as usize,
        terminal_additive_bits: terminal_additive_bound.bits() as usize,
        terminal_output_bits: terminal_output_bound.bits() as usize,
        rounding,
        output_bound,
    })
}

/// One CRT slot's refresh parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSlotNoiseParameters {
    plaintext_modulus: BigUint,
    /// κ_t = q/q_t, retained as a semantic CRT target and spacing factor.
    kappa: BigUint,
    gamma_kappa: BigUint,
    fresh_error_route_gain: BigUint,
    decoder_action_gain: BigUint,
}

/// Exact CRT-local action gains for one refresh route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshActionGains {
    pub gamma_kappa: Vec<BigUint>,
    pub gamma_kappa_l2_sq: Vec<BigUint>,
    pub mask_digit_gains: Vec<BigUint>,
    pub mask_digit_l2_sq_gains: Vec<BigUint>,
    pub fresh_error_digit_gains: Vec<Vec<BigUint>>,
    pub fresh_error_digit_l2_sq_gains: Vec<Vec<BigUint>>,
    pub mask_route_gain: BigUint,
    pub mask_route_l2_sq_gain: BigUint,
    pub fresh_error_route_gains: Vec<BigUint>,
    pub fresh_error_route_l2_sq_gains: Vec<BigUint>,
}

/// Returns the balanced base-`beta` digits used by ordinary DCRT decomposition.
/// The tie at `beta/2` is rounded to an even quotient, matching the CPU path.
pub fn balanced_digits_for_tower(
    value: &BigUint,
    modulus: &BigUint,
    beta: &BigUint,
    digits: usize,
) -> Result<Vec<BigInt>, NoiseSimulationError> {
    if modulus.is_zero() || beta <= &BigUint::one() || digits == 0 {
        return Err(NoiseSimulationError::InvalidRefresh("invalid balanced decomposition inputs"));
    }
    let modulus_i = BigInt::from_biguint(Sign::Plus, modulus.clone());
    let beta_i = BigInt::from_biguint(Sign::Plus, beta.clone());
    let residue = value % modulus;
    let residue_i = BigInt::from_biguint(Sign::Plus, residue);
    let mut current = if &residue_i * 2 > modulus_i { residue_i - &modulus_i } else { residue_i };
    let half = &beta_i / 2;
    let mut result = Vec::with_capacity(digits);
    for _ in 0..digits {
        let quotient = current.div_euclid(&beta_i);
        let remainder = current.rem_euclid(&beta_i);
        let (digit, next) = if remainder < half {
            (remainder, quotient)
        } else if remainder > half {
            (remainder.clone() - &beta_i, quotient + 1)
        } else if (quotient.clone() % 2u8).is_zero() {
            (half.clone(), quotient)
        } else {
            (half.clone() - &beta_i, quotient + 1)
        };
        result.push(digit);
        current = next;
    }
    if !current.is_zero() {
        return Err(NoiseSimulationError::InvalidRefresh(
            "balanced decomposition carry does not vanish",
        ));
    }
    Ok(result)
}

/// Returns the exact L1 norm of the balanced decomposition in one CRT tower.
pub fn balanced_digit_l1_for_tower(
    value: &BigUint,
    modulus: &BigUint,
    beta: &BigUint,
    digits: usize,
) -> Result<BigUint, NoiseSimulationError> {
    Ok(balanced_digits_for_tower(value, modulus, beta, digits)?
        .into_iter()
        .map(|digit| digit.abs().to_biguint().unwrap_or_else(BigUint::zero))
        .fold(BigUint::zero(), |sum, digit| sum + digit))
}

/// Returns the exact squared L2 contribution of a balanced decomposition in
/// one CRT tower. This uses the same public target and digit shape as the
/// worst-case L1 gain; no caller-supplied average gain is used.
pub fn balanced_digit_l2_sq_for_tower(
    value: &BigUint,
    modulus: &BigUint,
    beta: &BigUint,
    digits: usize,
) -> Result<BigUint, NoiseSimulationError> {
    Ok(balanced_digits_for_tower(value, modulus, beta, digits)?
        .into_iter()
        .map(|digit| {
            let magnitude = digit.abs().to_biguint().unwrap_or_else(BigUint::zero);
            &magnitude * &magnitude
        })
        .fold(BigUint::zero(), |sum, digit| sum + digit))
}

fn rank_one_target_gain(
    target: &BigUint,
    crt_moduli: &[BigUint],
    beta: &BigUint,
    digits_per_tower: usize,
) -> Result<BigUint, NoiseSimulationError> {
    crt_moduli
        .iter()
        .map(|modulus| balanced_digit_l1_for_tower(target, modulus, beta, digits_per_tower))
        .try_fold(BigUint::zero(), |sum, gain| Ok(sum + gain?))
}

fn rank_one_target_l2_sq(
    target: &BigUint,
    crt_moduli: &[BigUint],
    beta: &BigUint,
    digits_per_tower: usize,
) -> Result<BigUint, NoiseSimulationError> {
    crt_moduli
        .iter()
        .map(|modulus| balanced_digit_l2_sq_for_tower(target, modulus, beta, digits_per_tower))
        .try_fold(BigUint::zero(), |sum, gain| Ok(sum + gain?))
}

/// Computes target-aware CRT-local gains for the refresh routes.
pub fn refresh_action_gains(
    full_modulus: &BigUint,
    crt_moduli: &[BigUint],
    base_p: &BigUint,
    mask_digits: usize,
    fresh_digits: usize,
    gadget_base: &BigUint,
    gadget_digit_count: usize,
    coefficient_count: usize,
) -> Result<RefreshActionGains, NoiseSimulationError> {
    if full_modulus.is_zero() ||
        crt_moduli.is_empty() ||
        base_p < &BigUint::from(2u8) ||
        gadget_base <= &BigUint::one() ||
        (gadget_base & (gadget_base - BigUint::one())) != BigUint::zero() ||
        gadget_digit_count == 0 ||
        mask_digits == 0 ||
        fresh_digits == 0 ||
        coefficient_count == 0
    {
        return Err(NoiseSimulationError::InvalidRefresh("invalid refresh gain inputs"));
    }
    validate_crt_factorization(full_modulus, crt_moduli)?;
    if !gadget_digit_count.is_multiple_of(crt_moduli.len()) {
        return Err(NoiseSimulationError::InvalidRefresh(
            "gadget digit count must be divisible by CRT tower count",
        ));
    }
    let digits_per_tower = gadget_digit_count / crt_moduli.len();
    if digits_per_tower == 0 {
        return Err(NoiseSimulationError::InvalidRefresh("zero digits per CRT tower"));
    }
    let mut gamma_kappa = Vec::with_capacity(crt_moduli.len());
    let mut gamma_kappa_l2_sq = Vec::with_capacity(crt_moduli.len());
    let mut fresh_error_digit_gains = Vec::with_capacity(crt_moduli.len());
    let mut fresh_error_digit_l2_sq_gains = Vec::with_capacity(crt_moduli.len());
    for modulus in crt_moduli {
        let kappa = full_modulus / modulus;
        let mut gamma = BigUint::zero();
        let mut gamma_l2_sq = BigUint::zero();
        let mut gadget_power = BigUint::one();
        for _ in 0..digits_per_tower {
            let target = &kappa * &gadget_power;
            gamma = gamma.max(balanced_digit_l1_for_tower(
                &target,
                modulus,
                gadget_base,
                digits_per_tower,
            )?);
            gamma_l2_sq = gamma_l2_sq.max(balanced_digit_l2_sq_for_tower(
                &target,
                modulus,
                gadget_base,
                digits_per_tower,
            )?);
            gadget_power *= gadget_base;
        }
        gamma_kappa.push(gamma);
        gamma_kappa_l2_sq.push(gamma_l2_sq);
        let mut per_digit = Vec::with_capacity(fresh_digits);
        let mut per_digit_l2_sq = Vec::with_capacity(fresh_digits);
        let mut power = BigUint::one();
        for _ in 0..fresh_digits {
            let target = &kappa * &power;
            per_digit.push(rank_one_target_gain(
                &target,
                crt_moduli,
                gadget_base,
                digits_per_tower,
            )?);
            per_digit_l2_sq.push(rank_one_target_l2_sq(
                &target,
                crt_moduli,
                gadget_base,
                digits_per_tower,
            )?);
            power *= base_p;
        }
        fresh_error_digit_gains.push(per_digit);
        fresh_error_digit_l2_sq_gains.push(per_digit_l2_sq);
    }
    let mut mask_digit_gains = Vec::with_capacity(mask_digits);
    let mut mask_digit_l2_sq_gains = Vec::with_capacity(mask_digits);
    let mut power = BigUint::one();
    for _ in 0..mask_digits {
        mask_digit_gains.push(rank_one_target_gain(
            &power,
            crt_moduli,
            gadget_base,
            digits_per_tower,
        )?);
        mask_digit_l2_sq_gains.push(rank_one_target_l2_sq(
            &power,
            crt_moduli,
            gadget_base,
            digits_per_tower,
        )?);
        power *= base_p;
    }
    let mask_route_gain = BigUint::from(coefficient_count) *
        mask_digit_gains.iter().cloned().fold(BigUint::zero(), |sum, gain| sum + gain);
    let mask_route_l2_sq_gain = BigUint::from(coefficient_count) *
        mask_digit_l2_sq_gains.iter().cloned().fold(BigUint::zero(), |sum, gain| sum + gain);
    let fresh_error_route_gains = fresh_error_digit_gains
        .iter()
        .map(|gains| {
            BigUint::from(coefficient_count) *
                gains.iter().cloned().fold(BigUint::zero(), |sum, gain| sum + gain)
        })
        .collect();
    let fresh_error_route_l2_sq_gains = fresh_error_digit_l2_sq_gains
        .iter()
        .map(|gains| {
            BigUint::from(coefficient_count) *
                gains.iter().cloned().fold(BigUint::zero(), |sum, gain| sum + gain)
        })
        .collect();
    Ok(RefreshActionGains {
        gamma_kappa,
        gamma_kappa_l2_sq,
        mask_digit_gains,
        mask_digit_l2_sq_gains,
        fresh_error_digit_gains,
        fresh_error_digit_l2_sq_gains,
        mask_route_gain,
        mask_route_l2_sq_gain,
        fresh_error_route_gains,
        fresh_error_route_l2_sq_gains,
    })
}

/// Numeric inputs for exact refresh threshold checking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshNoiseParameters {
    full_modulus: BigUint,
    base_p: BigUint,
    /// Q_L, the sparse-LWR input modulus that every mask must cover.
    sparse_lwr_modulus: BigUint,
    mask_base_p_digit_count: usize,
    fresh_error_base_p_digit_count: usize,
    mask_statistical_security_bits: u64,
    mask_slot_count: usize,
    coefficient_count: usize,
    /// Number of public-key columns covered by each aggregate: 2*ell_beta.
    component_columns: usize,
    gamma_kappa_l2_sq: Vec<BigUint>,
    /// Decoder action dimensions retained from setup so AverageCase can
    /// derive its p=2 scale from the same structural inputs.
    decoder_columns: usize,
    decoder_ring_dimension: usize,
    decoder_preimage_bound: BigUint,
    mask_digit_gains: Vec<BigUint>,
    mask_digit_l2_sq_gains: Vec<BigUint>,
    mask_route_gain: BigUint,
    mask_route_l2_sq_gain: BigUint,
    fresh_error_digit_gains: Vec<Vec<BigUint>>,
    fresh_error_digit_l2_sq_gains: Vec<Vec<BigUint>>,
    fresh_error_route_l2_sq_gains: Vec<BigUint>,
    slots: Vec<RefreshSlotNoiseParameters>,
}

impl RefreshNoiseParameters {
    /// Builds route, scale, and decoder gains from the structural model.
    pub(crate) fn from_structural(
        model: &PowerLutNoiseParameters,
        full_modulus: BigUint,
        base_p: BigUint,
        sparse_lwr_modulus: BigUint,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: u64,
        mask_slot_count: usize,
        coefficient_count: usize,
        plaintext_moduli: Vec<BigUint>,
        decoder_preimage_bound: BigUint,
    ) -> Result<Self, NoiseSimulationError> {
        let decoder_columns = model
            .gadget_digit_count
            .checked_add(2)
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        if model.secret_dimension != 2 ||
            full_modulus.is_zero() ||
            base_p < BigUint::from(2u8) ||
            sparse_lwr_modulus.is_zero() ||
            mask_base_p_digit_count == 0 ||
            fresh_error_base_p_digit_count == 0 ||
            mask_slot_count == 0 ||
            coefficient_count == 0 ||
            plaintext_moduli.len() != mask_slot_count ||
            decoder_preimage_bound.is_zero()
        {
            return Err(NoiseSimulationError::InvalidRefresh("inconsistent dimensions"));
        }
        validate_crt_factorization(&full_modulus, &plaintext_moduli)?;
        let component_columns = model
            .gadget_digit_count
            .checked_mul(2)
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        let gains = refresh_action_gains(
            &full_modulus,
            &plaintext_moduli,
            &base_p,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
            &model.gadget_base,
            model.gadget_digit_count,
            coefficient_count,
        )?;
        let decoder_action_gain = BigUint::from(2u8) *
            BigUint::from(decoder_columns) *
            BigUint::from(model.ring_dimension) *
            decoder_preimage_bound.clone();
        let slots = plaintext_moduli
            .into_iter()
            .enumerate()
            .map(|(slot, plaintext_modulus)| RefreshSlotNoiseParameters {
                kappa: &full_modulus / &plaintext_modulus,
                gamma_kappa: gains.gamma_kappa[slot].clone(),
                fresh_error_route_gain: gains.fresh_error_route_gains[slot].clone(),
                plaintext_modulus,
                decoder_action_gain: decoder_action_gain.clone(),
            })
            .collect();
        Ok(Self {
            full_modulus,
            base_p,
            sparse_lwr_modulus,
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
            mask_statistical_security_bits,
            mask_slot_count,
            coefficient_count,
            component_columns,
            gamma_kappa_l2_sq: gains.gamma_kappa_l2_sq,
            decoder_columns,
            decoder_ring_dimension: model.ring_dimension,
            decoder_preimage_bound,
            mask_digit_gains: gains.mask_digit_gains,
            mask_digit_l2_sq_gains: gains.mask_digit_l2_sq_gains,
            mask_route_gain: gains.mask_route_gain,
            mask_route_l2_sq_gain: gains.mask_route_l2_sq_gain,
            fresh_error_digit_gains: gains.fresh_error_digit_gains,
            fresh_error_digit_l2_sq_gains: gains.fresh_error_digit_l2_sq_gains,
            fresh_error_route_l2_sq_gains: gains.fresh_error_route_l2_sq_gains,
            slots,
        })
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSlotNoiseReport {
    pub slot: usize,
    pub spacing: BigUint,
    /// B_m = p^d_m - 1, the maximum mask value before refresh.
    pub mask_bound: BigUint,
    /// B_e = p^d_e - 1, the maximum fresh error after refresh.
    pub fresh_error_bound: BigUint,
    /// Exact κ_t = q/q_t.
    pub kappa: BigUint,
    pub gamma_kappa: BigUint,
    pub state_term: BigUint,
    pub prf_mask_term: BigUint,
    pub prf_fresh_term: BigUint,
    pub fresh_error_route_gain: BigUint,
    pub decoder_term: BigUint,
    /// F_t, excluding the semantic mask B_m.
    pub operation_noise_bound: BigUint,
    pub pre_rounding_bound: BigUint,
    pub twice_pre_rounding_bound: BigUint,
    /// spacing - 2*(B_m + F_t); positive means strict rounding passes.
    pub rounding_margin: BigInt,
    /// q_t - B_e; positive means strict fresh-error check passes.
    pub fresh_error_margin: BigInt,
    pub fresh_error_below_plaintext_modulus: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshNoiseReport {
    pub prf_output_bound: BigUint,
    pub mask_digit_gains: Vec<BigUint>,
    pub mask_route_gain: BigUint,
    pub fresh_error_digit_gains: Vec<Vec<BigUint>>,
    pub mask_base_p_digit_count: usize,
    pub fresh_error_base_p_digit_count: usize,
    pub sparse_lwr_modulus: BigUint,
    pub mask_modulus: BigUint,
    pub mask_bound: BigUint,
    pub fresh_error_modulus: BigUint,
    pub fresh_error_bound: BigUint,
    /// mask_slot_count * (2*ell_beta) * coefficient_count.
    pub exposed_transcript_coordinates: BigUint,
    pub mask_statistical_security_bits: u64,
    /// 2^(lambda+1) * D * max_t(F_t). Equality is accepted.
    pub mask_statistical_required: BigUint,
    /// M_m - required. Non-negative means the hiding test passes.
    pub mask_statistical_margin: BigInt,
    pub mask_statistical_accepted: bool,
    /// M_m - Q_L; non-negative means the mask range covers the PRF domain.
    pub mask_domain_margin: BigInt,
    pub mask_domain_accepted: bool,
    pub fresh_error_accepted: bool,
    pub hard_authority: RefreshHardAuthority,
    pub slots: Vec<RefreshSlotNoiseReport>,
    pub refreshed_error_bound: Option<BigUint>,
    pub accepted: bool,
}

/// Independent WorstCase predicates that form the production refresh
/// authority.  Rounding is intentionally not part of this authority so an
/// AverageCase correctness estimate cannot accidentally redefine security or
/// smudging acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshHardAuthority {
    pub mask_domain_accepted: bool,
    pub mask_statistical_accepted: bool,
    pub fresh_error_accepted: bool,
    pub accepted: bool,
}

/// A setup-bound, secret-free input to the complete sparse-PRF/refresh
/// simulator.  The constructor is crate-visible so setup code, rather than an
/// external caller, owns the identity and structural parameters.
#[derive(Clone, Debug)]
pub struct PowerLutNoiseSnapshot {
    setup_identity: [u8; 32],
    prf_program: SparseLwrPrfProgram,
    pbc_layout: PbcPublicLayout,
    model: PowerLutNoiseParameters,
    refresh: RefreshNoiseParameters,
    initial_state_bound: BigUint,
    initial_average_variance: AverageVariance,
}

/// Combined report returned by a setup-bound simulation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PowerLutNoiseReport {
    pub prf: SparsePrfNoiseReport,
    pub refresh: RefreshNoiseReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PowerLutAverageNoiseReport {
    pub snapshot_identity: [u8; 32],
    pub prf: AverageSparsePrfNoiseReport,
    pub refresh: AverageRefreshNoiseReport,
    /// Lattice security is decided by the parameter-search harness, not this
    /// AverageCase production report.
    pub security_authority: &'static str,
    pub correctness_authority: NoiseModelKind,
    pub hard_authority_accepted: bool,
    pub correctness_accepted: bool,
    pub accepted: bool,
}

impl PowerLutNoiseSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_setup(
        setup_identity: [u8; 32],
        prf_program: SparseLwrPrfProgram,
        pbc_layout: PbcPublicLayout,
        model: PowerLutNoiseParameters,
        full_modulus: BigUint,
        base_p: BigUint,
        mask_base_p_digit_count: usize,
        fresh_error_base_p_digit_count: usize,
        mask_statistical_security_bits: u64,
        mask_slot_count: usize,
        coefficient_count: usize,
        plaintext_moduli: Vec<BigUint>,
        decoder_preimage_bound: BigUint,
        initial_state_bound: BigUint,
        initial_average_variance: AverageVariance,
    ) -> Result<Self, NoiseSimulationError> {
        pbc_layout
            .validate()
            .map_err(|error| NoiseSimulationError::InvalidPbcLayout(error.to_string()))?;
        if prf_program.profile().ring_dimension() != model.ring_dimension ||
            prf_program.bucket_width() != pbc_layout.bucket_width
        {
            return Err(NoiseSimulationError::InvalidRefresh("PRF/layout mismatch"));
        }
        let refresh = RefreshNoiseParameters::from_structural(
            &model,
            full_modulus,
            base_p.clone(),
            BigUint::from(prf_program.profile().q_l()),
            mask_base_p_digit_count,
            fresh_error_base_p_digit_count,
            mask_statistical_security_bits,
            mask_slot_count,
            coefficient_count,
            plaintext_moduli,
            decoder_preimage_bound,
        )?;
        Ok(Self {
            setup_identity,
            prf_program,
            pbc_layout,
            model,
            refresh,
            initial_state_bound,
            initial_average_variance,
        })
    }

    pub fn setup_identity(&self) -> &[u8; 32] {
        &self.setup_identity
    }

    /// Returns the setup-derived doubled-coordinate starting variance used by
    /// the AverageCase channel.
    pub fn initial_average_variance(&self) -> &AverageVariance {
        &self.initial_average_variance
    }

    /// Simulates the complete sparse PRF and refresh using only setup-bound
    /// public layout and structural parameters.
    pub fn simulate(&self) -> Result<PowerLutNoiseReport, NoiseSimulationError> {
        let prf = simulate_sparse_prf(
            &self.prf_program,
            &self.model,
            &self.pbc_layout,
            self.initial_state_bound.clone(),
        )?;
        let refresh = simulate_refresh(
            &self.model,
            &self.refresh,
            self.initial_state_bound.clone(),
            prf.output_bound.clone(),
        )?;
        Ok(PowerLutNoiseReport { prf, refresh })
    }

    /// Runs AverageCase only when explicitly enabled. Security is deliberately
    /// left to the parameter-search harness; this report contains only the
    /// setup-bound hard gates plus AverageCase rounding/smudging authorities.
    pub fn simulate_average(
        &self,
        config: &AverageCaseConfig,
    ) -> Result<PowerLutAverageNoiseReport, NoiseSimulationError> {
        if !config.allow_average_acceptance {
            return Err(NoiseSimulationError::AverageAcceptanceDisabled);
        }
        let prf = simulate_average_sparse_prf(
            &self.prf_program,
            &self.model,
            &self.pbc_layout,
            self.initial_average_variance.clone(),
        )?;
        let refresh = simulate_average_refresh(
            &self.model,
            &self.refresh,
            self.initial_average_variance.clone(),
            prf.output_variance.clone(),
            config,
        )?;
        let hard_authority_accepted = refresh.domain_accepted && refresh.fresh_error_accepted;
        let correctness_accepted = refresh.rounding_accepted && refresh.mask_smudging_accepted;
        let accepted = hard_authority_accepted && correctness_accepted;
        Ok(PowerLutAverageNoiseReport {
            snapshot_identity: self.setup_identity,
            prf,
            refresh,
            security_authority: "harness",
            correctness_authority: NoiseModelKind::AverageCase,
            hard_authority_accepted,
            correctness_accepted,
            accepted,
        })
    }
}

/// Exact refresh simulation with strict per-slot inequalities.
pub fn simulate_refresh(
    parameters: &PowerLutNoiseParameters,
    refresh: &RefreshNoiseParameters,
    state_bound: BigUint,
    prf_output_bound: BigUint,
) -> Result<RefreshNoiseReport, NoiseSimulationError> {
    if parameters.secret_dimension != 2 {
        return Err(NoiseSimulationError::InvalidSecretDimension);
    }
    if refresh.full_modulus.is_zero() ||
        refresh.base_p < BigUint::from(2u8) ||
        refresh.mask_base_p_digit_count == 0 ||
        refresh.fresh_error_base_p_digit_count == 0 ||
        refresh.mask_slot_count == 0 ||
        refresh.coefficient_count == 0 ||
        refresh.slots.len() != refresh.mask_slot_count
    {
        return Err(NoiseSimulationError::InvalidRefresh("inconsistent dimensions"));
    }
    let mask_modulus = base_p_power(&refresh.base_p, refresh.mask_base_p_digit_count)?;
    let fresh_error_modulus =
        base_p_power(&refresh.base_p, refresh.fresh_error_base_p_digit_count)?;
    let mask_bound = &mask_modulus - BigUint::one();
    let fresh_error_bound = &fresh_error_modulus - BigUint::one();
    let mask_domain_margin = signed_difference(&mask_modulus, &refresh.sparse_lwr_modulus);
    let mask_domain_accepted = mask_domain_margin >= BigInt::zero();
    let exposed_transcript_coordinates = BigUint::from(refresh.mask_slot_count) *
        BigUint::from(refresh.component_columns) *
        BigUint::from(refresh.coefficient_count);
    let mut slots = Vec::with_capacity(refresh.slots.len());
    let mut max_operation_noise_bound = BigUint::zero();
    for (slot, slot_parameters) in refresh.slots.iter().enumerate() {
        if slot_parameters.plaintext_modulus <= BigUint::one() {
            return Err(NoiseSimulationError::InvalidRefresh("invalid slot parameters"));
        }
        let remainder = &refresh.full_modulus % &slot_parameters.plaintext_modulus;
        if !remainder.is_zero() {
            return Err(NoiseSimulationError::InvalidCrtDivision);
        }
        let spacing = &refresh.full_modulus / &slot_parameters.plaintext_modulus;
        let state_term = &slot_parameters.gamma_kappa * &state_bound;
        let prf_mask_term = &refresh.mask_route_gain * &prf_output_bound;
        let prf_fresh_term = &slot_parameters.fresh_error_route_gain * &prf_output_bound;
        let decoder_term = parameters.helper_error_bound() * &slot_parameters.decoder_action_gain;
        // F_t = γ_{κ,t} E_state + R_m E_PRF + R_{e,t} E_PRF + E_decoder,t,
        // where each route gain is the exact balanced DCRT action gain.
        let operation_noise_bound = &state_term + &prf_mask_term + &prf_fresh_term + &decoder_term;
        max_operation_noise_bound = max_operation_noise_bound.max(operation_noise_bound.clone());
        let pre_rounding_bound = &mask_bound + &operation_noise_bound;
        let twice_pre_rounding_bound = BigUint::from(2u8) * &pre_rounding_bound;
        let rounding_margin = signed_difference(&spacing, &twice_pre_rounding_bound);
        let fresh_error_margin =
            signed_difference(&slot_parameters.plaintext_modulus, &fresh_error_bound);
        let fresh_error_below_plaintext_modulus = fresh_error_margin > BigInt::zero();
        let slot_accepted = rounding_margin > BigInt::zero() && fresh_error_below_plaintext_modulus;
        slots.push(RefreshSlotNoiseReport {
            slot,
            spacing,
            mask_bound: mask_bound.clone(),
            fresh_error_bound: fresh_error_bound.clone(),
            kappa: slot_parameters.kappa.clone(),
            gamma_kappa: slot_parameters.gamma_kappa.clone(),
            state_term,
            prf_mask_term,
            prf_fresh_term,
            fresh_error_route_gain: slot_parameters.fresh_error_route_gain.clone(),
            decoder_term,
            operation_noise_bound,
            pre_rounding_bound,
            twice_pre_rounding_bound,
            rounding_margin,
            fresh_error_margin,
            fresh_error_below_plaintext_modulus,
            accepted: slot_accepted,
        });
    }
    let security_shift = refresh
        .mask_statistical_security_bits
        .checked_add(1)
        .and_then(|bits| usize::try_from(bits).ok())
        .ok_or(NoiseSimulationError::IntegerConversion)?;
    let mask_statistical_required = (BigUint::one() << security_shift) *
        &exposed_transcript_coordinates *
        &max_operation_noise_bound;
    let mask_statistical_margin = signed_difference(&mask_modulus, &mask_statistical_required);
    // M_m >= 2^(λ+1) D max_t(F_t); equality is intentionally accepted.
    let mask_statistical_accepted = mask_statistical_margin >= BigInt::zero();
    let fresh_error_accepted = slots.iter().all(|slot| slot.fresh_error_below_plaintext_modulus);
    let hard_authority = RefreshHardAuthority {
        mask_domain_accepted,
        mask_statistical_accepted,
        fresh_error_accepted,
        accepted: mask_domain_accepted && mask_statistical_accepted && fresh_error_accepted,
    };
    let accepted =
        mask_domain_accepted && mask_statistical_accepted && slots.iter().all(|slot| slot.accepted);
    Ok(RefreshNoiseReport {
        prf_output_bound,
        mask_digit_gains: refresh.mask_digit_gains.clone(),
        mask_route_gain: refresh.mask_route_gain.clone(),
        fresh_error_digit_gains: refresh.fresh_error_digit_gains.clone(),
        mask_base_p_digit_count: refresh.mask_base_p_digit_count,
        fresh_error_base_p_digit_count: refresh.fresh_error_base_p_digit_count,
        sparse_lwr_modulus: refresh.sparse_lwr_modulus.clone(),
        mask_modulus,
        mask_bound,
        fresh_error_modulus,
        fresh_error_bound: fresh_error_bound.clone(),
        exposed_transcript_coordinates,
        mask_statistical_security_bits: refresh.mask_statistical_security_bits,
        mask_statistical_required,
        mask_statistical_margin,
        mask_statistical_accepted,
        mask_domain_margin,
        mask_domain_accepted,
        fresh_error_accepted,
        hard_authority,
        slots,
        refreshed_error_bound: accepted.then_some(fresh_error_bound),
        accepted,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageRefreshSlotNoiseReport {
    pub slot: usize,
    pub spacing: BigUint,
    pub mask_bound: BigUint,
    pub fresh_error_bound: BigUint,
    pub deterministic_term: BigUint,
    pub state_variance: AverageVariance,
    pub mask_variance: AverageVariance,
    pub fresh_variance: AverageVariance,
    pub decoder_variance: AverageVariance,
    pub stochastic_variance: AverageVariance,
    pub z_sq: AverageVariance,
    pub favg: BigUint,
    pub squared_margin: BigInt,
    pub squared_deficit: BigInt,
    pub fresh_error_below_plaintext_modulus: bool,
    pub rounding_accepted: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageRefreshNoiseReport {
    pub schema_version: u32,
    pub mode: NoiseModelKind,
    pub authority: AcceptedUnder,
    pub centered_coordinates: bool,
    pub decoder_factor: BigUint,
    pub mask_bound: BigUint,
    pub fresh_error_bound: BigUint,
    pub event_budget: AverageEventBudget,
    /// Topology counts are evidence; `joint_event_count` is the sole tail
    /// budget used by all AverageCase inspections.
    pub mask_event_count: BigUint,
    pub fresh_event_count: BigUint,
    pub joint_event_count: BigUint,
    pub epsilon_joint: AverageVariance,
    pub masking_distance_bound: AverageVariance,
    pub z_sq: AverageVariance,
    pub tail_correction_bits: u32,
    pub mask_statistical_security_bits: u64,
    pub mask_smudging_max_favg: BigUint,
    pub mask_smudging_margin: BigInt,
    pub mask_smudging_accepted: bool,
    pub domain_margin: BigInt,
    pub domain_accepted: bool,
    pub fresh_error_accepted: bool,
    pub heuristics: Vec<HeuristicId>,
    pub slots: Vec<AverageRefreshSlotNoiseReport>,
    pub rounding_accepted: bool,
    pub accepted: bool,
}

/// Average-case refresh propagation and strict squared-domain threshold.
/// The p=2 centering transform and decoder factor are derived directly from
/// the structural refresh inputs; unsupported setups fail closed.
pub fn simulate_average_refresh(
    parameters: &PowerLutNoiseParameters,
    refresh: &RefreshNoiseParameters,
    state_variance: AverageVariance,
    prf_variance: AverageVariance,
    config: &AverageCaseConfig,
) -> Result<AverageRefreshNoiseReport, NoiseSimulationError> {
    if refresh.base_p != BigUint::from(2u8) || refresh.slots.is_empty() {
        return Err(NoiseSimulationError::UnsupportedAverageRefresh);
    }
    // The doubled-coordinate decoder scale is a direct function of the
    // actual base-p setup and decoder dimensions.  It is deliberately not a
    // caller-provided flag or proof field.
    let decoder_factor = refresh.base_p.clone();
    let decoder_action_gain = &decoder_factor *
        BigUint::from(refresh.decoder_columns) *
        BigUint::from(refresh.decoder_ring_dimension) *
        &refresh.decoder_preimage_bound;
    if refresh.full_modulus.is_zero() ||
        refresh.slots.len() != refresh.mask_slot_count ||
        refresh.mask_digit_gains.len() != refresh.mask_base_p_digit_count ||
        refresh.mask_digit_l2_sq_gains.len() != refresh.mask_base_p_digit_count ||
        refresh.fresh_error_digit_gains.len() != refresh.mask_slot_count ||
        refresh.fresh_error_digit_l2_sq_gains.len() != refresh.mask_slot_count ||
        refresh.fresh_error_route_l2_sq_gains.len() != refresh.mask_slot_count ||
        refresh.gamma_kappa_l2_sq.len() != refresh.mask_slot_count ||
        refresh
            .fresh_error_digit_l2_sq_gains
            .iter()
            .any(|gains| gains.len() != refresh.fresh_error_base_p_digit_count) ||
        refresh
            .fresh_error_digit_gains
            .iter()
            .any(|gains| gains.len() != refresh.fresh_error_base_p_digit_count)
    {
        return Err(NoiseSimulationError::InvalidRefresh("inconsistent dimensions"));
    }
    let events = config.event_budget(refresh.coefficient_count, refresh.mask_slot_count);
    let mask_event_count = BigUint::from(refresh.mask_slot_count) *
        BigUint::from(refresh.component_columns) *
        BigUint::from(refresh.coefficient_count) *
        BigUint::from(refresh.mask_base_p_digit_count);
    let fresh_event_count = BigUint::from(refresh.component_columns) *
        BigUint::from(refresh.coefficient_count) *
        BigUint::from(refresh.fresh_error_base_p_digit_count);
    let joint_event_count = config.joint_event_count(&mask_event_count, &fresh_event_count)?;
    let z_sq = config.z_squared_for_log2(ceil_log2(&joint_event_count)?)?;
    let epsilon_joint =
        AverageVariance::new(joint_event_count.clone(), BigUint::one() << config.failure_exponent)?;
    let masking_distance_bound = AverageVariance::new(
        BigUint::one(),
        BigUint::one() << refresh.mask_statistical_security_bits,
    )?
    .add(&epsilon_joint);
    let mask_modulus = base_p_power(&refresh.base_p, refresh.mask_base_p_digit_count)?;
    let fresh_modulus = base_p_power(&refresh.base_p, refresh.fresh_error_base_p_digit_count)?;
    let mask_bound = &mask_modulus - BigUint::one();
    let fresh_bound = &fresh_modulus - BigUint::one();
    let mask_digit_variances = refresh
        .mask_digit_l2_sq_gains
        .iter()
        .cloned()
        .fold(BigUint::zero(), |sum, value| sum + value) *
        BigUint::from(refresh.coefficient_count);
    let mask_label_l2_sq = refresh.mask_route_l2_sq_gain.clone();
    let mask_digit_variance = AverageVariance::new(mask_digit_variances, BigUint::one())?;
    // The grouped PRF output is the shared public source for both routes;
    // retaining it in each independently routed term is conservative and
    // prevents an unclassified shared-state addition from being cancelled.
    let mask_variance = mask_digit_variance.add(&prf_variance.scaled(&mask_label_l2_sq));
    let helper_variance = parameters.helper_doubled_variance();
    let mut slots = Vec::with_capacity(refresh.slots.len());
    let mut max_favg = BigUint::zero();
    let mut heuristics = vec![
        HeuristicId::H1StateUncorrelated,
        HeuristicId::H2DigitUniformFallback,
        HeuristicId::H4SlotRhsIndependence,
        HeuristicId::H6GaussianTailClosure,
    ];
    for (slot, slot_parameters) in refresh.slots.iter().enumerate() {
        if slot_parameters.plaintext_modulus <= BigUint::one() {
            return Err(NoiseSimulationError::InvalidRefresh("invalid slot parameters"));
        }
        let spacing = &refresh.full_modulus / &slot_parameters.plaintext_modulus;
        let fresh_digit_variances = refresh.fresh_error_digit_l2_sq_gains[slot]
            .iter()
            .cloned()
            .fold(BigUint::zero(), |sum, value| sum + value) *
            BigUint::from(refresh.coefficient_count);
        let fresh_label_l2_sq = refresh.fresh_error_route_l2_sq_gains[slot].clone();
        let fresh_variance = AverageVariance::new(fresh_digit_variances, BigUint::one())?
            .add(&prf_variance.scaled(&fresh_label_l2_sq));
        let state_gain_sq = &refresh.gamma_kappa_l2_sq[slot];
        let decoder_gain_sq = &decoder_action_gain * &decoder_action_gain;
        let state_term = state_variance.scaled(state_gain_sq);
        let routed_mask = mask_variance.clone();
        let routed_fresh = fresh_variance;
        let decoder_variance = helper_variance.scaled(&(BigUint::from(4u8) * decoder_gain_sq));
        let stochastic_variance =
            state_term.add(&routed_mask).add(&routed_fresh).add(&decoder_variance);
        let favg = average_favg(&z_sq, &stochastic_variance, config.tail_correction_bits)?;
        max_favg = max_favg.max(favg.clone());
        let squared_margin = signed_difference(&(&spacing * 2u8), &(&mask_bound * 2u8));
        let margin_squared = &squared_margin * &squared_margin;
        let tail_bits = config
            .tail_correction_bits
            .checked_mul(2)
            .ok_or(NoiseSimulationError::IntegerConversion)?;
        let lhs = margin_squared *
            BigInt::from_biguint(
                Sign::Plus,
                &z_sq.denominator * &stochastic_variance.denominator,
            );
        let rhs = BigInt::from_biguint(
            Sign::Plus,
            BigUint::from(4u8) *
                &z_sq.numerator *
                (BigUint::one() << tail_bits) *
                &stochastic_variance.numerator,
        );
        let squared_deficit = &lhs - &rhs;
        let stochastic_ok = squared_margin > BigInt::zero() && squared_deficit > BigInt::zero();
        let fresh_margin = signed_difference(&slot_parameters.plaintext_modulus, &fresh_bound);
        let fresh_ok = fresh_margin > BigInt::zero();
        let deterministic_term = mask_bound.clone();
        slots.push(AverageRefreshSlotNoiseReport {
            slot,
            spacing,
            mask_bound: mask_bound.clone(),
            fresh_error_bound: fresh_bound.clone(),
            deterministic_term,
            state_variance: state_term,
            mask_variance: routed_mask,
            fresh_variance: routed_fresh,
            decoder_variance,
            stochastic_variance,
            z_sq: z_sq.clone(),
            favg,
            squared_margin,
            squared_deficit,
            fresh_error_below_plaintext_modulus: fresh_ok,
            rounding_accepted: stochastic_ok,
            accepted: stochastic_ok && fresh_ok,
        });
    }
    let rounding_accepted = slots.iter().all(|slot| slot.rounding_accepted);
    let domain_margin = signed_difference(&mask_modulus, &refresh.sparse_lwr_modulus);
    let domain_accepted = domain_margin >= BigInt::zero();
    let fresh_error_accepted = slots.iter().all(|slot| slot.fresh_error_below_plaintext_modulus);
    let smudging_dimension = BigUint::from(refresh.mask_slot_count) *
        BigUint::from(refresh.component_columns) *
        BigUint::from(refresh.coefficient_count);
    let smudging_required = average_smudging_required(
        refresh.mask_statistical_security_bits,
        &smudging_dimension,
        &max_favg,
    )?;
    let mask_smudging_margin = signed_difference(&mask_modulus, &smudging_required);
    let mask_smudging_accepted = mask_smudging_margin >= BigInt::zero();
    let accepted =
        domain_accepted && fresh_error_accepted && mask_smudging_accepted && rounding_accepted;
    Ok(AverageRefreshNoiseReport {
        schema_version: AVERAGE_CASE_REPORT_SCHEMA_VERSION,
        mode: NoiseModelKind::AverageCase,
        authority: AcceptedUnder::AverageCase,
        centered_coordinates: true,
        decoder_factor,
        mask_bound,
        fresh_error_bound: fresh_bound,
        event_budget: events,
        mask_event_count,
        fresh_event_count,
        joint_event_count,
        epsilon_joint,
        masking_distance_bound,
        z_sq,
        tail_correction_bits: config.tail_correction_bits,
        mask_statistical_security_bits: refresh.mask_statistical_security_bits,
        mask_smudging_max_favg: max_favg,
        mask_smudging_margin,
        mask_smudging_accepted,
        domain_margin,
        domain_accepted,
        fresh_error_accepted,
        heuristics: {
            heuristics.sort_by_key(|id| *id as u8);
            heuristics.dedup();
            heuristics
        },
        slots,
        rounding_accepted,
        accepted,
    })
}

fn base_p_power(base_p: &BigUint, digits: usize) -> Result<BigUint, NoiseSimulationError> {
    let digits = u32::try_from(digits).map_err(|_| NoiseSimulationError::IntegerConversion)?;
    Ok(base_p.pow(digits))
}

fn signed_difference(left: &BigUint, right: &BigUint) -> BigInt {
    BigInt::from_biguint(Sign::Plus, left.clone()) - BigInt::from_biguint(Sign::Plus, right.clone())
}

fn validate_crt_factorization(
    full_modulus: &BigUint,
    plaintext_moduli: &[BigUint],
) -> Result<(), NoiseSimulationError> {
    if plaintext_moduli.is_empty() {
        return Err(NoiseSimulationError::InvalidCrtFactorization);
    }
    let mut product = BigUint::one();
    for (index, modulus) in plaintext_moduli.iter().enumerate() {
        if modulus <= &BigUint::one() || full_modulus % modulus != BigUint::zero() {
            if modulus <= &BigUint::one() {
                return Err(NoiseSimulationError::InvalidCrtFactorization);
            }
            return Err(NoiseSimulationError::InvalidCrtDivision);
        }
        if plaintext_moduli[..index].iter().any(|previous| gcd(previous, modulus) != BigUint::one())
        {
            return Err(NoiseSimulationError::InvalidCrtFactorization);
        }
        product *= modulus;
    }
    if product != *full_modulus {
        return Err(NoiseSimulationError::InvalidCrtFactorization);
    }
    Ok(())
}

fn gcd(left: &BigUint, right: &BigUint) -> BigUint {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pbc::{
            PbcActiveCellIndex, PbcParameters, PbcPublicLayout, PbcRootSeed, derive_attempt_seed,
        },
        program::PowerLutProgramBuilder,
    };

    fn parameters() -> PowerLutNoiseParameters {
        PowerLutNoiseParameters::dense(8, BigUint::from(4u8), 3, BigUint::from(2u8)).unwrap()
    }

    #[test]
    fn average_transfers_derive_l2_squared_gains_from_structural_model() {
        let parameters = parameters();
        let transfer = average_fixed_lut_transfer(&parameters, 2).unwrap();
        assert_eq!(transfer.gain_sq, BigUint::from(2u8) * parameters.gamma_c_l2_sq());
    }

    #[test]
    fn average_variance_is_reduced_and_stays_in_doubled_units() {
        let variance = AverageVariance::new(BigUint::from(8u8), BigUint::from(12u8)).unwrap();
        assert_eq!(
            variance,
            AverageVariance { numerator: BigUint::from(2u8), denominator: BigUint::from(3u8) }
        );
        // beta=2, ell=1 has doubled digit variance one exactly.
        let p = PowerLutNoiseParameters::dense(8, BigUint::from(2u8), 1, BigUint::one()).unwrap();
        assert_eq!(
            p.helper_doubled_variance(),
            AverageVariance::new(BigUint::one(), BigUint::one()).unwrap()
        );
    }

    #[test]
    fn average_tail_budget_is_exact_and_acceptance_is_strict() {
        let config = AverageCaseConfig {
            failure_exponent: 10,
            input_domain_log2: 2,
            extra_event_log2: 3,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        let events = AverageEventBudget {
            input_domain_log2: 2,
            coefficient_log2: 1,
            slot_log2: 2,
            inspection_event_log2: 3,
        };
        assert_eq!(events.log2_events().unwrap(), 8);
        assert_eq!(
            config.z_squared(events).unwrap(),
            AverageVariance::new(
                BigUint::from(2u8) * BigUint::from(693_148u32) * BigUint::from(19u8),
                BigUint::from(1_000_000u32),
            )
            .unwrap()
        );
        let variance = AverageVariance::new(BigUint::one(), BigUint::one()).unwrap();
        assert!(
            !average_refresh_accepts(
                &BigUint::from(2u8),
                &BigUint::one(),
                &BigUint::one(),
                &BigUint::one(),
                0,
                &variance
            )
            .unwrap()
        );
        assert_eq!(
            average_refresh_accepts(
                &BigUint::from(2u8),
                &BigUint::one(),
                &BigUint::one(),
                &BigUint::one(),
                u32::MAX,
                &variance,
            ),
            Err(NoiseSimulationError::IntegerConversion)
        );
    }

    #[test]
    fn average_favg_uses_exact_ceil_sqrt_and_zero_boundary() {
        assert_eq!(ceil_sqrt(&BigUint::zero()).unwrap(), 0u8.into());
        assert_eq!(ceil_sqrt(&BigUint::one()).unwrap(), 1u8.into());
        assert_eq!(ceil_sqrt(&BigUint::from(2u8)).unwrap(), 2u8.into());
        assert_eq!(ceil_sqrt(&BigUint::from(3u8)).unwrap(), 2u8.into());
        assert_eq!(ceil_sqrt(&BigUint::from(4u8)).unwrap(), 2u8.into());
        let z_sq = AverageVariance::new(1u8.into(), 1u8.into()).unwrap();
        let variance = AverageVariance::new(16u8.into(), 1u8.into()).unwrap();
        assert_eq!(average_favg(&z_sq, &AverageVariance::zero(), 0).unwrap(), 0u8.into());
        assert_eq!(average_favg(&z_sq, &variance, 0).unwrap(), 2u8.into());
        assert_eq!(average_favg(&z_sq, &variance, 1).unwrap(), 4u8.into());
    }

    #[test]
    fn average_tail_correction_overflow_fails_closed() {
        let z_sq = AverageVariance::new(1u8.into(), 1u8.into()).unwrap();
        let variance = AverageVariance::new(1u8.into(), 1u8.into()).unwrap();
        assert_eq!(
            average_favg(&z_sq, &variance, u32::MAX),
            Err(NoiseSimulationError::IntegerConversion)
        );
    }

    #[test]
    fn average_smudging_equality_is_accepted() {
        let required =
            average_smudging_required(3, &BigUint::from(2u8), &BigUint::from(4u8)).unwrap();
        assert_eq!(required, BigUint::from(128u8));
        assert!(signed_difference(&required, &required) >= BigInt::zero());
    }

    #[test]
    fn average_snapshot_pairs_structural_setup() {
        let pbc_parameters = PbcParameters::paper_evaluation(10, 4);
        let layout = PbcPublicLayout::build(
            &pbc_parameters,
            derive_attempt_seed(PbcRootSeed([12u8; 32]), 0),
            0,
        )
        .unwrap();
        let profile = crate::prf::SparseLwrPrfProfile::new(2, 2, 8, 8).unwrap();
        let program = crate::prf::SparseLwrPrfProgram::new(
            profile,
            layout.bucket_width,
            layout.parameters.bucket_count,
        )
        .unwrap();
        let model = parameters();
        let snapshot = PowerLutNoiseSnapshot::from_setup(
            [9u8; 32],
            program,
            layout,
            model,
            BigUint::from(60u8),
            BigUint::from(2u8),
            3,
            2,
            0,
            3,
            2,
            vec![BigUint::from(3u8), BigUint::from(4u8), BigUint::from(5u8)],
            BigUint::from(5u8),
            BigUint::zero(),
            AverageVariance::new(BigUint::from(4u8), BigUint::one()).unwrap(),
        )
        .unwrap();
        let report = snapshot
            .simulate_average(&AverageCaseConfig {
                allow_average_acceptance: true,
                ..AverageCaseConfig::default()
            })
            .unwrap();
        assert_eq!(
            snapshot.initial_average_variance(),
            &AverageVariance::new(BigUint::from(4u8), BigUint::one()).unwrap()
        );
        assert!(report.refresh.centered_coordinates);
        assert_eq!(report.refresh.mask_event_count, BigUint::from(108u8));
        assert_eq!(report.refresh.fresh_event_count, BigUint::from(24u8));
        assert_eq!(report.refresh.joint_event_count, BigUint::from(132u8));
        assert!(report.refresh.slots.iter().all(|slot| slot.z_sq == report.refresh.z_sq));
        assert_eq!(report.refresh.decoder_factor, BigUint::from(2u8));
        assert_eq!(report.security_authority, "harness");
        assert_eq!(report.correctness_authority, NoiseModelKind::AverageCase);
        assert_eq!(report.accepted, report.hard_authority_accepted && report.correctness_accepted);
    }

    #[test]
    fn average_refresh_rejects_non_binary_base_without_structural_centering() {
        let pbc_parameters = PbcParameters::paper_evaluation(10, 4);
        let layout = PbcPublicLayout::build(
            &pbc_parameters,
            derive_attempt_seed(PbcRootSeed([13u8; 32]), 0),
            0,
        )
        .unwrap();
        let profile = crate::prf::SparseLwrPrfProfile::new(2, 2, 8, 8).unwrap();
        let program = crate::prf::SparseLwrPrfProgram::new(
            profile,
            layout.bucket_width,
            layout.parameters.bucket_count,
        )
        .unwrap();
        let snapshot = PowerLutNoiseSnapshot::from_setup(
            [10u8; 32],
            program,
            layout,
            parameters(),
            BigUint::from(60u8),
            BigUint::from(2u8),
            3,
            2,
            0,
            3,
            2,
            vec![BigUint::from(3u8), BigUint::from(4u8), BigUint::from(5u8)],
            BigUint::from(5u8),
            BigUint::zero(),
            AverageVariance::zero(),
        )
        .unwrap();
        let mut refresh = snapshot.refresh.clone();
        refresh.base_p = BigUint::from(4u8);
        assert_eq!(
            simulate_average_refresh(
                &snapshot.model,
                &refresh,
                AverageVariance::zero(),
                AverageVariance::zero(),
                &AverageCaseConfig::default(),
            ),
            Err(NoiseSimulationError::UnsupportedAverageRefresh)
        );
    }

    #[test]
    fn dense_gain_is_regular_two_component_bound() {
        let p = parameters();
        // 2 * ell * n * Delta = 2 * 3 * 8 * 2.
        assert_eq!(p.regular_dense_gain(), BigUint::from(96u8));
        assert_eq!(p.delta(), BigUint::from(2u8));
    }

    #[test]
    fn non_power_of_two_base_is_rejected() {
        assert_eq!(
            PowerLutNoiseParameters::dense(8, BigUint::from(3u8), 3, BigUint::from(2u8)),
            Err(NoiseSimulationError::InvalidGadgetBase)
        );
    }

    #[test]
    fn balanced_digits_use_tie_to_even_for_beta_two_four_and_eight() {
        let cases = [
            (BigUint::from(2u8), BigUint::from(1u8), vec![BigInt::from(1)]),
            (
                BigUint::from(2u8),
                BigUint::from(3u8),
                vec![BigInt::from(-1), BigInt::from(0), BigInt::from(1)],
            ),
            (BigUint::from(2u8), BigUint::from(16u8), vec![BigInt::from(-1)]),
            (BigUint::from(4u8), BigUint::from(2u8), vec![BigInt::from(2)]),
            (BigUint::from(4u8), BigUint::from(6u8), vec![BigInt::from(-2), BigInt::from(2)]),
            (BigUint::from(4u8), BigUint::from(15u8), vec![BigInt::from(-2)]),
            (BigUint::from(8u8), BigUint::from(4u8), vec![BigInt::from(4)]),
            (BigUint::from(8u8), BigUint::from(13u8), vec![BigInt::from(-4)]),
        ];
        for (beta, value, expected) in cases {
            assert_eq!(
                balanced_digits_for_tower(&value, &BigUint::from(17u8), &beta, expected.len())
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn refresh_action_gains_are_target_aware_and_not_dense_scaled() {
        let gains = refresh_action_gains(
            &BigUint::from(15u8),
            &[BigUint::from(3u8), BigUint::from(5u8)],
            &BigUint::from(2u8),
            2,
            2,
            &BigUint::from(4u8),
            2,
            1,
        )
        .unwrap();
        assert_eq!(gains.mask_digit_gains, vec![BigUint::from(2u8), BigUint::from(3u8)]);
        assert_eq!(gains.mask_digit_l2_sq_gains, vec![BigUint::from(2u8), BigUint::from(5u8)]);
        assert_eq!(gains.mask_route_gain, BigUint::from(5u8));
        assert_eq!(gains.mask_route_l2_sq_gain, BigUint::from(7u8));
        assert_eq!(gains.gamma_kappa, vec![BigUint::one(), BigUint::from(2u8)]);
        assert_eq!(gains.gamma_kappa_l2_sq, vec![BigUint::one(), BigUint::from(4u8)]);
        assert_eq!(gains.fresh_error_route_gains, vec![BigUint::from(2u8), BigUint::from(3u8)]);
        assert_eq!(
            gains.fresh_error_route_l2_sq_gains,
            vec![BigUint::from(2u8), BigUint::from(5u8)]
        );
        assert_eq!(gains.fresh_error_digit_gains[0], vec![BigUint::one(), BigUint::one()]);
        assert_eq!(gains.fresh_error_digit_gains[1], vec![BigUint::from(2u8), BigUint::one()]);
        assert_eq!(gains.fresh_error_digit_l2_sq_gains[0], vec![BigUint::one(), BigUint::one()]);
        assert_eq!(
            gains.fresh_error_digit_l2_sq_gains[1],
            vec![BigUint::from(4u8), BigUint::one()]
        );
    }

    #[test]
    fn program_simulation_covers_unary_binary_and_fuse() {
        let mut builder = PowerLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let unary_lut =
            builder.lut(crate::program::LutTable::unary(2, 8, vec![0, 1]).unwrap()).unwrap();
        let binary_lut = builder
            .lut(crate::program::LutTable::binary(2, 2, 8, vec![0, 1, 2, 3]).unwrap())
            .unwrap();
        let unary = builder.unary(builder.input_wire(input).unwrap(), unary_lut).unwrap();
        let binary = builder.binary(builder.input_wire(input).unwrap(), rhs, binary_lut).unwrap();
        builder.output(unary).unwrap();
        builder.output(binary).unwrap();
        let program = builder.build().unwrap();
        let mut inputs = ProgramNoiseInputs::default();
        inputs.input_bounds.insert(input, BigUint::from(1u8));
        let report = simulate_program(&program, &parameters(), &inputs).unwrap();
        assert_eq!(report.steps.len(), 2);
        assert_eq!(report.steps[1].transfer.gain, BigUint::from(384u32) * BigUint::from(96u32));
        assert_eq!(
            report.steps[1].transfer.additive,
            BigUint::from(384u32) * BigUint::from(2u8) +
                BigUint::from(4u8) * BigUint::from(97u32) * BigUint::from(2u8)
        );
    }

    #[test]
    fn standalone_transfers_are_exact_and_explicit() {
        let parameters = parameters();
        let fuse = fixed_fuse_transfer(&parameters, BigUint::from(3u8));
        assert_eq!(fuse.gain, BigUint::from(96u8));
        assert_eq!(fuse.additive, BigUint::from(6u8));
        let lut = fixed_lut_transfer(&parameters, 4).unwrap();
        assert_eq!(lut.gain, BigUint::from(384u16));
        assert_eq!(lut.additive, BigUint::from(776u16));
        let two_input = two_input_lut_transfer(&parameters, 4, BigUint::from(3u8)).unwrap();
        assert_eq!(two_input.gain, BigUint::from(384u16) * BigUint::from(96u8));
        assert_eq!(
            two_input.additive,
            BigUint::from(384u16) * BigUint::from(6u8) + BigUint::from(776u16)
        );
        let monomial = monomial_one_hot_transfer(&parameters, 2, 4).unwrap();
        assert_eq!(monomial.gain, BigUint::from(384u16) * BigUint::from(192u8));
        assert_eq!(monomial.additive, BigUint::from(2312u16));
    }

    #[test]
    fn secret_dimension_is_checked_by_simulation() {
        let mut parameters = parameters();
        parameters.secret_dimension = 1;
        let mut builder = PowerLutProgramBuilder::new();
        let input = builder.input(2).unwrap();
        let lut = builder.lut(crate::program::LutTable::unary(2, 4, vec![0, 1]).unwrap()).unwrap();
        let output = builder.unary(builder.input_wire(input).unwrap(), lut).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut inputs = ProgramNoiseInputs::default();
        inputs.input_bounds.insert(input, BigUint::zero());
        assert_eq!(
            simulate_program(&program, &parameters, &inputs),
            Err(NoiseSimulationError::InvalidSecretDimension)
        );
    }

    #[test]
    fn active_widths_are_derived_from_public_layout_and_exclude_padding() {
        let parameters = PbcParameters::paper_evaluation(10, 4);
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(PbcRootSeed([7u8; 32]), 0), 0)
                .unwrap();
        let active = PbcActiveCellIndex::build(&layout).unwrap();
        assert!(
            active
                .bucket_active_widths()
                .zip(layout.cells.iter())
                .any(|(width, row)| width < row.len())
        );
    }

    #[test]
    fn sparse_prf_uses_layout_active_widths_internally() {
        let pbc_parameters = PbcParameters::paper_evaluation(10, 4);
        let layout = PbcPublicLayout::build(
            &pbc_parameters,
            derive_attempt_seed(PbcRootSeed([8u8; 32]), 0),
            0,
        )
        .unwrap();
        let profile = crate::prf::SparseLwrPrfProfile::new(2, 2, 8, 8).unwrap();
        let program = crate::prf::SparseLwrPrfProgram::new(
            profile,
            layout.bucket_width,
            layout.parameters.bucket_count,
        )
        .unwrap();
        let report =
            simulate_sparse_prf(&program, &parameters(), &layout, BigUint::zero()).unwrap();
        assert_eq!(report.bucket_bounds.len(), layout.parameters.bucket_count);
        assert_eq!(report.bucket_bounds.len(), layout.cells.len());
        assert_eq!(report.bucket_stages.len(), report.bucket_bounds.len());
        assert!(report.bucket_stages.iter().enumerate().all(|(index, stage)| {
            stage.bucket == index &&
                stage.one_hot_output_bound <= stage.lut_output_bound &&
                stage.active_count <= layout.bucket_width &&
                stage.selection_inherited_bits == stage.selection_inherited_bound.bits() as usize &&
                stage.selection_additive_bits == stage.selection_additive_bound.bits() as usize
        }));
        assert!(report.group_stages.iter().all(|group| {
            group.additive_bound == group.base_helper_additive.clone() + &group.gamma_a_additive &&
                group.output_bound == group.inherited_bound.clone() + &group.additive_bound
        }));
        assert_eq!(
            report.terminal_additive_bound,
            report.terminal_base_helper_additive.clone() + &report.terminal_gamma_a_additive
        );
        assert_eq!(report.terminal_additive_bits, report.terminal_additive_bound.bits() as usize);
        assert_eq!(report.terminal_inherited_bits, report.terminal_inherited_bound.bits() as usize);
        assert_eq!(report.terminal_lut_width, report.lut_width);
        assert_eq!(report.terminal_gamma_c, parameters().gamma_c().clone());
        assert_eq!(report.terminal_gamma_a, parameters().gamma_a().clone());
    }

    #[test]
    fn structural_refresh_route_scale_and_decoder_gains_are_exact() {
        let model = parameters();
        let refresh = RefreshNoiseParameters::from_structural(
            &model,
            BigUint::from(60u8),
            BigUint::from(2u8),
            BigUint::from(8u8),
            3,
            2,
            0,
            3,
            2,
            vec![BigUint::from(3u8), BigUint::from(4u8), BigUint::from(5u8)],
            BigUint::from(5u8),
        )
        .unwrap();
        assert_eq!(refresh.mask_route_gain, BigUint::from(20u8));
        assert_eq!(
            refresh.mask_digit_gains,
            vec![BigUint::from(3u8), BigUint::from(5u8), BigUint::from(2u8)]
        );
        assert_eq!(refresh.slots[0].gamma_kappa, BigUint::one());
        assert_eq!(refresh.slots[1].gamma_kappa, BigUint::one());
        assert_eq!(refresh.slots[2].gamma_kappa, BigUint::from(2u8));
        assert_eq!(refresh.slots[0].fresh_error_route_gain, BigUint::from(4u8));
        assert_eq!(refresh.slots[1].fresh_error_route_gain, BigUint::from(6u8));
        assert_eq!(refresh.slots[2].fresh_error_route_gain, BigUint::from(6u8));
        assert_eq!(refresh.component_columns, 6);
        assert_eq!(refresh.slots[0].kappa, BigUint::from(20u8));
        assert_eq!(refresh.slots[1].kappa, BigUint::from(15u8));
        assert_eq!(refresh.slots[2].kappa, BigUint::from(12u8));
        let report = simulate_refresh(&model, &refresh, BigUint::zero(), BigUint::zero()).unwrap();
        assert_eq!(report.exposed_transcript_coordinates, BigUint::from(36u8));
        assert_eq!(report.slots[0].operation_noise_bound, BigUint::from(800u16));
        assert_eq!(report.mask_statistical_required, BigUint::from(57_600u32));
    }

    #[test]
    fn generic_program_simulation_rejects_one_hot() {
        let mut builder = PowerLutProgramBuilder::new();
        let input = builder.input(4).unwrap();
        let selector = builder.rhs_family(4).unwrap();
        let values = builder.public_value_family(4).unwrap();
        let output =
            builder.one_hot_select(builder.input_wire(input).unwrap(), selector, values).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let mut inputs = ProgramNoiseInputs::default();
        inputs.input_bounds.insert(input, BigUint::from(3u8));
        inputs.one_hot_active_counts.insert(0, 2);
        assert_eq!(
            simulate_program(&program, &parameters(), &inputs),
            Err(NoiseSimulationError::UnsupportedOneHot)
        );
    }

    #[test]
    fn refresh_equality_is_rejected_and_success_resets() {
        let p = PowerLutNoiseParameters::dense(1, BigUint::from(4u8), 14, BigUint::one()).unwrap();
        // The largest CRT factor is chosen so its spacing is exactly the
        // strict rounding boundary. Other factors have strictly larger
        // spacing, so the rejection is attributable to equality itself.
        let pre = BigUint::from(8_254u32);
        let equality_factor = BigUint::from(8_255u32);
        let q = &pre * &equality_factor;
        let refresh = RefreshNoiseParameters::from_structural(
            &p,
            q,
            BigUint::from(2u8),
            BigUint::from(2u8),
            12,
            1,
            0,
            2,
            1,
            vec![pre.clone(), equality_factor],
            BigUint::one(),
        )
        .unwrap();
        let report = simulate_refresh(&p, &refresh, BigUint::zero(), BigUint::zero()).unwrap();
        assert!(!report.accepted);
        assert!(!report.slots[1].accepted);
        assert_eq!(report.slots[1].rounding_margin, BigInt::zero());

        let success_factor = BigUint::from(8_256u32);
        let success_spacing_factor = BigUint::from(8_255u32);
        let q = &success_spacing_factor * &success_factor;
        let refresh = RefreshNoiseParameters::from_structural(
            &p,
            q,
            BigUint::from(2u8),
            BigUint::from(2u8),
            12,
            1,
            0,
            2,
            1,
            vec![success_spacing_factor, success_factor],
            BigUint::one(),
        )
        .unwrap();
        let report = simulate_refresh(&p, &refresh, BigUint::zero(), BigUint::zero()).unwrap();
        assert!(report.accepted);
        assert_eq!(report.slots.len(), 2);
        assert!(report.slots.iter().all(|slot| slot.accepted));
        assert_eq!(report.slots[1].twice_pre_rounding_bound, BigUint::from(8_254u32));
        assert_eq!(report.slots[1].spacing, BigUint::from(8_255u32));
        assert_eq!(report.refreshed_error_bound, Some(BigUint::from(1u8)));
        assert_eq!(report.fresh_error_bound, BigUint::from(1u8));
    }

    #[test]
    fn mask_domain_boundary_is_exact() {
        let p = parameters();
        let build = |q_l| {
            RefreshNoiseParameters::from_structural(
                &p,
                BigUint::from(60u8),
                BigUint::from(3u8),
                q_l,
                2,
                1,
                0,
                3,
                1,
                vec![BigUint::from(3u8), BigUint::from(4u8), BigUint::from(5u8)],
                BigUint::one(),
            )
            .unwrap()
        };
        let equal =
            simulate_refresh(&p, &build(BigUint::from(9u8)), BigUint::zero(), BigUint::zero())
                .unwrap();
        assert!(equal.mask_domain_accepted);
        assert_eq!(equal.mask_domain_margin, BigInt::zero());
        let below =
            simulate_refresh(&p, &build(BigUint::from(10u8)), BigUint::zero(), BigUint::zero())
                .unwrap();
        assert!(!below.mask_domain_accepted);
        assert!(!below.hard_authority.accepted);
        assert_eq!(below.mask_domain_margin, BigInt::from(-1));
    }

    #[test]
    fn fresh_error_boundary_is_strict_per_crt_modulus() {
        let p =
            PowerLutNoiseParameters::dense(8, BigUint::from(4u8), 2, BigUint::from(2u8)).unwrap();
        let refresh = RefreshNoiseParameters::from_structural(
            &p,
            BigUint::from(6u8),
            BigUint::from(3u8),
            BigUint::from(8u8),
            2,
            1,
            0,
            2,
            1,
            vec![BigUint::from(2u8), BigUint::from(3u8)],
            BigUint::one(),
        )
        .unwrap();
        let report = simulate_refresh(&p, &refresh, BigUint::zero(), BigUint::zero()).unwrap();
        // B_e = 3^1 - 1 = 2: equality at q_0=2 rejects, while q_1=3
        // satisfies the exact one-below condition B_e=q_1-1.
        assert_eq!(report.fresh_error_bound, BigUint::from(2u8));
        assert!(!report.slots[0].fresh_error_below_plaintext_modulus);
        assert!(!report.fresh_error_accepted);
        assert!(!report.hard_authority.accepted);
        assert_eq!(report.slots[0].fresh_error_margin, BigInt::zero());
        assert!(report.slots[1].fresh_error_below_plaintext_modulus);
        assert_eq!(report.slots[1].fresh_error_margin, BigInt::one());
    }

    #[test]
    fn mask_hiding_equality_is_accepted_exactly() {
        // ell_beta=2, two CRT mask slots, and one coefficient give D=8.
        // With the target-aware gamma values, E_state=24 and E_decoder=8,
        // max(F_t)=32. For lambda=0 the exact required mask modulus is
        // 2^(0+1)*8*32 = 512 = 2^9.
        let model =
            PowerLutNoiseParameters::dense(1, BigUint::from(2u8), 2, BigUint::one()).unwrap();
        let refresh = RefreshNoiseParameters::from_structural(
            &model,
            BigUint::from(6u8),
            BigUint::from(2u8),
            BigUint::from(2u8),
            9,
            1,
            0,
            2,
            1,
            vec![BigUint::from(2u8), BigUint::from(3u8)],
            BigUint::one(),
        )
        .unwrap();
        let report =
            simulate_refresh(&model, &refresh, BigUint::from(24u8), BigUint::zero()).unwrap();
        assert_eq!(report.exposed_transcript_coordinates, BigUint::from(8u8));
        assert_eq!(report.mask_statistical_required, BigUint::from(512u16));
        assert_eq!(report.mask_modulus, BigUint::from(512u16));
        assert_eq!(report.mask_statistical_margin, BigInt::zero());
        assert!(report.mask_statistical_accepted);
        assert!(report.hard_authority.accepted);
    }
}
