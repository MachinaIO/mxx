//! Noise interpretation of an [`ExponentLutProgram`](crate::program::ExponentLutProgram).
//!
//! This module is independent of sparse-LWR and noise refresh.  It interprets
//! the same four primitive operations as the public-key and encoding backends,
//! but its wire value is an exact worst-case coefficient bound together with
//! an exact rational doubled-coordinate variance.

use std::collections::BTreeMap;

use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::program::{
    ExponentLutProgram, ExponentProgramLowering, FamilyRange, FamilyRhsSource, FamilyWireSource,
    Gate, LutId, ProgramBindings, ProgramFamilyBindings, ProgramGateInstance, ProgramInputId,
    ProgramLowerError, ProgramLoweringTypes, ProgramStep, RhsFamilyId, RhsId, WireFamilyId, WireId,
    lower_program,
};

/// Exact rational value used for doubled-coordinate variances.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AverageVariance {
    pub numerator: BigUint,
    pub denominator: BigUint,
}

impl AverageVariance {
    pub fn new(numerator: BigUint, denominator: BigUint) -> Result<Self, ProgramNoiseError> {
        if denominator.is_zero() {
            return Err(ProgramNoiseError::IntegerConversion);
        }
        let divisor = gcd(&numerator, &denominator);
        Ok(Self { numerator: numerator / &divisor, denominator: denominator / divisor })
    }

    pub fn zero() -> Self {
        Self { numerator: BigUint::zero(), denominator: BigUint::one() }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            &self.numerator * &other.denominator + &other.numerator * &self.denominator,
            &self.denominator * &other.denominator,
        )
        .expect("positive variance denominator")
    }

    pub fn scaled(&self, gain: &BigUint) -> Self {
        Self::new(&self.numerator * gain, self.denominator.clone())
            .expect("positive variance denominator")
    }
}

/// Coefficient norms of one matrix action on a noise row.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RightActionNoiseGain {
    pub l1: BigUint,
    pub l2_squared: BigUint,
}

impl RightActionNoiseGain {
    pub fn unit() -> Self {
        Self { l1: BigUint::one(), l2_squared: BigUint::one() }
    }
}

/// The source of a balanced decomposition used to derive right-action gains.
///
/// `Explicit` is retained for small, already-materialized decomposition
/// fixtures.  Production setup should use one of the compact variants: they
/// retain the target shape and evaluate the same balanced base decomposition
/// without allocating a `(digit_count * rows * columns * N)` tensor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseDecompositionSource {
    /// Explicit balanced coefficients in
    /// `[decomposed_row][output_column][ring_coefficient]` order.
    Explicit(Vec<BigInt>),
    /// One non-zero target ring coefficient.  `value` is decomposed with the
    /// same balanced power-of-two algorithm as the matrix implementation.
    SparseMonomial {
        target_row: usize,
        output_column: usize,
        ring_coefficient: usize,
        value: BigInt,
    },
    /// A unit target coefficient at the selected position.
    Unit { target_row: usize, output_column: usize, ring_coefficient: usize },
    /// A diagonal matrix with unit ring coefficients.
    DiagonalUnit,
    /// One target coefficient equal to `base^exponent`.
    /// The unit action gain here describes only the public matrix action; an
    /// online sampled encoding's own error remains a separate program input.
    PowerOfBase {
        target_row: usize,
        output_column: usize,
        ring_coefficient: usize,
        exponent: usize,
    },
    /// A scalar times the canonical gadget matrix.  This is the compact
    /// representation of deterministic `kappa_t G` targets used by refresh;
    /// its gain is evaluated column-wise without materializing the gadget.
    GadgetScalar { value: BigInt },
    /// A sampled bounded matrix whose entries are consumed directly (rather
    /// than as a gadget decomposition).  The bound is the enforced typed
    /// preimage bound, not a modulus-derived proxy.
    BoundedMatrix { coefficient_bound: BigUint },
    /// Unknown/dense target whose source coefficients have absolute value at
    /// most `coefficient_bound`.  Per-digit bounds are derived from the base,
    /// rather than treating the full modulus as a scalar gain.
    DenseBound { coefficient_bound: BigUint },
}

/// Balanced decomposition data used to derive right-action gains.
///
/// The shape is always retained, including the ring and output-column axes;
/// this prevents a compact gain from being paired with a different action.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseDecompositionPlan {
    pub base: BigUint,
    pub digit_count: usize,
    /// Number of coefficients in each ring element of the target matrix.
    pub ring_dimension: usize,
    pub source: NoiseDecompositionSource,
    /// CRT tower moduli used by compact target decomposition.  An empty list
    /// is the legacy one-tower form used only by explicit small fixtures.
    pub crt_moduli: Vec<BigUint>,
    /// Balanced digits emitted by each CRT tower.
    pub digits_per_tower: usize,
    pub target_rows: usize,
    pub target_columns: usize,
    /// Number of rows after all gadget digits are expanded.
    pub decomposed_rows: usize,
}

impl NoiseDecompositionPlan {
    pub fn new(
        base: BigUint,
        digit_count: usize,
        balanced_coefficients: Vec<BigInt>,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
    ) -> Result<Self, ProgramNoiseError> {
        if balanced_coefficients.is_empty() ||
            base < BigUint::from(2u8) ||
            digit_count == 0 ||
            ring_dimension == 0 ||
            target_rows == 0 ||
            target_columns == 0
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        let decomposed_rows =
            target_rows.checked_mul(digit_count).ok_or(ProgramNoiseError::InvalidDecomposition)?;
        let expected = decomposed_rows
            .checked_mul(target_columns)
            .and_then(|value| value.checked_mul(ring_dimension))
            .ok_or(ProgramNoiseError::InvalidDecomposition)?;
        if balanced_coefficients.len() != expected {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        let parameters = Self {
            base,
            digit_count,
            ring_dimension,
            source: NoiseDecompositionSource::Explicit(balanced_coefficients),
            crt_moduli: Vec::new(),
            digits_per_tower: digit_count,
            target_rows,
            target_columns,
            decomposed_rows,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    /// Creates a compact plan for one sparse target coefficient.
    pub fn sparse_monomial(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        target_row: usize,
        output_column: usize,
        ring_coefficient: usize,
        value: BigInt,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::SparseMonomial {
                target_row,
                output_column,
                ring_coefficient,
                value,
            },
        )
    }

    /// Creates a compact plan for a unit target coefficient.
    pub fn unit(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        target_row: usize,
        output_column: usize,
        ring_coefficient: usize,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::Unit { target_row, output_column, ring_coefficient },
        )
    }

    /// Creates a compact plan for a square unit diagonal target.
    pub fn diagonal_unit(
        base: BigUint,
        digit_count: usize,
        size: usize,
        ring_dimension: usize,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            size,
            size,
            ring_dimension,
            NoiseDecompositionSource::DiagonalUnit,
        )
    }

    /// Creates a compact plan for one power-of-base target coefficient.
    pub fn power_of_base(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        target_row: usize,
        output_column: usize,
        ring_coefficient: usize,
        exponent: usize,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::PowerOfBase {
                target_row,
                output_column,
                ring_coefficient,
                exponent,
            },
        )
    }

    /// Creates a compact plan for a scalar multiple of the canonical gadget.
    pub fn gadget_scalar(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        value: BigInt,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::GadgetScalar { value },
        )
    }

    /// Creates a compact plan for a directly consumed bounded matrix.
    pub fn bounded_matrix(
        base: BigUint,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        coefficient_bound: BigUint,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            1,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::BoundedMatrix { coefficient_bound },
        )
    }

    /// Creates a compact conservative plan for an unknown dense target.
    pub fn dense_bound(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        coefficient_bound: BigUint,
    ) -> Result<Self, ProgramNoiseError> {
        Self::from_source(
            base,
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            NoiseDecompositionSource::DenseBound { coefficient_bound },
        )
    }

    fn from_source(
        base: BigUint,
        digit_count: usize,
        target_rows: usize,
        target_columns: usize,
        ring_dimension: usize,
        source: NoiseDecompositionSource,
    ) -> Result<Self, ProgramNoiseError> {
        let decomposed_rows =
            target_rows.checked_mul(digit_count).ok_or(ProgramNoiseError::InvalidDecomposition)?;
        let parameters = Self {
            base,
            digit_count,
            ring_dimension,
            source,
            crt_moduli: Vec::new(),
            digits_per_tower: digit_count,
            target_rows,
            target_columns,
            decomposed_rows,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    /// Binds the actual CRT towers used by a compact decomposition.  Each
    /// tower is decomposed independently with `digits_per_tower` digits.
    pub fn bind_crt_layout(
        mut self,
        crt_moduli: Vec<BigUint>,
        digits_per_tower: usize,
    ) -> Result<Self, ProgramNoiseError> {
        if crt_moduli.is_empty() ||
            digits_per_tower == 0 ||
            crt_moduli.iter().any(|modulus| modulus <= &BigUint::one()) ||
            crt_moduli.len().checked_mul(digits_per_tower) != Some(self.digit_count)
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        self.crt_moduli = crt_moduli;
        self.digits_per_tower = digits_per_tower;
        self.validate()?;
        Ok(self)
    }

    fn tower_moduli(&self) -> Vec<Option<&BigUint>> {
        if self.crt_moduli.is_empty() {
            vec![None]
        } else {
            self.crt_moduli.iter().map(Some).collect()
        }
    }

    fn sparse_source_valid(&self, value: &BigInt) -> bool {
        self.tower_moduli().into_iter().all(|modulus| {
            let tower_value =
                modulus.map_or_else(|| value.clone(), |modulus| centered_residue(value, modulus));
            balanced_profile(&tower_value, &self.base, self.digits_per_tower).is_ok()
        })
    }

    /// Computes exact right-action gains from the balanced coefficients.
    ///
    /// A row vector multiplied on the right accumulates into output columns,
    /// therefore each output-column norm includes every digit-expanded row
    /// and every coefficient of its ring polynomial.  Coefficients use the
    /// canonical compact order `[decomposed_row][column][ring_coefficient]`.
    pub fn action_gain(&self) -> RightActionNoiseGain {
        match &self.source {
            NoiseDecompositionSource::Explicit(coefficients) => {
                let mut column_l1 = vec![BigUint::zero(); self.target_columns];
                let mut column_l2_squared = vec![BigUint::zero(); self.target_columns];
                for row in 0..self.decomposed_rows {
                    for column in 0..self.target_columns {
                        let start = (row * self.target_columns + column) * self.ring_dimension;
                        let end = start + self.ring_dimension;
                        for coefficient in &coefficients[start..end] {
                            column_l1[column] += coefficient.magnitude();
                            column_l2_squared[column] += coefficient.magnitude().pow(2u32);
                        }
                    }
                }
                RightActionNoiseGain {
                    l1: column_l1.into_iter().max().unwrap_or_default(),
                    l2_squared: column_l2_squared.into_iter().max().unwrap_or_default(),
                }
            }
            NoiseDecompositionSource::SparseMonomial { output_column, value, .. } => {
                self.sparse_gain(value, *output_column)
            }
            NoiseDecompositionSource::Unit { output_column, .. } => {
                self.sparse_gain(&BigInt::one(), *output_column)
            }
            NoiseDecompositionSource::DiagonalUnit => self.sparse_gain(&BigInt::one(), 0),
            NoiseDecompositionSource::PowerOfBase { output_column, exponent, .. } => {
                debug_assert!(*exponent < self.digits_per_tower);
                let value = BigInt::from_biguint(
                    num_bigint::Sign::Plus,
                    self.base.pow(u32::try_from(*exponent).expect("validated exponent")),
                );
                self.sparse_gain(&value, *output_column)
            }
            NoiseDecompositionSource::GadgetScalar { value } => {
                let mut best =
                    RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() };
                // CRT-bound gadget layouts repeat the local digit axis in
                // each tower and target row.  Evaluate one representative of
                // each local digit class; scanning every repeated output
                // column is mathematically redundant and quadratic in the
                // effective modulus dimensions.
                for digit in 0..self.digits_per_tower {
                    let coefficient = value *
                        BigInt::from_biguint(
                            num_bigint::Sign::Plus,
                            self.base.pow(u32::try_from(digit).expect("validated gadget digit")),
                        );
                    let gain = self.sparse_gain(&coefficient, 0);
                    if gain.l1 > best.l1 {
                        best = gain;
                    }
                }
                best
            }
            NoiseDecompositionSource::BoundedMatrix { coefficient_bound } => {
                let count = BigUint::from(
                    self.target_rows
                        .checked_mul(self.ring_dimension)
                        .expect("validated bounded matrix shape"),
                );
                RightActionNoiseGain {
                    l1: coefficient_bound * &count,
                    l2_squared: coefficient_bound.pow(2u32) * count,
                }
            }
            NoiseDecompositionSource::DenseBound { coefficient_bound } => {
                let mut per_coefficient_l1 = BigUint::zero();
                let mut per_coefficient_l2_squared = BigUint::zero();
                for modulus in self.tower_moduli() {
                    let tower_bound = modulus.map_or_else(
                        || coefficient_bound.clone(),
                        |modulus| coefficient_bound.min(&(modulus / 2u8)).clone(),
                    );
                    let (l1, l2) = self.dense_coefficient_gain(&tower_bound, self.digits_per_tower);
                    per_coefficient_l1 += l1;
                    per_coefficient_l2_squared += l2;
                }
                let count = BigUint::from(
                    self.target_rows
                        .checked_mul(self.ring_dimension)
                        .expect("validated dense decomposition shape"),
                );
                RightActionNoiseGain {
                    l1: per_coefficient_l1 * &count,
                    l2_squared: per_coefficient_l2_squared * count,
                }
            }
        }
    }

    fn sparse_gain(&self, value: &BigInt, output_column: usize) -> RightActionNoiseGain {
        debug_assert!(output_column < self.target_columns);
        let mut l1 = BigUint::zero();
        let mut l2_squared = BigUint::zero();
        for modulus in self.tower_moduli() {
            let tower_value =
                modulus.map_or_else(|| value.clone(), |modulus| centered_residue(value, modulus));
            let (tower_l1, tower_l2_squared) =
                balanced_profile(&tower_value, &self.base, self.digits_per_tower)
                    .expect("validated compact decomposition must fit its digit count");
            l1 += tower_l1;
            l2_squared += tower_l2_squared;
        }
        RightActionNoiseGain { l1, l2_squared }
    }

    fn dense_coefficient_gain(
        &self,
        coefficient_bound: &BigUint,
        digit_count: usize,
    ) -> (BigUint, BigUint) {
        let half_base = &self.base >> 1usize;
        let mut l1 = BigUint::zero();
        let mut l2_squared = BigUint::zero();
        let mut state_bound = coefficient_bound.clone();
        for _ in 0..digit_count {
            let digit_bound = state_bound.clone().min(half_base.clone());
            l1 += &digit_bound;
            l2_squared += digit_bound.pow(2u32);
            state_bound = balanced_interval_next_bound(&state_bound, &self.base);
        }
        (l1, l2_squared)
    }

    pub(crate) fn validate(&self) -> Result<(), ProgramNoiseError> {
        if self.base < BigUint::from(2u8) ||
            !is_power_of_two(&self.base) ||
            self.digit_count == 0 ||
            self.ring_dimension == 0 ||
            self.target_rows == 0 ||
            self.target_columns == 0 ||
            self.decomposed_rows != self.target_rows.checked_mul(self.digit_count).unwrap_or(0) ||
            self.target_rows.checked_mul(self.ring_dimension).is_none() ||
            !match &self.source {
                NoiseDecompositionSource::Explicit(coefficients) => {
                    coefficients.len() ==
                        self.decomposed_rows
                            .checked_mul(self.target_columns)
                            .and_then(|value| value.checked_mul(self.ring_dimension))
                            .unwrap_or(0)
                }
                NoiseDecompositionSource::SparseMonomial {
                    target_row,
                    output_column,
                    ring_coefficient,
                    value,
                } => {
                    *target_row < self.target_rows &&
                        *output_column < self.target_columns &&
                        *ring_coefficient < self.ring_dimension &&
                        !value.is_zero() &&
                        self.sparse_source_valid(value)
                }
                NoiseDecompositionSource::Unit { target_row, output_column, ring_coefficient } => {
                    *target_row < self.target_rows &&
                        *output_column < self.target_columns &&
                        *ring_coefficient < self.ring_dimension
                }
                NoiseDecompositionSource::DiagonalUnit => self.target_rows == self.target_columns,
                NoiseDecompositionSource::PowerOfBase {
                    target_row,
                    output_column,
                    ring_coefficient,
                    exponent,
                } => {
                    *target_row < self.target_rows &&
                        *output_column < self.target_columns &&
                        *ring_coefficient < self.ring_dimension &&
                        *exponent < self.digits_per_tower &&
                        u32::try_from(*exponent).is_ok()
                }
                NoiseDecompositionSource::GadgetScalar { value } => {
                    Some(self.target_columns) == self.target_rows.checked_mul(self.digit_count) &&
                        !value.is_zero()
                }
                NoiseDecompositionSource::BoundedMatrix { coefficient_bound } => {
                    !coefficient_bound.is_zero()
                }
                NoiseDecompositionSource::DenseBound { coefficient_bound } => {
                    !coefficient_bound.is_zero() &&
                        self.tower_moduli().into_iter().all(|modulus| {
                            let tower_bound = modulus.map_or_else(
                                || coefficient_bound.clone(),
                                |modulus| coefficient_bound.min(&(modulus / 2u8)).clone(),
                            );
                            balanced_interval_representable(
                                &tower_bound,
                                &self.base,
                                self.digits_per_tower,
                            )
                        })
                }
            } && (self.crt_moduli.is_empty() && self.digits_per_tower == self.digit_count ||
                !self.crt_moduli.is_empty() &&
                    self.crt_moduli.len().checked_mul(self.digits_per_tower) ==
                        Some(self.digit_count) &&
                    self.crt_moduli.iter().all(|modulus| modulus > &BigUint::one()))
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        Ok(())
    }
}

/// Returns the exact L1 and squared-L2 contribution of one source
/// coefficient under the balanced power-of-two decomposition.
pub(crate) fn balanced_profile(
    value: &BigInt,
    base: &BigUint,
    digit_count: usize,
) -> Result<(BigUint, BigUint), ProgramNoiseError> {
    let base = BigInt::from_biguint(num_bigint::Sign::Plus, base.clone());
    let mut current = value.clone();
    let mut l1 = BigUint::zero();
    let mut l2_squared = BigUint::zero();
    for _ in 0..digit_count {
        let (digit, next) = balanced_digit_step(&current, &base);
        current = next;
        l1 += digit.magnitude();
        l2_squared += digit.magnitude().pow(2u32);
    }
    if !current.is_zero() {
        return Err(ProgramNoiseError::InvalidDecomposition);
    }
    Ok((l1, l2_squared))
}

pub(crate) fn centered_residue(value: &BigInt, modulus: &BigUint) -> BigInt {
    let modulus = BigInt::from_biguint(num_bigint::Sign::Plus, modulus.clone());
    let mut residue = value % &modulus;
    if residue.sign() == num_bigint::Sign::Minus {
        residue += &modulus;
    }
    if &residue * 2u8 > modulus { residue - modulus } else { residue }
}

/// Exact interval recurrence for the balanced round-to-even carry.  The
/// balanced next-state function is monotone, so its maximum absolute value on
/// `[-state_bound, state_bound]` is attained at an endpoint.
pub(crate) fn balanced_interval_next_bound(state_bound: &BigUint, base: &BigUint) -> BigUint {
    let bound = BigInt::from_biguint(num_bigint::Sign::Plus, state_bound.clone());
    let negative = -&bound;
    let base = BigInt::from_biguint(num_bigint::Sign::Plus, base.clone());
    let (_, next_positive) = balanced_digit_step(&bound, &base);
    let (_, next_negative) = balanced_digit_step(&negative, &base);
    next_positive.magnitude().max(next_negative.magnitude()).clone()
}

pub(crate) fn balanced_interval_representable(
    state_bound: &BigUint,
    base: &BigUint,
    digit_count: usize,
) -> bool {
    let mut bound = state_bound.clone();
    for _ in 0..digit_count {
        bound = balanced_interval_next_bound(&bound, base);
    }
    bound.is_zero()
}

pub(crate) fn balanced_digit_step(value: &BigInt, base: &BigInt) -> (BigInt, BigInt) {
    let mut quotient = value / base;
    let mut remainder = value % base;
    if remainder.sign() == num_bigint::Sign::Minus {
        remainder += base;
        quotient -= 1u8;
    }
    let half = base / 2u8;
    let carry = remainder > half || (remainder == half && !(&quotient % 2u8).is_zero());
    let digit = if carry { &remainder - base } else { remainder };
    (digit, quotient + u8::from(carry))
}

fn is_power_of_two(value: &BigUint) -> bool {
    !value.is_zero() && (value & (value - 1u8)).is_zero()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GateNoiseOperation {
    MulPrivate,
    MulPublic,
    Add,
    Lut,
}

/// Setup-derived balanced decompositions for one primitive gate. LUT actions
/// carry a second plan for their automorphism-helper error.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GateNoiseDecomposition {
    pub action: NoiseDecompositionPlan,
    pub helper_action: Option<NoiseDecompositionPlan>,
}

/// Noise parameters for one primitive operation occurrence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GateNoiseGains {
    pub step_index: usize,
    pub operation_index: usize,
    pub branch_index: usize,
    pub operation: GateNoiseOperation,
    pub gate: String,
    pub action: RightActionNoiseGain,
    pub helper_action: Option<RightActionNoiseGain>,
    pub inherited_bound: BigUint,
    pub fresh_bound: BigUint,
    pub output_bound: BigUint,
    pub inherited_variance: AverageVariance,
    pub fresh_variance: AverageVariance,
    pub output_variance: AverageVariance,
}

/// Gate-only gain plan consumed by every Exponent-LUT program simulation.
/// Readout and refresh routes are intentionally outside this type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseGainPlan {
    gates: Vec<GateNoiseGains>,
}

impl ProgramNoiseGainPlan {
    pub fn new(gates: Vec<GateNoiseGains>) -> Self {
        Self { gates }
    }

    pub fn gates(&self) -> &[GateNoiseGains] {
        &self.gates
    }

    pub fn topology_event_count(
        program: &ExponentLutProgram,
        family_active_counts: &[usize],
    ) -> Result<usize, ProgramNoiseError> {
        let mut family_index = 0usize;
        let mut count = 0usize;
        for step in program.steps() {
            let contribution = match step {
                ProgramStep::Gate(Gate::Lut { lut, .. }) => {
                    program.lut(*lut).ok_or(ProgramNoiseError::MissingLut)?.values().len()
                }
                ProgramStep::Gate(_) => 1,
                ProgramStep::Family { body, .. } => {
                    let active = *family_active_counts
                        .get(family_index)
                        .ok_or(ProgramNoiseError::InvalidDecomposition)?;
                    family_index += 1;
                    active
                        .checked_mul(family_body_action_count(program, body)?)
                        .and_then(|value| value.checked_add(1))
                        .ok_or(ProgramNoiseError::IntegerConversion)?
                }
            };
            count = count.checked_add(contribution).ok_or(ProgramNoiseError::IntegerConversion)?;
        }
        if family_index != family_active_counts.len() || count == 0 {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        Ok(count)
    }

    pub fn validate(
        &self,
        program: &ExponentLutProgram,
        family_active_counts: &[usize],
    ) -> Result<(), ProgramNoiseError> {
        let mut expected = 0usize;
        let mut family_index = 0usize;
        for (step_index, step) in program.steps().iter().enumerate() {
            let expected_operations = match step {
                ProgramStep::Gate(gate) => gate_noise_operations(program, gate)?,
                ProgramStep::Family { body, .. } => {
                    let active = *family_active_counts
                        .get(family_index)
                        .ok_or(ProgramNoiseError::InvalidDecomposition)?;
                    if active == 0 {
                        return Err(ProgramNoiseError::InvalidDecomposition);
                    }
                    family_index += 1;
                    let mut operations = Vec::new();
                    for gate in body.gates() {
                        operations.extend(gate_noise_operations(program, gate)?);
                    }
                    operations.push(GateNoiseOperation::Add);
                    operations
                }
            };
            let start = expected;
            expected = expected
                .checked_add(expected_operations.len())
                .ok_or(ProgramNoiseError::InvalidDecomposition)?;
            if expected > self.gates.len() ||
                self.gates[start..expected].iter().enumerate().any(|(branch, gain)| {
                    gain.step_index != step_index ||
                        gain.branch_index != branch ||
                        gain.operation != expected_operations[branch]
                })
            {
                return Err(ProgramNoiseError::InvalidDecomposition);
            }
        }
        if family_index != family_active_counts.len() ||
            self.gates.len() != expected ||
            self.gates.iter().enumerate().any(|(index, gain)| {
                gain.operation_index != index ||
                    gain.action.l1.is_zero() ||
                    gain.action.l2_squared.is_zero() ||
                    (gain.operation == GateNoiseOperation::Lut) != gain.helper_action.is_some() ||
                    gain.helper_action
                        .as_ref()
                        .is_some_and(|helper| helper.l1.is_zero() || helper.l2_squared.is_zero())
            })
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        Ok(())
    }
}

fn gate_noise_operations(
    program: &ExponentLutProgram,
    gate: &Gate,
) -> Result<Vec<GateNoiseOperation>, ProgramNoiseError> {
    let (operation, count) = match gate {
        Gate::MulPrivate { .. } => (GateNoiseOperation::MulPrivate, 1),
        Gate::MulPublic { .. } => (GateNoiseOperation::MulPublic, 1),
        Gate::Add { .. } => (GateNoiseOperation::Add, 1),
        Gate::Lut { lut, .. } => (
            GateNoiseOperation::Lut,
            program.lut(*lut).ok_or(ProgramNoiseError::MissingLut)?.values().len(),
        ),
    };
    Ok(vec![operation; count])
}

fn family_body_action_count(
    program: &ExponentLutProgram,
    body: &crate::program::FamilyProgram,
) -> Result<usize, ProgramNoiseError> {
    body.gates().iter().try_fold(0usize, |count, gate| {
        count
            .checked_add(gate_noise_operations(program, gate)?.len())
            .ok_or(ProgramNoiseError::IntegerConversion)
    })
}

/// Complete error state carried by one simulated exponent-program wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseState {
    pub bound: BigUint,
    pub doubled_variance: AverageVariance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FamilyNoiseState {
    /// A scalar broadcast, or a uniform input family, remains one shared state.
    /// Its declared capacity is deliberately not materialized.
    Uniform { state: ProgramNoiseState, active_count: usize },
    /// Member states are allocated only for genuinely heterogeneous inputs.
    Members(Vec<ProgramNoiseState>),
}

impl FamilyNoiseState {
    fn active_count(&self) -> usize {
        match self {
            Self::Uniform { active_count, .. } => *active_count,
            Self::Members(values) => values.len(),
        }
    }

    fn map(
        &self,
        mut operation: impl FnMut(&ProgramNoiseState) -> PrimitiveNoiseTransfer,
    ) -> (Self, Vec<PrimitiveNoiseTransfer>) {
        match self {
            Self::Uniform { state, active_count } => {
                let transfer = operation(state);
                (
                    Self::Uniform { state: transfer.output(), active_count: *active_count },
                    vec![transfer],
                )
            }
            Self::Members(values) => {
                let transfers = values.iter().map(operation).collect::<Vec<_>>();
                let outputs = transfers.iter().map(PrimitiveNoiseTransfer::output).collect();
                (Self::Members(outputs), transfers)
            }
        }
    }

    fn zip_map(
        &self,
        other: &Self,
        mut operation: impl FnMut(&ProgramNoiseState, &ProgramNoiseState) -> PrimitiveNoiseTransfer,
    ) -> Result<(Self, Vec<PrimitiveNoiseTransfer>), ProgramNoiseError> {
        if self.active_count() != other.active_count() {
            return Err(ProgramNoiseError::FamilyShapeMismatch);
        }
        match (self, other) {
            (Self::Uniform { state: left, active_count }, Self::Uniform { state: right, .. }) => {
                let transfer = operation(left, right);
                Ok((
                    Self::Uniform { state: transfer.output(), active_count: *active_count },
                    vec![transfer],
                ))
            }
            (Self::Members(left), Self::Members(right)) => {
                let transfers = left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| operation(left, right))
                    .collect::<Vec<_>>();
                let outputs = transfers.iter().map(PrimitiveNoiseTransfer::output).collect();
                Ok((Self::Members(outputs), transfers))
            }
            (Self::Uniform { state, .. }, Self::Members(values)) => {
                let transfers =
                    values.iter().map(|value| operation(state, value)).collect::<Vec<_>>();
                let outputs = transfers.iter().map(PrimitiveNoiseTransfer::output).collect();
                Ok((Self::Members(outputs), transfers))
            }
            (Self::Members(values), Self::Uniform { state, .. }) => {
                let transfers =
                    values.iter().map(|value| operation(value, state)).collect::<Vec<_>>();
                let outputs = transfers.iter().map(PrimitiveNoiseTransfer::output).collect();
                Ok((Self::Members(outputs), transfers))
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NoiseProgramWire {
    Scalar(ProgramNoiseState),
    Family(FamilyNoiseState),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NoiseProgramRhs {
    Scalar,
    Family { active_count: usize },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseInputs {
    pub input_bounds: BTreeMap<ProgramInputId, BigUint>,
    pub input_variances: BTreeMap<ProgramInputId, AverageVariance>,
    #[serde(default)]
    pub public_family_uniform_bounds: BTreeMap<WireFamilyId, BigUint>,
    #[serde(default)]
    pub public_family_uniform_variances: BTreeMap<WireFamilyId, AverageVariance>,
    pub public_family_bounds: BTreeMap<WireFamilyId, Vec<BigUint>>,
    pub public_family_variances: BTreeMap<WireFamilyId, Vec<AverageVariance>>,
}

impl Default for ProgramNoiseInputs {
    fn default() -> Self {
        Self {
            input_bounds: BTreeMap::new(),
            input_variances: BTreeMap::new(),
            public_family_uniform_bounds: BTreeMap::new(),
            public_family_uniform_variances: BTreeMap::new(),
            public_family_bounds: BTreeMap::new(),
            public_family_variances: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FamilyBodyNoiseStep {
    pub body_gate_index: usize,
    pub operation: GateNoiseOperation,
    pub inherited_bound: BigUint,
    pub fresh_bound: BigUint,
    pub output_bound: BigUint,
    pub inherited_variance: AverageVariance,
    pub fresh_variance: AverageVariance,
    pub output_variance: AverageVariance,
    pub inherited_bits: usize,
    pub fresh_bits: usize,
    pub output_bits: usize,
    pub bit_growth: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseStep {
    pub step_index: usize,
    pub inherited_bound: BigUint,
    pub fresh_bound: BigUint,
    pub output_bound: BigUint,
    pub inherited_variance: AverageVariance,
    pub fresh_variance: AverageVariance,
    pub output_variance: AverageVariance,
    pub constituent_bounds: Vec<BigUint>,
    pub constituent_variances: Vec<AverageVariance>,
    pub inherited_bits: usize,
    pub fresh_bits: usize,
    pub output_bits: usize,
    pub family_body: Vec<FamilyBodyNoiseStep>,
    #[serde(default)]
    pub family_member_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseReport {
    pub wire_bounds: BTreeMap<WireId, BigUint>,
    pub wire_variances: BTreeMap<WireId, AverageVariance>,
    pub steps: Vec<ProgramNoiseStep>,
    pub output_bound: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ProgramNoiseError {
    #[error("program input bound is missing: {0:?}")]
    MissingInputBound(ProgramInputId),
    #[error("program input variance is missing: {0:?}")]
    MissingInputVariance(ProgramInputId),
    #[error("public family member error state is missing: family {family:?}, member {member}")]
    MissingPublicFamilyInput { family: WireFamilyId, member: usize },
    #[error("program output is missing")]
    MissingOutput,
    #[error("integer conversion overflow")]
    IntegerConversion,
    #[error("decomposition schema is invalid or incomplete")]
    InvalidDecomposition,
    #[error("program LUT declaration is missing")]
    MissingLut,
    #[error("noise interpreter received incompatible scalar/family wire kinds")]
    WireKindMismatch,
    #[error("noise interpreter received incompatible family shapes")]
    FamilyShapeMismatch,
    #[error("program traversal failed: {0}")]
    ProgramTraversal(String),
}

#[derive(Clone, Debug)]
struct PrimitiveNoiseTransfer {
    inherited_bound: BigUint,
    fresh_bound: BigUint,
    output_bound: BigUint,
    inherited_variance: AverageVariance,
    fresh_variance: AverageVariance,
    output_variance: AverageVariance,
}

impl PrimitiveNoiseTransfer {
    fn output(&self) -> ProgramNoiseState {
        ProgramNoiseState {
            bound: self.output_bound.clone(),
            doubled_variance: self.output_variance.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct OperationRecord {
    operation: GateNoiseOperation,
    transfers: Vec<PrimitiveNoiseTransfer>,
    constituents: Vec<ProgramNoiseState>,
    active_count: usize,
}

struct ProgramNoiseInterpreter<'a> {
    program: &'a ExponentLutProgram,
    helper_cutoff: &'a BigUint,
    helper_variance: &'a AverageVariance,
    actions: &'a [GateNoiseGains],
    records: BTreeMap<ProgramGateInstance, OperationRecord>,
}

impl ProgramNoiseInterpreter<'_> {
    fn action(
        &self,
        gate: ProgramGateInstance,
        operation: GateNoiseOperation,
    ) -> Result<&GateNoiseGains, ProgramNoiseError> {
        let (step, range) = action_range(self.program, gate)?;
        if range.len() != 1 {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        let branch = range.start;
        self.actions
            .iter()
            .find(|gain| {
                gain.step_index == step &&
                    gain.branch_index == branch &&
                    gain.operation == operation
            })
            .ok_or(ProgramNoiseError::InvalidDecomposition)
    }

    fn record(
        &mut self,
        gate: ProgramGateInstance,
        operation: GateNoiseOperation,
        transfers: Vec<PrimitiveNoiseTransfer>,
        active_count: usize,
    ) {
        let constituents = transfers.iter().map(PrimitiveNoiseTransfer::output).collect();
        self.records
            .insert(gate, OperationRecord { operation, transfers, constituents, active_count });
    }

    fn record_with_constituents(
        &mut self,
        gate: ProgramGateInstance,
        operation: GateNoiseOperation,
        transfers: Vec<PrimitiveNoiseTransfer>,
        constituents: Vec<ProgramNoiseState>,
        active_count: usize,
    ) {
        self.records
            .insert(gate, OperationRecord { operation, transfers, constituents, active_count });
    }
}

impl ProgramLoweringTypes for ProgramNoiseInterpreter<'_> {
    type Wire = NoiseProgramWire;
    type Rhs = NoiseProgramRhs;
    type HelperSet = ();
    type Error = ProgramNoiseError;
}

impl ExponentProgramLowering for ProgramNoiseInterpreter<'_> {
    fn mul_private(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Rhs,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let action = self.action(gate, GateNoiseOperation::MulPrivate)?.action.clone();
        match (lhs, rhs) {
            (NoiseProgramWire::Scalar(state), NoiseProgramRhs::Scalar) => {
                let transfer =
                    transfer_mul_private(state, &action, self.helper_cutoff, self.helper_variance);
                let output = transfer.output();
                self.record(gate, GateNoiseOperation::MulPrivate, vec![transfer], 1);
                Ok(NoiseProgramWire::Scalar(output))
            }
            (NoiseProgramWire::Family(states), NoiseProgramRhs::Family { active_count })
                if states.active_count() == *active_count =>
            {
                let (output, transfers) = states.map(|state| {
                    transfer_mul_private(state, &action, self.helper_cutoff, self.helper_variance)
                });
                self.record(gate, GateNoiseOperation::MulPrivate, transfers, *active_count);
                Ok(NoiseProgramWire::Family(output))
            }
            _ => Err(ProgramNoiseError::WireKindMismatch),
        }
    }

    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let action = self.action(gate, GateNoiseOperation::MulPublic)?.action.clone();
        match (private, public) {
            (NoiseProgramWire::Scalar(private), NoiseProgramWire::Scalar(public)) => {
                let transfer = transfer_mul_public(private, public, &action);
                let output = transfer.output();
                self.record(gate, GateNoiseOperation::MulPublic, vec![transfer], 1);
                Ok(NoiseProgramWire::Scalar(output))
            }
            (NoiseProgramWire::Family(private), NoiseProgramWire::Family(public)) => {
                let active_count = private.active_count();
                let (output, transfers) = private.zip_map(public, |private, public| {
                    transfer_mul_public(private, public, &action)
                })?;
                self.record(gate, GateNoiseOperation::MulPublic, transfers, active_count);
                Ok(NoiseProgramWire::Family(output))
            }
            _ => Err(ProgramNoiseError::WireKindMismatch),
        }
    }

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let _ = self.action(gate, GateNoiseOperation::Add)?;
        match (lhs, rhs) {
            (NoiseProgramWire::Scalar(lhs), NoiseProgramWire::Scalar(rhs)) => {
                let transfer = transfer_add(lhs, rhs);
                let output = transfer.output();
                self.record(gate, GateNoiseOperation::Add, vec![transfer], 1);
                Ok(NoiseProgramWire::Scalar(output))
            }
            (NoiseProgramWire::Family(lhs), NoiseProgramWire::Family(rhs)) => {
                let active_count = lhs.active_count();
                let (output, transfers) = lhs.zip_map(rhs, transfer_add)?;
                self.record(gate, GateNoiseOperation::Add, transfers, active_count);
                Ok(NoiseProgramWire::Family(output))
            }
            _ => Err(ProgramNoiseError::WireKindMismatch),
        }
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        _helpers: &Self::HelperSet,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let table = program.lut(lut).ok_or(ProgramNoiseError::MissingLut)?;
        let (step, range) = action_range(program, gate)?;
        let branch_actions = range
            .map(|branch| {
                self.actions.iter().find(|gain| {
                    gain.step_index == step &&
                        gain.branch_index == branch &&
                        gain.operation == GateNoiseOperation::Lut
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(ProgramNoiseError::InvalidDecomposition)?;
        if branch_actions.len() != table.values().len() ||
            branch_actions.iter().any(|gain| gain.operation != GateNoiseOperation::Lut)
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
        let evaluate = |state: &ProgramNoiseState| {
            let branches = branch_actions
                .iter()
                .map(|gain| {
                    transfer_lut_branch(
                        state,
                        &gain.action,
                        gain.helper_action
                            .as_ref()
                            .ok_or(ProgramNoiseError::InvalidDecomposition)?,
                        self.helper_cutoff,
                        self.helper_variance,
                    )
                })
                .collect::<Result<Vec<_>, ProgramNoiseError>>()?;
            let aggregate = aggregate_lut_branches(state, &branches);
            Ok::<_, ProgramNoiseError>((aggregate, branches))
        };
        match input {
            NoiseProgramWire::Scalar(state) => {
                let (transfer, branches) = evaluate(state)?;
                let output = transfer.output();
                let constituents = branches.iter().map(PrimitiveNoiseTransfer::output).collect();
                self.record_with_constituents(
                    gate,
                    GateNoiseOperation::Lut,
                    vec![transfer],
                    constituents,
                    1,
                );
                Ok(NoiseProgramWire::Scalar(output))
            }
            NoiseProgramWire::Family(states) => {
                let active_count = states.active_count();
                let (output, transfers, constituents) = match states {
                    FamilyNoiseState::Uniform { state, active_count } => {
                        let (transfer, branches) = evaluate(state)?;
                        let constituents =
                            branches.iter().map(PrimitiveNoiseTransfer::output).collect();
                        (
                            FamilyNoiseState::Uniform {
                                state: transfer.output(),
                                active_count: *active_count,
                            },
                            vec![transfer],
                            constituents,
                        )
                    }
                    FamilyNoiseState::Members(values) => {
                        let evaluated =
                            values.iter().map(evaluate).collect::<Result<Vec<_>, _>>()?;
                        let transfers = evaluated
                            .iter()
                            .map(|(transfer, _)| transfer.clone())
                            .collect::<Vec<_>>();
                        let outputs = transfers
                            .iter()
                            .map(PrimitiveNoiseTransfer::output)
                            .collect::<Vec<_>>();
                        let constituents = evaluated
                            .first()
                            .map(|(_, branches)| {
                                branches.iter().map(PrimitiveNoiseTransfer::output).collect()
                            })
                            .unwrap_or_default();
                        (FamilyNoiseState::Members(outputs), transfers, constituents)
                    }
                };
                self.record_with_constituents(
                    gate,
                    GateNoiseOperation::Lut,
                    transfers,
                    constituents,
                    active_count,
                );
                Ok(NoiseProgramWire::Family(output))
            }
        }
    }

    fn public_attribute_available(&self, _wire: &Self::Wire) -> bool {
        true
    }

    fn broadcast_wire(
        &mut self,
        wire: &Self::Wire,
        range: &FamilyRange,
    ) -> Result<Self::Wire, Self::Error> {
        let NoiseProgramWire::Scalar(state) = wire else {
            return Err(ProgramNoiseError::WireKindMismatch);
        };
        let active_count = range_count(range)?;
        // A broadcast is one shared scalar state, not `capacity` copies.  The
        // multiplicity is applied only when an active family is reduced.
        Ok(NoiseProgramWire::Family(FamilyNoiseState::Uniform {
            state: state.clone(),
            active_count,
        }))
    }

    fn reduce_add(
        &mut self,
        value: &Self::Wire,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let NoiseProgramWire::Family(values) = value else {
            return Err(ProgramNoiseError::WireKindMismatch);
        };
        let active_count = range_count(range)?;
        if values.active_count() != active_count {
            return Err(ProgramNoiseError::FamilyShapeMismatch);
        }
        // Only active members participate.  Explicit Program Add uses the
        // worst-case coherent standard-deviation bound, represented here as
        // `r * sum_i V_i`; a uniform family therefore becomes `r^2 V`.
        let (bound_sum, variance_sum, constituents) = match values {
            FamilyNoiseState::Uniform { state, .. } => (
                &state.bound * active_count,
                state.doubled_variance.scaled(&BigUint::from(active_count)),
                vec![state.clone()],
            ),
            FamilyNoiseState::Members(values) => {
                let bound = values.iter().fold(BigUint::zero(), |sum, state| sum + &state.bound);
                let variance = values
                    .iter()
                    .fold(AverageVariance::zero(), |sum, state| sum.add(&state.doubled_variance));
                (bound, variance, values.clone())
            }
        };
        let output_variance = variance_sum.scaled(&BigUint::from(active_count));
        let transfer = PrimitiveNoiseTransfer {
            inherited_bound: bound_sum.clone(),
            fresh_bound: BigUint::zero(),
            output_bound: bound_sum,
            inherited_variance: variance_sum,
            fresh_variance: AverageVariance::zero(),
            output_variance,
        };
        let output = transfer.output();
        self.record_with_constituents(
            gate,
            GateNoiseOperation::Add,
            vec![transfer],
            constituents,
            active_count,
        );
        Ok(NoiseProgramWire::Scalar(output))
    }
}

fn transfer_mul_private(
    inherited: &ProgramNoiseState,
    action: &RightActionNoiseGain,
    helper_cutoff: &BigUint,
    helper_variance: &AverageVariance,
) -> PrimitiveNoiseTransfer {
    PrimitiveNoiseTransfer {
        inherited_bound: inherited.bound.clone(),
        fresh_bound: helper_cutoff.clone(),
        output_bound: &action.l1 * &inherited.bound + helper_cutoff,
        inherited_variance: inherited.doubled_variance.clone(),
        fresh_variance: helper_variance.clone(),
        output_variance: inherited.doubled_variance.scaled(&action.l2_squared).add(helper_variance),
    }
}

fn transfer_mul_public(
    private: &ProgramNoiseState,
    public: &ProgramNoiseState,
    action: &RightActionNoiseGain,
) -> PrimitiveNoiseTransfer {
    // The private accumulator itself is acted on by a public monomial (unit
    // norm), while the public-label encoding error is acted on by K_private.
    PrimitiveNoiseTransfer {
        inherited_bound: private.bound.clone(),
        fresh_bound: &action.l1 * &public.bound,
        output_bound: &private.bound + &action.l1 * &public.bound,
        inherited_variance: private.doubled_variance.clone(),
        fresh_variance: public.doubled_variance.scaled(&action.l2_squared),
        output_variance: private
            .doubled_variance
            .add(&public.doubled_variance.scaled(&action.l2_squared)),
    }
}

fn transfer_add(lhs: &ProgramNoiseState, rhs: &ProgramNoiseState) -> PrimitiveNoiseTransfer {
    let variance_sum = lhs.doubled_variance.add(&rhs.doubled_variance);
    PrimitiveNoiseTransfer {
        inherited_bound: lhs.bound.clone(),
        fresh_bound: rhs.bound.clone(),
        output_bound: &lhs.bound + &rhs.bound,
        inherited_variance: lhs.doubled_variance.clone(),
        fresh_variance: rhs.doubled_variance.clone(),
        output_variance: variance_sum.scaled(&BigUint::from(2u8)),
    }
}

fn transfer_lut_branch(
    input: &ProgramNoiseState,
    direct_action: &RightActionNoiseGain,
    helper_action: &RightActionNoiseGain,
    helper_cutoff: &BigUint,
    helper_variance: &AverageVariance,
) -> Result<PrimitiveNoiseTransfer, ProgramNoiseError> {
    let helper_weight = BigUint::one() + &helper_action.l1;
    let helper_variance_gain = BigUint::one() + &helper_action.l2_squared;
    Ok(PrimitiveNoiseTransfer {
        inherited_bound: input.bound.clone(),
        fresh_bound: helper_cutoff * &helper_weight,
        output_bound: &direct_action.l1 * &input.bound + helper_cutoff * &helper_weight,
        inherited_variance: input.doubled_variance.clone(),
        fresh_variance: helper_variance.scaled(&helper_variance_gain),
        output_variance: input
            .doubled_variance
            .scaled(&direct_action.l2_squared)
            .add(&helper_variance.scaled(&helper_variance_gain)),
    })
}

fn aggregate_lut_branches(
    input: &ProgramNoiseState,
    branches: &[PrimitiveNoiseTransfer],
) -> PrimitiveNoiseTransfer {
    // Flat LUT branch errors use the accepted H3 independent-sum heuristic.
    // This is intentionally not the conservative transfer for an explicit
    // Program `Add` gate.
    PrimitiveNoiseTransfer {
        inherited_bound: input.bound.clone(),
        fresh_bound: branches.iter().fold(BigUint::zero(), |sum, branch| sum + &branch.fresh_bound),
        output_bound: branches
            .iter()
            .fold(BigUint::zero(), |sum, branch| sum + &branch.output_bound),
        inherited_variance: input.doubled_variance.clone(),
        fresh_variance: branches
            .iter()
            .fold(AverageVariance::zero(), |sum, branch| sum.add(&branch.fresh_variance)),
        output_variance: branches
            .iter()
            .fold(AverageVariance::zero(), |sum, branch| sum.add(&branch.output_variance)),
    }
}

/// Interprets a complete Exponent-LUT program and records every wire state.
pub fn simulate_program(
    program: &ExponentLutProgram,
    inputs: &ProgramNoiseInputs,
    helper_cutoff: &BigUint,
    helper_variance: &AverageVariance,
    actions: &[GateNoiseGains],
    family_active_counts: &[usize],
) -> Result<ProgramNoiseReport, ProgramNoiseError> {
    let mut scalar_inputs = BTreeMap::new();
    for (input, _) in program.input_wires() {
        scalar_inputs.insert(
            *input,
            NoiseProgramWire::Scalar(ProgramNoiseState {
                bound: inputs
                    .input_bounds
                    .get(input)
                    .cloned()
                    .ok_or(ProgramNoiseError::MissingInputBound(*input))?,
                doubled_variance: inputs
                    .input_variances
                    .get(input)
                    .cloned()
                    .ok_or(ProgramNoiseError::MissingInputVariance(*input))?,
            }),
        );
    }
    let rhs_inputs = program
        .rhs_inputs()
        .keys()
        .map(|id| (*id, NoiseProgramRhs::Scalar))
        .collect::<BTreeMap<RhsId, _>>();
    let mut rhs_families = BTreeMap::<RhsFamilyId, NoiseProgramRhs>::new();
    let mut wire_values = BTreeMap::<WireFamilyId, NoiseProgramWire>::new();
    let mut helpers = BTreeMap::<LutId, ()>::new();
    let mut family_ranges = ProgramFamilyBindings::new();
    let mut family_index = 0usize;
    for step in program.steps() {
        match step {
            ProgramStep::Gate(Gate::Lut { lut, .. }) => {
                helpers.insert(*lut, ());
            }
            ProgramStep::Gate(_) => {}
            ProgramStep::Family { id, capacity, body, .. } => {
                let count = *family_active_counts
                    .get(family_index)
                    .ok_or(ProgramNoiseError::FamilyShapeMismatch)?;
                family_index += 1;
                if count == 0 || count > *capacity {
                    return Err(ProgramNoiseError::FamilyShapeMismatch);
                }
                family_ranges.bind(
                    *id,
                    FamilyRange::bounded(0usize, count, *capacity)
                        .map_err(|error| ProgramNoiseError::ProgramTraversal(error.to_string()))?,
                );
                for source in body.rhs_inputs().values() {
                    let FamilyRhsSource::Member(family) = source;
                    rhs_families.insert(*family, NoiseProgramRhs::Family { active_count: count });
                }
                for source in body.wire_inputs().values() {
                    let FamilyWireSource::Member(family) = source else { continue };
                    wire_values.insert(
                        *family,
                        NoiseProgramWire::Family(program_family_input(inputs, *family, count)?),
                    );
                }
                for gate in body.gates() {
                    if let crate::program::Gate::Lut { lut, .. } = gate {
                        helpers.insert(*lut, ());
                    }
                }
            }
        }
    }
    if family_index != family_active_counts.len() {
        return Err(ProgramNoiseError::FamilyShapeMismatch);
    }

    let bindings =
        ProgramBindings::new(&scalar_inputs, &rhs_inputs, &rhs_families, &wire_values, &helpers);
    let mut interpreter = ProgramNoiseInterpreter {
        program,
        helper_cutoff,
        helper_variance,
        actions,
        records: BTreeMap::new(),
    };
    let wires = lower_program(program, &bindings, &family_ranges, &mut interpreter)
        .map_err(map_lower_error)?;
    let output_wire = *program.outputs().last().ok_or(ProgramNoiseError::MissingOutput)?;
    let NoiseProgramWire::Scalar(output) =
        wires.get(&output_wire).cloned().ok_or(ProgramNoiseError::MissingOutput)?
    else {
        return Err(ProgramNoiseError::WireKindMismatch);
    };
    let mut wire_bounds = BTreeMap::new();
    let mut wire_variances = BTreeMap::new();
    for (wire, value) in &wires {
        if let NoiseProgramWire::Scalar(state) = value {
            wire_bounds.insert(*wire, state.bound.clone());
            wire_variances.insert(*wire, state.doubled_variance.clone());
        }
    }
    let steps = build_step_report(program, &interpreter.records)?;
    Ok(ProgramNoiseReport { wire_bounds, wire_variances, steps, output_bound: output.bound })
}

fn program_family_input(
    inputs: &ProgramNoiseInputs,
    family: WireFamilyId,
    count: usize,
) -> Result<FamilyNoiseState, ProgramNoiseError> {
    // Binding a uniform family keeps one shared scalar state. Only an
    // explicitly index-dependent bound or variance allocates active members;
    // the declared family capacity is never materialized here.
    let member_bounds = inputs.public_family_bounds.get(&family);
    let member_variances = inputs.public_family_variances.get(&family);
    let uniform_bound = inputs.public_family_uniform_bounds.get(&family);
    let uniform_variance = inputs.public_family_uniform_variances.get(&family);
    match (member_bounds, member_variances, uniform_bound, uniform_variance) {
        (Some(bounds), Some(variances), _, _)
            if bounds.len() == count && variances.len() == count =>
        {
            Ok(FamilyNoiseState::Members(
                bounds
                    .iter()
                    .cloned()
                    .zip(variances.iter().cloned())
                    .map(|(bound, doubled_variance)| ProgramNoiseState { bound, doubled_variance })
                    .collect(),
            ))
        }
        (None, None, Some(bound), Some(doubled_variance)) => Ok(FamilyNoiseState::Uniform {
            state: ProgramNoiseState {
                bound: bound.clone(),
                doubled_variance: doubled_variance.clone(),
            },
            active_count: count,
        }),
        (None, Some(variances), Some(bound), _) if variances.len() == count => {
            Ok(FamilyNoiseState::Members(
                variances
                    .iter()
                    .cloned()
                    .map(|doubled_variance| ProgramNoiseState {
                        bound: bound.clone(),
                        doubled_variance,
                    })
                    .collect(),
            ))
        }
        (Some(bounds), None, _, Some(doubled_variance)) if bounds.len() == count => {
            Ok(FamilyNoiseState::Members(
                bounds
                    .iter()
                    .cloned()
                    .map(|bound| ProgramNoiseState {
                        bound,
                        doubled_variance: doubled_variance.clone(),
                    })
                    .collect(),
            ))
        }
        _ => Err(ProgramNoiseError::MissingPublicFamilyInput { family, member: count }),
    }
}

fn build_step_report(
    program: &ExponentLutProgram,
    records: &BTreeMap<ProgramGateInstance, OperationRecord>,
) -> Result<Vec<ProgramNoiseStep>, ProgramNoiseError> {
    program
        .steps()
        .iter()
        .enumerate()
        .map(|(step_index, step)| match step {
            ProgramStep::Gate(_) => {
                let record = records
                    .get(&ProgramGateInstance::Gate { step: step_index })
                    .ok_or(ProgramNoiseError::InvalidDecomposition)?;
                noise_step(step_index, &record.transfers[0], &record.constituents, Vec::new(), 0)
            }
            ProgramStep::Family { body, .. } => {
                let reduction = records
                    .get(&ProgramGateInstance::FamilyReduction { step: step_index })
                    .ok_or(ProgramNoiseError::InvalidDecomposition)?;
                let family_body = body
                    .gates()
                    .iter()
                    .enumerate()
                    .map(|(gate_index, _)| {
                        let record = records
                            .get(&ProgramGateInstance::FamilyGate {
                                step: step_index,
                                gate: gate_index,
                            })
                            .ok_or(ProgramNoiseError::InvalidDecomposition)?;
                        Ok(family_body_noise_step(
                            gate_index,
                            record.operation,
                            &record.transfers[0],
                        ))
                    })
                    .collect::<Result<Vec<_>, ProgramNoiseError>>()?;
                noise_step(
                    step_index,
                    &reduction.transfers[0],
                    &reduction.constituents,
                    family_body,
                    reduction.active_count,
                )
            }
        })
        .collect()
}

fn noise_step(
    step_index: usize,
    transfer: &PrimitiveNoiseTransfer,
    constituents: &[ProgramNoiseState],
    family_body: Vec<FamilyBodyNoiseStep>,
    family_member_count: usize,
) -> Result<ProgramNoiseStep, ProgramNoiseError> {
    Ok(ProgramNoiseStep {
        step_index,
        inherited_bound: transfer.inherited_bound.clone(),
        fresh_bound: transfer.fresh_bound.clone(),
        output_bound: transfer.output_bound.clone(),
        inherited_variance: transfer.inherited_variance.clone(),
        fresh_variance: transfer.fresh_variance.clone(),
        output_variance: transfer.output_variance.clone(),
        constituent_bounds: constituents.iter().map(|state| state.bound.clone()).collect(),
        constituent_variances: constituents
            .iter()
            .map(|state| state.doubled_variance.clone())
            .collect(),
        inherited_bits: bit_length(&transfer.inherited_bound),
        fresh_bits: bit_length(&transfer.fresh_bound),
        output_bits: bit_length(&transfer.output_bound),
        family_body,
        family_member_count,
    })
}

fn family_body_noise_step(
    body_gate_index: usize,
    operation: GateNoiseOperation,
    transfer: &PrimitiveNoiseTransfer,
) -> FamilyBodyNoiseStep {
    let inherited_bits = bit_length(&transfer.inherited_bound);
    let fresh_bits = bit_length(&transfer.fresh_bound);
    let output_bits = bit_length(&transfer.output_bound);
    FamilyBodyNoiseStep {
        body_gate_index,
        operation,
        inherited_bound: transfer.inherited_bound.clone(),
        fresh_bound: transfer.fresh_bound.clone(),
        output_bound: transfer.output_bound.clone(),
        inherited_variance: transfer.inherited_variance.clone(),
        fresh_variance: transfer.fresh_variance.clone(),
        output_variance: transfer.output_variance.clone(),
        inherited_bits,
        fresh_bits,
        output_bits,
        bit_growth: output_bits.saturating_sub(inherited_bits.max(fresh_bits)),
    }
}

fn action_range(
    program: &ExponentLutProgram,
    instance: ProgramGateInstance,
) -> Result<(usize, std::ops::Range<usize>), ProgramNoiseError> {
    let step_index = match instance {
        ProgramGateInstance::Gate { step } |
        ProgramGateInstance::FamilyGate { step, .. } |
        ProgramGateInstance::FamilyReduction { step } => step,
    };
    let step = program.steps().get(step_index).ok_or(ProgramNoiseError::InvalidDecomposition)?;
    match (instance, step) {
        (ProgramGateInstance::Gate { .. }, ProgramStep::Gate(gate)) => {
            Ok((step_index, 0..gate_noise_operations(program, gate)?.len()))
        }
        (ProgramGateInstance::FamilyGate { gate, .. }, ProgramStep::Family { body, .. }) => {
            let mut start = 0usize;
            for body_gate in body.gates().iter().take(gate) {
                start = start
                    .checked_add(gate_noise_operations(program, body_gate)?.len())
                    .ok_or(ProgramNoiseError::IntegerConversion)?;
            }
            let count = gate_noise_operations(
                program,
                body.gates().get(gate).ok_or(ProgramNoiseError::InvalidDecomposition)?,
            )?
            .len();
            Ok((step_index, start..start + count))
        }
        (ProgramGateInstance::FamilyReduction { .. }, ProgramStep::Family { body, .. }) => {
            let start = family_body_action_count(program, body)?;
            Ok((step_index, start..start + 1))
        }
        _ => Err(ProgramNoiseError::InvalidDecomposition),
    }
}

fn range_count(range: &FamilyRange) -> Result<usize, ProgramNoiseError> {
    range
        .count()
        .evaluate(&ParamEnv::default())
        .map_err(|_| ProgramNoiseError::FamilyShapeMismatch)?
        .to_usize()
        .filter(|count| *count != 0)
        .ok_or(ProgramNoiseError::FamilyShapeMismatch)
}

fn map_lower_error(error: ProgramLowerError<ProgramNoiseError>) -> ProgramNoiseError {
    match error {
        ProgramLowerError::Operation { source, .. } => source,
        ProgramLowerError::Validation(error) => {
            ProgramNoiseError::ProgramTraversal(error.to_string())
        }
    }
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

fn bit_length(value: &BigUint) -> usize {
    value.bits() as usize
}
