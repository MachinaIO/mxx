//! Noise interpretation of an [`ExponentLutProgram`](crate::program::ExponentLutProgram).
//!
//! BGG-owned primitive transfers and gadget/residual arithmetic. This module
//! interprets the same four primitive operations as the public-key and encoding
//! backends, using exact worst-case bounds and doubled-coordinate variances.
//! Refresh setup adapters derive compact plans from concrete BGG artifact schemas;
//! the application retains digit scheduling and preprocessing/online orchestration.

use super::{BggRefreshParameters, setup_identity_digest};
use crate::{
    layout::ExpLayout,
    noise_refresh::{
        pbc::{PbcActiveCellIndex, PbcPublicLayout},
        prf::SparseLwrPrfProgram,
        refresh::RefreshPrfContract,
        setup::RefreshSetupError,
        simulation::{
            AverageCaseConfig, AverageRefreshNoiseReport, AverageRefreshSlotNoiseReport,
            AverageVarianceBreakdown, ExponentLutAverageNoiseReport, ExponentLutNoiseReport,
            MaskSmudgingParameters, NoiseModelKind, NoiseSimulationError,
            PreRefreshNoiseEvaluation, RefreshNoiseParameters, RefreshNoiseReport,
            RefreshRouteNoiseGains, RefreshSlotNoiseReport, TraceNoiseGains,
            ValueReadoutNoiseGains, average_favg, ceil_div, fresh_error_below_every_prime_factor,
            switched_error_bound,
        },
    },
    noise_utils::{NonnegativeRational, RightActionNoiseGain},
};
use std::collections::BTreeMap;

use super::BggPreparedReadout;
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::program::{
    ExponentLutProgram, ExponentProgramLowering, FamilyRange, FamilyRhsSource, FamilyWireSource,
    Gate, LutId, ProgramBindings, ProgramFamilyBindings, ProgramGateInstance, ProgramInputId,
    ProgramLowerError, ProgramLoweringTypes, ProgramStep, RhsFamilyId, RhsId, WireFamilyId, WireId,
};

/// Setup-produced decomposition plans for the refresh routes.
///
/// Each plan must contain either the concrete balanced coefficients or a
/// validated compact source/bound for the corresponding target.  The setup
/// boundary derives all route gains through
/// [`NoiseDecompositionPlan::action_gain`]; callers cannot provide a detached
/// scalar gain in its place.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefreshNoiseGainDecompositions {
    /// One compact decomposition for each CRT slot's `G^-1(kappa_t G)` action.
    pub state_scaling: Vec<NoiseDecompositionPlan>,
    /// One compact decomposition for every P-adic mask digit `P^c`.
    pub mask: Vec<Vec<NoiseDecompositionPlan>>,
    /// Number of ring-coefficient positions represented by each mask plan.
    /// The compact search path stores one plan per P-adic digit and uses this
    /// multiplicity instead of materializing `coefficient_count` copies.
    pub mask_multiplicity: usize,
    /// One compact decomposition family for every slot's `kappa_t P^c` fresh digit.
    pub fresh: Vec<Vec<NoiseDecompositionPlan>>,
    /// Common ring-coordinate multiplicity for the CRT-shared fresh source.
    /// Each slot retains its own weighted decomposition plans.
    pub fresh_multiplicity: usize,
    pub decoder: Vec<NoiseDecompositionPlan>,
}

impl RefreshNoiseGainDecompositions {
    pub fn derive_route_gains(
        &self,
        slot_count: usize,
    ) -> Result<RefreshRouteNoiseGains, NoiseSimulationError> {
        if self.state_scaling.len() != slot_count ||
            self.mask.len() != slot_count ||
            self.mask.iter().any(|plans| plans.is_empty()) ||
            self.fresh.len() != slot_count ||
            self.mask_multiplicity == 0 ||
            self.fresh_multiplicity == 0 ||
            self.fresh.iter().any(|plans| plans.is_empty()) ||
            self.decoder.len() != slot_count
        {
            return Err(NoiseSimulationError::InvalidRefresh(
                "refresh decomposition slot count mismatch",
            ));
        }
        let derive = |plan: &NoiseDecompositionPlan| {
            NoiseGainPlanBuilder::derive_route_gains(std::slice::from_ref(plan))
                .map(|mut gains| gains.remove(0))
                .map_err(|_| NoiseSimulationError::InvalidRefresh("invalid refresh decomposition"))
        };
        let derive_sum = |plans: &[NoiseDecompositionPlan], multiplicity: usize| {
            let gains = plans.iter().map(derive).collect::<Result<Vec<_>, _>>()?;
            let mut aggregate =
                RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() };
            for gain in gains {
                aggregate.l1 += gain.l1 * multiplicity;
                aggregate.l2_squared += gain.l2_squared * multiplicity;
            }
            if aggregate.l1.is_zero() != aggregate.l2_squared.is_zero() {
                return Err(NoiseSimulationError::InvalidRefresh("invalid refresh decomposition"));
            }
            Ok(aggregate)
        };
        Ok(RefreshRouteNoiseGains {
            state_scaling: self.state_scaling.iter().map(derive).collect::<Result<Vec<_>, _>>()?,
            mask: self
                .mask
                .iter()
                .map(|plans| derive_sum(plans, self.mask_multiplicity))
                .collect::<Result<Vec<_>, _>>()?,
            fresh: self
                .fresh
                .iter()
                .map(|plans| derive_sum(plans, self.fresh_multiplicity))
                .collect::<Result<Vec<_>, _>>()?,
            decoder: self.decoder.iter().map(derive).collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// Setup-produced gain data for the ordered program and value readout.
/// Every scalar in this structure is produced by [`NoiseGainPlanBuilder`]
/// from its corresponding concrete decomposition plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseGainPlan {
    pub(crate) program: ProgramNoiseGainPlan,
    pub(crate) readout: ValueReadoutNoiseGains,
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
                if gain.l1.is_zero() != gain.l2_squared.is_zero() {
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
                NoiseDecompositionSource::GadgetScalar { .. } => {
                    Some(self.target_columns) == self.target_rows.checked_mul(self.digit_count)
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
    /// Public plaintext multiplier gain in MulPublic. Preparation supplies
    /// its integer-lift polynomial norms; native PRF monomials use unit gain.
    pub public_plaintext_action: RightActionNoiseGain,
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
    pub inherited_variance: NonnegativeRational,
    pub fresh_variance: NonnegativeRational,
    pub output_variance: NonnegativeRational,
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
                ProgramStep::Gate(Gate::Lut { lut_id: lut, .. }) => {
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
        Gate::Lut { lut_id: lut, .. } => (
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
    pub doubled_variance: NonnegativeRational,
}

/// Return the padded digits per CRT tower and the number of retained towers.
/// The same ordinary decomposition layout is used by target preparation and
/// its compact noise model.
pub(crate) fn ordinary_gadget_towers(
    moduli: &[BigUint],
    base: &BigUint,
    digits: usize,
) -> Result<(usize, usize), &'static str> {
    let bits = base.bits().checked_sub(1).filter(|bits| *bits > 0).ok_or("invalid gadget base")?;
    if base != &(BigUint::one() << bits as usize) {
        return Err("gadget base must be binary");
    }
    let per_tower = moduli
        .iter()
        .map(|p| p.bits().div_ceil(bits) as usize)
        .max()
        .ok_or("missing CRT towers")?;
    if digits == 0 || digits % per_tower != 0 || digits / per_tower > moduli.len() {
        return Err("invalid ordinary gadget width");
    }
    Ok((per_tower, digits / per_tower))
}

/// Secret-weighted reconstruction error of the ordinary shortened gadget.
/// The average model intentionally treats centered discarded CRT residues as
/// independent uniforms and the augmented secret coefficients as ternary.
/// Stored variance is Var(2e), four times physical variance.
pub(crate) fn approximate_gadget_residual(
    moduli: &[BigUint],
    base: &BigUint,
    digits: usize,
    secret_dimension: usize,
    coefficient_count: usize,
) -> Result<Option<ProgramNoiseState>, &'static str> {
    let (_, retained) = ordinary_gadget_towers(moduli, base, digits)?;
    if retained == moduli.len() {
        return Ok(None);
    }
    if secret_dimension != 2 {
        return Err("approximate residual requires augmented ternary dimension two");
    }
    let omitted = &moduli[retained..];
    let product = omitted.iter().product::<BigUint>();
    let epsilon = omitted.iter().map(|p| (&product / p) * (p / 2u8)).sum::<BigUint>();
    let variance_sum =
        omitted.iter().map(|p| (&product / p).pow(2) * (p.pow(2) - 1u8)).sum::<BigUint>();
    Ok(Some(ProgramNoiseState {
        bound: epsilon * BigUint::from(coefficient_count + 1),
        doubled_variance: NonnegativeRational::new(
            variance_sum * BigUint::from(3 + 2 * coefficient_count),
            9u8.into(),
        )
        .expect("positive residual variance denominator"),
    }))
}

/// An even representative prevents multiplication by the native half-integer
/// readout payload from introducing a modular half-wrap. The corrected target
/// is divisible by every omitted tower, hence ordinary approximate D is exact
/// on this target. This is public target construction, not a modified gadget.
pub(crate) fn even_route_correction(value: &BigInt, dropped_product: &BigUint) -> BigInt {
    if dropped_product.is_one() {
        return BigInt::zero();
    }
    let period = BigInt::from(dropped_product * 2u8);
    let mut residue = ((value % &period) + &period) % &period;
    if residue.magnitude() % 2u8 == BigUint::one() {
        residue += BigInt::from(dropped_product.clone());
        residue %= &period;
    }
    if residue > BigInt::from(dropped_product.clone()) {
        residue -= period;
    }
    residue
}

/// Scalar column gains and residual of an explicitly modeled even-residual
/// bridge from a shortened gadget to the full gadget. Starting from the
/// shared low-CRT correction (ePrint 2024/909, Section 3.2), an odd residual
/// is shifted by plus or minus the odd low modulus to make it even. This
/// preserves every dropped residue and allows half-integer readout payloads
/// to weight the residual without modular inverse-of-two wraparound.
/// This custom bridge is not the default primitive decomposition algorithm.
/// The matrix is block diagonal across secret rows and constant in the ring,
/// so neither a ring-convolution nor a secret-dimension factor applies.
pub fn even_residual_gadget_bridge(
    moduli: &[u64],
    retained: usize,
    base: &BigUint,
    digits_per_tower: usize,
) -> Result<(RightActionNoiseGain, BigUint), ProgramNoiseError> {
    if retained == 0 ||
        retained > moduli.len() ||
        digits_per_tower == 0 ||
        digits_per_tower > 64 ||
        base.bits() > 64 ||
        base < &BigUint::from(2u8) ||
        !is_power_of_two(base)
    {
        return Err(ProgramNoiseError::InvalidDecomposition);
    }
    let largest_digit = base.pow((digits_per_tower - 1) as u32);
    if moduli.iter().any(|p| *p <= 1 || *p % 2 == 0 || BigUint::from(*p) <= largest_digit) {
        return Err(ProgramNoiseError::InvalidDecomposition);
    }
    let low: BigUint = moduli[retained..].iter().map(|p| BigUint::from(*p)).product();
    let mut gain = RightActionNoiseGain::unit();
    // Centered lifting can change even a retained radix column near p/2;
    // derive these columns too rather than assuming every one is a unit.
    for modulus in &moduli[..retained] {
        for digit in 0..digits_per_tower {
            let value = BigInt::from(base.pow(digit as u32));
            let centered = centered_residue(&value, &BigUint::from(*modulus));
            let (l1, l2_squared) = balanced_profile(&centered, base, digits_per_tower)?;
            gain.l1 = gain.l1.max(l1);
            gain.l2_squared = gain.l2_squared.max(l2_squared);
        }
    }
    let mut residual_bound = BigUint::zero();
    for p in &moduli[retained..] {
        let weight = &low / p;
        let inverse = mxx_primitives::utils::mod_inverse(
            (&weight % p).to_u64().ok_or(ProgramNoiseError::IntegerConversion)?,
            *p,
        )
        .ok_or(ProgramNoiseError::InvalidDecomposition)?;
        for digit in 0..digits_per_tower {
            let value =
                base.pow(u32::try_from(digit).map_err(|_| ProgramNoiseError::IntegerConversion)?);
            let twisted = (&value * inverse) % p;
            let residual = even_route_correction(
                &(BigInt::from(weight.clone()) *
                    centered_residue(&BigInt::from(twisted), &BigUint::from(*p))),
                &low,
            );
            residual_bound = residual_bound.max(residual.magnitude().clone());
            let mut l1 = BigUint::zero();
            let mut l2_squared = BigUint::zero();
            for modulus in &moduli[..retained] {
                let corrected = centered_residue(&(-&residual), &BigUint::from(*modulus));
                let (column_l1, column_l2) = balanced_profile(&corrected, base, digits_per_tower)?;
                l1 += column_l1;
                l2_squared += column_l2;
            }
            gain.l1 = gain.l1.max(l1);
            gain.l2_squared = gain.l2_squared.max(l2_squared);
        }
    }
    Ok((gain, residual_bound))
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[test]
    fn even_route_targets_preserve_omitted_residues_and_half_payload() {
        for product in [5u32, 13, 65] {
            let p = BigUint::from(product);
            for scalar in -150i32..150 {
                let value = BigInt::from(scalar);
                let residual = even_route_correction(&value, &p);
                assert_eq!(residual.magnitude() % 2u8, BigUint::zero());
                assert!(residual.magnitude() <= &p);
                assert_eq!((&value - &residual) % BigInt::from(p.clone()), BigInt::zero());
                // Every p=32 native half-payload has a genuine integer error.
                for odd in (-31i32..32).step_by(2) {
                    assert_eq!((&residual * odd) % 2u8, BigInt::zero());
                }
            }
        }
    }

    #[test]
    fn even_residual_bridge_preserves_small_crt_columns() {
        // Dropped unit column: even residual -72, corrected retained
        // residues 4 mod17 and -10 mod41, base16 digits (4,0),(6,-1).
        // Dropped radix16 column: residual16, digits (1,0),(0,-1).
        let (gain, residual) =
            even_residual_gadget_bridge(&[17, 41, 73], 2, &16u8.into(), 2).unwrap();
        assert_eq!(gain, RightActionNoiseGain { l1: 11u8.into(), l2_squared: 53u8.into() });
        assert_eq!(residual, 72u8.into());
        for (value, error, digits) in
            [(1i64, -72i64, [[4i64, 0], [6, -1]]), (16, 16, [[1, 0], [0, -1]])]
        {
            assert_eq!(error % 2, 0);
            assert_eq!((error - value).rem_euclid(73), 0);
            for (p, column) in [17i64, 41].into_iter().zip(digits) {
                assert_eq!((column[0] + 16 * column[1] + error).rem_euclid(p), 0);
            }
        }
        let (exact_gain, exact_error) =
            even_residual_gadget_bridge(&[17, 41, 73], 3, &16u8.into(), 2).unwrap();
        assert_eq!(exact_gain, RightActionNoiseGain::unit());
        assert!(exact_error.is_zero());
        assert!(even_residual_gadget_bridge(&[0, 41], 1, &16u8.into(), 2).is_err());
        assert_eq!(
            even_residual_gadget_bridge(&[19], 1, &16u8.into(), 2).unwrap().0.l1,
            3u8.into()
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FamilyNoiseState {
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

pub type NoiseProgramWire = crate::program::ProgramValue<ProgramNoiseState, FamilyNoiseState>;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseInputs {
    /// Modeled additive errors keyed by the setup operation index. These are
    /// applied after the operation's ordinary right actions. For approximate
    /// gadgets the caller must supply the secret-weighted reconstruction error,
    /// including both direct and helper residuals for each LUT branch.
    /// Average-case composition intentionally assumes residual independence
    /// from the ordinary gate error, following the approximate-gadget model.
    /// This heuristic does not apply to explicit Add or reused-wire sums.
    /// An empty map preserves exact-gadget simulation.
    #[serde(default)]
    pub operation_residuals: BTreeMap<usize, ProgramNoiseState>,
    /// Error of a prepared scalar RHS (including all weighted selector errors).
    /// Preparation must bind every RHS used by the Program. Payload one-hotness
    /// does not reduce this error; reused sources must have their actions combined.
    pub rhs_errors: BTreeMap<RhsId, ProgramNoiseState>,
    /// Prepared RHS errors for structural families. Uniform families retain
    /// one compact state; heterogeneous families retain each member's error.
    pub rhs_family_errors: BTreeMap<RhsFamilyId, FamilyNoiseState>,
    pub input_errors: BTreeMap<ProgramInputId, ProgramNoiseState>,
    pub public_family_errors: BTreeMap<WireFamilyId, FamilyNoiseState>,
}

impl Default for ProgramNoiseInputs {
    fn default() -> Self {
        Self {
            operation_residuals: BTreeMap::new(),
            rhs_errors: BTreeMap::new(),
            rhs_family_errors: BTreeMap::new(),
            input_errors: BTreeMap::new(),
            public_family_errors: BTreeMap::new(),
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
    pub inherited_variance: NonnegativeRational,
    pub fresh_variance: NonnegativeRational,
    pub output_variance: NonnegativeRational,
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
    pub inherited_variance: NonnegativeRational,
    pub fresh_variance: NonnegativeRational,
    pub output_variance: NonnegativeRational,
    pub constituent_bounds: Vec<BigUint>,
    pub constituent_variances: Vec<NonnegativeRational>,
    pub inherited_bits: usize,
    pub fresh_bits: usize,
    pub output_bits: usize,
    pub family_body: Vec<FamilyBodyNoiseStep>,
    #[serde(default)]
    pub family_member_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProgramNoiseReport {
    pub dependencies: crate::noise_utils::NoiseDependencyGraph,
    /// The evaluator's states, retaining bounds and doubled-coordinate moments together.
    pub wire_states: BTreeMap<WireId, NoiseProgramWire>,
    pub steps: Vec<ProgramNoiseStep>,
    /// Declared scalar output; it need not be the last evaluated gate.
    pub output: WireId,
}

impl ProgramNoiseReport {
    /// Read the output validated by Program simulation from the same wire trace.
    pub fn output_state(&self) -> &ProgramNoiseState {
        self.wire_states[&self.output].scalar().expect("validated scalar Program output")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ProgramNoiseError {
    #[error(transparent)]
    Arithmetic(#[from] crate::noise_utils::NoiseArithmeticError),
    #[error("program input error state is missing: {0:?}")]
    MissingInputError(ProgramInputId),
    #[error("prepared RHS error state is missing: {0:?}")]
    MissingRhsError(RhsId),
    #[error("prepared RHS family error state is missing: {0:?}")]
    MissingRhsFamilyError(RhsFamilyId),
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
    inherited_variance: NonnegativeRational,
    fresh_variance: NonnegativeRational,
    output_variance: NonnegativeRational,
}

impl PrimitiveNoiseTransfer {
    fn add_residual(mut self, residual: Option<&ProgramNoiseState>) -> Self {
        if let Some(residual) = residual {
            self.fresh_bound += &residual.bound;
            self.output_bound += &residual.bound;
            // The accepted average-case model treats reconstruction residuals
            // as independent additive errors. This is a heuristic, not an
            // independence guarantee about the underlying ciphertext.
            self.fresh_variance = self.fresh_variance.add(&residual.doubled_variance);
            self.output_variance = self.output_variance.add(&residual.doubled_variance);
        }
        self
    }

    fn output(&self) -> ProgramNoiseState {
        ProgramNoiseState {
            bound: self.output_bound.clone(),
            doubled_variance: self.output_variance.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct OperationRecord {
    operation: GateNoiseOperation,
    transfers: Vec<PrimitiveNoiseTransfer>,
    constituents: Vec<ProgramNoiseState>,
    active_count: usize,
}

/// Public BGG transfer parameters. Helper variance is Var(2e), not Var(e).
pub struct BggNoiseModel<'a> {
    pub helper_cutoff: &'a BigUint,
    pub helper_variance: &'a NonnegativeRational,
    pub actions: &'a [GateNoiseGains],
    pub operation_residuals: &'a BTreeMap<usize, ProgramNoiseState>,
}

pub struct BggNoiseEvaluator<'a> {
    pub(super) ranges: &'a ProgramFamilyBindings,
    pub(super) program: &'a ExponentLutProgram,
    pub(super) helper_cutoff: &'a BigUint,
    pub(super) helper_variance: &'a NonnegativeRational,
    pub(super) actions: &'a [GateNoiseGains],
    pub(super) operation_residuals: &'a BTreeMap<usize, ProgramNoiseState>,
    pub(super) records: BTreeMap<ProgramGateInstance, OperationRecord>,
}

impl BggNoiseEvaluator<'_> {
    fn scoped<'a>(
        &self,
        value: &'a NoiseProgramWire,
        gate: ProgramGateInstance,
    ) -> Result<std::borrow::Cow<'a, NoiseProgramWire>, ProgramNoiseError> {
        match (value, gate) {
            (NoiseProgramWire::Scalar(state), ProgramGateInstance::FamilyGate { family, .. }) => {
                let range =
                    self.ranges.range(family).ok_or(ProgramNoiseError::FamilyShapeMismatch)?;
                Ok(std::borrow::Cow::Owned(NoiseProgramWire::Family(FamilyNoiseState::Uniform {
                    state: state.clone(),
                    active_count: range_count(range)?,
                })))
            }
            _ => Ok(std::borrow::Cow::Borrowed(value)),
        }
    }

    pub(super) fn reduce_add(
        &mut self,
        value: &NoiseProgramWire,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<NoiseProgramWire, ProgramNoiseError> {
        let active_count = range_count(range)?;
        let values = match value {
            NoiseProgramWire::Family(values) => std::borrow::Cow::Borrowed(values),
            NoiseProgramWire::Scalar(state) => std::borrow::Cow::Owned(FamilyNoiseState::Uniform {
                state: state.clone(),
                active_count,
            }),
        };
        if values.active_count() != active_count {
            return Err(ProgramNoiseError::FamilyShapeMismatch);
        }
        // Only active members participate. Shared/unknown covariance remains
        // coherent: uniform families use r^2 V. Heterogeneous families use
        // Cauchy bounds without amplifying noise by deterministic members.
        let (bound_sum, variance_sum, constituents) = match values.as_ref() {
            FamilyNoiseState::Uniform { state, .. } => (
                &state.bound * active_count,
                state.doubled_variance.scaled(&BigUint::from(active_count)),
                vec![state.clone()],
            ),
            FamilyNoiseState::Members(values) => {
                let bound = values.iter().fold(BigUint::zero(), |sum, state| sum + &state.bound);
                let variance = values.iter().fold(NonnegativeRational::zero(), |sum, state| {
                    sum.add(&state.doubled_variance)
                });
                (bound, variance, values.clone())
            }
        };
        let output_variance = match values.as_ref() {
            FamilyNoiseState::Uniform { .. } => variance_sum.scaled(&BigUint::from(active_count)),
            FamilyNoiseState::Members(members) => NonnegativeRational::sum_variance_bounds(
                members.iter().map(|member| &member.doubled_variance),
            ),
        };
        let transfer = PrimitiveNoiseTransfer {
            inherited_bound: bound_sum.clone(),
            fresh_bound: BigUint::zero(),
            output_bound: bound_sum,
            inherited_variance: variance_sum,
            fresh_variance: NonnegativeRational::zero(),
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

impl ProgramLoweringTypes for BggNoiseEvaluator<'_> {
    type Wire = NoiseProgramWire;
    type Rhs = NoiseProgramWire;
    type HelperSet = ();
    type Error = ProgramNoiseError;
}

impl ExponentProgramLowering for BggNoiseEvaluator<'_> {
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let private = self.scoped(private, gate)?;
        let public = self.scoped(public, gate)?;
        let gain = self.action(gate, GateNoiseOperation::MulPublic)?;
        let action = gain.action.clone();
        let plaintext_action = gain.public_plaintext_action.clone();
        let residual = self.operation_residuals.get(&gain.operation_index);
        match (private.as_ref(), public.as_ref()) {
            (NoiseProgramWire::Scalar(private), NoiseProgramWire::Scalar(public)) => {
                let transfer = transfer_mul_public(private, public, &action, &plaintext_action)
                    .add_residual(residual);
                let output = transfer.output();
                self.record(gate, GateNoiseOperation::MulPublic, vec![transfer], 1);
                Ok(NoiseProgramWire::Scalar(output))
            }
            (NoiseProgramWire::Family(private), NoiseProgramWire::Family(public)) => {
                let active_count = private.active_count();
                let (output, transfers) = private.zip_map(public, |private, public| {
                    transfer_mul_public(private, public, &action, &plaintext_action)
                        .add_residual(residual)
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
        let lhs = self.scoped(lhs, gate)?;
        let rhs = self.scoped(rhs, gate)?;
        let _ = self.action(gate, GateNoiseOperation::Add)?;
        match (lhs.as_ref(), rhs.as_ref()) {
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

    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        _: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let input = self.scoped(input, gate)?;
        let rhs = self.scoped(rhs, gate)?;
        let gain = self.action(gate, GateNoiseOperation::MulPrivate)?;
        let residual = self.operation_residuals.get(&gain.operation_index);
        let transfer = |state: &ProgramNoiseState, rhs: &ProgramNoiseState| {
            transfer_mul_private(state, &gain.action, &rhs.bound, &rhs.doubled_variance)
                .add_residual(residual)
        };
        let (output, transfers, count) = match (input.as_ref(), rhs.as_ref()) {
            (NoiseProgramWire::Scalar(state), NoiseProgramWire::Scalar(rhs)) => {
                let value = transfer(state, rhs);
                (NoiseProgramWire::Scalar(value.output()), vec![value], 1)
            }
            (NoiseProgramWire::Family(states), NoiseProgramWire::Scalar(rhs)) => {
                let (output, transfers) = states.map(|state| transfer(state, rhs));
                (NoiseProgramWire::Family(output), transfers, states.active_count())
            }
            (NoiseProgramWire::Scalar(state), NoiseProgramWire::Family(rhs)) => {
                let (output, transfers) = rhs.map(|rhs| transfer(state, rhs));
                (NoiseProgramWire::Family(output), transfers, rhs.active_count())
            }
            (NoiseProgramWire::Family(states), NoiseProgramWire::Family(rhs)) => {
                let (output, transfers) = states.zip_map(rhs, transfer)?;
                (NoiseProgramWire::Family(output), transfers, states.active_count())
            }
        };
        // Shared RHS errors retain their source through the dependency graph;
        // reductions do not treat repeated uses as fresh samples.
        self.record(gate, GateNoiseOperation::MulPrivate, transfers, count);
        Ok(output)
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        _helpers: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let input = self.scoped(input, gate)?;
        let input = input.as_ref();
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
                    .map(|transfer| {
                        transfer.add_residual(self.operation_residuals.get(&gain.operation_index))
                    })
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
}

fn transfer_mul_private(
    inherited: &ProgramNoiseState,
    action: &RightActionNoiseGain,
    helper_cutoff: &BigUint,
    helper_variance: &NonnegativeRational,
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
    plaintext_action: &RightActionNoiseGain,
) -> PrimitiveNoiseTransfer {
    // e_out = nu_public * e_private + e_public * K_private.
    // H1 supplies the coefficient variance model for the public polynomial;
    // the decomposition action on the label error is accounted separately.
    PrimitiveNoiseTransfer {
        inherited_bound: private.bound.clone(),
        fresh_bound: &action.l1 * &public.bound,
        output_bound: &plaintext_action.l1 * &private.bound + &action.l1 * &public.bound,
        inherited_variance: private.doubled_variance.clone(),
        fresh_variance: public.doubled_variance.scaled(&action.l2_squared),
        output_variance: private
            .doubled_variance
            .scaled(&plaintext_action.l2_squared)
            .add(&public.doubled_variance.scaled(&action.l2_squared)),
    }
}

fn transfer_add(lhs: &ProgramNoiseState, rhs: &ProgramNoiseState) -> PrimitiveNoiseTransfer {
    PrimitiveNoiseTransfer {
        inherited_bound: lhs.bound.clone(),
        fresh_bound: rhs.bound.clone(),
        output_bound: &lhs.bound + &rhs.bound,
        inherited_variance: lhs.doubled_variance.clone(),
        fresh_variance: rhs.doubled_variance.clone(),
        output_variance: lhs.doubled_variance.sum_variance_bound(&rhs.doubled_variance),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_approximate_gadget_residual_matches_uniform_ternary_model() {
        let moduli = vec![BigUint::from(3u8), BigUint::from(5u8)];
        let residual =
            super::approximate_gadget_residual(&moduli, &2u8.into(), 3, 2, 1).unwrap().unwrap();
        // Enumerate the model's two independent discarded residues and one
        // ternary secret coefficient; use its exact moments as the oracle.
        let errors = (-2i64..=2)
            .flat_map(|r0| (-2i64..=2).flat_map(move |r1| (-1i64..=1).map(move |s| r0 + s * r1)))
            .collect::<Vec<_>>();
        assert_eq!(errors.iter().sum::<i64>(), 0);
        assert_eq!(
            residual.bound,
            BigUint::from(errors.iter().map(|e| e.unsigned_abs()).max().unwrap())
        );
        assert_eq!(
            residual.doubled_variance,
            NonnegativeRational::new(
                BigUint::from(4 * errors.iter().map(|e| e * e).sum::<i64>() as u64),
                BigUint::from(errors.len()),
            )
            .unwrap()
        );
        assert!(
            super::approximate_gadget_residual(&moduli, &2u8.into(), 6, 2, 1).unwrap().is_none()
        );
    }

    #[test]
    fn test_operation_residual_uses_heuristic_independence() {
        let zero = ProgramNoiseState {
            bound: BigUint::zero(),
            doubled_variance: NonnegativeRational::zero(),
        };
        let variance = NonnegativeRational::new(2u8.into(), 1u8.into()).unwrap();
        let ordinary = super::transfer_mul_private(
            &zero,
            &RightActionNoiseGain::unit(),
            &4u8.into(),
            &variance,
        );
        // Under the accepted residual-independence model, equal marginal
        // variances add. Explicit shared-wire sums use a different transfer.
        let residue = ProgramNoiseState { bound: 4u8.into(), doubled_variance: variance.clone() };
        let modeled = ordinary.clone().add_residual(Some(&residue));
        assert_eq!(modeled.output_bound, BigUint::from(8u8));
        assert_eq!(modeled.output_variance, variance.scaled(&2u8.into()));
        assert_eq!(modeled.fresh_variance, modeled.output_variance);
        let constant = ProgramNoiseState { bound: 5u8.into(), ..zero };
        let shifted = ordinary.clone().add_residual(Some(&constant));
        assert_eq!(shifted.output_bound, BigUint::from(9u8));
        assert_eq!(shifted.output_variance, ordinary.output_variance);
        assert_eq!(shifted.fresh_variance, ordinary.fresh_variance);
    }

    #[test]
    fn refresh_route_gains_require_actual_per_slot_decompositions() {
        let plan =
            NoiseDecompositionPlan::new(BigUint::from(2u8), 1, vec![BigInt::from(2u8)], 1, 1, 1)
                .expect("concrete balanced decomposition");
        let decompositions = RefreshNoiseGainDecompositions {
            state_scaling: vec![plan.clone(), plan.clone()],
            mask: vec![vec![plan.clone()]; 2],
            mask_multiplicity: 1,
            fresh: vec![vec![plan.clone()], vec![plan.clone()]],
            fresh_multiplicity: 1,
            decoder: vec![plan.clone(), plan.clone()],
        };
        let gains = decompositions.derive_route_gains(2).expect("per-slot route gains");
        assert_eq!(gains.state_scaling[0].l1, BigUint::from(2u8));
        assert_eq!(gains.state_scaling[0].l2_squared, BigUint::from(4u8));
        assert_eq!(gains.fresh.len(), 2);
        assert!(
            RefreshNoiseGainDecompositions { fresh: vec![], ..decompositions }
                .derive_route_gains(2)
                .is_err()
        );
    }

    #[test]
    fn refresh_route_gains_sum_each_p_digit_and_ring_coordinate() {
        // P=2 is intentionally distinct from gadget beta=4.  Each of three
        // P-adic digits is emitted for each of three ring coordinates, while
        // state scaling remains one exact action per unequal CRT slot.
        let crt = vec![BigUint::from(101u16), BigUint::from(103u16), BigUint::from(107u16)];
        let route = |value: BigInt, ring_coefficient: usize| {
            NoiseDecompositionPlan::sparse_monomial(
                BigUint::from(4u8),
                12,
                2,
                4,
                8,
                1,
                0,
                ring_coefficient,
                value,
            )
            .unwrap()
            .bind_crt_layout(crt.clone(), 4)
            .unwrap()
        };
        let mask: Vec<_> = (0u32..3)
            .flat_map(|digit| (0..3).map(move |j| route(BigUint::from(2u8).pow(digit).into(), j)))
            .collect();
        let fresh: Vec<Vec<_>> = [BigUint::from(7u8), BigUint::from(3u8)]
            .into_iter()
            .map(|kappa| {
                (0u32..3)
                    .flat_map(|digit| {
                        let value = &kappa * BigUint::from(2u8).pow(digit);
                        (0..3).map(move |j| route(value.clone().into(), j))
                    })
                    .collect()
            })
            .collect();
        let state_scaling = [BigUint::from(7u8), BigUint::from(3u8)]
            .into_iter()
            .map(|kappa| {
                NoiseDecompositionPlan::gadget_scalar(
                    BigUint::from(4u8),
                    12,
                    2,
                    24,
                    8,
                    kappa.into(),
                )
                .unwrap()
                .bind_crt_layout(crt.clone(), 4)
                .unwrap()
            })
            .collect();
        let decoder = vec![route(1.into(), 0), route(1.into(), 0)];
        let decompositions = RefreshNoiseGainDecompositions {
            state_scaling,
            mask: vec![mask; 2],
            mask_multiplicity: 1,
            fresh_multiplicity: 1,
            fresh,
            decoder,
        };
        let gains = decompositions.derive_route_gains(2).unwrap();
        let sum = |plans: &[NoiseDecompositionPlan]| {
            plans.iter().fold(
                RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() },
                |mut total, plan| {
                    let gain = plan.action_gain();
                    total.l1 += gain.l1;
                    total.l2_squared += gain.l2_squared;
                    total
                },
            )
        };
        assert_eq!(gains.mask[0], sum(&decompositions.mask[0]));
        assert_eq!(gains.mask[1], sum(&decompositions.mask[1]));
        assert_eq!(gains.fresh[0], sum(&decompositions.fresh[0]));
        assert_eq!(gains.fresh[1], sum(&decompositions.fresh[1]));
        assert_ne!(gains.fresh[0], gains.fresh[1]);
        assert_ne!(gains.state_scaling[0], gains.state_scaling[1]);

        // Compact route plans represent all ring-coordinate routes with a
        // multiplicity; they must produce exactly the same gain as the
        // explicit per-coordinate plans.  The PRF component axis is already
        // the target-column axis of `sparse_monomial` and is therefore not a
        // second multiplicity factor.
        // A single action per P-adic digit is the compact representation used
        // by the graph-free path.  Repeat it for each ring coordinate and
        // compare against the corresponding explicit route sum.
        let compact = RefreshNoiseGainDecompositions {
            state_scaling: decompositions.state_scaling.clone(),
            mask: vec![vec![route(1.into(), 0), route(2.into(), 0), route(4.into(), 0)]; 2],
            mask_multiplicity: 3,
            fresh: vec![
                vec![route(7.into(), 0), route(14.into(), 0), route(28.into(), 0)],
                vec![route(3.into(), 0), route(6.into(), 0), route(12.into(), 0)],
            ],
            fresh_multiplicity: 3,
            decoder: decompositions.decoder.clone(),
        };
        let explicit = decompositions.derive_route_gains(2).unwrap();
        let compact_gains = compact.derive_route_gains(2).unwrap();
        assert_eq!(explicit.mask, compact_gains.mask);
        assert_eq!(explicit.fresh, compact_gains.fresh);
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
            output_variance: NonnegativeRational::zero(),
        };
        let gate = GateNoiseGains {
            public_plaintext_action: RightActionNoiseGain::unit(),
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
            inherited_variance: NonnegativeRational::zero(),
            fresh_variance: NonnegativeRational::zero(),
            output_variance: NonnegativeRational::zero(),
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

    use super::*;
    use crate::layout::ExpLayout;

    struct TestTransfer {
        output_bound: BigUint,
        fresh_bound: BigUint,
        output_variance: NonnegativeRational,
    }

    fn transfer_mul_private(
        input_bound: &BigUint,
        input_variance: &NonnegativeRational,
        action: &RightActionNoiseGain,
        helper_bound: &BigUint,
        helper_variance: &NonnegativeRational,
    ) -> TestTransfer {
        TestTransfer {
            output_bound: &action.l1 * input_bound + helper_bound,
            fresh_bound: helper_bound.clone(),
            output_variance: input_variance.scaled(&action.l2_squared).add(helper_variance),
        }
    }

    fn transfer_mul_public(
        private_bound: &BigUint,
        private_variance: &NonnegativeRational,
        public_bound: &BigUint,
        public_variance: &NonnegativeRational,
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
        input_variance: &NonnegativeRational,
        direct: &RightActionNoiseGain,
        helper: &RightActionNoiseGain,
        helper_bound: &BigUint,
        helper_variance: &NonnegativeRational,
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
        inherited_variance: NonnegativeRational,
        fresh_variance: NonnegativeRational,
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
    fn decoder_preimage_plans_preserve_slot_shape_and_bound() {
        let ring = mxx_dsl::Ring::new(15, 4);
        let preimages = [
            ring.preimage_input("decoder-slot-0", (8, 4), 2),
            ring.preimage_input("decoder-slot-1", (7, 5), 3),
        ];
        let schemas =
            preimages.iter().map(compact_preimage_schema).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(schemas[0].max_coefficient_bound, BigUint::from(2u8));
        assert_eq!(schemas[1].max_coefficient_bound, BigUint::from(3u8));
        let plans = schemas
            .iter()
            .map(|schema| bounded_plan_from_schema(&BigUint::from(4u8), schema))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!((plans[0].target_rows, plans[0].target_columns), (8, 4));
        assert_eq!((plans[1].target_rows, plans[1].target_columns), (7, 5));
        assert_eq!(plans[0].action_gain().l1, BigUint::from(64u8));
        assert_eq!(plans[1].action_gain().l1, BigUint::from(84u8));
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
    fn lut_branch_includes_direct_and_acted_helper_terms() {
        let direct =
            RightActionNoiseGain { l1: BigUint::from(2u8), l2_squared: BigUint::from(9u8) };
        let helper =
            RightActionNoiseGain { l1: BigUint::from(4u8), l2_squared: BigUint::from(16u8) };
        let branch = transfer_lut(
            &BigUint::from(5u8),
            &NonnegativeRational::new(BigUint::from(7u8), BigUint::one()).unwrap(),
            &direct,
            &helper,
            &BigUint::from(2u8),
            &NonnegativeRational::new(BigUint::from(2u8), BigUint::one()).unwrap(),
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
            &NonnegativeRational::new(BigUint::from(2u8), BigUint::one()).unwrap(),
            &BigUint::from(5u8),
            &NonnegativeRational::new(BigUint::from(5u8), BigUint::one()).unwrap(),
            &action,
        );
        assert_eq!(transfer.output_bound, BigUint::from(22u8));
        assert_eq!(transfer.output_variance.numerator, BigUint::from(47u8));
    }

    #[test]
    fn flat_branches_use_l2_squared_not_l1_squared() {
        let variance = NonnegativeRational::new(BigUint::from(7u8), BigUint::one()).unwrap();
        let direct = RightActionNoiseGain { l1: 2u8.into(), l2_squared: 9u8.into() };
        let helper = RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() };
        let helper_variance = NonnegativeRational::zero();
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
            NonnegativeRational::new(BigUint::from(63u8), BigUint::one()).unwrap()
        );
    }

    #[test]
    fn binary_lut_noise_propagates_product_error_and_residual() {
        use crate::program::{
            AttributeKnowledge, ExponentLutProgramBuilder, Gate, LutTable, ProgramWireType,
        };
        let build = |automatic| {
            let mut builder = ExponentLutProgramBuilder::new();
            builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 2 }, 16).unwrap();
            let input = builder
                .input(
                    2,
                    ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private },
                )
                .unwrap();
            let family = builder.rhs_family(2).unwrap();
            let rhs = builder.rhs_input(family, 2, 2).unwrap();
            let binary =
                builder.register_lut(LutTable::binary(2, 2, 2, vec![0, 1, 1, 0]).unwrap()).unwrap();
            let wire = builder.input_wire(input).unwrap();
            let output = if automatic {
                builder.lut(wire, binary, Some(rhs)).unwrap()
            } else {
                let unary =
                    builder.register_lut(LutTable::unary(4, 2, vec![0, 1, 1, 0]).unwrap()).unwrap();
                let product = builder.private_mul(wire, rhs).unwrap();
                builder.lut(product, unary, None).unwrap()
            };
            builder.output(output).unwrap();
            (builder.build().unwrap(), input, rhs)
        };
        let (automatic, input, rhs) = build(true);
        let (explicit, _, _) = build(false);
        assert_eq!(automatic.id(), explicit.id());
        let unit = NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap();
        let state = |bound| ProgramNoiseState {
            bound: BigUint::from(bound),
            doubled_variance: unit.clone(),
        };
        let inputs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(input, state(3u8))]),
            rhs_errors: BTreeMap::from([(rhs, state(5u8))]),
            operation_residuals: BTreeMap::from([(0, state(7u8))]),
            ..Default::default()
        };
        let actions = (0..5)
            .map(|index| GateNoiseGains {
                public_plaintext_action: RightActionNoiseGain::unit(),
                step_index: usize::from(index != 0),
                operation_index: index,
                branch_index: if index == 0 { 0 } else { index - 1 },
                operation: if index == 0 {
                    GateNoiseOperation::MulPrivate
                } else {
                    GateNoiseOperation::Lut
                },
                gate: format!("separated-{index}"),
                action: RightActionNoiseGain { l1: 2u8.into(), l2_squared: 4u8.into() },
                helper_action: (index != 0).then(RightActionNoiseGain::unit),
                inherited_bound: 0u8.into(),
                fresh_bound: 0u8.into(),
                output_bound: 0u8.into(),
                inherited_variance: NonnegativeRational::zero(),
                fresh_variance: NonnegativeRational::zero(),
                output_variance: NonnegativeRational::zero(),
            })
            .collect::<Vec<_>>();
        let run = |program| {
            simulate_program(program, &inputs, &2u8.into(), &unit, &actions, &[]).unwrap()
        };
        let actual = run(&automatic);
        let expected = run(&explicit);
        assert_eq!(actual.dependencies, expected.dependencies);
        assert_eq!(actual.wire_states, expected.wire_states);
        let Gate::MulPrivate { output: product, .. } = automatic.gates()[0] else {
            panic!("product")
        };
        // The residual belongs inside the LUT input: 2*3 + 5 + 7 = 18.
        assert_eq!(actual.wire_states[&product].scalar().unwrap().bound, BigUint::from(18u8));
        // Four branches each contribute 2*18 + 2*(1+1) = 40.
        assert_eq!(actual.output_state().bound, BigUint::from(160u16));
        // H3 aggregates four branches of variance 4*(4*1+1+1)+(1+1).
        assert_eq!(actual.output_state().doubled_variance, unit.scaled(&104u8.into()));
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

        let unit = NonnegativeRational::new(BigUint::one(), BigUint::one()).unwrap();
        let action = |operation, operation_index, branch_index, l1, l2_squared| GateNoiseGains {
            public_plaintext_action: RightActionNoiseGain::unit(),
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
            inherited_variance: NonnegativeRational::zero(),
            fresh_variance: NonnegativeRational::zero(),
            output_variance: NonnegativeRational::zero(),
        };
        let actions = vec![
            action(GateNoiseOperation::MulPrivate, 0, 0, 2u8.into(), 4u8.into()),
            action(GateNoiseOperation::MulPublic, 1, 1, 3u8.into(), 9u8.into()),
            action(GateNoiseOperation::Add, 2, 0, 1u8.into(), 1u8.into()),
            action(GateNoiseOperation::Add, 3, 0, 1u8.into(), 1u8.into()),
        ];
        let inputs = ProgramNoiseInputs {
            operation_residuals: BTreeMap::new(),
            rhs_errors: BTreeMap::new(),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::one(),
                        doubled_variance: unit.clone(),
                    },
                    active_count: 2,
                },
            )]),
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState { bound: BigUint::from(3u8), doubled_variance: unit.clone() },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Members(vec![
                    ProgramNoiseState {
                        bound: 5u8.into(),
                        doubled_variance: NonnegativeRational::new(3u8.into(), 1u8.into()).unwrap()
                    };
                    2
                ]),
            )]),
        };
        let report =
            simulate_program(&program, &inputs, &BigUint::one(), &unit, &actions, &[2]).unwrap();
        // A public polynomial with norms ||nu||_1=3, ||nu||_2^2=5
        // acts on the inherited error separately from the decomposition.
        let mut polynomial_actions = actions.clone();
        polynomial_actions[1].public_plaintext_action =
            RightActionNoiseGain { l1: 3u8.into(), l2_squared: 5u8.into() };
        let polynomial_report =
            simulate_program(&program, &inputs, &BigUint::one(), &unit, &polynomial_actions, &[2])
                .unwrap();
        assert_eq!(polynomial_report.output_state().bound, BigUint::from(72u8));
        assert_eq!(
            polynomial_report.output_state().doubled_variance,
            unit.scaled(&BigUint::from(208u16))
        );
        let dependencies = &report.dependencies;
        let crate::noise_utils::NoiseDependencyNode::Add { lhs, rhs, .. } =
            &dependencies.nodes[dependencies.wires[&trailing].0]
        else {
            panic!("the trailing primitive is Add")
        };
        assert_eq!(lhs, rhs);
        assert_eq!(*lhs, dependencies.wires[&program.input_wires()[&input]]);
        assert!(dependencies.nodes.iter().any(|node| matches!(node,
            crate::noise_utils::NoiseDependencyNode::MulPrivate { input: parent, .. } if parent == lhs
        )));
        let restored: crate::noise_utils::NoiseDependencyGraph =
            serde_json::from_slice(&serde_json::to_vec(dependencies).unwrap()).unwrap();
        assert_eq!(&restored, dependencies);
        assert_eq!(report.steps[0].family_body.len(), 2);
        assert_eq!(report.steps[0].family_member_count, 2);
        assert_eq!(report.steps[0].family_body[0].output_bound, BigUint::from(7u8));
        assert_eq!(report.steps[0].family_body[1].fresh_variance.numerator, BigUint::from(27u8));
        // A family member is not a separate noise semantic: the recorded
        // rows must equal the same primitive transfers used by ordinary
        // MulPrivate / MulPublic operations.
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
            &NonnegativeRational::new(BigUint::from(3u8), BigUint::one()).unwrap(),
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
            inherited_variance: NonnegativeRational::new(BigUint::from(64u8), BigUint::one())
                .unwrap(),
            fresh_variance: NonnegativeRational::zero(),
            operand_count: 2,
        });
        assert_eq!(report.steps[0].output_bound, expected_reduction.output_bound);
        assert_eq!(report.steps[0].output_variance, expected_reduction.output_variance);
        assert_ne!(
            report.steps.last().unwrap().output_variance,
            report.wire_states[&output].scalar().unwrap().doubled_variance
        );
        assert_eq!(
            report.output_state().bound,
            report.wire_states[&output].scalar().unwrap().bound
        );
        let _ = trailing;

        // Prepared family errors must survive binding rather than reverting
        // to one helper's error. Isolate them from inherited/public errors;
        // the second member is deterministic, so reduction keeps variance 1.
        let prepared_rhs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState {
                    bound: BigUint::zero(),
                    doubled_variance: NonnegativeRational::zero(),
                },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::zero(),
                        doubled_variance: NonnegativeRational::zero(),
                    },
                    active_count: 2,
                },
            )]),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Members(vec![
                    ProgramNoiseState { bound: 3u8.into(), doubled_variance: unit.clone() },
                    ProgramNoiseState {
                        bound: 5u8.into(),
                        doubled_variance: NonnegativeRational::zero(),
                    },
                ]),
            )]),
            ..Default::default()
        };
        let prepared_rhs =
            serde_json::from_slice(&serde_json::to_vec(&prepared_rhs).unwrap()).unwrap();
        let prepared =
            simulate_program(&program, &prepared_rhs, &BigUint::one(), &unit, &actions, &[2])
                .unwrap();
        assert_eq!(prepared.output_state().bound, BigUint::from(8u8));
        assert_eq!(prepared.wire_states[&output].scalar().unwrap().doubled_variance, unit);

        // Equal bounds with different member variances remain heterogeneous;
        // they cannot use a single representative's closed-form transfer.
        let mixed_inputs = ProgramNoiseInputs {
            operation_residuals: BTreeMap::new(),
            rhs_errors: BTreeMap::new(),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::one(),
                        doubled_variance: unit.clone(),
                    },
                    active_count: 2,
                },
            )]),
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState { bound: BigUint::from(3u8), doubled_variance: unit.clone() },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Members(
                    [3u8, 7]
                        .into_iter()
                        .map(|v| ProgramNoiseState {
                            bound: 5u8.into(),
                            doubled_variance: NonnegativeRational::new(v.into(), 1u8.into())
                                .unwrap(),
                        })
                        .collect(),
                ),
            )]),
        };
        let mixed_report =
            simulate_program(&program, &mixed_inputs, &BigUint::one(), &unit, &actions, &[2])
                .unwrap();
        // Marginal variances are 32 and 68. Cauchy gives
        // 100 + 2*sqrt(2176), rounded outward to 194 rather than 2*100.
        assert_eq!(mixed_report.steps[0].output_variance.numerator, BigUint::from(194u8));
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

        let unit = NonnegativeRational::new(BigUint::one(), BigUint::one()).unwrap();
        let private_action = RightActionNoiseGain { l1: 2u8.into(), l2_squared: 4u8.into() };
        let public_action = RightActionNoiseGain { l1: 3u8.into(), l2_squared: 9u8.into() };
        let public_bound = BigUint::from(5u8);
        let public_variance = NonnegativeRational::new(BigUint::from(3u8), BigUint::one()).unwrap();
        let actions = vec![
            GateNoiseGains {
                public_plaintext_action: RightActionNoiseGain::unit(),
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
                inherited_variance: NonnegativeRational::zero(),
                fresh_variance: NonnegativeRational::zero(),
                output_variance: NonnegativeRational::zero(),
            },
            GateNoiseGains {
                public_plaintext_action: RightActionNoiseGain::unit(),
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
                inherited_variance: NonnegativeRational::zero(),
                fresh_variance: NonnegativeRational::zero(),
                output_variance: NonnegativeRational::zero(),
            },
            GateNoiseGains {
                public_plaintext_action: RightActionNoiseGain::unit(),
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
                inherited_variance: NonnegativeRational::zero(),
                fresh_variance: NonnegativeRational::zero(),
                output_variance: NonnegativeRational::zero(),
            },
        ];
        let base_inputs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState { bound: BigUint::from(3u8), doubled_variance: unit.clone() },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: public_bound.clone(),
                        doubled_variance: public_variance.clone(),
                    },
                    active_count: 3,
                },
            )]),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::one(),
                        doubled_variance: unit.clone(),
                    },
                    active_count: 3,
                },
            )]),
            ..ProgramNoiseInputs::default()
        };
        let explicit_inputs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState { bound: BigUint::from(3u8), doubled_variance: unit.clone() },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Members(vec![
                    ProgramNoiseState {
                        bound: public_bound.clone(),
                        doubled_variance: public_variance.clone()
                    };
                    3
                ]),
            )]),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::one(),
                        doubled_variance: unit.clone(),
                    },
                    active_count: 3,
                },
            )]),
            ..ProgramNoiseInputs::default()
        };
        let uniform =
            simulate_program(&program, &base_inputs, &BigUint::one(), &unit, &actions, &[3])
                .unwrap();
        let explicit =
            simulate_program(&program, &explicit_inputs, &BigUint::one(), &unit, &actions, &[3])
                .unwrap();
        assert_eq!(uniform.wire_states, explicit.wire_states);
        assert_eq!(uniform.output_state().bound, explicit.output_state().bound);
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
        let mut aggregate_variance = NonnegativeRational::zero();
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
                &NonnegativeRational::new(BigUint::from(variance), BigUint::one()).unwrap(),
                &public_action,
            );
            aggregate_bound += public.output_bound;
            aggregate_variance = aggregate_variance.add(&public.output_variance);
        }
        let reduced = transfer_add(AddInputSummary {
            inherited_bound: aggregate_bound,
            fresh_bound: BigUint::zero(),
            inherited_variance: aggregate_variance.clone(),
            fresh_variance: NonnegativeRational::zero(),
            operand_count: public_bounds.len(),
        });
        assert_eq!(reduced.output_bound, BigUint::from(75u8));
        assert_eq!(reduced.output_variance.numerator, BigUint::from(369u32));

        let heterogeneous_inputs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(
                input,
                ProgramNoiseState { bound: BigUint::from(3u8), doubled_variance: unit.clone() },
            )]),
            public_family_errors: BTreeMap::from([(
                public_family,
                FamilyNoiseState::Members(
                    public_bounds
                        .iter()
                        .zip(&public_variances)
                        .map(|(&b, &v)| ProgramNoiseState {
                            bound: b.into(),
                            doubled_variance: NonnegativeRational::new(v.into(), 1u8.into())
                                .unwrap(),
                        })
                        .collect(),
                ),
            )]),
            rhs_family_errors: BTreeMap::from([(
                selector_family,
                FamilyNoiseState::Uniform {
                    state: ProgramNoiseState {
                        bound: BigUint::one(),
                        doubled_variance: unit.clone(),
                    },
                    active_count: 3,
                },
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
        // Marginals 32, 41, 50: the first pair is bounded by 146;
        // 146 + 50 + 2*ceil(sqrt(146*50)) = 368, below the old 369.
        let correlated = NonnegativeRational::new(368u32.into(), 1u8.into()).unwrap();
        assert_eq!(heterogeneous.steps[0].output_variance, correlated);
        assert_eq!(
            heterogeneous.wire_states[&output].scalar().unwrap().bound,
            reduced.output_bound
        );
        assert_eq!(
            heterogeneous.wire_states[&output].scalar().unwrap().doubled_variance,
            correlated
        );

        // One variable public-label error and two deterministic members.
        // The sole variable term has gain 3, hence variance 9, independently
        // of the two constant errors' possibly nonzero means.
        let mut one_variable = heterogeneous_inputs;
        one_variable.input_errors.get_mut(&input).unwrap().doubled_variance =
            NonnegativeRational::zero();
        let FamilyNoiseState::Members(members) =
            one_variable.public_family_errors.get_mut(&public_family).unwrap()
        else {
            panic!("member inputs")
        };
        for (index, member) in members.iter_mut().enumerate() {
            member.doubled_variance =
                if index == 0 { unit.clone() } else { NonnegativeRational::zero() };
        }
        let FamilyNoiseState::Uniform { state, .. } =
            one_variable.rhs_family_errors.get_mut(&selector_family).unwrap()
        else {
            panic!("uniform selector inputs")
        };
        state.doubled_variance = NonnegativeRational::zero();
        let one_variable = simulate_program(
            &program,
            &one_variable,
            &BigUint::one(),
            &NonnegativeRational::zero(),
            &actions,
            &[3],
        )
        .unwrap();
        assert_eq!(
            one_variable.wire_states[&output].scalar().unwrap().doubled_variance,
            unit.scaled(&9u8.into())
        );
    }

    #[test]
    fn test_shared_rhs_family_preserves_coherent_noise() {
        use crate::program::{AttributeKnowledge, ExponentLutProgramBuilder, ProgramWireType};
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 4 }, 16).unwrap();
        let input_id = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs = builder.rhs_input(family, 2, 2).unwrap();
        let output = builder.private_mul(builder.input_wire(input_id).unwrap(), rhs).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let action = RightActionNoiseGain::unit();
        let zero = NonnegativeRational::zero();
        let actions = [GateNoiseGains {
            public_plaintext_action: RightActionNoiseGain::unit(),
            step_index: 0,
            operation_index: 0,
            branch_index: 0,
            operation: GateNoiseOperation::MulPrivate,
            gate: "shared-rhs".into(),
            action: action.clone(),
            helper_action: None,
            inherited_bound: 0u8.into(),
            fresh_bound: 0u8.into(),
            output_bound: 0u8.into(),
            inherited_variance: zero.clone(),
            fresh_variance: zero.clone(),
            output_variance: zero,
        }];
        let cutoff = BigUint::from(2u8);
        let variance = NonnegativeRational::new(1u8.into(), 2u8.into()).unwrap();
        let residuals = BTreeMap::new();
        let ranges = ProgramFamilyBindings::new();
        let mut evaluator = BggNoiseEvaluator {
            ranges: &ranges,
            program: &program,
            helper_cutoff: &cutoff,
            helper_variance: &variance,
            actions: &actions,
            operation_residuals: &residuals,
            records: BTreeMap::new(),
        };
        let state = ProgramNoiseState { bound: 3u8.into(), doubled_variance: variance.clone() };
        let input = NoiseProgramWire::Family(FamilyNoiseState::Uniform {
            state: state.clone(),
            active_count: 3,
        });
        let product = evaluator
            .mul_private(
                &input,
                &NoiseProgramWire::Scalar(ProgramNoiseState {
                    bound: cutoff.clone(),
                    doubled_variance: variance.clone(),
                }),
                None,
                ProgramGateInstance::Gate { step: 0 },
            )
            .unwrap();
        let expected =
            super::transfer_mul_private(&state, &action, &cutoff, evaluator.helper_variance)
                .output();
        let reduced = evaluator
            .reduce_add(
                &product,
                &FamilyRange::full(3).unwrap(),
                ProgramGateInstance::FamilyReduction { step: 0 },
            )
            .unwrap();
        let NoiseProgramWire::Scalar(reduced) = reduced else { panic!("scalar reduction") };
        assert_eq!(reduced.bound, expected.bound * 3u8);
        assert_eq!(reduced.doubled_variance, expected.doubled_variance.scaled(&9u8.into()));

        // Three independent selector errors with monomial weights survive
        // even though one-hotness selects only one payload. H4 permits their
        // variances to add before the direct gate.
        let inputs = ProgramNoiseInputs {
            input_errors: BTreeMap::from([(
                input_id,
                ProgramNoiseState {
                    bound: state.bound.clone(),
                    doubled_variance: state.doubled_variance.clone(),
                },
            )]),
            rhs_errors: BTreeMap::from([(
                rhs,
                ProgramNoiseState {
                    bound: &cutoff * 3u8,
                    doubled_variance: variance.scaled(&3u8.into()),
                },
            )]),
            ..Default::default()
        };
        let report =
            simulate_program(&program, &inputs, &cutoff, &variance, &actions, &[]).unwrap();
        assert_eq!(report.wire_states[&output].scalar().unwrap().bound, BigUint::from(9u8));
        assert_eq!(
            report.wire_states[&output].scalar().unwrap().doubled_variance,
            variance.scaled(&4u8.into())
        );
    }

    #[test]
    fn test_add_noise_retains_repeated_wire_and_deterministic_operand() {
        let state = ProgramNoiseState {
            bound: 3u8.into(),
            doubled_variance: NonnegativeRational::new(5u8.into(), 2u8.into()).unwrap(),
        };
        let repeated = super::transfer_add(&state, &state);
        assert_eq!(repeated.output_bound, &state.bound * 2u8);
        assert_eq!(repeated.output_variance, state.doubled_variance.scaled(&4u8.into()));
        let deterministic =
            ProgramNoiseState { bound: 1u8.into(), doubled_variance: NonnegativeRational::zero() };
        let shifted = super::transfer_add(&state, &deterministic);
        assert_eq!(shifted.output_bound, &state.bound + &deterministic.bound);
        assert_eq!(shifted.output_variance, state.doubled_variance);
    }
}

fn transfer_lut_branch(
    input: &ProgramNoiseState,
    direct_action: &RightActionNoiseGain,
    helper_action: &RightActionNoiseGain,
    helper_cutoff: &BigUint,
    helper_variance: &NonnegativeRational,
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
            .fold(NonnegativeRational::zero(), |sum, branch| sum.add(&branch.fresh_variance)),
        output_variance: branches
            .iter()
            .fold(NonnegativeRational::zero(), |sum, branch| sum.add(&branch.output_variance)),
    }
}

/// Interprets a complete Exponent-LUT program and records every wire state.
pub fn simulate_program(
    program: &ExponentLutProgram,
    inputs: &ProgramNoiseInputs,
    helper_cutoff: &BigUint,
    helper_variance: &NonnegativeRational,
    actions: &[GateNoiseGains],
    family_active_counts: &[usize],
) -> Result<ProgramNoiseReport, ProgramNoiseError> {
    // Reject stale or misclassified residual metadata instead of silently
    // dropping a residual intended for another setup action.
    for (index, residual) in &inputs.operation_residuals {
        if residual.doubled_variance.denominator.is_zero() ||
            !actions.iter().any(|gain| {
                gain.operation_index == *index && gain.operation != GateNoiseOperation::Add
            })
        {
            return Err(ProgramNoiseError::InvalidDecomposition);
        }
    }
    let mut scalar_inputs = BTreeMap::new();
    for (input, _) in program.input_wires() {
        scalar_inputs.insert(
            *input,
            NoiseProgramWire::Scalar(
                inputs
                    .input_errors
                    .get(input)
                    .cloned()
                    .ok_or(ProgramNoiseError::MissingInputError(*input))?,
            ),
        );
    }
    let rhs_inputs = program
        .rhs_inputs()
        .keys()
        .map(|id| {
            let state = inputs.rhs_errors.get(id).ok_or(ProgramNoiseError::MissingRhsError(*id))?;
            Ok((*id, NoiseProgramWire::Scalar(state.clone())))
        })
        .collect::<Result<BTreeMap<RhsId, _>, ProgramNoiseError>>()?;
    let mut rhs_families = BTreeMap::<RhsFamilyId, NoiseProgramWire>::new();
    let mut wire_values = BTreeMap::<WireFamilyId, NoiseProgramWire>::new();
    let mut helpers = BTreeMap::<LutId, ()>::new();
    let mut family_ranges = ProgramFamilyBindings::new();
    let mut family_index = 0usize;
    for step in program.steps() {
        match step {
            ProgramStep::Gate(Gate::Lut { lut_id: lut, .. }) => {
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
                    let FamilyRhsSource::Member(family) = source else { continue };
                    rhs_families.insert(
                        *family,
                        NoiseProgramWire::Family(
                            inputs
                                .rhs_family_errors
                                .get(family)
                                .cloned()
                                .ok_or(ProgramNoiseError::MissingRhsFamilyError(*family))?,
                        ),
                    );
                }
                for source in body.wire_inputs().values() {
                    let FamilyWireSource::Member(family) = source else { continue };
                    wire_values.insert(
                        *family,
                        NoiseProgramWire::Family(program_family_input(inputs, *family, count)?),
                    );
                }
                for gate in body.gates() {
                    if let crate::program::Gate::Lut { lut_id: lut, .. } = gate {
                        helpers.insert(*lut, ());
                    }
                }
            }
        }
    }
    if family_index != family_active_counts.len() {
        return Err(ProgramNoiseError::FamilyShapeMismatch);
    }

    let bindings = scalar_inputs
        .into_iter()
        .fold(ProgramBindings::new(), |bindings, (id, value)| bindings.input(id, value));
    let bindings =
        rhs_inputs.into_iter().fold(bindings, |bindings, (id, value)| bindings.rhs(id, value));
    let bindings = rhs_families
        .into_iter()
        .fold(bindings, |bindings, (id, value)| bindings.rhs_family(id, value));
    let bindings = wire_values
        .into_iter()
        .fold(bindings, |bindings, (id, value)| bindings.wire_family(id, value));
    let bindings =
        helpers.into_iter().fold(bindings, |bindings, (id, value)| bindings.helper(id, value));
    let (wires, interpreter, dependencies) = crate::scheme::trace_noise::<super::BggScheme>(
        program,
        BggNoiseModel {
            helper_cutoff,
            helper_variance,
            actions,
            operation_residuals: &inputs.operation_residuals,
        },
        &bindings,
        &family_ranges,
    )
    .map_err(map_lower_error)?;
    let output_wire = *program.outputs().last().ok_or(ProgramNoiseError::MissingOutput)?;
    let NoiseProgramWire::Scalar(_) =
        wires.get(&output_wire).ok_or(ProgramNoiseError::MissingOutput)?
    else {
        return Err(ProgramNoiseError::WireKindMismatch);
    };
    let steps = build_step_report(program, &interpreter.records)?;
    Ok(ProgramNoiseReport { dependencies, wire_states: wires, steps, output: output_wire })
}

fn program_family_input(
    inputs: &ProgramNoiseInputs,
    family: WireFamilyId,
    count: usize,
) -> Result<FamilyNoiseState, ProgramNoiseError> {
    inputs
        .public_family_errors
        .get(&family)
        .filter(|state| state.active_count() == count)
        .cloned()
        .ok_or(ProgramNoiseError::MissingPublicFamilyInput { family, member: count })
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
            ProgramStep::Family { id, body, .. } => {
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
                                family: *id,
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
        ProgramLowerError::Graph(error) => ProgramNoiseError::ProgramTraversal(error.to_string()),
    }
}

fn bit_length(value: &BigUint) -> usize {
    value.bits() as usize
}

/// BGG value-readout transfer from an already evaluated Program output.
/// The input payload is a signed monomial. Gains and residuals are bound by
/// readout preparation; H1/H3/H4 and the additive reconstruction-residual
/// heuristic are the same models used by the primitive evaluator. This
/// operation does not inspect a PRF profile, label schedule or refresh plan.
pub fn evaluate_value_readout_noise(
    output: &ProgramNoiseState,
    helper_bound: &BigUint,
    helper_variance: &NonnegativeRational,
    gains: &ValueReadoutNoiseGains,
    target_plaintext_gain: &RightActionNoiseGain,
    residuals: Option<&(ProgramNoiseState, Vec<ProgramNoiseState>, ProgramNoiseState)>,
) -> Result<ProgramNoiseState, NoiseSimulationError> {
    let zero =
        ProgramNoiseState { bound: BigUint::zero(), doubled_variance: NonnegativeRational::zero() };
    let (target_residual, trace_residuals, normalization_residual) = match residuals {
        Some((target, trace, normalization)) if trace.len() == gains.nonidentity_trace.len() => {
            (target, trace.as_slice(), normalization)
        }
        Some(_) => return Err(NoiseSimulationError::InvalidDecomposition),
        None => (&zero, &[][..], &zero),
    };
    if std::iter::once(target_residual)
        .chain(trace_residuals)
        .chain(std::iter::once(normalization_residual))
        .any(|residual| residual.doubled_variance.denominator.is_zero())
    {
        return Err(NoiseSimulationError::InvalidDecomposition);
    }
    let target_bound = &gains.target.l1 * &output.bound + &target_residual.bound;
    let mut trace_bound = &gains.identity_trace.l1 * &target_bound;
    for gain in &gains.nonidentity_trace {
        trace_bound += &gain.action.l1 * &target_bound;
        // The switch error is multiplied by sigma(mu*T'), where the
        // native accumulator mu is a signed monomial. The online mask
        // error separately acts through its cached public decomposition.
        trace_bound += &(&target_plaintext_gain.l1 + &gain.helper_action.l1) * helper_bound;
    }
    for residual in trace_residuals {
        trace_bound += &residual.bound;
    }
    let readout_bound = &gains.normalization.l1 * trace_bound + &normalization_residual.bound;
    let target_variance = output
        .doubled_variance
        .scaled(&gains.target.l2_squared)
        .add(&target_residual.doubled_variance);
    let mut trace_variance = target_variance.clone();
    for gain in &gains.nonidentity_trace {
        trace_variance = trace_variance.add(&target_variance.scaled(&gain.action.l2_squared));
        trace_variance = trace_variance.add(
            &helper_variance
                .scaled(&(&target_plaintext_gain.l2_squared + &gain.helper_action.l2_squared)),
        );
    }
    for residual in trace_residuals {
        trace_variance = trace_variance.add(&residual.doubled_variance);
    }
    let readout_variance = trace_variance
        .scaled(&gains.normalization.l2_squared)
        .add(&normalization_residual.doubled_variance);
    Ok(ProgramNoiseState { bound: readout_bound, doubled_variance: readout_variance })
}

#[derive(Clone)]
pub struct BggNoiseSnapshot {
    pub authority: NoiseModelKind,
    pub layout: ExpLayout,
    pub program: ExponentLutProgram,
    pub initial_state_bound: BigUint,
    pub initial_state_doubled_variance: NonnegativeRational,
    pub helper_error_cutoff: BigUint,
    pub helper_doubled_variance: NonnegativeRational,
    /// Input and prepared-operation errors bound by preparation to `program`.
    /// Evaluation consumes these bindings without interpreting a PRF schedule.
    pub inputs: ProgramNoiseInputs,
    pub gate_gains: Vec<GateNoiseGains>,
    pub readout_gains: ValueReadoutNoiseGains,
    pub refresh: RefreshNoiseParameters,
    pub gain_plan: NoiseGainPlan,
    pub family_active_counts: Vec<usize>,
    /// Optional modeled errors after the target, each nonidentity trace branch,
    /// and normalization, respectively. They are already secret-weighted.
    /// This metadata does not establish an executable approximate setup.
    pub readout_residuals: Option<(ProgramNoiseState, Vec<ProgramNoiseState>, ProgramNoiseState)>,
    target_plaintext_gain: RightActionNoiseGain,
    native_modulus: BigUint,
    identity: [u8; 32],
}

impl BggNoiseSnapshot {
    /// Bind the ordinary BGG gadget's residual sources to its actual gate and
    /// readout equations. Setup calls this once on a newly constructed snapshot.
    pub(crate) fn apply_approximate_gadget_residual(
        &mut self,
        moduli: &[BigUint],
        base: &BigUint,
        digits: usize,
        secret_dimension: usize,
        coefficient_count: usize,
    ) -> Result<(), &'static str> {
        if let Some(residual) =
            approximate_gadget_residual(moduli, base, digits, secret_dimension, coefficient_count)?
        {
            for gate in &self.gate_gains {
                if gate.operation == GateNoiseOperation::Add {
                    continue;
                }
                let sources = if gate.operation == GateNoiseOperation::Lut { 2u8 } else { 1u8 };
                self.inputs.operation_residuals.insert(
                    gate.operation_index,
                    ProgramNoiseState {
                        bound: &residual.bound * sources,
                        doubled_variance: residual.doubled_variance.scaled(&BigUint::from(sources)),
                    },
                );
            }
            let zero = ProgramNoiseState {
                bound: BigUint::zero(),
                doubled_variance: NonnegativeRational::zero(),
            };
            let trace = ProgramNoiseState {
                bound: &residual.bound * (&self.target_plaintext_gain().l1 + 1u8),
                doubled_variance: residual
                    .doubled_variance
                    .scaled(&(&self.target_plaintext_gain().l2_squared + 1u8)),
            };
            self.readout_residuals =
                Some((zero.clone(), vec![trace; self.readout_gains.nonidentity_trace.len()], zero));
        }
        Ok(())
    }

    /// Bind noise evaluation to the exact frozen accumulator supplied by setup.
    /// The readout is a separate terminal declaration; it must not select or
    /// reconstruct an alternate PRF accumulation schedule inside this adapter.
    pub fn from_setup(
        identity: [u8; 32],
        program: ExponentLutProgram,
        readout: &crate::readout::ValueReadout,
        inputs: ProgramNoiseInputs,
        initial_state_bound: BigUint,
        initial_variance: NonnegativeRational,
        helper_error_cutoff: BigUint,
        helper_variance: NonnegativeRational,
        refresh: RefreshNoiseParameters,
        gain_plan: NoiseGainPlan,
        family_active_counts: Vec<usize>,
    ) -> Result<Self, NoiseSimulationError> {
        let program_layout = program.layout().ok_or(NoiseSimulationError::InvalidDecomposition)?;
        let layout = program_layout.exponent;
        if readout.layout != layout {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        gain_plan.validate(&program, &family_active_counts)?;
        refresh.validate()?;
        let ring_dimension = 1usize
            .checked_shl(program_layout.ring_dimension_log2)
            .ok_or(NoiseSimulationError::InvalidDecomposition)?;
        layout.validate(ring_dimension).map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        let target_plaintext_gain = readout
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
        let trace = readout
            .trace_plan(ring_dimension)
            .map_err(|_| NoiseSimulationError::InvalidDecomposition)?;
        if gain_plan.readout.nonidentity_trace.len() != trace.helper_count() {
            return Err(NoiseSimulationError::InvalidDecomposition);
        }
        Ok(Self {
            authority: NoiseModelKind::WorstCase,
            layout,
            program,
            initial_state_bound,
            initial_state_doubled_variance: initial_variance,
            helper_error_cutoff: helper_error_cutoff.clone(),
            helper_doubled_variance: helper_variance.clone(),
            inputs,
            gate_gains: gain_plan.program.gates().to_vec(),
            readout_gains: gain_plan.readout.clone(),
            refresh,
            gain_plan,
            family_active_counts,
            readout_residuals: None,
            target_plaintext_gain,
            native_modulus,
            identity,
        })
    }

    pub fn setup_identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Plaintext target action used to weight trace-switch residuals.
    pub fn target_plaintext_gain(&self) -> &RightActionNoiseGain {
        &self.target_plaintext_gain
    }

    pub fn simulate(
        &self,
    ) -> Result<ExponentLutNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
        // Use the same topology-derived public-family map as AverageCase so
        // Family body MulPublic never silently drops online input errors.
        self.simulate_with_inputs(&self.inputs)
    }

    /// Evaluates the fixed program/readout portion using the snapshot's
    /// topology-derived inputs.  Refresh digit counts do not affect this
    /// result and callers may safely reuse it across a `d_m` scan.
    pub fn evaluate_pre_refresh(
        &self,
    ) -> Result<PreRefreshNoiseEvaluation<ProgramNoiseReport>, NoiseSimulationError> {
        self.evaluate_pre_refresh_with_inputs(&self.inputs)
    }

    pub fn evaluate_pre_refresh_with_inputs(
        &self,
        inputs: &ProgramNoiseInputs,
    ) -> Result<PreRefreshNoiseEvaluation<ProgramNoiseReport>, NoiseSimulationError> {
        let program = simulate_program(
            &self.program,
            inputs,
            &self.helper_error_cutoff,
            &self.helper_doubled_variance,
            self.gain_plan.program.gates(),
            &self.family_active_counts,
        )?;
        let readout = evaluate_value_readout_noise(
            program.output_state(),
            &self.helper_error_cutoff,
            &self.helper_doubled_variance,
            &self.readout_gains,
            &self.target_plaintext_gain,
            self.readout_residuals.as_ref(),
        )?;
        Ok(PreRefreshNoiseEvaluation {
            program,
            readout_bound: readout.bound,
            readout_variance: readout.doubled_variance,
            readout_bias_bound: BigUint::zero(),
        })
    }

    pub fn simulate_with_inputs(
        &self,
        inputs: &ProgramNoiseInputs,
    ) -> Result<ExponentLutNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
        let pre_refresh = self.evaluate_pre_refresh_with_inputs(inputs)?;
        self.simulate_worst_from_pre_refresh(pre_refresh)
    }

    pub fn simulate_worst_from_pre_refresh(
        &self,
        pre_refresh: PreRefreshNoiseEvaluation<ProgramNoiseReport>,
    ) -> Result<ExponentLutNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
        let PreRefreshNoiseEvaluation { program, readout_bound: output_bound, .. } = pre_refresh;
        let refresh = self.simulate_worst_refresh(&output_bound)?;
        Ok(ExponentLutNoiseReport { program, readout_bound: output_bound, refresh })
    }

    /// Evaluates only the d_m-dependent worst-case refresh equations.  This
    /// is the search fast path: it does not clone or materialize the program
    /// report for every mask-digit candidate.
    pub fn worst_refresh_accepted_from_pre_refresh(
        &self,
        pre_refresh: &PreRefreshNoiseEvaluation<ProgramNoiseReport>,
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
                    (&routes.mask[slot].l1 + &routes.fresh[slot].l1) * output_bound +
                    self.refresh.route_bias_bounds.get(slot).cloned().unwrap_or_default();
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
    ) -> Result<ExponentLutAverageNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
        self.simulate_average_with_inputs(config, &self.inputs)
    }

    pub fn simulate_average_with_inputs(
        &self,
        config: &AverageCaseConfig,
        inputs: &ProgramNoiseInputs,
    ) -> Result<ExponentLutAverageNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
        if !config.allow_average_acceptance {
            return Err(NoiseSimulationError::AverageAcceptanceDisabled);
        }
        let pre_refresh = self.evaluate_pre_refresh_with_inputs(inputs)?;
        self.simulate_average_with_pre_refresh(config, pre_refresh)
    }

    pub fn simulate_average_with_pre_refresh(
        &self,
        config: &AverageCaseConfig,
        pre_refresh: PreRefreshNoiseEvaluation<ProgramNoiseReport>,
    ) -> Result<ExponentLutAverageNoiseReport<ProgramNoiseReport>, NoiseSimulationError> {
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
        pre_refresh: &PreRefreshNoiseEvaluation<ProgramNoiseReport>,
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
        pre_refresh: &PreRefreshNoiseEvaluation<ProgramNoiseReport>,
        materialize_program: bool,
    ) -> Result<
        (bool, Option<ExponentLutAverageNoiseReport<ProgramNoiseReport>>),
        NoiseSimulationError,
    > {
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
            let switched = |variance: NonnegativeRational| {
                NonnegativeRational::new(variance.numerator, variance.denominator * lift * lift)
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
                mask_uniform_moments: self.refresh.mask_uniform_moments(slot),
                fresh_uniform_moments: self.refresh.fresh_uniform_moments(),
            };
            let variance = breakdown
                .readout_mask
                .add(&breakdown.readout_fresh)
                .add(&breakdown.decoder)
                .add(&breakdown.state);
            Ok::<(NonnegativeRational, AverageVarianceBreakdown), NoiseSimulationError>((
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
                    ceil_div(&exact.rounding_numerator, &exact.rounding_denominator)? +
                    ceil_div(
                        &(&pre_refresh.readout_bias_bound *
                            (&routes.mask[slot].l1 + &routes.fresh[slot].l1) +
                            self.refresh
                                .route_bias_bounds
                                .get(slot)
                                .cloned()
                                .unwrap_or_default()),
                        &exact.lift,
                    )?;
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

impl From<ProgramNoiseError> for NoiseSimulationError {
    fn from(error: ProgramNoiseError) -> Self {
        Self::Program(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompactPreimageSchema {
    rows: usize,
    columns: usize,
    ring_dimension: usize,
    max_coefficient_bound: BigUint,
}

fn compact_preimage_schema(
    preimage: &mxx_dsl::Preimage,
) -> Result<CompactPreimageSchema, RefreshSetupError> {
    let matrix = preimage.matrix_type();
    let concrete = |value: &IntExpr, what: &'static str| {
        value
            .evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters(what))?
            .to_usize()
            .ok_or(RefreshSetupError::InvalidParameters(what))
    };
    let bound = preimage
        .max_coefficient_bound()
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("preimage bound must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("preimage bound must be non-negative"))?;
    Ok(CompactPreimageSchema {
        rows: concrete(&matrix.rows, "preimage rows must be concrete")?,
        columns: concrete(&matrix.columns, "preimage columns must be concrete")?,
        ring_dimension: concrete(
            &matrix.ring_dimension,
            "preimage ring dimension must be concrete",
        )?,
        max_coefficient_bound: bound,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompactReadoutPlanSchema {
    target: CompactPreimageSchema,
    normalization: CompactPreimageSchema,
    trace: Vec<(CompactPreimageSchema, CompactPreimageSchema)>,
    pub(crate) decoder: Vec<CompactPreimageSchema>,
}

pub(crate) fn compact_readout_plan_schema(
    parameters: &BggRefreshParameters,
    program: &SparseLwrPrfProgram,
) -> Result<CompactReadoutPlanSchema, RefreshSetupError> {
    parameters.validate_prf(&RefreshPrfContract::from_program(program))?;
    let ring_dimension = parameters
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let digit_count = parameters.layout.digit_count;
    let decomposed_rows = parameters
        .layout
        .secret_dimension
        .checked_mul(digit_count)
        .ok_or(RefreshSetupError::InvalidParameters("decomposed readout rows overflow"))?;
    let direct_bound = parameters
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))? /
        2u8;
    let direct = |rows: usize, columns: usize| CompactPreimageSchema {
        rows,
        columns,
        ring_dimension,
        max_coefficient_bound: direct_bound.clone(),
    };
    let normalization_size = parameters
        .component_count
        .checked_mul(digit_count)
        .ok_or(RefreshSetupError::InvalidParameters("normalization shape overflow"))?;
    let trace_count = program
        .readout
        .trace_plan(ring_dimension)
        .map_err(|_| RefreshSetupError::InvalidManifest)?
        .helper_count();
    let decoder_bound = parameters
        .resolve_decoder_preimage_bound()?
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("decoder bound must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("decoder bound must be non-negative"))?;
    Ok(CompactReadoutPlanSchema {
        target: direct(decomposed_rows, parameters.layout.public_key_columns()),
        normalization: direct(normalization_size, normalization_size),
        trace: (0..trace_count)
            .map(|_| {
                (
                    direct(decomposed_rows, parameters.layout.public_key_columns()),
                    direct(decomposed_rows, parameters.layout.public_key_columns()),
                )
            })
            .collect(),
        decoder: (0..parameters.refresh.crt_plaintext_moduli.len())
            .map(|slot| {
                let decoder_rows = parameters
                    .decoder_digit_count(slot)?
                    .checked_add(2)
                    .and_then(|digits| parameters.component_count.checked_mul(digits))
                    .ok_or(RefreshSetupError::InvalidParameters("decoder shape overflow"))?;
                Ok(CompactPreimageSchema {
                    rows: decoder_rows,
                    columns: parameters.layout.public_key_columns(),
                    ring_dimension,
                    max_coefficient_bound: decoder_bound.clone(),
                })
            })
            .collect::<Result<Vec<_>, RefreshSetupError>>()?,
    })
}

pub(crate) fn bounded_plan_from_schema(
    base: &BigUint,
    schema: &CompactPreimageSchema,
) -> Result<NoiseDecompositionPlan, RefreshSetupError> {
    NoiseDecompositionPlan::bounded_matrix(
        base.clone(),
        schema.rows,
        schema.columns,
        schema.ring_dimension,
        schema.max_coefficient_bound.clone(),
    )
    .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact preimage action"))
}

pub(crate) fn producer_readout_plan_schema(
    parameters: &BggRefreshParameters,
    artifacts: &BggPreparedReadout,
    decoder: &[mxx_dsl::Preimage],
) -> Result<CompactReadoutPlanSchema, RefreshSetupError> {
    let direct_bound = parameters
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))? /
        2u8;
    let family_schema = |matrix: &mxx_ir_core::types::MatrixType| {
        let concrete = |value: &IntExpr, what: &'static str| {
            value
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters(what))?
                .to_usize()
                .ok_or(RefreshSetupError::InvalidParameters(what))
        };
        Ok::<_, RefreshSetupError>(CompactPreimageSchema {
            rows: concrete(&matrix.rows, "readout helper rows must be concrete")?,
            columns: concrete(&matrix.columns, "readout helper columns must be concrete")?,
            ring_dimension: concrete(
                &matrix.ring_dimension,
                "readout helper ring dimension must be concrete",
            )?,
            max_coefficient_bound: direct_bound.clone(),
        })
    };
    let switch_schema = family_schema(artifacts.helper_switch_decompositions().element_type())?;
    let public_schema = family_schema(artifacts.helper_public_decompositions().element_type())?;
    let trace = (0..artifacts.helper_sigmas().len())
        .map(|_| (switch_schema.clone(), public_schema.clone()))
        .collect();
    let decoder = decoder.iter().map(compact_preimage_schema).collect::<Result<Vec<_>, _>>()?;
    if decoder.len() != parameters.refresh.crt_plaintext_moduli.len() {
        return Err(RefreshSetupError::InvalidManifest);
    }
    Ok(CompactReadoutPlanSchema {
        target: compact_preimage_schema(&artifacts.target_decompositions().at(0))?,
        normalization: compact_preimage_schema(artifacts.normalization_decomposition())?,
        trace,
        decoder,
    })
}

/// Derives the compact noise plans used by the application-specific snapshot
/// without constructing any label-major executable graph.  The dimensions are
/// the same dimensions produced by the public producer: a decomposed BGG
/// matrix has `secret_dimension * gadget_digit_count` rows and the original
/// public-key columns, while normalization has `component_count * digits`
/// rows and columns.  Directly consumed preimages intentionally use the
/// bounded-matrix source with their typed decomposition bound.
pub(crate) fn compact_noise_snapshot_plans(
    parameters: &BggRefreshParameters,
    program: &ExponentLutProgram,
    family_active_counts: &[usize],
    readout_schema: &CompactReadoutPlanSchema,
) -> Result<(NoiseGainPlan, RefreshNoiseGainDecompositions), RefreshSetupError> {
    let p = parameters;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let target_columns = p.layout.public_key_columns();
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (digits_per_tower, retained_towers) =
        ordinary_gadget_towers(&crt_moduli, &base, digit_count)
            .map_err(RefreshSetupError::InvalidParameters)?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli[..retained_towers].to_vec(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let public_bound = p
        .refresh
        .full_modulus
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("full modulus must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("full modulus must be positive"))? /
        2u8;
    let public_action = || {
        if retained_towers < crt_moduli.len() {
            return NoiseDecompositionPlan::bounded_matrix(
                base.clone(),
                target_rows * digit_count,
                target_columns,
                ring_dimension,
                &base / 2u8,
            )
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid approximate public action plan")
            });
        }
        NoiseDecompositionPlan::dense_bound(
            base.clone(),
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            public_bound.clone(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid public action plan"))
        .and_then(&bind_crt)
    };
    let unit_action = || {
        NoiseDecompositionPlan::unit(
            base.clone(),
            digit_count,
            target_rows,
            target_columns,
            ring_dimension,
            0,
            0,
            0,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid unit action plan"))
        .and_then(&bind_crt)
    };
    let mut gate_decompositions = Vec::new();
    let mut gates = Vec::new();
    let mut operation_index = 0usize;
    let mut add_gate = |step_index: usize,
                        branch_index: usize,
                        operation: GateNoiseOperation,
                        action: NoiseDecompositionPlan,
                        helper_action: Option<NoiseDecompositionPlan>| {
        gate_decompositions.push(GateNoiseDecomposition { action, helper_action });
        gates.push(GateNoiseGains {
            public_plaintext_action: RightActionNoiseGain::unit(),
            step_index,
            operation_index,
            branch_index,
            operation,
            gate: "refresh-producer".to_owned(),
            action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
            helper_action: None,
            inherited_bound: BigUint::zero(),
            fresh_bound: BigUint::zero(),
            output_bound: BigUint::zero(),
            inherited_variance: NonnegativeRational::zero(),
            fresh_variance: NonnegativeRational::zero(),
            output_variance: NonnegativeRational::zero(),
        });
        operation_index += 1;
    };
    let mut family_index = 0usize;
    for (step_index, step) in program.steps().iter().enumerate() {
        match step {
            ProgramStep::Family { body, .. } => {
                let count = family_active_counts
                    .get(family_index)
                    .copied()
                    .ok_or(RefreshSetupError::InvalidParameters("family layout mismatch"))?;
                family_index += 1;
                if count == 0 {
                    return Err(RefreshSetupError::InvalidParameters("invalid program family body"));
                }
                // All active family members use the same typed public/private
                // action schema.  Keep one pair of gains for the structural
                // body and apply it to `count` members in the simulator;
                // materializing one pair per member makes search scale with
                // the PBC family rather than with the program shape.
                for (body_gate, gate) in body.gates().iter().enumerate() {
                    let (operation, action) = match gate {
                        crate::program::Gate::MulPrivate { .. } => {
                            (GateNoiseOperation::MulPrivate, public_action()?)
                        }
                        crate::program::Gate::MulPublic { .. } => {
                            (GateNoiseOperation::MulPublic, public_action()?)
                        }
                        crate::program::Gate::Add { .. } => {
                            (GateNoiseOperation::Add, unit_action()?)
                        }
                        crate::program::Gate::Lut { .. } => {
                            (GateNoiseOperation::Lut, public_action()?)
                        }
                    };
                    add_gate(step_index, body_gate, operation, action, None);
                }
                add_gate(
                    step_index,
                    body.gates().len(),
                    GateNoiseOperation::Add,
                    unit_action()?,
                    None,
                );
            }
            ProgramStep::Gate(gate) => match gate {
                Gate::MulPrivate { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::MulPrivate, public_action()?, None)
                }
                Gate::MulPublic { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::MulPublic, public_action()?, None)
                }
                Gate::Add { .. } => {
                    add_gate(step_index, 0, GateNoiseOperation::Add, unit_action()?, None)
                }
                Gate::Lut { lut_id: lut, .. } => {
                    let branches = program
                        .lut(*lut)
                        .ok_or(RefreshSetupError::InvalidParameters("missing LUT"))?
                        .values()
                        .len();
                    for branch_index in 0..branches {
                        add_gate(
                            step_index,
                            branch_index,
                            GateNoiseOperation::Lut,
                            public_action()?,
                            Some(public_action()?),
                        );
                    }
                }
            },
        }
    }
    if family_index != family_active_counts.len() {
        return Err(RefreshSetupError::InvalidParameters("program family count mismatch"));
    }

    let target_plan = bounded_plan_from_schema(&base, &readout_schema.target)?;
    let normalization_plan = bounded_plan_from_schema(&base, &readout_schema.normalization)?;
    let trace_plan = readout_schema
        .trace
        .iter()
        .map(|(action, helper)| {
            Ok((bounded_plan_from_schema(&base, action)?, bounded_plan_from_schema(&base, helper)?))
        })
        .collect::<Result<Vec<_>, RefreshSetupError>>()?;
    let readout = ValueReadoutNoiseGains {
        target: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        identity_trace: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        nonidentity_trace: trace_plan
            .iter()
            .map(|_| TraceNoiseGains {
                action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
                helper_action: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
            })
            .collect(),
        normalization: RightActionNoiseGain { l1: 1u8.into(), l2_squared: 1u8.into() },
        inherited_bound: BigUint::zero(),
        output_bound: BigUint::zero(),
        output_variance: NonnegativeRational::zero(),
    };
    let gain_plan =
        NoiseGainPlanBuilder::new(gate_decompositions, target_plan, trace_plan, normalization_plan)
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid compact producer gain plan")
            })?
            .build(gates, readout)
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("invalid compact producer gain plan")
            })?;
    Ok((gain_plan, compact_noise_snapshot_routes(p, readout_schema)?))
}

/// Rebuilds only the mask/fresh route decomposition for a changed `d_m`.
/// Gate/readout plans are independent of the mask digit count and are cached
/// by the parameter-search context.
/// The ordinary gadget retains a prefix of the full CRT idempotents. Decoder
/// trapdoors remain exact in their independently selected modulus.
pub(crate) fn corrected_route_scalars(
    p: &BggRefreshParameters,
    fresh: bool,
) -> Result<(Vec<Vec<BigInt>>, Vec<BigUint>), RefreshSetupError> {
    let evaluate = |x: &IntExpr| {
        x.evaluate(&ParamEnv::default())
            .map_err(|_| RefreshSetupError::InvalidParameters("nonconcrete route parameter"))
    };
    let moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|x| evaluate(x)?.to_biguint().ok_or(RefreshSetupError::InvalidManifest))
        .collect::<Result<Vec<_>, _>>()?;
    let base =
        evaluate(&p.layout.gadget_base)?.to_biguint().ok_or(RefreshSetupError::InvalidManifest)?;
    let (_, retained) = ordinary_gadget_towers(&moduli, &base, p.layout.digit_count)
        .map_err(RefreshSetupError::InvalidParameters)?;
    let dropped = moduli[retained..].iter().product::<BigUint>();
    let full = moduli.iter().product::<BigUint>();
    let mut values = Vec::new();
    let mut biases = Vec::new();
    for slot in 0..moduli.len() {
        let scale = if fresh {
            evaluate(&p.refresh.scale_expression(slot)?)?
        } else {
            BigInt::from(&full / &p.switch_moduli[slot])
        };
        let digits =
            if fresh { p.fresh_error_base_p_digit_count } else { p.mask_base_p_digit_counts[slot] };
        let mut bias = BigUint::zero();
        let scalars = (0..digits)
            .map(|digit| {
                let value = &scale * BigInt::from(BigUint::from(p.base_p).pow(digit as u32));
                let residual = even_route_correction(&value, &dropped);
                bias += residual.magnitude();
                value - residual
            })
            .collect();
        values.push(scalars);
        biases.push(bias * BigUint::from(p.base_p - 1) / 2u8);
    }
    Ok((values, biases))
}

pub(crate) fn compact_mask_routes(
    parameters: &BggRefreshParameters,
) -> Result<(Vec<Vec<NoiseDecompositionPlan>>, usize), RefreshSetupError> {
    let p = parameters;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (digits_per_tower, retained_towers) =
        ordinary_gadget_towers(&crt_moduli, &base, digit_count)
            .map_err(RefreshSetupError::InvalidParameters)?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli[..retained_towers].to_vec(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let dropped_product = crt_moduli[retained_towers..].iter().product::<BigUint>();
    let retained_product = BigInt::from(crt_moduli[..retained_towers].iter().product::<BigUint>());
    let route_plan = |value: BigInt| {
        // Gain-only representative: retained residues (and therefore every
        // balanced digit) are unchanged. Runtime keeps the full-q target.
        let mut value =
            (&value - even_route_correction(&value, &dropped_product)) % &retained_product;
        if value.magnitude() * 2u8 > *retained_product.magnitude() {
            value = if value < BigInt::zero() {
                value + &retained_product
            } else {
                value - &retained_product
            };
        }
        NoiseDecompositionPlan::sparse_monomial(
            base.clone(),
            digit_count,
            target_rows,
            p.prf_component_count(),
            ring_dimension,
            target_rows.saturating_sub(1),
            0,
            0,
            value,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid PRF route plan"))
        .and_then(bind_crt)
    };
    let full_modulus: BigUint = crt_moduli.iter().product();
    if p.switch_moduli.len() != crt_moduli.len() ||
        p.switch_moduli
            .iter()
            .any(|modulus| modulus.is_zero() || &full_modulus % modulus != BigUint::zero())
    {
        return Err(RefreshSetupError::InvalidParameters("invalid switching moduli"));
    }
    let mask = p
        .switch_moduli
        .iter()
        .zip(&p.mask_base_p_digit_counts)
        .map(|(modulus, digits)| {
            let lift = &full_modulus / modulus;
            (0..*digits)
                .map(|c| {
                    let c = u32::try_from(c).map_err(|_| {
                        RefreshSetupError::InvalidParameters("mask digit count overflow")
                    })?;
                    route_plan(-BigInt::from(&lift * BigUint::from(p.base_p).pow(c)))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((mask, p.coefficient_count))
}

pub(crate) fn compact_noise_snapshot_routes(
    parameters: &BggRefreshParameters,
    readout_schema: &CompactReadoutPlanSchema,
) -> Result<RefreshNoiseGainDecompositions, RefreshSetupError> {
    let p = parameters;
    let ring_dimension = p
        .layout
        .ring_dimension
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("ring dimension must be concrete"))?
        .to_usize()
        .ok_or(RefreshSetupError::InvalidParameters("ring dimension must fit usize"))?;
    let base = p
        .layout
        .gadget_base
        .evaluate(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("gadget base must be concrete"))?
        .to_biguint()
        .ok_or(RefreshSetupError::InvalidParameters("gadget base must be positive"))?;
    let digit_count = p.layout.digit_count;
    let target_rows = p.layout.secret_dimension;
    let crt_moduli = p
        .refresh
        .crt_plaintext_moduli
        .iter()
        .map(|modulus| {
            modulus
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters("CRT modulus must be concrete"))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("CRT modulus must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (digits_per_tower, retained_towers) =
        ordinary_gadget_towers(&crt_moduli, &base, digit_count)
            .map_err(RefreshSetupError::InvalidParameters)?;
    let bind_crt = |plan: NoiseDecompositionPlan| {
        plan.bind_crt_layout(crt_moduli[..retained_towers].to_vec(), digits_per_tower)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid compact CRT decomposition"))
    };
    let scales = (0..p.refresh.crt_plaintext_moduli.len())
        .map(|slot| {
            let expression = p.refresh.scale_expression(slot)?;
            expression
                .evaluate(&ParamEnv::default())
                .map_err(|_| {
                    RefreshSetupError::InvalidParameters("refresh scale must be concrete")
                })?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters("refresh scale must be positive"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let state_scaling = scales
        .iter()
        .map(|scale| {
            NoiseDecompositionPlan::gadget_scalar(
                base.clone(),
                digit_count,
                target_rows,
                target_rows * digit_count,
                ring_dimension,
                BigInt::from(scale % crt_moduli[..retained_towers].iter().product::<BigUint>()),
            )
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh scale plan"))
            .and_then(&bind_crt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dropped_product = crt_moduli[retained_towers..].iter().product::<BigUint>();
    let retained_product = BigInt::from(crt_moduli[..retained_towers].iter().product::<BigUint>());
    let route_plan = |value: BigInt, ring_coefficient: usize| {
        let mut value =
            (&value - even_route_correction(&value, &dropped_product)) % &retained_product;
        if value.magnitude() * 2u8 > *retained_product.magnitude() {
            value = if value < BigInt::zero() {
                value + &retained_product
            } else {
                value - &retained_product
            };
        }
        NoiseDecompositionPlan::sparse_monomial(
            base.clone(),
            digit_count,
            target_rows,
            p.prf_component_count(),
            ring_dimension,
            target_rows.saturating_sub(1),
            0,
            ring_coefficient,
            value,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid PRF route plan"))
        .and_then(&bind_crt)
    };
    let (mask, _) = compact_mask_routes(p)?;
    let fresh = scales
        .iter()
        .map(|scale| {
            (0..p.fresh_error_base_p_digit_count)
                .map(|c| {
                    route_plan(-BigInt::from(scale * BigUint::from(p.base_p).pow(c as u32)), 0)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decoder = readout_schema
        .decoder
        .iter()
        .map(|schema| bounded_plan_from_schema(&base, schema))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RefreshNoiseGainDecompositions {
        state_scaling,
        mask,
        mask_multiplicity: p.coefficient_count,
        fresh_multiplicity: p.coefficient_count,
        fresh,
        decoder,
    })
}

impl BggRefreshParameters {
    /// Returns the setup-derived doubled-coordinate variance of one sampled
    /// encoding-error coefficient. AverageCase cannot start from a caller
    /// supplied variance; unsupported symbolic/non-rational distributions
    /// fail closed.
    pub fn average_initial_variance(&self) -> Result<NonnegativeRational, RefreshSetupError> {
        self.validate()?;
        average_variance_from_encoding_sigma(&self.encoding_error_sigma)
    }

    /// Constructs the complete refresh-noise schema from setup parameters.
    ///
    /// This is intentionally independent of any PRF evaluator.  It binds the
    /// full modulus and every CRT modulus, derives both the joint tail event
    /// count and the mask-only smudging dimension from the refresh topology.
    /// Operational route gains remain unattached until preprocessing supplies
    /// the balanced decomposition plans (explicit or compact). Nominal payload
    /// moments depend only on the native base and digit counts.
    pub fn refresh_noise_parameters(&self) -> Result<RefreshNoiseParameters, RefreshSetupError> {
        self.validate()?;
        let eval_uint = |value: &mxx_ir_core::IntExpr,
                         message: &'static str|
         -> Result<BigUint, RefreshSetupError> {
            value
                .evaluate(&ParamEnv::default())
                .map_err(|_| RefreshSetupError::InvalidParameters(message))?
                .to_biguint()
                .ok_or(RefreshSetupError::InvalidParameters(message))
        };
        let full_modulus = eval_uint(
            &self.refresh.full_modulus,
            "full refresh modulus must be a positive concrete integer",
        )?;
        let plaintext_moduli = self
            .refresh
            .crt_plaintext_moduli
            .iter()
            .map(|value| eval_uint(value, "CRT plaintext modulus must be positive"))
            .collect::<Result<Vec<_>, _>>()?;
        let topology = self.transcript_topology();
        // The joint mask/fresh event count belongs only to the average tail
        // budget. Smudging uses the mask transcript dimension D, without the
        // P-adic digit axis or shared fresh labels.
        let smudging_coordinates = topology
            .mask_smudging_coordinates()
            .map_err(|_| RefreshSetupError::InvalidParameters("refresh topology overflows"))?;
        let mut refresh = RefreshNoiseParameters::new(
            plaintext_moduli,
            self.switch_moduli.clone(),
            self.base_p.into(),
            self.mask_base_p_digit_counts.clone(),
            self.fresh_error_base_p_digit_count,
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh noise parameters"))?;
        refresh = refresh
            .with_global_modulus(full_modulus)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid full refresh modulus"))?
            .with_transcript_topology(topology.clone())
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh topology"))?
            .with_mask_smudging_parameters(MaskSmudgingParameters {
                lambda_mask: self.mask_statistical_security_bits,
                transcript_coordinates: smudging_coordinates,
            })
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid mask smudging schema"))?;
        Ok(refresh)
    }

    /// Builds the setup-owned noise snapshot from the exact producer plans.
    /// Every plan is derived from the producer's program/readout/refresh
    /// descriptors; geometry-only gains are rejected by the snapshot schema.
    pub(crate) fn build_noise_snapshot(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
        gain_plan: crate::bgg::noise::NoiseGainPlan,
        route_decompositions: RefreshNoiseGainDecompositions,
    ) -> Result<BggNoiseSnapshot, RefreshSetupError> {
        let family_active_counts = PbcActiveCellIndex::build(pbc_layout)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid PBC active-cell index"))?
            .bucket_active_widths()
            .collect::<Vec<_>>();
        self.build_noise_snapshot_with_family_active_counts(
            prf_program,
            pbc_layout,
            gain_plan,
            route_decompositions,
            family_active_counts,
        )
    }

    pub(crate) fn build_noise_snapshot_with_family_active_counts(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
        gain_plan: crate::bgg::noise::NoiseGainPlan,
        route_decompositions: RefreshNoiseGainDecompositions,
        family_active_counts: Vec<usize>,
    ) -> Result<BggNoiseSnapshot, RefreshSetupError> {
        self.validate()?;
        let decoder_preimage_bound = self.resolve_decoder_preimage_bound()?;
        let contract = RefreshPrfContract::from_program(&prf_program);
        // The producer declaration commits to the exact value-readout
        // semantics as well as the geometry.  Use the same readout-inclusive
        // identity here so the production snapshot can be compared directly
        // with that declaration.
        let setup_identity = setup_identity_digest(
            self,
            &contract,
            pbc_layout.layout_id,
            &decoder_preimage_bound,
            prf_program.readout.identity(),
        );
        let helper_error_bound = self
            .encoding_error_bound
            .evaluate(&ParamEnv::default())
            .map_err(|_| {
                RefreshSetupError::InvalidParameters("encoding error bound must be concrete")
            })?
            .to_biguint()
            .ok_or(RefreshSetupError::InvalidParameters(
                "encoding error bound must be non-negative",
            ))?;
        let expected_mask_route_counts = self
            .mask_base_p_digit_counts
            .iter()
            .map(|digits| {
                self.coefficient_count
                    .checked_mul(*digits)
                    .ok_or(RefreshSetupError::InvalidParameters("mask route count overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_fresh_route_count = Some(self.coefficient_count)
            .and_then(|value| value.checked_mul(self.fresh_error_base_p_digit_count))
            .ok_or(RefreshSetupError::InvalidParameters("fresh route count overflow"))?;
        let actual_mask_route_counts = route_decompositions
            .mask
            .iter()
            .map(|plans| {
                plans
                    .len()
                    .checked_mul(route_decompositions.mask_multiplicity)
                    .ok_or(RefreshSetupError::InvalidParameters("mask route count overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let actual_fresh_route_counts = route_decompositions
            .fresh
            .iter()
            .map(|plans| {
                plans
                    .len()
                    .checked_mul(route_decompositions.fresh_multiplicity)
                    .ok_or(RefreshSetupError::InvalidParameters("fresh route count overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if actual_mask_route_counts.len() != self.switch_moduli.len() ||
            actual_mask_route_counts != expected_mask_route_counts ||
            route_decompositions.fresh.len() != self.refresh.crt_plaintext_moduli.len() ||
            actual_fresh_route_counts.iter().any(|count| *count != expected_fresh_route_count)
        {
            return Err(RefreshSetupError::InvalidParameters(
                "refresh route count does not match P-adic topology",
            ));
        }
        let route_gains = route_decompositions
            .derive_route_gains(self.refresh.crt_plaintext_moduli.len())
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh decomposition"))?;
        let mut refresh = self
            .refresh_noise_parameters()?
            .with_route_gains(route_gains)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid refresh route gains"))?;
        let (_, mask_bias) = corrected_route_scalars(self, false)?;
        let (_, fresh_bias) = corrected_route_scalars(self, true)?;
        refresh.route_bias_bounds =
            mask_bias.into_iter().zip(fresh_bias).map(|(mask, fresh)| mask + fresh).collect();
        // The executed PRF composes all selector ciphertexts before one
        // external product per block. One-hotness constrains the payload,
        // never the sum of ciphertext errors. Signed-monomial labels have
        // unit l1/l2 gain; distinct fresh selector draws use H1--H4 here.
        let program = prf_program.accumulation;
        let initial_variance = self.average_initial_variance()?;
        let sampled_error = ProgramNoiseState {
            bound: helper_error_bound.clone(),
            doubled_variance: initial_variance.clone(),
        };
        let inputs = ProgramNoiseInputs {
            input_errors: program.inputs().keys().map(|id| (*id, sampled_error.clone())).collect(),
            rhs_errors: program
                .rhs_inputs()
                .keys()
                .zip(&family_active_counts)
                .map(|(id, count)| {
                    (
                        *id,
                        ProgramNoiseState {
                            bound: &helper_error_bound * BigUint::from(*count),
                            doubled_variance: initial_variance.scaled(&BigUint::from(*count)),
                        },
                    )
                })
                .collect(),
            ..Default::default()
        };
        let mut snapshot = BggNoiseSnapshot::from_setup(
            setup_identity,
            program,
            &prf_program.readout,
            inputs,
            helper_error_bound.clone(),
            initial_variance.clone(),
            helper_error_bound,
            initial_variance,
            refresh,
            gain_plan,
            Vec::new(),
        )
        .map_err(|_| RefreshSetupError::InvalidParameters("invalid noise snapshot"))?;
        let moduli = self
            .refresh
            .crt_plaintext_moduli
            .iter()
            .map(|p| {
                p.evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|p| p.to_biguint())
                    .ok_or(RefreshSetupError::InvalidManifest)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let base = self
            .layout
            .gadget_base
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|base| base.to_biguint())
            .ok_or(RefreshSetupError::InvalidManifest)?;
        snapshot
            .apply_approximate_gadget_residual(
                &moduli,
                &base,
                self.layout.digit_count,
                self.layout.secret_dimension,
                self.coefficient_count,
            )
            .map_err(RefreshSetupError::InvalidParameters)?;
        Ok(snapshot)
    }

    /// Evaluates one AverageCase search candidate using the same complete,
    /// setup-owned snapshot construction used by final graph assembly.
    ///
    /// Only the candidate mask digit count is varied; all other setup fields,
    /// including the fresh-error digit count, remain fixed.  The returned
    /// report carries the canonical snapshot identity and both the independent
    /// WorstCase hard authority and AverageCase correctness result, so a
    /// selector can compare the report with the final bundle without building
    /// a graph or supplying a proxy variance.
    pub fn evaluate_average_candidate(
        &self,
        prf_program: SparseLwrPrfProgram,
        pbc_layout: PbcPublicLayout,
        candidate_mask_base_p_digit_count: usize,
        config: &AverageCaseConfig,
    ) -> Result<ExponentLutAverageNoiseReport<ProgramNoiseReport>, RefreshSetupError> {
        let mut candidate = self.clone();
        candidate.mask_base_p_digit_counts.fill(candidate_mask_base_p_digit_count);
        // The mask digit count changes the route topology.  Rebuild both the
        // compact route plans and the gain plan from the candidate instead of
        // reusing plans generated for the previous digit count.
        let readout_schema = compact_readout_plan_schema(&candidate, &prf_program)?;
        let (gain_plan, route_decompositions) = compact_noise_snapshot_plans(
            &candidate,
            &prf_program.accumulation,
            &[],
            &readout_schema,
        )?;
        let snapshot = candidate.build_noise_snapshot(
            prf_program,
            &pbc_layout,
            gain_plan,
            route_decompositions,
        )?;
        snapshot
            .simulate_average(config)
            .map_err(|_| RefreshSetupError::InvalidParameters("invalid AverageCase candidate"))
    }

    /// Evaluates the supplied execution Program without constructing executable graphs.
    /// A cached gain plan may optionally include invariant state/fresh routes.
    /// These caches must belong to this profile, layout and encoding parameters;
    /// only mask digit counts may vary. Decoder plans use the current parameters.
    pub fn build_noise_snapshot_from_public_layout(
        &self,
        program: &SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
        active_cells: &PbcActiveCellIndex,
        cached: Option<(&NoiseGainPlan, Option<&RefreshNoiseGainDecompositions>)>,
    ) -> Result<BggNoiseSnapshot, RefreshSetupError> {
        self.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let schema = compact_readout_plan_schema(self, program)?;
        let (gain_plan, routes) = match cached {
            None => compact_noise_snapshot_plans(self, &program.accumulation, &[], &schema)?,
            Some((gain_plan, None)) => {
                (gain_plan.clone(), compact_noise_snapshot_routes(self, &schema)?)
            }
            Some((gain_plan, Some(invariants))) => {
                let (mask, mask_multiplicity) = compact_mask_routes(self)?;
                let base = self
                    .layout
                    .gadget_base
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_biguint())
                    .ok_or(RefreshSetupError::InvalidParameters("invalid gadget base"))?;
                let decoder = schema
                    .decoder
                    .iter()
                    .map(|entry| bounded_plan_from_schema(&base, entry))
                    .collect::<Result<Vec<_>, _>>()?;
                (
                    gain_plan.clone(),
                    RefreshNoiseGainDecompositions {
                        state_scaling: invariants.state_scaling.clone(),
                        mask,
                        mask_multiplicity,
                        fresh: invariants.fresh.clone(),
                        fresh_multiplicity: invariants.fresh_multiplicity,
                        decoder,
                    },
                )
            }
        };
        self.build_noise_snapshot_with_family_active_counts(
            program.clone(),
            pbc_layout,
            gain_plan,
            routes,
            active_cells.bucket_active_widths().collect(),
        )
    }

    /// Returns the invariant decomposition routes together with their gain
    /// plan.  The caller may reuse these routes while scanning `d_m`; only the
    /// P-adic mask route changes with that parameter.
    pub fn noise_gain_and_route_invariants_from_public_layout(
        &self,
        program: &SparseLwrPrfProgram,
        pbc_layout: &PbcPublicLayout,
    ) -> Result<(NoiseGainPlan, RefreshNoiseGainDecompositions), RefreshSetupError> {
        self.validate()?;
        pbc_layout.validate().map_err(|error| RefreshSetupError::Pbc(error.to_string()))?;
        let readout_schema = compact_readout_plan_schema(self, program)?;
        compact_noise_snapshot_plans(self, &program.accumulation, &[], &readout_schema)
    }

    /// Derives the exact aggregate mask route gain for one `d_m` without
    /// constructing a program, snapshot, or executable graph.  This is the
    /// only route component that changes during a mask-digit scan.
    pub fn mask_route_gain_for_digit_count(
        &self,
        mask_digit_count: usize,
    ) -> Result<Vec<crate::noise_utils::RightActionNoiseGain>, RefreshSetupError> {
        let mut candidate = self.clone();
        candidate.mask_base_p_digit_counts.fill(mask_digit_count);
        let (plans, multiplicity) = compact_mask_routes(&candidate)?;
        plans
            .iter()
            .map(|plans| {
                let gains = NoiseGainPlanBuilder::derive_route_gains(plans)
                    .map_err(|_| RefreshSetupError::InvalidParameters("invalid mask route plan"))?;
                let mut aggregate = crate::noise_utils::RightActionNoiseGain {
                    l1: 0u8.into(),
                    l2_squared: 0u8.into(),
                };
                for gain in gains {
                    aggregate.l1 += gain.l1 * multiplicity;
                    aggregate.l2_squared += gain.l2_squared * multiplicity;
                }
                Ok(aggregate)
            })
            .collect()
    }
}

fn average_variance_from_encoding_sigma(
    sigma: &mxx_ir_core::RealExpr,
) -> Result<NonnegativeRational, RefreshSetupError> {
    let sigma = sigma
        .evaluate_rational(&ParamEnv::default())
        .map_err(|_| RefreshSetupError::InvalidParameters("encoding sigma must be rational"))?;
    let numerator = sigma
        .numerator()
        .to_biguint()
        .filter(|value| !value.is_zero())
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma must be positive"))?;
    let denominator = sigma
        .denominator()
        .to_biguint()
        .filter(|value| !value.is_zero())
        .ok_or(RefreshSetupError::InvalidParameters("encoding sigma denominator invalid"))?;
    NonnegativeRational::new(
        BigUint::from(4u8) * &numerator * &numerator,
        &denominator * &denominator,
    )
    .map_err(|_| RefreshSetupError::InvalidParameters("encoding variance is invalid"))
}
