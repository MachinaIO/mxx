//! Exact noise arithmetic shared by encoding schemes.
//!
//! These values have no implicit ciphertext coordinate units. In particular,
//! BGG stores variance in doubled coordinates at its adapter boundary.
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum NoiseArithmeticError {
    #[error("rational denominator must be positive")]
    ZeroDenominator,
}

/// Exact nonnegative rational. Coordinate units belong to its owning noise trace.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NonnegativeRational {
    pub numerator: BigUint,
    pub denominator: BigUint,
}

impl NonnegativeRational {
    pub fn new(numerator: BigUint, denominator: BigUint) -> Result<Self, NoiseArithmeticError> {
        if denominator.is_zero() {
            return Err(NoiseArithmeticError::ZeroDenominator);
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

    /// Bound a heterogeneous sum without assuming independence. Zero-variance
    /// terms introduce no covariance, including when their means are nonzero.
    /// Compose pairwise Cauchy bounds and cap outward-rounding accumulation by
    /// the Cauchy bound r * sum(V_i), counting only nonzero variances. Equal
    /// variances and rational proportional standard deviations remain exact.
    pub fn sum_variance_bounds<'a>(variances: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut total = Self::zero();
        let mut bound = Self::zero();
        let mut count = 0usize;
        for variance in variances {
            if variance.numerator.is_zero() {
                continue;
            }
            total = total.add(variance);
            bound = bound.sum_variance_bound(variance);
            count += 1;
        }
        let cap = total.scaled(&BigUint::from(count));
        if &bound.numerator * &cap.denominator <= &cap.numerator * &bound.denominator {
            bound
        } else {
            cap
        }
    }

    /// Upper bound on the variance of a sum with unknown covariance.
    /// Cauchy--Schwarz gives V(X+Y) <= V(X)+V(Y)+2 sqrt(V(X)V(Y)).
    /// Round the square root outward in the common rational denominator;
    /// cap that rounding by 2(V(X)+V(Y)), which is always another valid bound.
    /// Equal variances retain the exact coherent bound 4V, and a zero-variance
    /// operand leaves the other variance unchanged. No independence is assumed.
    pub fn sum_variance_bound(&self, other: &Self) -> Self {
        let denominator = &self.denominator * &other.denominator;
        let left = &self.numerator * &other.denominator;
        let right = &other.numerator * &self.denominator;
        let product = &left * &right;
        let floor = product.sqrt();
        let ceil = if &floor * &floor == product { floor } else { floor + 1u8 };
        let sum = left + right;
        let numerator = (&sum + ceil * 2u8).min(&sum * 2u8);
        Self::new(numerator, denominator).expect("positive variance denominator")
    }
}

#[cfg(test)]
mod covariance_tests {
    use super::*;

    #[test]
    fn tail_envelope_rounds_the_combined_mean_and_deviation_once() {
        let moments = NoiseMoments::uniform_interval((-1).into(), 4u8.into());
        assert_eq!(moments.tail_bound(&NonnegativeRational::zero()), 1u8.into());
        assert_eq!(
            moments.tail_bound(&NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap()),
            2u8.into()
        );
        assert_eq!(
            moments.tail_bound(&NonnegativeRational::new(4u8.into(), 1u8.into()).unwrap()),
            3u8.into()
        );
    }

    #[test]
    fn test_mean_enclosure_preserves_native_bias_and_linear_images() {
        let native = NoiseMoments::uniform_interval((-1).into(), 4u8.into());
        assert_eq!(native.mean_radius, NonnegativeRational::zero());
        let doubled = native.add(&native);
        assert_eq!(doubled.mean_numerator, BigInt::one());
        assert_eq!(doubled.mean_denominator, BigUint::one());
        assert_eq!(doubled, native.scaled(&2u8.into()));
        let image = native.linear_image(
            &NonnegativeRational::new(3u8.into(), 1u8.into()).unwrap(),
            native.variance.scaled(&9u8.into()),
        );
        // The wrapped/nonwrapped coefficients of multiplication by 2-X
        // contain 2*d0+d1 or 2*d0-d1. Enumerate the actual finite digit law.
        for sign in [-1i64, 1] {
            let values = (-1i64..=2)
                .flat_map(|a| (-1i64..=2).map(move |b| 2 * a + sign * b))
                .collect::<Vec<_>>();
            let sum = values.iter().sum::<i64>();
            let squares = values.iter().map(|x| x * x).sum::<i64>();
            let count = values.len() as i64;
            assert!(
                BigUint::from(sum.unsigned_abs()) * &image.mean_radius.denominator <=
                    &image.mean_radius.numerator * BigUint::from(count as u64)
            );
            let variance_numerator = BigUint::from((squares * count - sum * sum) as u64);
            assert!(
                variance_numerator * &image.variance.denominator <=
                    &image.variance.numerator * BigUint::from((count * count) as u64)
            );
        }
        let zero = native.linear_image(&NonnegativeRational::zero(), NonnegativeRational::zero());
        assert!(zero.mean_radius.numerator.is_zero());
        assert!(zero.variance.numerator.is_zero());
    }

    #[test]
    fn test_moment_clearance_includes_bias_and_strict_boundary() {
        let unit = NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap();
        let moments = NoiseMoments {
            mean_numerator: (-1).into(),
            mean_denominator: 2u8.into(),
            mean_radius: NonnegativeRational::new(1u8.into(), 4u8.into()).unwrap(),
            variance: NonnegativeRational::new(9u8.into(), 16u8.into()).unwrap(),
        };
        // |mean| + radius + sigma = 1/2 + 1/4 + 3/4 = 3/2.
        assert!(!moments.fits(&NonnegativeRational::new(3u8.into(), 2u8.into()).unwrap(), &unit));
        assert!(moments.fits(&NonnegativeRational::new(7u8.into(), 4u8.into()).unwrap(), &unit));
        assert!(!moments.fits(&NonnegativeRational::new(1u8.into(), 2u8.into()).unwrap(), &unit));
        let deterministic = NoiseMoments { variance: NonnegativeRational::zero(), ..moments };
        assert!(
            !deterministic.fits(&NonnegativeRational::new(3u8.into(), 4u8.into()).unwrap(), &unit)
        );
        assert!(deterministic.fits(&unit, &unit));
    }

    #[test]
    fn test_sum_variance_bound_preserves_coherent_and_zero_variance() {
        let variance = NonnegativeRational::new(7u8.into(), 3u8.into()).unwrap();
        assert_eq!(variance.sum_variance_bound(&variance), variance.scaled(&4u8.into()));
        assert_eq!(variance.sum_variance_bound(&NonnegativeRational::zero()), variance);
        assert_eq!(NonnegativeRational::zero().sum_variance_bound(&variance), variance);
        let amplified = variance.scaled(&9u8.into());
        assert_eq!(variance.sum_variance_bound(&amplified), variance.scaled(&16u8.into()));
    }

    #[test]
    fn test_family_variance_bound_preserves_shared_sources_and_constant_terms() {
        let variance = NonnegativeRational::new(7u8.into(), 3u8.into()).unwrap();
        let zero = NonnegativeRational::zero();
        assert_eq!(NonnegativeRational::sum_variance_bounds([]), zero);
        assert_eq!(NonnegativeRational::sum_variance_bounds([&zero, &variance, &zero]), variance);
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&variance; 3]),
            variance.scaled(&9u8.into())
        );
        // If all errors are multiples of the same E, the tight worst
        // covariance is attained: Var(E + 2E + 3E) = 36 Var(E).
        let twice = variance.scaled(&4u8.into());
        let thrice = variance.scaled(&9u8.into());
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&variance, &twice, &zero, &thrice]),
            variance.scaled(&36u8.into())
        );
        assert_eq!(
            NonnegativeRational::sum_variance_bounds([&thrice, &zero, &twice, &variance]),
            variance.scaled(&36u8.into())
        );
    }

    #[test]
    fn test_sum_variance_bound_rounds_covariance_outward() {
        for left in 1u32..12 {
            for right in 1u32..12 {
                let x = NonnegativeRational::new(left.into(), 3u8.into()).unwrap();
                let y = NonnegativeRational::new(right.into(), 5u8.into()).unwrap();
                let bound = x.sum_variance_bound(&y);
                let sum = x.add(&y);
                // Verify the defining Cauchy inequality exactly after clearing
                // denominators; no floating-point reference or tail model.
                let excess =
                    &bound.numerator * &sum.denominator - &sum.numerator * &bound.denominator;
                let excess_denominator = &bound.denominator * &sum.denominator;
                assert!(
                    &excess * &excess * &x.denominator * &y.denominator >=
                        &x.numerator *
                            &y.numerator *
                            4u8 *
                            &excess_denominator *
                            &excess_denominator
                );
                assert!(
                    &bound.numerator * &sum.denominator <=
                        &sum.numerator * 2u8 * &bound.denominator
                );
            }
        }
    }
}

/// Physical moment enclosure, conditional on the caller's stated distribution
/// model. Each coefficient mean lies in center +/- mean_radius; a zero radius
/// records an exact scalar mean. Variance is a centered variance bound, not a
/// second moment. Neither field implies independence or a tail law.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseMoments {
    pub mean_numerator: BigInt,
    pub mean_denominator: BigUint,
    pub mean_radius: NonnegativeRational,
    pub variance: NonnegativeRational,
}

impl NoiseMoments {
    /// Outward integer envelope for the caller's explicit tail model.
    pub fn tail_bound(&self, z_squared: &NonnegativeRational) -> BigUint {
        let bias = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator")
        .add(&self.mean_radius);
        let numerator = &z_squared.numerator * &self.variance.numerator;
        let denominator = &z_squared.denominator * &self.variance.denominator;
        let squared = (&numerator + &denominator - 1u8) / denominator;
        let floor = squared.sqrt();
        let tail = if &floor * &floor == squared { floor } else { floor + 1u8 };
        let upper = (&bias.numerator + &bias.denominator - 1u8) / &bias.denominator + tail;
        if upper.is_zero() {
            return upper;
        }
        // The separate ceilings exceed ceil(bias + sqrt(V*z^2)) by at most
        // one. Test that final integer exactly, without a floating-point tail.
        let lower = &upper - 1u8;
        let scaled = &lower * &bias.denominator;
        if scaled >= bias.numerator &&
            (&scaled - &bias.numerator).pow(2) *
                &z_squared.denominator *
                &self.variance.denominator >=
                &z_squared.numerator * &self.variance.numerator * bias.denominator.pow(2)
        {
            lower
        } else {
            upper
        }
    }

    /// Strict clearance for a mean interval plus a caller-selected tail
    /// multiplier: |center| + radius + sqrt(z_squared * variance) < limit.
    /// The multiplier's distribution assumption and event budget belong to
    /// the audit, not to this exact rational comparison.
    pub fn fits(&self, limit: &NonnegativeRational, z_squared: &NonnegativeRational) -> bool {
        let bias = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator")
        .add(&self.mean_radius);
        let available = &limit.numerator * &bias.denominator;
        let occupied = &bias.numerator * &limit.denominator;
        if available <= occupied {
            return false;
        }
        let margin = available - occupied;
        let denominator = &limit.denominator * &bias.denominator;
        margin.pow(2) * &z_squared.denominator * &self.variance.denominator >
            &z_squared.numerator * &self.variance.numerator * denominator.pow(2)
    }

    /// Enclose a linear image with a known row-l1 bound. The caller derives
    /// its centered variance transfer under the applicable coefficient model.
    /// A uniform scalar mean need not stay uniform after negacyclic wrapping;
    /// retain its image as an interval instead of assigning it to every output.
    pub fn linear_image(&self, l1: &NonnegativeRational, variance: NonnegativeRational) -> Self {
        let center = NonnegativeRational::new(
            self.mean_numerator.magnitude().clone(),
            self.mean_denominator.clone(),
        )
        .expect("positive mean denominator");
        let magnitude = center.add(&self.mean_radius);
        Self {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::new(
                magnitude.numerator * &l1.numerator,
                magnitude.denominator * &l1.denominator,
            )
            .expect("positive mean and action denominators"),
            variance,
        }
    }

    /// Sum mean intervals and retain unknown/shared-source covariance.
    pub fn add(&self, rhs: &Self) -> Self {
        let numerator = &self.mean_numerator * BigInt::from(rhs.mean_denominator.clone()) +
            &rhs.mean_numerator * BigInt::from(self.mean_denominator.clone());
        let denominator = &self.mean_denominator * &rhs.mean_denominator;
        let divisor = gcd(numerator.magnitude(), &denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: denominator / divisor,
            mean_radius: self.mean_radius.add(&rhs.mean_radius),
            variance: self.variance.sum_variance_bound(&rhs.variance),
        }
    }

    /// Scale one source, including repeated family captures. This is not an
    /// independent sum of newly sampled errors.
    pub fn scaled(&self, count: &BigUint) -> Self {
        let numerator = &self.mean_numerator * BigInt::from(count.clone());
        let divisor = gcd(numerator.magnitude(), &self.mean_denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: &self.mean_denominator / divisor,
            mean_radius: self.mean_radius.scaled(count),
            variance: self.variance.scaled(&count.pow(2)),
        }
    }

    /// Uniform integer interval starting at `lower` with positive `size`.
    /// The support contract is established by parameter construction.
    pub fn uniform_interval(lower: BigInt, size: BigUint) -> Self {
        let numerator = lower * 2u8 + BigInt::from(&size - 1u8);
        let denominator = BigUint::from(2u8);
        let divisor = gcd(numerator.magnitude(), &denominator);
        Self {
            mean_numerator: numerator / BigInt::from(divisor.clone()),
            mean_denominator: denominator / divisor,
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(&size * &size - 1u8, 12u8.into())
                .expect("positive constant denominator"),
        }
    }
}

/// Coefficient norms of one matrix action on a noise row.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RightActionNoiseGain {
    pub l1: BigUint,
    pub l2_squared: BigUint,
}

impl RightActionNoiseGain {
    /// Convolution gains of distinct coefficients of an integer polynomial lift.
    pub(crate) fn from_polynomial_coefficients(
        coefficients: impl IntoIterator<Item = BigInt>,
    ) -> Self {
        let mut gain = Self { l1: BigUint::zero(), l2_squared: BigUint::zero() };
        for coefficient in coefficients {
            let magnitude = coefficient.magnitude();
            gain.l1 += magnitude;
            gain.l2_squared += magnitude * magnitude;
        }
        gain
    }

    pub fn unit() -> Self {
        Self { l1: BigUint::one(), l2_squared: BigUint::one() }
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

use crate::program::{
    ExponentLutProgram, ExponentProgramLowering, FamilyRange, LutId, ProgramBindings,
    ProgramFamilyBindings, ProgramGateInstance, ProgramInputId, ProgramLowerError,
    ProgramLoweringTypes, RhsFamilyId, RhsId, WireFamilyId, WireId,
};
use std::collections::BTreeMap;

/// An index into one trace's flat dependency arena, never a sampler identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseNodeId(pub usize);

/// Program binding provenance. Distinct bindings do not imply independent
/// noise: sampling contracts must establish that separately. Repeated uses of
/// a binding retain the same node even when the backend clones its carrier.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseBinding {
    Input(ProgramInputId),
    Family(WireFamilyId),
    Rhs(RhsId),
    RhsFamily(RhsFamilyId),
    Helper(LutId),
    GateHelper(ProgramGateInstance),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseDependencyNode {
    Binding(NoiseBinding),
    Add {
        gate: ProgramGateInstance,
        lhs: NoiseNodeId,
        rhs: NoiseNodeId,
    },
    MulPublic {
        gate: ProgramGateInstance,
        private: NoiseNodeId,
        public: NoiseNodeId,
    },
    MulPrivate {
        gate: ProgramGateInstance,
        input: NoiseNodeId,
        rhs: NoiseNodeId,
        helper: Option<NoiseNodeId>,
    },
    Lut {
        gate: ProgramGateInstance,
        table: LutId,
        input: NoiseNodeId,
        helper: Option<NoiseNodeId>,
    },
    /// A symbolic family reduction preserves the active range and ancestors;
    /// it does not assert independence between family indices.
    Reduction {
        gate: ProgramGateInstance,
        input: NoiseNodeId,
        range: FamilyRange,
    },
}

/// Dependencies recorded during the actual noise interpreter traversal.
/// Storage is proportional to bindings and static operations, not family
/// capacity or coefficient count. Serialization stores the arena once.
/// This graph supplies provenance for moment transfers; by itself it supplies
/// neither per-sampler distributions nor covariance estimates.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct NoiseDependencyGraph {
    pub nodes: Vec<NoiseDependencyNode>,
    pub wires: BTreeMap<WireId, NoiseNodeId>,
}

#[derive(Clone)]
struct Dependent<V> {
    value: V,
    node: NoiseNodeId,
}

struct DependencyEvaluator<E> {
    inner: E,
    graph: NoiseDependencyGraph,
}

impl<E> DependencyEvaluator<E> {
    fn record<V>(&mut self, value: V, node: NoiseDependencyNode) -> Dependent<V> {
        let id = NoiseNodeId(self.graph.nodes.len());
        self.graph.nodes.push(node);
        Dependent { value, node: id }
    }

    fn bind<V: Clone>(&mut self, value: &V, binding: NoiseBinding) -> Dependent<V> {
        self.record(value.clone(), NoiseDependencyNode::Binding(binding))
    }
}

impl<E: ExponentProgramLowering> ProgramLoweringTypes for DependencyEvaluator<E> {
    type Wire = Dependent<E::Wire>;
    type Rhs = Dependent<E::Rhs>;
    type HelperSet = Dependent<E::HelperSet>;
    type Error = E::Error;
}

impl<E: ExponentProgramLowering> ExponentProgramLowering for DependencyEvaluator<E> {
    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let value = self.inner.add(&lhs.value, &rhs.value, gate)?;
        Ok(self.record(value, NoiseDependencyNode::Add { gate, lhs: lhs.node, rhs: rhs.node }))
    }
    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let value = self.inner.mul_public(&private.value, &public.value, gate)?;
        Ok(self.record(
            value,
            NoiseDependencyNode::MulPublic { gate, private: private.node, public: public.node },
        ))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        helper: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let value =
            self.inner.mul_private(&input.value, &rhs.value, helper.map(|h| &h.value), gate)?;
        Ok(self.record(
            value,
            NoiseDependencyNode::MulPrivate {
                gate,
                input: input.node,
                rhs: rhs.node,
                helper: helper.map(|h| h.node),
            },
        ))
    }
    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        table: LutId,
        input: &Self::Wire,
        helper: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let value = self.inner.lut(program, table, &input.value, helper.map(|h| &h.value), gate)?;
        Ok(self.record(
            value,
            NoiseDependencyNode::Lut {
                gate,
                table,
                input: input.node,
                helper: helper.map(|h| h.node),
            },
        ))
    }
}

pub(crate) fn trace_dependencies<E: ExponentProgramLowering>(
    program: &ExponentLutProgram,
    bindings: &ProgramBindings<'_, E::Wire, E::Rhs, E::HelperSet>,
    ranges: &ProgramFamilyBindings,
    evaluator: E,
    mut reduce: impl FnMut(
        &mut E,
        &E::Wire,
        &FamilyRange,
        ProgramGateInstance,
    ) -> Result<E::Wire, E::Error>,
) -> Result<(BTreeMap<WireId, E::Wire>, E, NoiseDependencyGraph), ProgramLowerError<E::Error>>
where
    E::HelperSet: Clone,
{
    let mut evaluator =
        DependencyEvaluator { inner: evaluator, graph: NoiseDependencyGraph::default() };
    let inputs = bindings
        .inputs
        .iter()
        .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::Input(*id))))
        .collect();
    let rhs = bindings
        .rhs_inputs
        .iter()
        .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::Rhs(*id))))
        .collect();
    let families = bindings
        .wire_values
        .iter()
        .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::Family(*id))))
        .collect();
    let rhs_families = bindings
        .rhs_families
        .iter()
        .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::RhsFamily(*id))))
        .collect();
    let helpers = bindings
        .helpers
        .iter()
        .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::Helper(*id))))
        .collect();
    let gate_helpers = bindings.gate_helpers.map(|values| {
        values
            .iter()
            .map(|(id, v)| (*id, evaluator.bind(v, NoiseBinding::GateHelper(*id))))
            .collect()
    });
    let bindings = ProgramBindings::new(
        &inputs,
        &rhs,
        &rhs_families,
        &families,
        &helpers,
        gate_helpers.as_ref(),
    );
    let wires = crate::program::lower_program(
        program,
        &bindings,
        ranges,
        &mut evaluator,
        |evaluator, input, range, gate| {
            let value = reduce(&mut evaluator.inner, &input.value, range, gate)?;
            Ok(evaluator.record(
                value,
                NoiseDependencyNode::Reduction { gate, input: input.node, range: range.clone() },
            ))
        },
    )?;
    let wires = wires
        .into_iter()
        .map(|(id, value)| {
            evaluator.graph.wires.insert(id, value.node);
            (id, value.value)
        })
        .collect();
    Ok((wires, evaluator.inner, evaluator.graph))
}
