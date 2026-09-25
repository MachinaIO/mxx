//! Source-aware average-case noise contributions.
//!
//! A noise value's moments are the image of the sampled errors and rounding
//! residuals that produced it. This module keeps those images apart, keyed by
//! the sampled object they came from, so that composition can add the
//! variances of genuinely independent samples while retaining the Cauchy
//! (coherent) bound for repeated or unknown ones. See
//! `docs/correctness/average-case-noise-implementation-design.md`.
//!
//! Independence rules, all conservative:
//!
//! * Contributions of the same sampled object (equal source and member coordinates) are combined
//!   coherently.
//! * Sampled errors of distinct objects are uncorrelated. Every transfer that produces them applies
//!   a fixed public or prepared action (the fixed-RHS conditional model, H4), and the caller
//!   records that assumption.
//! * A source whose members were summed over a lane, or read through an unknown index relation,
//!   keeps all of its contributions in one coherent group: overlapping sums of the same members are
//!   not independent.
//! * Rounding residuals of distinct gates are uncorrelated with each other and with sampled errors
//!   (H2). Residuals of one gate across family lanes are coherent.
//! * An opaque contribution (a boundary that only has a total envelope) is coherent with
//!   everything.

use crate::noise_utils::{NoiseCorrelation, NoiseMoments, NonnegativeRational};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::{collections::BTreeMap, sync::Arc};

/// Coordinate of a sampled member.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum SampleCoordinate {
    /// Member `index`, read inside `depth` enclosing parallel families.
    Index { depth: usize, index: Arc<IntExpr> },
    /// The sum over the members addressed by the lanes of a family at
    /// `depth`.
    Reduced { depth: usize },
    /// A member related to other reads in an unknown way.
    Unknown,
}

/// What produced a contribution. Independence also depends on the sampling
/// contract and the transfer; the kind alone does not establish it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ContributionKind {
    SampledError,
    RoundingResidual,
    Opaque,
}

/// The image of one sampled object (or residual) in a noise value.
#[derive(Clone, Debug)]
pub(crate) struct NoiseContribution {
    /// Sampling namespace of the original object, never the consuming gate.
    pub source: [u8; 32],
    /// Diagnostic name of the producing transfer; not part of the identity.
    pub origin: &'static str,
    /// Member coordinates of the sampled object.
    pub sample: Vec<SampleCoordinate>,
    pub kind: ContributionKind,
    /// Every member of the enclosing input family has its own sample, so a
    /// member read appends its index to `sample`.
    pub per_member: bool,
    /// The coefficients are uncorrelated with a common variance (a fresh
    /// sample, or its image under signed monomials), so a fixed action may
    /// use its coefficient energy instead of its squared L1 norm (H1/H4).
    pub white: bool,
    pub moments: NoiseMoments,
}

impl NoiseContribution {
    fn key(&self) -> (ContributionKind, [u8; 32], &[SampleCoordinate]) {
        (self.kind, self.source, &self.sample)
    }

    fn summarized(&self) -> bool {
        self.sample.iter().any(|coordinate| !matches!(coordinate, SampleCoordinate::Index { .. }))
    }
}

/// Aggregate reduction counts, for diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReductionCounts {
    pub coherent: usize,
    pub independent: usize,
    pub histogram: usize,
    pub unknown_index: usize,
}

/// The contributions of one noise value.
#[derive(Clone, Debug, Default)]
pub(crate) struct NoiseBreakdown {
    terms: Vec<NoiseContribution>,
    /// Depth of the family lane whose coordinates this element's terms
    /// carry, when the element was produced by a parallel body.
    lane_depth: Option<usize>,
}

/// Fraction bits kept by the outward rounding of propagated moments. The
/// exact alignment and scale ratios otherwise grow every denominator along a
/// modulus chain; `2^-128` is far below every noise budget.
const PRECISION: u32 = 128;

fn zero_moments() -> NoiseMoments {
    NoiseMoments {
        mean_numerator: num_bigint::BigInt::zero(),
        mean_denominator: BigUint::one(),
        mean_radius: NonnegativeRational::zero(),
        variance: NonnegativeRational::zero(),
    }
}

impl NoiseBreakdown {
    /// One fresh sampled object.
    pub(crate) fn sampled(
        source: [u8; 32],
        origin: &'static str,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: Vec::new(),
                kind: ContributionKind::SampledError,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// One sampled member of a sampled family, at coordinate `index`.
    pub(crate) fn sampled_member(
        source: [u8; 32],
        origin: &'static str,
        depth: usize,
        index: IntExpr,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: vec![SampleCoordinate::Index { depth, index: Arc::new(index) }],
                kind: ContributionKind::SampledError,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// One rounding residual of a gate.
    pub(crate) fn rounding(source: [u8; 32], origin: &'static str, moments: NoiseMoments) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample: Vec::new(),
                kind: ContributionKind::RoundingResidual,
                per_member: false,
                white: false,
                moments,
            }],
        }
    }

    /// A total envelope with no known decomposition, coherent with every
    /// other contribution.
    /// A rounding residual of one sampled object, with that object's member
    /// coordinates: residuals of distinct objects are uncorrelated (H2).
    pub(crate) fn object_residual(
        source: [u8; 32],
        origin: &'static str,
        sample: Vec<SampleCoordinate>,
        moments: NoiseMoments,
        white: bool,
    ) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source,
                origin,
                sample,
                kind: ContributionKind::RoundingResidual,
                per_member: false,
                white,
                moments,
            }],
        }
    }

    /// The source and member coordinates when these contributions are those
    /// of exactly one sampled object.
    pub(crate) fn single_sample(&self) -> Option<([u8; 32], &[SampleCoordinate])> {
        match self.terms.as_slice() {
            [term] if term.kind == ContributionKind::SampledError && !term.per_member => {
                Some((term.source, &term.sample))
            }
            _ => None,
        }
    }

    pub(crate) fn opaque(origin: &'static str, moments: NoiseMoments) -> Self {
        Self {
            lane_depth: None,
            terms: vec![NoiseContribution {
                source: [0; 32],
                origin,
                sample: Vec::new(),
                kind: ContributionKind::Opaque,
                per_member: false,
                white: false,
                moments,
            }],
        }
    }

    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Mark every sampled error as the sample of one family member each.
    pub(crate) fn independent_members(mut self) -> Self {
        for term in &mut self.terms {
            if term.kind == ContributionKind::SampledError {
                term.per_member = true;
            }
        }
        self
    }

    /// The contributions of one member read at `index` inside `depth`
    /// enclosing families. Per-member samples take the member's coordinate.
    /// Coordinates of a produced family's lane keep their meaning only when
    /// the member is read by the same lane (`identity`) at the same depth;
    /// otherwise they are no longer related to the reading lane and become
    /// unknown.
    pub(crate) fn member(&self, depth: usize, index: &IntExpr, identity: bool) -> Self {
        let lane_depth = self.lane_depth;
        let related = |d: usize| match lane_depth {
            Some(lane_depth) => d < lane_depth || (d == depth && lane_depth == depth && identity),
            None => d <= depth,
        };
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let mut term = term.clone();
                for coordinate in &mut term.sample {
                    if matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if !related(*d))
                    {
                        *coordinate = SampleCoordinate::Unknown;
                    }
                }
                if term.per_member {
                    term.sample
                        .push(SampleCoordinate::Index { depth, index: Arc::new(index.clone()) });
                    term.per_member = false;
                }
                term
            })
            .collect();
        Self { terms, lane_depth: None }
    }

    /// Mark this element of a family produced by a parallel body at `depth`.
    pub(crate) fn with_lane_depth(mut self, depth: usize) -> Self {
        self.lane_depth = Some(depth);
        self
    }

    /// Apply a fixed linear action to every contribution. `action` returns
    /// the mean gain, the variance of the image, and whether the image stays
    /// white, from the contribution before the action.
    pub(crate) fn transform(
        &self,
        mut action: impl FnMut(&NoiseContribution) -> (NonnegativeRational, NonnegativeRational, bool),
    ) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let (mean_gain, variance, white) = action(term);
                NoiseContribution {
                    moments: term.moments.linear_image(&mean_gain, variance).rounded_up(PRECISION),
                    white,
                    ..term.clone()
                }
            })
            .collect();
        Self { terms, lane_depth: self.lane_depth }
    }

    /// Both collections; contributions of the same object are combined
    /// coherently.
    pub(crate) fn merge(&self, other: &Self) -> Self {
        let mut merged =
            Self { terms: self.terms.clone(), lane_depth: self.lane_depth.or(other.lane_depth) };
        merged.terms.extend(other.terms.iter().cloned());
        merged.normalize()
    }

    fn normalize(self) -> Self {
        let mut grouped = BTreeMap::<
            (ContributionKind, [u8; 32], Vec<SampleCoordinate>),
            NoiseContribution,
        >::new();
        let lane_depth = self.lane_depth;
        for term in self.terms {
            let key = (term.kind, term.source, term.sample.clone());
            match grouped.get_mut(&key) {
                Some(existing) => {
                    existing.moments = existing
                        .moments
                        .add_with_covariance(&term.moments, NoiseCorrelation::SharedSource)
                        .rounded_up(PRECISION);
                    existing.white &= term.white;
                    existing.per_member &= term.per_member;
                }
                None => {
                    let mut term = term;
                    term.moments = term.moments.rounded_up(PRECISION);
                    grouped.insert(key, term);
                }
            }
        }
        Self { terms: grouped.into_values().collect(), lane_depth }
    }

    /// Sum a family of `count` lanes at `depth` whose lane index is `lane`.
    pub(crate) fn reduce(
        &self,
        depth: usize,
        lane: &IntExpr,
        count: &BigUint,
        counts: &mut ReductionCounts,
    ) -> Self {
        let lane = lane.clone().canonicalize();
        let terms = self
            .terms
            .iter()
            .map(|term| {
                let at_depth = term
                    .sample
                    .iter()
                    .filter(|coordinate| {
                        matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if *d == depth)
                    })
                    .collect::<Vec<_>>();
                let lane_dependent = at_depth.iter().any(|coordinate| match coordinate {
                    SampleCoordinate::Index { index, .. } => index.contains_loop_index(),
                    _ => false,
                });
                let mut reduced = term.clone();
                // This lane's coordinates are summed away.
                for coordinate in &mut reduced.sample {
                    if matches!(coordinate, SampleCoordinate::Index { depth: d, .. } if *d == depth) {
                        *coordinate = SampleCoordinate::Reduced { depth };
                    }
                }
                // Only a contribution carrying member coordinates can be a
                // distinct object per lane; a gate residual has none.
                let independent =
                    term.kind != ContributionKind::Opaque && lane_dependent && at_depth.len() == 1;
                let multiplicity = match (independent, at_depth.first()) {
                    (true, Some(SampleCoordinate::Index { index, .. })) => {
                        lane_multiplicity(index, &lane, count)
                    }
                    _ => None,
                };
                match multiplicity {
                    Some(LaneMultiplicity::Injective) => {
                        counts.independent += 1;
                        reduced.moments = independent_sum(&term.moments, count, count);
                    }
                    Some(LaneMultiplicity::Histogram(squares)) => {
                        counts.histogram += 1;
                        reduced.moments = independent_sum(&term.moments, count, &squares);
                    }
                    None => {
                        if lane_dependent && term.kind != ContributionKind::Opaque {
                            counts.unknown_index += 1;
                            for coordinate in &mut reduced.sample {
                                if matches!(coordinate, SampleCoordinate::Reduced { depth: d } if *d == depth) {
                                    *coordinate = SampleCoordinate::Unknown;
                                }
                            }
                        } else {
                            counts.coherent += 1;
                        }
                        reduced.moments = term.moments.scaled(count);
                    }
                }
                reduced.white = false;
                reduced
            })
            .collect::<Vec<_>>();
        Self { terms, lane_depth: None }.normalize()
    }

    /// Total moments. Contributions are grouped into classes of possible
    /// covariance: within a class variances combine coherently; distinct
    /// sampled and residual classes add; the opaque class is coherent with
    /// the rest.
    pub(crate) fn total(&self) -> NoiseMoments {
        let summarized_sources = self
            .terms
            .iter()
            .filter(|term| term.kind != ContributionKind::Opaque && term.summarized())
            .map(|term| (term.kind, term.source))
            .collect::<std::collections::BTreeSet<_>>();
        let mut classes =
            BTreeMap::<(ContributionKind, [u8; 32], Vec<SampleCoordinate>), NoiseMoments>::new();
        let mut opaque = None::<NoiseMoments>;
        for term in &self.terms {
            if term.kind == ContributionKind::Opaque {
                opaque = Some(match opaque {
                    None => term.moments.clone(),
                    Some(previous) => {
                        previous.add_with_covariance(&term.moments, NoiseCorrelation::Unknown)
                    }
                });
                continue;
            }
            let (kind, source, sample) = term.key();
            let class_sample = if summarized_sources.contains(&(kind, source)) {
                Vec::new()
            } else {
                sample.to_vec()
            };
            classes
                .entry((kind, source, class_sample))
                .and_modify(|moments| {
                    *moments =
                        moments.add_with_covariance(&term.moments, NoiseCorrelation::SharedSource)
                })
                .or_insert_with(|| term.moments.clone());
        }
        let independent = classes.into_values().fold(zero_moments(), |total, class| {
            total.add_with_covariance(&class, NoiseCorrelation::Independent)
        });
        match opaque {
            None => independent,
            Some(opaque) => independent.add_with_covariance(&opaque, NoiseCorrelation::Unknown),
        }
    }

    /// Variance by contribution kind, for diagnostics.
    /// The contribution with the largest variance, for diagnostics.
    pub(crate) fn dominant(&self) -> Option<&NoiseContribution> {
        self.terms.iter().max_by(|left, right| {
            let (left, right) = (&left.moments.variance, &right.moments.variance);
            (&left.numerator * &right.denominator).cmp(&(&right.numerator * &left.denominator))
        })
    }

    pub(crate) fn variance_by_kind(&self) -> BTreeMap<ContributionKind, NonnegativeRational> {
        let mut by_kind = BTreeMap::<ContributionKind, NonnegativeRational>::new();
        for term in &self.terms {
            let entry = by_kind.entry(term.kind).or_insert_with(NonnegativeRational::zero);
            *entry = entry.add(&term.moments.variance);
        }
        by_kind
    }
}

enum LaneMultiplicity {
    /// Every lane reads a distinct member.
    Injective,
    /// Lanes read members with multiplicities `m_s`; the value is `sum m_s²`.
    Histogram(BigUint),
}

/// How the lanes of a family read the members addressed by `index`.
fn lane_multiplicity(index: &IntExpr, lane: &IntExpr, count: &BigUint) -> Option<LaneMultiplicity> {
    let index = index.clone().canonicalize();
    if let IntExpr::Select { selector, branches } = &index {
        if selector.clone().canonicalize() != *lane || BigUint::from(branches.len()) != *count {
            return None;
        }
        let mut histogram = BTreeMap::new();
        for branch in branches {
            *histogram.entry(branch.evaluate(&ParamEnv::default()).ok()?).or_insert(0u64) += 1;
        }
        if histogram.len() == branches.len() {
            return Some(LaneMultiplicity::Injective);
        }
        return Some(LaneMultiplicity::Histogram(
            histogram.values().map(|multiplicity| BigUint::from(multiplicity * multiplicity)).sum(),
        ));
    }
    // A constant offset of the lane index reads distinct members.
    IntExpr::Sub(Box::new(index), Box::new(lane.clone()))
        .canonicalize()
        .evaluate(&ParamEnv::default())
        .ok()
        .map(|_| LaneMultiplicity::Injective)
}

/// Sum of `count` equally distributed members whose variances combine as
/// `squares` copies of one member's (the count itself for independent
/// members, `sum m_s²` for repeated groups).
fn independent_sum(moments: &NoiseMoments, count: &BigUint, squares: &BigUint) -> NoiseMoments {
    let mean = moments.scaled(count);
    NoiseMoments { variance: moments.variance.scaled(squares), ..mean }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(variance: u64) -> NoiseMoments {
        NoiseMoments {
            variance: NonnegativeRational::new(variance.into(), 1u8.into()).unwrap(),
            ..zero_moments()
        }
    }

    fn variance(breakdown: &NoiseBreakdown) -> NonnegativeRational {
        breakdown.total().variance
    }

    fn rational(value: u64) -> NonnegativeRational {
        NonnegativeRational::new(value.into(), 1u8.into()).unwrap()
    }

    #[test]
    fn identical_sources_add_coherently_and_distinct_samples_independently() {
        let a = NoiseBreakdown::sampled([1; 32], "test", moments(3), true);
        let b = NoiseBreakdown::sampled([2; 32], "test", moments(3), true);
        assert_eq!(variance(&a.merge(&a)), rational(12));
        assert_eq!(variance(&a.merge(&b)), rational(6));
    }

    #[test]
    fn a_family_sum_keeps_shared_coherent_and_fresh_independent() {
        let lane = IntExpr::LoopIndex(0);
        let shared = NoiseBreakdown::sampled([1; 32], "test", moments(2), true);
        let fresh =
            NoiseBreakdown::sampled_member([2; 32], "test", 1, lane.clone(), moments(5), true);
        let mut counts = ReductionCounts::default();
        let reduced = shared.merge(&fresh).reduce(1, &lane, &BigUint::from(4u8), &mut counts);
        // 4² * 2 + 4 * 5.
        assert_eq!(variance(&reduced), rational(52));
        assert_eq!((counts.coherent, counts.independent), (1, 1));
    }

    #[test]
    fn repeated_selections_use_their_multiplicity_histogram() {
        let lane = IntExpr::LoopIndex(0);
        let index = IntExpr::Select {
            selector: Box::new(lane.clone()),
            branches: vec![IntExpr::constant(0), IntExpr::constant(0), IntExpr::constant(1)],
        };
        let member = NoiseBreakdown::sampled_member([1; 32], "test", 1, index, moments(1), true);
        let mut counts = ReductionCounts::default();
        let reduced = member.reduce(1, &lane, &BigUint::from(3u8), &mut counts);
        // 2² + 1², not 3.
        assert_eq!(variance(&reduced), rational(5));
        let unknown = IntExpr::Mul(Box::new(lane.clone()), Box::new(lane.clone()));
        let member = NoiseBreakdown::sampled_member([1; 32], "test", 1, unknown, moments(1), true);
        let reduced = member.reduce(1, &lane, &BigUint::from(3u8), &mut counts);
        assert_eq!(variance(&reduced), rational(9), "an unknown index map stays coherent");
    }

    #[test]
    fn reused_members_stay_shared_and_distinct_members_are_independent() {
        let first = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(3),
            moments(1),
            true,
        );
        let other = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(4),
            moments(1),
            true,
        );
        assert_eq!(variance(&first.merge(&first)), rational(4));
        assert_eq!(variance(&first.merge(&other)), rational(2));
    }

    #[test]
    fn overlapping_family_sums_of_one_source_stay_coherent() {
        let lane = IntExpr::LoopIndex(0);
        let member =
            NoiseBreakdown::sampled_member([1; 32], "test", 1, lane.clone(), moments(1), true);
        let mut counts = ReductionCounts::default();
        let reduced = member.reduce(1, &lane, &BigUint::from(4u8), &mut counts);
        // Two sums over the same members: (sqrt 4 + sqrt 4)² = 16, not 8.
        assert_eq!(variance(&reduced.merge(&reduced)), rational(16));
        // Another single member of the same source joins the coherent class.
        let single = NoiseBreakdown::sampled_member(
            [1; 32],
            "test",
            0,
            IntExpr::constant(0),
            moments(4),
            true,
        );
        assert_eq!(variance(&reduced.merge(&single)), rational(16));
    }

    #[test]
    fn per_member_samples_take_the_read_coordinate() {
        let family =
            NoiseBreakdown::sampled([1; 32], "test", moments(1), true).independent_members();
        let a = family.member(0, &IntExpr::constant(0), false);
        let b = family.member(0, &IntExpr::constant(1), false);
        assert_eq!(variance(&a.merge(&b)), rational(2));
        let shared = NoiseBreakdown::sampled([1; 32], "test", moments(1), true);
        let a = shared.member(0, &IntExpr::constant(0), false);
        let b = shared.member(0, &IntExpr::constant(1), false);
        assert_eq!(
            variance(&a.merge(&b)),
            rational(4),
            "a family without the contract is one sample"
        );
    }

    #[test]
    fn rounding_residuals_are_coherent_across_lanes_and_opaque_terms_with_everything() {
        let lane = IntExpr::LoopIndex(0);
        let rounding = NoiseBreakdown::rounding([7; 32], "test", moments(1));
        let mut counts = ReductionCounts::default();
        assert_eq!(
            variance(&rounding.reduce(1, &lane, &BigUint::from(3u8), &mut counts)),
            rational(9)
        );
        let fresh = NoiseBreakdown::sampled([1; 32], "test", moments(4), true);
        assert_eq!(variance(&fresh.merge(&rounding)), rational(5));
        let opaque = NoiseBreakdown::opaque("test", moments(4));
        assert_eq!(variance(&fresh.merge(&opaque)), rational(16));
    }
}
