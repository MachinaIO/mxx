//! Noise of Dynamic package evaluation, term by term with
//! [`crate::rns_bgg::dynamic`] and the encoding lowering's
//! `dynamic_mul`/`dynamic_lut`.
//!
//! The deterministic bounds are the fixed-basis recurrences of
//! [`RnsBggParameters::dynamic_mul_noise_bound`] and
//! [`RnsBggParameters::dynamic_lut_noise_bound`]; no operation rounds. Every
//! package error is bounded by the setup's contract: every gadget ciphertext's,
//! companion's, and helper's by its Gaussian cutoff. Every gadget ciphertext's
//! top row is public, so a product's companions cover its private digit rows
//! only.
//!
//! Sources are identified by package, never by consuming gate or level: a
//! package read at several gates, and the same automorphism packages read by
//! every LUT, contribute the same sampled object each time, so their images
//! add coherently. The helpers of `k` and `k^-1` enter one LUT through the
//! mask switch and through the derived `C_k`; one helper source covers both.
//! Moments use the squared `l1` gain of every action unless a contribution
//! is a fresh sample against a fixed public action, whose energy bounds its
//! variance; acceptance uses the deterministic bound.
use super::*;

impl RnsBggNoiseEvaluator<'_> {
    fn dynamic_contracts(&self) -> Result<&RnsBggDynamicNoise, RnsBggError> {
        self.bindings
            .dynamic
            .as_ref()
            .ok_or(RnsBggError::Binding("the Dynamic package contracts are not bound"))
    }

    /// The source identity of a package component.
    fn package_source(
        &self,
        component: &str,
        coordinate: Option<u32>,
    ) -> Result<[u8; 32], RnsBggError> {
        let namespace = self.dynamic_contracts()?.namespace;
        Ok(stable_source_id(&(
            b"rns-bgg/dynamic-package".as_slice(),
            namespace,
            component,
            coordinate,
        )))
    }

    /// One package component read with the sample coordinate of `member`.
    fn package_term(
        &self,
        source: [u8; 32],
        origin: &'static str,
        member: Option<&IntExpr>,
        variance: NonnegativeRational,
    ) -> NoiseBreakdown {
        let moments = zero_mean_moments(variance);
        match member {
            Some(index) => NoiseBreakdown::sampled_member(
                source,
                origin,
                self.lane_indices.len(),
                index.clone(),
                moments,
                false,
            ),
            None => NoiseBreakdown::sampled(source, origin, moments, false),
        }
    }

    /// The error bound of a package component of contract cutoff `cutoff`:
    /// its expanded bound with compact encodings, else the cutoff.
    fn package_error(
        &self,
        contracts: &RnsBggDynamicNoise,
        cutoff: &BigUint,
    ) -> Result<BigUint, RnsBggError> {
        match &contracts.compact_extension {
            Some(extension) => {
                crate::rns_bgg::dynamic::compact_expansion_bound(self.parameters, extension, cutoff)
            }
            None => Ok(cutoff.clone()),
        }
    }

    /// The contribution of expanded compact package errors whose share of the
    /// bound is at most `bound`: an opaque term of mean radius `bound` and
    /// variance `bound^2` (`|X| <= b` gives `|E X| <= b` and `Var X <= b^2`),
    /// aggregated with every other term under an unknown covariance. Expansion
    /// repeats each sampled error across columns, so independent-entry
    /// energies and zero means do not hold for it.
    fn compact_term(origin: &'static str, bound: &BigUint) -> NoiseBreakdown {
        NoiseBreakdown::opaque(
            origin,
            NoiseMoments {
                mean_numerator: 0.into(),
                mean_denominator: BigUint::one(),
                mean_radius: NonnegativeRational::new(bound.clone(), BigUint::one())
                    .expect("unit denominator"),
                variance: NonnegativeRational::new(bound.pow(2u32), BigUint::one())
                    .expect("unit denominator"),
            },
        )
    }

    /// The variance of a Gaussian contract of standard deviation `sigma`.
    fn contract_variance(sigma: &BigDecimal) -> NonnegativeRational {
        encoding_variance_from_sigma(sigma)
    }

    /// The Dynamic product of `input` by the package of `rhs`.
    pub(super) fn dynamic_mul(
        &mut self,
        input: &(WireId, RnsBggNoiseValue),
        rhs: &RhsRef<IntExpr>,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let contracts = self.dynamic_contracts()?.clone();
        let level = self.level(input.0)?;
        let wire = self.site()?.output;
        self.parameters.dynamic_route(level)?;
        self.parameters.dynamic_route(self.level(wire)?)?;
        let coordinate = *contracts
            .rhs
            .get(&rhs.id)
            .ok_or(RnsBggError::Binding("missing RHS package contract"))?;
        let (contract, sparse_count) = match self
            .bindings
            .rhs
            .get(&rhs.id)
            .ok_or(RnsBggError::Binding("missing RHS noise contract"))?
        {
            InputBinding::Scalar(value) => (value.clone(), None),
            InputBinding::Family(family) => {
                (family.element.clone(), Some(family.max_nonzero_payloads.clone()))
            }
        };
        let distributions = &contracts.distributions;
        let ciphertext_error = distributions.ciphertext.cutoff().clone();
        let ciphertext_sigma = distributions.ciphertext.sigma();
        let companion = &distributions.companion;
        let companion_error = self.package_error(&contracts, companion.cutoff())?;
        let output_l1 = &input.1.bound.payload_l1 * &contract.payload_l1;
        // The public top row's digit rows have no companion.
        let public_rows = crate::rns_bgg::automorphism_public_rows(self.parameters, level)?;
        let bound = self.parameters.dynamic_mul_noise_bound(
            level,
            &input.1.bound,
            &ciphertext_error,
            &companion_error,
            public_rows,
            output_l1,
        )?;
        let compact = contracts.compact_extension.is_some();
        // The companions' share `(m - p) K beta'` of the bound.
        let companion_share =
            BigUint::from(2 * self.parameters.component_columns(level)? - public_rows) *
                self.parameters.dynamic_decomposition_gain(level)? *
                &companion_error;
        let k = self.parameters.dynamic_decomposition_gain(level)?;
        let gain = NonnegativeRational::new(k.clone(), BigUint::one()).expect("unit denominator");
        let k_squared = k.pow(2u32);
        let ciphertext_source = self.package_source("rhs-ciphertext", Some(coordinate))?;
        let companion_source = self.package_source("rhs-companion", Some(coordinate))?;
        let companion_variance = encoding_variance_from_sigma(companion.sigma())
            .scaled(&self.parameters.dynamic_companion_energy(level, public_rows)?);
        let ciphertext_variance =
            Self::contract_variance(ciphertext_sigma).scaled(&input.1.payload_l2_squared);
        let breakdown = input.1.contributions("dynamic-mul-input").map(|input_terms| {
            // `e Dec(C)`: `Dec(C)` depends on the ciphertext, so the input
            // error keeps the squared `l1` gain.
            input_terms
                .transform(|term| (gain.clone(), term.moments.variance.scaled(&k_squared), false))
                .merge(&self.package_term(
                    ciphertext_source,
                    "dynamic-rhs-ciphertext-error",
                    rhs.index.as_ref(),
                    ciphertext_variance.clone(),
                ))
                .merge(&if compact {
                    Self::compact_term("dynamic-rhs-compact-companion-error", &companion_share)
                } else {
                    self.package_term(
                        companion_source,
                        "dynamic-rhs-companion-error",
                        rhs.index.as_ref(),
                        companion_variance.clone(),
                    )
                })
        });
        let sources = [ciphertext_source, companion_source]
            .into_iter()
            .map(|source| (source, NoiseSourceProvenance { source, route: vec![level] }))
            .collect::<BTreeMap<_, _>>();
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound,
                payload_l2_squared: &input.1.payload_l2_squared * &contract.payload_l2_squared,
                moments: None,
                provenance: BTreeMap::new(),
                heuristics: input.1.heuristics.clone(),
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone(), sources],
        );
        let node = crate::noise_utils::NoiseDependencyNode::MulPrivate {
            gate: Some(wire),
            input: self.node_for_wire(input.0)?,
            rhs: self.rhs_node(rhs),
            helper: None,
        };
        if let (Some(count), Some(index), Some(lane)) =
            (sparse_count, rhs.index.as_ref(), self.lane_indices.last())
        {
            if super::evaluator::lane_injective(index, lane) {
                self.set_lane_support(wire, count);
            }
        }
        self.operation_node(wire, value, node)
    }

    /// The Dynamic LUT of `input` by `table`.
    pub(super) fn dynamic_lut(
        &mut self,
        input: &(WireId, RnsBggNoiseValue),
        table: &LutTable,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let contracts = self.dynamic_contracts()?.clone();
        let level = self.level(input.0)?;
        let wire = self.site()?.output;
        let lut = self.program.lut_id_at(wire).map_err(|_| RnsBggError::Lut)?;
        let normalized = level + usize::from(table.output_representation().is_coefficient());
        self.parameters.dynamic_route(normalized.max(self.level(wire)?))?;
        let automorphisms = crate::rns_bgg::dynamic::lut_automorphisms(
            self.parameters,
            &contracts.automorphisms,
            level,
            table,
        )?;
        let branches = automorphisms.len();
        // `depths[d - 1]` branches are switched `d` times.
        let mut depths = Vec::new();
        for k in &automorphisms[1..] {
            let depth = contracts.depths.get(k).copied().unwrap_or(1);
            if depths.len() < depth {
                depths.resize(depth, 0);
            }
            depths[depth - 1] += 1;
        }
        let distributions = &contracts.distributions;
        let output_payload_l1 = match table.output_representation() {
            crate::program::LutOutputRepresentation::Exponent(_) => BigUint::one(),
            crate::program::LutOutputRepresentation::Coefficient => {
                BigUint::from(table.max_output())
            }
        };
        let companion_error = self.package_error(&contracts, distributions.companion.cutoff())?;
        let helper_error = self.package_error(&contracts, distributions.helper.cutoff())?;
        let lut_bound = |companion_error: &BigUint, helper_error: &BigUint| {
            if depths.len() <= 1 {
                self.parameters.dynamic_lut_noise_bound(
                    level,
                    &input.1.bound,
                    branches,
                    companion_error,
                    helper_error,
                    output_payload_l1.clone(),
                )
            } else {
                self.parameters.dynamic_radix_lut_noise_bound(
                    level,
                    &input.1.bound,
                    &depths,
                    companion_error,
                    helper_error,
                    output_payload_l1.clone(),
                )
            }
        };
        let bound = lut_bound(&companion_error, &helper_error)?;
        // The package errors' share of the bound, for compact encodings.
        let package_share = contracts
            .compact_extension
            .is_some()
            .then(|| {
                lut_bound(&BigUint::zero(), &BigUint::zero())
                    .map(|without| &bound.error - without.error)
            })
            .transpose()?;
        let k = self.parameters.dynamic_decomposition_gain(level)?;
        let s = self.parameters.dynamic_scalar_action_gain(level)?;
        // The input reaches the output through the identity branch (`S`) and
        // every branch switched `d` times (`S K^d`), coherently.
        let input_gain = depths.iter().enumerate().fold(s.clone(), |gain, (depth, count)| {
            gain + BigUint::from(*count) * &s * k.pow(depth as u32 + 1)
        });
        // The fresh errors of a branch's `j`-th of `d` switches pass the
        // decompositions of the `d - j` switches after it (at most `K` each
        // in `l1`) as well.
        let switched = depths.iter().enumerate().fold(BigUint::zero(), |sum, (depth, count)| {
            sum + BigUint::from(*count) *
                (0..=depth).map(|later| k.pow(later as u32)).sum::<BigUint>()
        });
        let mean_gain =
            NonnegativeRational::new(input_gain.clone(), BigUint::one()).expect("unit denominator");
        let input_variance_gain = input_gain.pow(2u32);
        // A fresh companion or helper error against the fixed public actions
        // `Dec(T_k)` then `V_D` has the variance of their energies; the
        // switched branches add coherently, one source per component.
        let energy = self.parameters.dynamic_scalar_action_energy(level)?;
        let coherent = switched.pow(2u32) * &energy;
        let companion_variance = encoding_variance_from_sigma(distributions.companion.sigma())
            .scaled(&(self.parameters.dynamic_automorphism_companion_energy(level)? * &coherent));
        let decomposition_energy = BigUint::from(2 * self.parameters.component_columns(level)?) *
            self.parameters.level(level)?.ring_dimension() *
            ((BigUint::one() << self.parameters.level(level)?.base_bits()) / 2u8).pow(2u32);
        let helper_variance = encoding_variance_from_sigma(distributions.helper.sigma())
            .scaled(&((decomposition_energy + &input.1.payload_l2_squared) * &coherent));
        let companion_source = self.package_source("automorphism-companion", None)?;
        let helper_source = self.package_source("automorphism-helper", None)?;
        let breakdown = input.1.contributions("dynamic-lut-input").map(|input_terms| {
            let terms = input_terms.transform(|term| {
                (mean_gain.clone(), term.moments.variance.scaled(&input_variance_gain), false)
            });
            if branches == 1 {
                return terms;
            }
            if let Some(share) = &package_share {
                return terms.merge(&Self::compact_term(
                    "dynamic-automorphism-compact-package-error",
                    share,
                ));
            }
            terms
                .merge(&self.package_term(
                    companion_source,
                    "dynamic-automorphism-companion-error",
                    None,
                    companion_variance.clone(),
                ))
                .merge(&self.package_term(
                    helper_source,
                    "dynamic-automorphism-helper-error",
                    None,
                    helper_variance.clone(),
                ))
        });
        let sources = if branches == 1 {
            BTreeMap::new()
        } else {
            [companion_source, helper_source]
                .into_iter()
                .map(|source| (source, NoiseSourceProvenance { source, route: vec![level] }))
                .collect::<BTreeMap<_, _>>()
        };
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound,
                payload_l2_squared: output_payload_l1.pow(2u32),
                moments: None,
                provenance: BTreeMap::new(),
                heuristics: input.1.heuristics.clone(),
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone(), sources],
        );
        let node = crate::noise_utils::NoiseDependencyNode::Lut {
            gate: Some(wire),
            table: lut,
            input: self.node_for_wire(input.0)?,
            helper: None,
        };
        self.operation_node_with_multiplier(
            wire,
            value,
            node,
            self.event_multiplier.clone() * BigUint::from(branches),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// `value >= bound` for a nonnegative rational and an integer.
    fn at_least(value: &NonnegativeRational, bound: &BigUint) -> bool {
        value.numerator >= bound * &value.denominator
    }

    /// Compact package contributions are opaque enclosures: each carries mean radius `b` and
    /// variance `b^2` for its deterministic share `b`; a breakdown with them hides its source
    /// identity (no sampled class); and its total, aggregated with an unknown covariance among
    /// them and against a sampled term, has mean radius at least `sum b_i`, variance at least
    /// `(sum b_i)^2` (the fully correlated enclosure), and a tail bound at least `sum b_i`.
    #[test]
    fn compact_terms_enclose_their_deterministic_shares() {
        let shares = [3u32, 5, 11].map(BigUint::from);
        let sampled = NoiseBreakdown::sampled(
            [7; 32],
            "sampled",
            zero_mean_moments(NonnegativeRational::new(4u8.into(), BigUint::one()).unwrap()),
            false,
        );
        assert!(sampled.operand().is_some());
        let mut breakdown = sampled.clone();
        for share in &shares {
            let term = RnsBggNoiseEvaluator::compact_term("compact", share);
            let moments = term.total();
            assert!(at_least(&moments.mean_radius, &share.clone()));
            assert!(at_least(&moments.variance, &share.pow(2u32)));
            assert!(term.operand().is_none(), "a compact term is not a sampled source");
            breakdown = breakdown.merge(&term);
        }
        // The same package read again keeps the enclosure (repeated reads add coherently).
        breakdown = breakdown.merge(&RnsBggNoiseEvaluator::compact_term("compact", &shares[0]));
        let sum = shares.iter().sum::<BigUint>() + &shares[0];
        let total = breakdown.total();
        assert!(breakdown.operand().is_none());
        assert!(at_least(&total.mean_radius, &sum.clone()));
        assert!(at_least(&total.variance, &sum.pow(2u32)));
        let z_squared = NonnegativeRational::new(1u8.into(), BigUint::one()).unwrap();
        assert!(total.tail_bound(&z_squared) >= sum);
    }
}
