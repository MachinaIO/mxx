//! Noise of Dynamic package evaluation, term by term with
//! [`crate::rns_bgg::dynamic`] and the encoding lowering's
//! `dynamic_mul`/`dynamic_lut`.
//!
//! The deterministic bounds are the fixed-basis recurrences of
//! [`RnsBggParameters::dynamic_mul_noise_bound`] and
//! [`RnsBggParameters::dynamic_lut_noise_bound`]; no operation rounds. Every
//! package error is bounded by the setup's contract: a sampled gadget
//! ciphertext's and every companion's and helper's by its Gaussian cutoff, a
//! supplied gadget ciphertext's by its declared bound.
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

    /// The variance of a Gaussian contract, or of a supplied bound `B`
    /// (`B^2` bounds the variance of any variable bounded by `B`).
    fn contract_variance(sigma: Option<&BigDecimal>, bound: &BigUint) -> NonnegativeRational {
        match sigma {
            Some(sigma) => encoding_variance_from_sigma(sigma),
            None => {
                NonnegativeRational::new(bound.pow(2u32), BigUint::one()).expect("unit denominator")
            }
        }
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
        let package = contracts
            .rhs
            .get(&rhs.id)
            .ok_or(RnsBggError::Binding("missing RHS package contract"))?
            .clone();
        let coordinate = package.coordinate;
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
        let ciphertext_error = package.ciphertext_error.clone();
        let ciphertext_sigma = package.ciphertext_sigma.as_ref();
        let companion = &distributions.companion;
        let output_l1 = &input.1.bound.payload_l1 * &contract.payload_l1;
        let bound = self.parameters.dynamic_mul_noise_bound(
            level,
            &input.1.bound,
            &ciphertext_error,
            companion.cutoff(),
            output_l1,
        )?;
        let k = self.parameters.dynamic_decomposition_gain(level)?;
        let gain = NonnegativeRational::new(k.clone(), BigUint::one()).expect("unit denominator");
        let k_squared = k.pow(2u32);
        let ciphertext_source = self.package_source("rhs-ciphertext", Some(coordinate))?;
        let companion_source = self.package_source("rhs-companion", Some(coordinate))?;
        let companion_variance = encoding_variance_from_sigma(companion.sigma())
            .scaled(&self.parameters.dynamic_companion_energy(level)?);
        let ciphertext_variance = Self::contract_variance(ciphertext_sigma, &ciphertext_error)
            .scaled(&input.1.payload_l2_squared);
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
                .merge(&self.package_term(
                    companion_source,
                    "dynamic-rhs-companion-error",
                    rhs.index.as_ref(),
                    companion_variance.clone(),
                ))
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
        let branches = crate::rns_bgg::dynamic::lut_automorphisms(
            self.parameters,
            &contracts.automorphisms,
            level,
            table,
        )?
        .len();
        let distributions = &contracts.distributions;
        let output_payload_l1 = match table.output_representation() {
            crate::program::LutOutputRepresentation::Exponent(_) => BigUint::one(),
            crate::program::LutOutputRepresentation::Coefficient => {
                BigUint::from(table.max_output())
            }
        };
        let bound = self.parameters.dynamic_lut_noise_bound(
            level,
            &input.1.bound,
            branches,
            distributions.companion.cutoff(),
            distributions.helper.cutoff(),
            output_payload_l1.clone(),
        )?;
        let k = self.parameters.dynamic_decomposition_gain(level)?;
        let s = self.parameters.dynamic_scalar_action_gain(level)?;
        let switched = BigUint::from(branches - 1);
        // The input reaches the output through the identity branch (`S`) and
        // every switched branch (`S K`), coherently.
        let input_gain = &s + &switched * &s * &k;
        let mean_gain =
            NonnegativeRational::new(input_gain.clone(), BigUint::one()).expect("unit denominator");
        let input_variance_gain = input_gain.pow(2u32);
        // A fresh companion or helper error against the fixed public actions
        // `Dec(T_k)` then `V_D` has the variance of their energies; the
        // switched branches add coherently, one source per component.
        let energy = self.parameters.dynamic_scalar_action_energy(level)?;
        let coherent = switched.pow(2u32) * &energy;
        let companion_variance = encoding_variance_from_sigma(distributions.companion.sigma())
            .scaled(&(self.parameters.dynamic_companion_energy(level)? * &coherent));
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
