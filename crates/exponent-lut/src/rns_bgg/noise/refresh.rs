//! Refresh noise sources and gains, the evaluator's refresh lowering, and
//! its `NoiseRefresh` implementation.
use super::*;

fn refresh_channel_mod_down_bound(
    error: &BigUint,
    ratio: &BigUint,
) -> Result<BigUint, RnsBggError> {
    if ratio.is_zero() {
        return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
    }
    Ok(ceil_div_nonnegative(error, ratio) +
        if *ratio == BigUint::one() { BigUint::zero() } else { BigUint::one() })
}

fn refresh_fresh_source(
    commitment: &[u8; 32],
    component: usize,
    coefficient: usize,
    digit: usize,
) -> [u8; 32] {
    stable_source_id(&(b"rns-bgg-refresh/fresh", commitment, component, coefficient, digit))
}

fn refresh_mask_source(
    commitment: &[u8; 32],
    slot: usize,
    component: usize,
    coefficient: usize,
    digit: usize,
) -> [u8; 32] {
    stable_source_id(&(b"rns-bgg-refresh/mask", commitment, slot, component, coefficient, digit))
}

/// Stable source identity for the per-slot public refresh anchor `A_star`.
///
/// The anchor is reduced to `q_t` as a centered carrier.  It is therefore not
/// a `p -> q_t` ModDown event and must not acquire a channel-rounding source.
/// The decoder still uses the corresponding slot's `A_sum - H*A_star` target,
/// so keep an explicit route identity for that public contribution.
fn refresh_astar_source(commitment: &[u8; 32], slot: usize) -> [u8; 32] {
    stable_source_id(&(b"rns-bgg-refresh/a-star", commitment, slot))
}

/// Gain of one prepared refresh label action.
///
/// The public refresh action is `base^digit * X^coefficient * unit_row(component)`;
/// the rotation and unit-row routing preserve the coefficient norm, so only the
/// positional digit weight contributes to the scalar action gain.  Fresh labels
/// additionally carry the slot's `H = p/q_t` factor.  Keeping this per-label
/// avoids applying an aggregate route gain after the prepared weight has already
/// has been accounted for; the persisted aggregate decomposition gain is split
/// over these weights by `refresh_label_route_gain`.
fn refresh_label_action_gain(
    base: &BigUint,
    label: &RefreshPrfLabel,
    p: &BigUint,
    qt: &BigUint,
) -> Result<RightActionNoiseGain, RnsBggError> {
    if *base < BigUint::from(2u8) ||
        base.is_zero() ||
        qt.is_zero() ||
        p < qt ||
        p % qt != BigUint::zero()
    {
        return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
    }
    let digit = match label {
        RefreshPrfLabel::Mask { digit, .. } | RefreshPrfLabel::FreshError { digit, .. } => *digit,
    };
    let fresh = matches!(label, RefreshPrfLabel::FreshError { .. });
    let mut l1 = base.pow(digit as u32);
    if fresh {
        l1 *= p / qt;
    }
    Ok(RightActionNoiseGain { l2_squared: l1.pow(2u32), l1 })
}

fn refresh_label_weight_sums(
    base: &BigUint,
    digits: usize,
    component_columns: usize,
    coefficient_count: usize,
    p: &BigUint,
    qt: &BigUint,
    fresh: bool,
) -> Result<(BigUint, BigUint), RnsBggError> {
    if digits == 0 ||
        component_columns == 0 ||
        coefficient_count == 0 ||
        (fresh && (qt.is_zero() || p < qt || p % qt != BigUint::zero()))
    {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let h = if fresh { p / qt } else { BigUint::one() };
    let mut l1 = BigUint::zero();
    let mut l2_squared = BigUint::zero();
    for digit in 0..digits {
        let weight = base.pow(digit as u32) * &h;
        l1 += &weight;
        l2_squared += weight.pow(2u32);
    }
    let multiplicity = BigUint::from(component_columns) * BigUint::from(coefficient_count);
    Ok((l1 * &multiplicity, l2_squared * multiplicity))
}

/// Split a setup aggregate route gain over its canonical label weights.
/// Rounding outward keeps the persisted aggregate contract authoritative while
/// allowing each label to carry its own digit/H action exactly once.
fn refresh_label_route_gain(
    label_gain: &RightActionNoiseGain,
    aggregate: &RightActionNoiseGain,
    weight_l1_sum: &BigUint,
    weight_l2_squared_sum: &BigUint,
) -> Result<RightActionNoiseGain, RnsBggError> {
    if weight_l1_sum.is_zero() || weight_l2_squared_sum.is_zero() {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let unit_l1 = ceil_div_nonnegative(&aggregate.l1, weight_l1_sum);
    let unit_l2 = ceil_div_nonnegative(&aggregate.l2_squared, weight_l2_squared_sum);
    Ok(RightActionNoiseGain {
        l1: &label_gain.l1 * unit_l1,
        l2_squared: &label_gain.l2_squared * unit_l2,
    })
}

/// Add one deterministic affine route offset to a noise carrier.
///
/// Refresh digit selectors are represented by canonical positive coordinates;
/// the concrete mask/fresh route then adds a public half-offset.  The offset
/// is an affine constant, not another sampled digit: it changes the integer
/// support and mean, but never variance, provenance, or event count.
fn refresh_add_deterministic_offset(
    value: &mut RnsBggNoiseValue,
    offset_numerator: BigInt,
    offset_denominator: &BigUint,
) {
    if offset_numerator.is_zero() {
        return;
    }
    let magnitude = offset_numerator.magnitude().clone();
    value.bound.error += ceil_div_nonnegative(&magnitude, offset_denominator);
    if let Some(moments) = value.moments.as_mut() {
        let numerator = &moments.mean_numerator * BigInt::from(offset_denominator.clone()) +
            &offset_numerator * BigInt::from(moments.mean_denominator.clone());
        let denominator = &moments.mean_denominator * offset_denominator;
        let divisor = gcd(numerator.magnitude(), &denominator);
        let divisor = if divisor.is_zero() { BigUint::one() } else { divisor };
        moments.mean_numerator = numerator / BigInt::from(divisor.clone());
        moments.mean_denominator = denominator / divisor;
    }
}

/// Return the aggregate canonical-positive affine offset numerator for one
/// mask/fresh route.
///
/// The common contract gives mask `-(S_d + b^d/2)` and fresh
/// `-(b/2)S_d`, where `S_d = sum(b^digit)`.  The numerator is over two.  The
/// q-ring route scale is `q/p` for masks and `q/q_t` for fresh errors; the
/// label action's fresh `H = p/q_t` remains in `weight_l1_sum` and is not
/// applied a second time to the offset.
fn refresh_canonical_offset_numerator(
    base: &BigUint,
    digits: usize,
    component_columns: usize,
    coefficient_count: usize,
    aggregate: &RightActionNoiseGain,
    p: &BigUint,
    qt: &BigUint,
    offset_scale: &BigUint,
    fresh: bool,
) -> Result<BigInt, RnsBggError> {
    let (weight_l1_sum, _) = refresh_label_weight_sums(
        base,
        digits,
        component_columns,
        coefficient_count,
        p,
        qt,
        fresh,
    )?;
    let unit_gain = ceil_div_nonnegative(&aggregate.l1, &weight_l1_sum);
    let offset = crate::noise_refresh::refresh::canonical_digit_offset(base, digits, !fresh)
        .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let multiplicity = BigUint::from(component_columns) * BigUint::from(coefficient_count);
    Ok(offset * BigInt::from(offset_scale * multiplicity * unit_gain))
}

fn refresh_decoder_provenance(
    a_sum_values: &[&RnsBggNoiseValue],
    commitment: &[u8; 32],
    slot: usize,
) -> BTreeMap<[u8; 32], NoiseSourceProvenance> {
    let mut provenance: BTreeMap<[u8; 32], NoiseSourceProvenance> = BTreeMap::new();
    // The decoder carrier is s*ModDown_qp(A_sum - H*A_star) + e_dec.
    // A_sum shares the mask/fresh PRF labels with C_sum, which is essential
    // for preserving the covariance introduced by subtracting the decoder
    // after q -> p.  The slot's A_star source also represents the correlated
    // data-public route A_star*W included in A_sum.  The input/state carrier
    // is not part of this public target.  In particular, do not copy state
    // provenance here: it would claim a correlation between input ciphertext
    // noise and the independently prepared public anchor.
    for value in a_sum_values {
        for (source, route) in &value.provenance {
            let mut route = route.clone();
            route.route.push(slot);
            if let Some(existing) = provenance.get_mut(source) {
                existing.route.extend(route.route);
            } else {
                provenance.insert(*source, route);
            }
        }
    }
    let source = refresh_astar_source(commitment, slot);
    provenance.insert(source, NoiseSourceProvenance { source, route: vec![slot] });
    provenance
}

impl<'a> RnsBggNoiseEvaluator<'a> {
    /// Attach the checked refresh noise contract to this existing builder.
    /// The contract is stage metadata; no carrier or second evaluator is
    /// introduced.  Refresh composition fails closed when this method has not
    /// been called before the common refresh callback.
    pub fn with_refresh_noise_parameters(
        &mut self,
        parameters: RefreshNoiseParameters,
    ) -> Result<(), RnsBggError> {
        parameters.validate().map_err(|error| RnsBggError::Graph(error.to_string()))?;
        self.refresh_parameters = Some(parameters);
        Ok(())
    }

    fn refresh_binding_node(&mut self, wire: WireId) -> crate::noise_utils::NoiseNodeId {
        let node = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Binding(
            crate::noise_utils::NoiseBinding::GateHelper(wire),
        ));
        self.trace.node_event_counts.insert(node.0, BigUint::zero());
        node
    }

    fn refresh_add_node(
        &mut self,
        gate: WireId,
        lhs: crate::noise_utils::NoiseNodeId,
        rhs: crate::noise_utils::NoiseNodeId,
    ) -> crate::noise_utils::NoiseNodeId {
        let node = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Add {
            gate: Some(gate),
            lhs,
            rhs,
        });
        self.trace.node_event_counts.insert(node.0, BigUint::zero());
        node
    }

    fn route_value(
        &self,
        value: &RnsBggNoiseValue,
        weight: &BigUint,
        gain: &RightActionNoiseGain,
        payload: bool,
    ) -> RnsBggNoiseValue {
        let action = weight * &gain.l1;
        let action_squared = weight.pow(2u32) * &gain.l2_squared;
        let mut routed = value.clone();
        routed.bound.error *= &action;
        if !payload {
            routed.bound.payload_l1 = BigUint::zero();
            routed.payload_l2_squared = BigUint::zero();
        } else {
            routed.bound.payload_l1 *= weight;
            routed.payload_l2_squared *= weight.pow(2u32);
        }
        if let Some(moments) = value.moments.as_ref() {
            routed.moments = Some(
                moments.linear_image(
                    &NonnegativeRational::new(action.clone(), BigUint::one())
                        .expect("positive refresh route action"),
                    moments.variance.scaled(&action_squared),
                ),
            );
        }
        routed
    }

    fn zero_route_value() -> RnsBggNoiseValue {
        RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error: BigUint::zero(), payload_l1: BigUint::zero() },
            payload_l2_squared: BigUint::zero(),
            moments: Some(NoiseMoments {
                mean_numerator: BigInt::zero(),
                mean_denominator: BigUint::one(),
                mean_radius: NonnegativeRational::zero(),
                variance: NonnegativeRational::zero(),
            }),
            provenance: BTreeMap::new(),
            heuristics: NoiseHeuristicLedger::new(),
            breakdown: None,
        }
    }

    fn combine_route(&self, left: RnsBggNoiseValue, right: RnsBggNoiseValue) -> RnsBggNoiseValue {
        let correlation =
            if left.provenance.keys().any(|source| right.provenance.contains_key(source)) {
                NoiseCorrelation::SharedSource
            } else if !left.provenance.is_empty() && !right.provenance.is_empty() {
                NoiseCorrelation::Independent
            } else {
                NoiseCorrelation::Unknown
            };
        left.combine(&right, correlation)
    }

    fn mod_down_channel(
        &self,
        value: RnsBggNoiseValue,
        ratio: &BigUint,
        policy: &NoiseMomentPolicy,
        commitment: &[u8; 32],
        slot: usize,
        stage: &'static [u8],
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        if ratio.is_zero() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        let mut reduced = value.clone();
        reduced.bound.error = refresh_channel_mod_down_bound(&value.bound.error, ratio)?;
        reduced.moments = value.moments.as_ref().map(|moments| {
            moments.linear_image(
                &NonnegativeRational::new(BigUint::one(), ratio.clone())
                    .expect("positive channel modulus ratio"),
                moments.variance.scaled_ratio(&BigUint::one(), &ratio.pow(2u32)),
            )
        });
        if *ratio == BigUint::one() {
            return Ok(reduced);
        }
        let rounding = match policy {
            NoiseMomentPolicy::Discrete { .. } => NoiseMoments::discrete_rounding_residue(ratio),
            NoiseMomentPolicy::Continuous => Ok(NoiseMoments::continuous_rounding_residue()),
        }
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        reduced.moments = reduced
            .moments
            .map(|moments| moments.add_with_covariance(&rounding, NoiseCorrelation::Independent));
        // The channel switch introduces a slot-specific rounding source. Keep
        // it in the source set so a later CRT sum never treats residues from
        // different channels as one shared sample.
        let source =
            stable_source_id(&(b"rns-bgg-refresh/channel-mod-down", stage, commitment, slot));
        reduced.provenance.insert(source, NoiseSourceProvenance { source, route: vec![slot] });
        match policy {
            NoiseMomentPolicy::Discrete { .. } => {
                reduced.heuristics.insert(NoiseHeuristicId::ExactDistribution);
            }
            NoiseMomentPolicy::Continuous => {
                reduced.heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
            }
        }
        Ok(reduced)
    }

    fn refresh_raw_target(
        &mut self,
        inputs: &BTreeMap<usize, (WireId, RnsBggNoiseValue)>,
        declaration: &RefreshDeclaration,
        outputs: &BTreeMap<PrfLabel, BTreeMap<usize, (WireId, RnsBggNoiseValue)>>,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        let refresh = self
            .refresh_parameters
            .clone()
            .ok_or(RnsBggError::Parameters("refresh noise parameters are not bound"))?;
        refresh.validate().map_err(|error| RnsBggError::Graph(error.to_string()))?;
        if declaration.output_schema().is_empty() || declaration.slots().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        let topology = refresh
            .transcript_topology
            .as_ref()
            .ok_or(RnsBggError::Parameters("refresh transcript topology is missing"))?;
        let gains = refresh
            .route_gains
            .as_ref()
            .ok_or(RnsBggError::Parameters("refresh route gains are missing"))?;
        let decoder_bounds = if refresh.route_bias_bounds.len() == declaration.slots().len() {
            &refresh.route_bias_bounds
        } else {
            return Err(RnsBggError::Parameters("refresh decoder noise is missing"));
        };
        let global_modulus = refresh
            .global_modulus
            .clone()
            .ok_or(RnsBggError::Parameters("refresh global modulus is missing"))?;
        if topology.crt_slot_count != declaration.slots().len() ||
            topology.mask_digit_counts != declaration.digits().mask_counts ||
            topology.fresh_digit_count != declaration.digits().fresh_count ||
            topology.component_columns == 0 ||
            topology.coefficient_count == 0
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        if refresh.base_p != BigUint::from(declaration.digits().base) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }

        let ordered = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|entry| entry.1)
            .collect::<Vec<_>>();
        if inputs.len() != declaration.slots().len() ||
            inputs.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let source_wire = declaration.source_output();
        let source = inputs.get(&0).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        if source.0 != source_wire || inputs.values().any(|(wire, _)| *wire != source_wire) {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let source_payload = source.1.clone();
        let mut masks = std::iter::repeat_with(Self::zero_route_value)
            .take(declaration.slots().len())
            .collect::<Vec<_>>();
        let mut fresh = std::iter::repeat_with(Self::zero_route_value)
            .take(declaration.slots().len())
            .collect::<Vec<_>>();
        let mut refresh_nodes = Vec::with_capacity(declaration.slots().len() + outputs.len() + 1);
        for (index, expected_wire) in ordered.into_iter().enumerate() {
            let channel_outputs = outputs
                .get(&declaration.prf_labels()[index])
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            match label {
                RefreshPrfLabel::Mask { slot, digit, component, coefficient, .. } => {
                    if slot >= masks.len() ||
                        digit >= declaration.digits().mask_counts[slot] ||
                        component >= topology.component_columns ||
                        coefficient >= topology.coefficient_count
                    {
                        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
                    }
                    if channel_outputs.len() != 1 || !channel_outputs.contains_key(&slot) {
                        return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                    }
                    let (wire, value) = channel_outputs
                        .get(&slot)
                        .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                    if *wire != expected_wire {
                        return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                    }
                    refresh_nodes.push(self.refresh_binding_node(*wire));
                    let slot_metadata = declaration
                        .slots()
                        .get(slot)
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                    let p = slot_metadata
                        .p
                        .to_biguint()
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
                    let qt = slot_metadata
                        .q_t
                        .to_biguint()
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
                    let label_gain = refresh_label_action_gain(
                        &BigUint::from(declaration.digits().base),
                        &label,
                        &p,
                        &qt,
                    )?;
                    let (weight_l1_sum, weight_l2_squared_sum) = refresh_label_weight_sums(
                        &BigUint::from(declaration.digits().base),
                        declaration.digits().mask_counts[slot],
                        topology.component_columns,
                        topology.coefficient_count,
                        &p,
                        &qt,
                        false,
                    )?;
                    let route_gain = refresh_label_route_gain(
                        &label_gain,
                        &gains.mask[slot],
                        &weight_l1_sum,
                        &weight_l2_squared_sum,
                    )?;
                    let mut routed = self.route_value(value, &BigUint::one(), &route_gain, false);
                    routed.provenance = value.provenance.clone();
                    let mask_source = refresh_mask_source(
                        declaration.commitment(),
                        slot,
                        component,
                        coefficient,
                        digit,
                    );
                    routed.provenance.insert(
                        mask_source,
                        NoiseSourceProvenance {
                            source: mask_source,
                            route: vec![slot, component, coefficient, digit],
                        },
                    );
                    masks[slot] = self.combine_route(masks[slot].clone(), routed);
                }
                RefreshPrfLabel::FreshError { digit, component, coefficient, .. } => {
                    if digit >= declaration.digits().fresh_count ||
                        component >= topology.component_columns ||
                        coefficient >= topology.coefficient_count
                    {
                        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
                    }
                    if channel_outputs.len() != declaration.slots().len() ||
                        channel_outputs.keys().copied().collect::<Vec<_>>() !=
                            (0..declaration.slots().len()).collect::<Vec<_>>()
                    {
                        return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                    }
                    for slot in 0..declaration.slots().len() {
                        let (wire, value) = channel_outputs
                            .get(&slot)
                            .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                        if *wire != expected_wire {
                            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                        }
                        refresh_nodes.push(self.refresh_binding_node(*wire));
                        // Apply the prepared label action once.  The PRF output
                        // is the unweighted label carrier; the rotation and
                        // unit-row component route preserve its norm, while
                        // the digit (and fresh H) contributes this scalar gain.
                        let slot_metadata = declaration
                            .slots()
                            .get(slot)
                            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                        let p = slot_metadata
                            .p
                            .to_biguint()
                            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
                        let qt = slot_metadata
                            .q_t
                            .to_biguint()
                            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
                        let label_gain = refresh_label_action_gain(
                            &BigUint::from(declaration.digits().base),
                            &label,
                            &p,
                            &qt,
                        )?;
                        let (weight_l1_sum, weight_l2_squared_sum) = refresh_label_weight_sums(
                            &BigUint::from(declaration.digits().base),
                            declaration.digits().fresh_count,
                            topology.component_columns,
                            topology.coefficient_count,
                            &p,
                            &qt,
                            true,
                        )?;
                        let route_gain = refresh_label_route_gain(
                            &label_gain,
                            &gains.fresh[slot],
                            &weight_l1_sum,
                            &weight_l2_squared_sum,
                        )?;
                        let mut routed =
                            self.route_value(value, &BigUint::one(), &route_gain, false);
                        let fresh_source = refresh_fresh_source(
                            declaration.commitment(),
                            component,
                            coefficient,
                            digit,
                        );
                        routed.provenance.insert(
                            fresh_source,
                            NoiseSourceProvenance {
                                source: fresh_source,
                                route: vec![slot, component, coefficient, digit],
                            },
                        );
                        fresh[slot] = self.combine_route(fresh[slot].clone(), routed);
                    }
                }
            }
        }

        // Canonical-positive digit routes carry one deterministic affine
        // offset per aggregate mask/fresh route.  Apply it after all label
        // actions so each offset is emitted exactly once; unlike a PRF label,
        // it contributes no fresh variance, provenance, or event count.
        for slot in 0..declaration.slots().len() {
            let slot_metadata = declaration
                .slots()
                .get(slot)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let p = slot_metadata
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let qt = slot_metadata
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let q_to_p = &global_modulus / &p;
            let q_to_qt = &global_modulus / &qt;
            let mask_offset = refresh_canonical_offset_numerator(
                &BigUint::from(declaration.digits().base),
                declaration.digits().mask_counts[slot],
                topology.component_columns,
                topology.coefficient_count,
                &gains.mask[slot],
                &p,
                &qt,
                &q_to_p,
                false,
            )?;
            refresh_add_deterministic_offset(&mut masks[slot], mask_offset, &BigUint::from(2u8));
            let fresh_offset = refresh_canonical_offset_numerator(
                &BigUint::from(declaration.digits().base),
                declaration.digits().fresh_count,
                topology.component_columns,
                topology.coefficient_count,
                &gains.fresh[slot],
                &p,
                &qt,
                &q_to_qt,
                true,
            )?;
            refresh_add_deterministic_offset(&mut fresh[slot], fresh_offset, &BigUint::from(2u8));
        }

        let mut raw = Self::zero_route_value();
        let plaintext_moduli = declaration
            .slots()
            .iter()
            .map(|slot| {
                slot.q_t
                    .to_biguint()
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let recomposition_modulus = plaintext_moduli.iter().product::<BigUint>();
        for (slot_index, slot) in declaration.slots().iter().enumerate() {
            let slot_input =
                inputs.get(&slot_index).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let p = slot
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let qt = slot
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let declared_q = slot
                .q
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            if declared_q != global_modulus {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            if p < qt || &p % &qt != BigUint::zero() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            if global_modulus < p || &global_modulus % &p != BigUint::zero() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            let ratio = &p / &qt;
            let crt_weight = declaration.crt_weight(slot_index)?;
            // C_sum is assembled in the full q coordinate.  State action
            // scaling is setup-owned; label actions were applied once above.
            let state = self.route_value(
                &slot_input.1,
                &BigUint::one(),
                &gains.state_scaling[slot_index],
                true,
            );
            refresh_nodes.push(self.node_for_wire(slot_input.0)?);
            let mask = masks[slot_index].clone();
            let fresh_slot = fresh[slot_index].clone();
            // A_star is a centered q_t carrier, not a p -> q_t ModDown.  The
            // decoder target nevertheless records its slot-local public route
            // alongside the A_sum label sources; state remains only in C_sum.
            let decoder_provenance = refresh_decoder_provenance(
                &[&mask, &fresh_slot],
                declaration.commitment(),
                slot_index,
            );
            let c_sum = self.combine_route(self.combine_route(state, mask), fresh_slot);
            let q_to_p = &global_modulus / &p;
            let c_sum = self.mod_down_channel(
                c_sum,
                &q_to_p,
                declaration.moment_policy(),
                declaration.commitment(),
                slot_index,
                b"q-to-p",
            )?;
            // The decoder is
            // s*ModDown_qp(A_sum-H*A_star)+e_dec.  route_bias_bounds is the
            // setup-owned deterministic q-ring bound for that complete public
            // target (including data_public=A_star*W and all PRF labels).
            // Apply q -> p to this target before the decoder action, with a
            // distinct rounding source from the raw C_sum switch.  The final
            // p -> q_t switch below is applied only after decoder subtraction.
            let decoder_gain = &gains.decoder[slot_index];
            let decoder_target = RnsBggNoiseValue {
                bound: RnsBggNoiseBound {
                    error: decoder_bounds[slot_index].clone(),
                    payload_l1: BigUint::zero(),
                },
                payload_l2_squared: BigUint::zero(),
                moments: Some(NoiseMoments {
                    mean_numerator: BigInt::zero(),
                    mean_denominator: BigUint::one(),
                    mean_radius: NonnegativeRational::new(
                        decoder_bounds[slot_index].clone(),
                        BigUint::one(),
                    )
                    .expect("positive decoder residual bound"),
                    variance: NonnegativeRational::zero(),
                }),
                provenance: decoder_provenance,
                heuristics: NoiseHeuristicLedger::new(),
                breakdown: None,
            };
            let decoder_target = self.mod_down_channel(
                decoder_target,
                &q_to_p,
                declaration.moment_policy(),
                declaration.commitment(),
                slot_index,
                b"decoder-q-to-p",
            )?;
            let decoder_target =
                self.route_value(&decoder_target, &BigUint::one(), decoder_gain, false);
            let decoder_source = stable_source_id(&(
                b"rns-bgg-refresh/decoder",
                declaration.commitment(),
                slot_index,
            ));
            let decoder_error_bound = self.parameters.encoding_error_bound() * &decoder_gain.l1;
            let decoder_error = RnsBggNoiseValue {
                bound: RnsBggNoiseBound {
                    error: decoder_error_bound.clone(),
                    payload_l1: BigUint::zero(),
                },
                payload_l2_squared: BigUint::zero(),
                moments: Some(NoiseMoments {
                    mean_numerator: BigInt::zero(),
                    mean_denominator: BigUint::one(),
                    mean_radius: NonnegativeRational::new(decoder_error_bound, BigUint::one())
                        .expect("positive decoder error bound"),
                    variance: self
                        .parameters
                        .encoding_error_variance()
                        .scaled(&decoder_gain.l2_squared),
                }),
                provenance: BTreeMap::from([(
                    decoder_source,
                    NoiseSourceProvenance { source: decoder_source, route: vec![slot_index] },
                )]),
                heuristics: NoiseHeuristicLedger::new(),
                breakdown: None,
            };
            let decoder = self.combine_route(decoder_target, decoder_error);
            refresh_nodes.push(self.refresh_binding_node(slot_input.0));
            let decoded = self.combine_route(c_sum, decoder);
            let channel = self.mod_down_channel(
                decoded,
                &ratio,
                declaration.moment_policy(),
                declaration.commitment(),
                slot_index,
                b"p-to-qt",
            )?;
            let channel =
                self.route_value(&channel, &crt_weight, &RightActionNoiseGain::unit(), true);
            raw = self.combine_route(raw, channel);
        }

        let target_modulus = self.parameters.level(declaration.target_level())?.modulus();
        if recomposition_modulus != *target_modulus.as_ref() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        let mut refreshed = raw.clone();
        refreshed.bound.payload_l1 = source_payload.bound.payload_l1;
        refreshed.payload_l2_squared = source_payload.payload_l2_squared;
        let dependency = refresh_nodes
            .into_iter()
            .reduce(|lhs, rhs| self.refresh_add_node(source_wire, lhs, rhs))
            .ok_or(RnsBggError::Refresh(RefreshError::GraphLink(
                "refresh dependency graph is empty",
            )))?;
        self.trace.dependencies.wires.insert(source_wire, dependency);
        self.trace.wires.insert(source_wire, refreshed.clone());
        let refresh_events = BigUint::from(
            topology.joint_event_count().map_err(|error| RnsBggError::Graph(error.to_string()))?,
        );
        self.trace.node_event_counts.insert(dependency.0, refresh_events.clone());
        self.trace.joint_event_count += refresh_events;
        self.trace.refreshed_target_level = Some(declaration.target_level());
        self.trace.refreshed_declaration_commitment = Some(*declaration.commitment());
        self.trace.refreshed_target_modulus = Some(target_modulus.as_ref().clone());
        self.trace.refreshed_output_schema = Some(declaration.output_schema().to_vec());
        self.trace.refreshed_output = Some(refreshed.clone());
        self.trace.refreshed_moment_policy = Some(declaration.moment_policy().clone());
        Ok(refreshed)
    }
}

/// Noise-side implementation of the common one-shot refresh protocol.
///
/// PRF results are symbolic graph carriers, not sampled plaintexts. Every
/// mask label carries one canonical slot channel; every fresh-error label
/// carries all CRT channels. The shared fresh-error source is retained by
/// combining carriers according to their provenance instead of treating each
/// label or slot as an independent term.
impl NoiseRefresh for super::super::RnsBggRefresh<'_, RnsBggNoiseEvaluator<'_>> {
    type PrfOutput = BTreeMap<usize, (WireId, RnsBggNoiseValue)>;
    type Output = (WireId, RnsBggNoiseValue);
    type Error = RnsBggError;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error> {
        let inputs = self.inputs().clone();
        let evaluator = self.lowerer();
        let value = evaluator.refresh_raw_target(&inputs, declaration, &outputs)?;
        Ok((declaration.source_output(), value))
    }
}

#[cfg(test)]
mod test {
    use super::{super::test::value, *};
    use std::collections::BTreeSet;

    #[test]
    fn equal_modulus_finalization_adds_no_rounding_noise() {
        assert_eq!(
            refresh_channel_mod_down_bound(&BigUint::from(17u8), &BigUint::one()).unwrap(),
            BigUint::from(17u8)
        );
    }

    #[test]
    fn refresh_fresh_routes_share_one_provenance_source() {
        let commitment = [17u8; 32];
        let source = refresh_fresh_source(&commitment, 0, 0, 0);
        let mut left = value(&[1]);
        left.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source,
            route: vec![0, 0],
        });
        let mut right = value(&[1]);
        right.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source,
            route: vec![1, 0],
        });
        let combined = left.combine(&right, NoiseCorrelation::SharedSource);
        assert!(combined.provenance.contains_key(&source));
        assert_eq!(combined.provenance[&source].route.len(), 4);
    }

    #[test]
    fn refresh_multiple_labels_keep_slot_shared_sources_and_covariance() {
        let commitment = [23u8; 32];
        let first = refresh_fresh_source(&commitment, 0, 0, 0);
        let second = refresh_fresh_source(&commitment, 0, 0, 1);
        let moments = NoiseMoments {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
        };
        let fresh = |source, slot| {
            let mut value = value(&[1]);
            value.moments = Some(moments.clone());
            value.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
                source,
                route: vec![slot],
            });
            value
        };

        // Each fresh label is one sample shared by every CRT slot. Distinct
        // labels remain independent, even after the two labels are merged.
        let merged = fresh(first, 0)
            .combine(&fresh(first, 1), NoiseCorrelation::Independent)
            .combine(&fresh(second, 0), NoiseCorrelation::Independent)
            .combine(&fresh(second, 1), NoiseCorrelation::Independent);
        assert_eq!(merged.provenance.len(), 2);
        assert_eq!(merged.provenance[&first].route, vec![0, 1]);
        assert_eq!(merged.provenance[&second].route, vec![0, 1]);
        // The source map retains both shared identities.  Once a partial
        // aggregate contains two sources, the generic moment enclosure uses
        // the documented Cauchy covariance bound for the shared component.
        assert_eq!(
            merged.moments.unwrap().variance,
            NonnegativeRational::new(12u8.into(), 1u8.into()).unwrap()
        );
    }

    #[test]
    fn refresh_fresh_source_is_domain_bound_to_declaration() {
        assert_ne!(
            refresh_fresh_source(&[1u8; 32], 0, 0, 0),
            refresh_fresh_source(&[2u8; 32], 0, 0, 0)
        );
        assert_eq!(
            refresh_fresh_source(&[1u8; 32], 0, 0, 0),
            refresh_fresh_source(&[1u8; 32], 0, 0, 0)
        );
        assert_ne!(
            refresh_fresh_source(&[1u8; 32], 0, 0, 0),
            refresh_fresh_source(&[1u8; 32], 1, 0, 0)
        );
        assert_ne!(
            refresh_fresh_source(&[1u8; 32], 0, 0, 0),
            refresh_fresh_source(&[1u8; 32], 0, 1, 0)
        );
    }

    #[test]
    fn refresh_channel_oracle_switches_before_crt_idempotent() {
        // Distinct prime plaintext slots use independent idempotents in Q*=15.
        let slots = [3u8, 5]
            .into_iter()
            .enumerate()
            .map(|(slot, q_t)| crate::noise_refresh::refresh::RefreshSlot {
                slot,
                q: 15.into(),
                p: 15.into(),
                q_t: q_t.into(),
            })
            .collect::<Vec<_>>();
        let weight = |slot| crate::noise_refresh::refresh::refresh_crt_weight(&slots, slot);
        assert_eq!(weight(0).unwrap(), BigUint::from(10u8));
        assert_eq!(weight(1).unwrap(), BigUint::from(6u8));

        // A p=15 to q_t=5 switch first divides inherited error and adds one
        // centered residue unit; CRT weighting happens afterwards.
        assert_eq!(
            refresh_channel_mod_down_bound(&BigUint::from(17u8), &BigUint::from(3u8)).unwrap(),
            BigUint::from(7u8)
        );
        assert_eq!(
            &refresh_channel_mod_down_bound(&BigUint::from(17u8), &BigUint::from(3u8)).unwrap() *
                6u8,
            BigUint::from(42u8)
        );
        let after_q_to_p =
            refresh_channel_mod_down_bound(&BigUint::from(100u8), &BigUint::from(7u8)).unwrap();
        let after_p_to_qt =
            refresh_channel_mod_down_bound(&after_q_to_p, &BigUint::from(3u8)).unwrap();
        assert_eq!(after_q_to_p, BigUint::from(16u8));
        assert_eq!(after_p_to_qt, BigUint::from(7u8));
        assert_ne!(
            stable_source_id(&(b"rns-bgg-refresh/channel-mod-down", b"q-to-p", [9u8; 32], 0)),
            stable_source_id(&(b"rns-bgg-refresh/channel-mod-down", b"p-to-qt", [9u8; 32], 0))
        );
        assert_ne!(
            stable_source_id(&(b"rns-bgg-refresh/channel-mod-down", b"q-to-p", [9u8; 32], 0,)),
            stable_source_id(&(
                b"rns-bgg-refresh/channel-mod-down",
                b"decoder-q-to-p",
                [9u8; 32],
                0,
            ))
        );
    }

    #[test]
    fn refresh_label_action_gain_tracks_digit_and_fresh_h_only() {
        let mask = RefreshPrfLabel::Mask {
            refresh_id: [3u8; 32],
            slot: 0,
            component: 0,
            coefficient: 0,
            digit: 2,
        };
        let fresh = RefreshPrfLabel::FreshError {
            refresh_id: [3u8; 32],
            component: 1,
            coefficient: 7,
            digit: 2,
        };
        let mask_gain = refresh_label_action_gain(
            &BigUint::from(4u8),
            &mask,
            &BigUint::from(15u8),
            &BigUint::from(5u8),
        )
        .unwrap();
        let fresh_gain = refresh_label_action_gain(
            &BigUint::from(4u8),
            &fresh,
            &BigUint::from(15u8),
            &BigUint::from(5u8),
        )
        .unwrap();
        // X^coefficient and unit_row(component) preserve the norm; only the
        // radix digit and fresh H change the scalar action.
        assert_eq!(mask_gain.l1, BigUint::from(16u8));
        assert_eq!(mask_gain.l2_squared, BigUint::from(256u16));
        assert_eq!(fresh_gain.l1, BigUint::from(48u8));
        assert_eq!(fresh_gain.l2_squared, BigUint::from(2304u16));
    }

    #[test]
    fn refresh_label_route_gain_splits_aggregate_once() {
        let label = RefreshPrfLabel::Mask {
            refresh_id: [4u8; 32],
            slot: 0,
            component: 1,
            coefficient: 2,
            digit: 1,
        };
        let label_gain = refresh_label_action_gain(
            &BigUint::from(2u8),
            &label,
            &BigUint::from(15u8),
            &BigUint::from(5u8),
        )
        .unwrap();
        let (sum_l1, sum_l2) = refresh_label_weight_sums(
            &BigUint::from(2u8),
            2,
            2,
            3,
            &BigUint::from(15u8),
            &BigUint::from(5u8),
            false,
        )
        .unwrap();
        let route = refresh_label_route_gain(
            &label_gain,
            &RightActionNoiseGain { l1: BigUint::from(36u8), l2_squared: BigUint::from(60u8) },
            &sum_l1,
            &sum_l2,
        )
        .unwrap();
        // Aggregate decomposition gain is 2 per unit label weight.  The
        // digit weight is applied once, not once before and once after split.
        assert_eq!(sum_l1, BigUint::from(18u8));
        assert_eq!(sum_l2, BigUint::from(30u8));
        assert_eq!(route.l1, BigUint::from(4u8));
        assert_eq!(route.l2_squared, BigUint::from(8u8));
    }

    #[test]
    fn refresh_affine_offsets_match_canonical_positive_supports() {
        for base in [2u8, 4, 8] {
            for digits in 1..=2usize {
                let base = BigUint::from(base);
                let support = base.pow(digits as u32);
                let digit_sum = (&support - 1u8) / (&base - 1u8);
                let multiplicity = BigUint::from(2u8);
                let mask_weights = &digit_sum * &multiplicity;
                let fresh_weights = &digit_sum * 5u8 * &multiplicity;
                let aggregate_mask = RightActionNoiseGain {
                    l1: mask_weights.clone(),
                    l2_squared: mask_weights.pow(2u32),
                };
                let aggregate_fresh = RightActionNoiseGain {
                    l1: fresh_weights.clone(),
                    l2_squared: fresh_weights.pow(2u32),
                };
                let mask_offset = refresh_canonical_offset_numerator(
                    &base,
                    digits,
                    1,
                    2,
                    &aggregate_mask,
                    &BigUint::from(5u8),
                    &BigUint::one(),
                    &BigUint::from(3u8),
                    false,
                )
                .unwrap();
                let fresh_offset = refresh_canonical_offset_numerator(
                    &base,
                    digits,
                    1,
                    2,
                    &aggregate_fresh,
                    &BigUint::from(5u8),
                    &BigUint::one(),
                    &BigUint::from(15u8),
                    true,
                )
                .unwrap();
                let common_mask_offset = -BigInt::from(&digit_sum + &support / 2u8);
                let common_fresh_offset =
                    -BigInt::from(&base / 2u8) * BigInt::from(digit_sum.clone());
                assert_eq!(mask_offset, common_mask_offset * BigInt::from(&multiplicity * 3u8));
                assert_eq!(fresh_offset, common_fresh_offset * BigInt::from(&multiplicity * 15u8));

                let mut mask = value(&[0]);
                mask.bound.error = &support - 1u8;
                mask.moments =
                    Some(NoiseMoments::uniform_interval(BigInt::zero(), support.clone()));
                let mask_source = stable_source_id(&(b"offset-mask", base.clone(), digits));
                mask.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
                    source: mask_source,
                    route: vec![digits],
                });
                let mask_variance = mask.moments.as_ref().unwrap().variance.clone();
                refresh_add_deterministic_offset(
                    &mut mask,
                    mask_offset.clone(),
                    &BigUint::from(2u8),
                );
                assert_eq!(
                    mask.bound.error,
                    &support - 1u8 +
                        ceil_div_nonnegative(mask_offset.magnitude(), &BigUint::from(2u8))
                );
                assert_eq!(mask.moments.as_ref().unwrap().variance, mask_variance);
                assert_eq!(mask.provenance.len(), 1);
                let mut mask_mean_numerator = BigInt::from(&support - 1u8) + &mask_offset;
                let mask_mean_denominator = BigUint::from(2u8);
                let mask_divisor = gcd(mask_mean_numerator.magnitude(), &mask_mean_denominator);
                mask_mean_numerator /= BigInt::from(mask_divisor.clone());
                assert_eq!(mask.moments.as_ref().unwrap().mean_numerator, mask_mean_numerator);
                assert_eq!(
                    mask.moments.as_ref().unwrap().mean_denominator,
                    mask_mean_denominator / mask_divisor
                );

                let mut fresh = value(&[0]);
                fresh.moments =
                    Some(NoiseMoments::uniform_interval(BigInt::zero(), support.clone()));
                let fresh_source = stable_source_id(&(b"offset-fresh", base, digits));
                fresh.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
                    source: fresh_source,
                    route: vec![digits],
                });
                let fresh_variance = fresh.moments.as_ref().unwrap().variance.clone();
                refresh_add_deterministic_offset(
                    &mut fresh,
                    fresh_offset.clone(),
                    &BigUint::from(2u8),
                );
                assert_eq!(fresh.moments.as_ref().unwrap().variance, fresh_variance);
                assert_eq!(fresh.provenance.len(), 1);
                let fresh_mean_numerator = BigInt::from(&support - 1u8) + &fresh_offset;
                let fresh_mean_denominator = BigUint::from(2u8);
                let fresh_divisor = gcd(fresh_mean_numerator.magnitude(), &fresh_mean_denominator);
                assert_eq!(
                    fresh.moments.as_ref().unwrap().mean_numerator,
                    fresh_mean_numerator / BigInt::from(fresh_divisor.clone())
                );
                assert_eq!(
                    fresh.moments.as_ref().unwrap().mean_denominator,
                    fresh_mean_denominator / fresh_divisor
                );
            }
        }
    }

    #[test]
    fn refresh_fresh_affine_offsets_keep_shared_sources_across_slots() {
        let source = refresh_fresh_source(&[12u8; 32], 0, 0, 0);
        let mut first = value(&[0]);
        first.moments = Some(NoiseMoments::uniform_interval(BigInt::zero(), 8u8.into()));
        first.provenance =
            RnsBggNoiseValue::single_provenance(NoiseSourceProvenance { source, route: vec![0] });
        let mut second = first.clone();
        second.provenance =
            RnsBggNoiseValue::single_provenance(NoiseSourceProvenance { source, route: vec![1] });
        refresh_add_deterministic_offset(&mut first, 12u8.into(), &2u8.into());
        refresh_add_deterministic_offset(&mut second, 20u8.into(), &2u8.into());
        let first_variance = first.moments.as_ref().unwrap().variance.clone();
        let second_variance = second.moments.as_ref().unwrap().variance.clone();
        assert_eq!(
            first.provenance.keys().collect::<Vec<_>>(),
            second.provenance.keys().collect::<Vec<_>>()
        );
        assert_eq!(first_variance, second_variance);
        assert_ne!(
            first.moments.as_ref().unwrap().mean_numerator,
            second.moments.as_ref().unwrap().mean_numerator
        );
    }

    #[test]
    fn refresh_decoder_provenance_keeps_c_sum_sources_shared() {
        let mut state = value(&[0]);
        let state_source = stable_source_id(&(b"state", 0usize));
        state.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source: state_source,
            route: vec![0],
        });
        let mut fresh = value(&[0]);
        let fresh_source = stable_source_id(&(b"fresh", 0usize));
        fresh.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source: fresh_source,
            route: vec![0, 1],
        });
        let commitment = [8u8; 32];
        let decoder = refresh_decoder_provenance(&[&fresh], &commitment, 1);
        // The state argument is intentionally not part of A_sum-H*A_star.
        assert!(!decoder.contains_key(&state_source));
        assert_eq!(decoder[&fresh_source].route, vec![0, 1, 1]);
        assert_eq!(decoder[&refresh_astar_source(&commitment, 1)].route, vec![1]);
    }

    #[test]
    fn refresh_decoder_provenance_has_no_anchor_moddown_source() {
        let commitment = [9u8; 32];
        let mask = value(&[0]);
        let decoder = refresh_decoder_provenance(&[&mask], &commitment, 0);
        let anchor = refresh_astar_source(&commitment, 0);
        let final_switch = stable_source_id(&(
            b"rns-bgg-refresh/channel-mod-down",
            b"p-to-qt",
            commitment,
            0usize,
        ));
        assert!(decoder.contains_key(&anchor));
        assert_ne!(anchor, final_switch);
    }

    #[test]
    fn refresh_decoder_target_preserves_a_sum_covariance_only() {
        let source = refresh_mask_source(&[10u8; 32], 0, 0, 0, 0);
        let mut c_sum = value(&[0]);
        c_sum.moments = Some(NoiseMoments {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
        });
        c_sum.provenance =
            RnsBggNoiseValue::single_provenance(NoiseSourceProvenance { source, route: vec![0] });
        let mut decoder = c_sum.clone();
        decoder.provenance = refresh_decoder_provenance(&[&c_sum], &[11u8; 32], 0);
        let shared = c_sum.combine(&decoder, NoiseCorrelation::Independent);

        // The common A_sum source forces the conservative shared-source
        // covariance envelope even when the caller's nominal classification
        // says independent.  A state-only source is not present in decoder.
        assert_eq!(
            shared.moments.unwrap().variance,
            NonnegativeRational::new(4u8.into(), 1u8.into()).unwrap()
        );
        assert!(!decoder.provenance.contains_key(&stable_source_id(&(b"state", 0usize))));
    }

    #[test]
    fn refresh_label_sources_distinguish_component_coefficient_and_digit() {
        let commitment = [7u8; 32];
        let sources = [
            refresh_mask_source(&commitment, 0, 0, 0, 0),
            refresh_mask_source(&commitment, 1, 0, 0, 0),
            refresh_mask_source(&commitment, 0, 1, 0, 0),
            refresh_mask_source(&commitment, 0, 0, 1, 0),
            refresh_mask_source(&commitment, 0, 0, 0, 1),
            refresh_fresh_source(&commitment, 0, 0, 0),
            refresh_fresh_source(&commitment, 1, 0, 0),
            refresh_fresh_source(&commitment, 0, 1, 0),
            refresh_fresh_source(&commitment, 0, 0, 1),
        ];
        assert_eq!(sources.iter().collect::<BTreeSet<_>>().len(), sources.len());
    }
}
