//! The symbolic noise evaluator: per-operation lowering, families, and its
//! trace bookkeeping.
use super::*;

/// Trace one noise operation's output, for diagnosing budget failures.
fn trace_noise_value(
    wire: WireId,
    level: Option<usize>,
    node: &crate::noise_utils::NoiseDependencyNode,
    value: &RnsBggNoiseValue,
) {
    if tracing::enabled!(tracing::Level::TRACE) {
        let kind = format!("{node:?}");
        tracing::trace!(
            wire = wire.0,
            level,
            operation = kind.split(|c: char| !c.is_alphanumeric()).next().unwrap_or(""),
            variance_log2 = value.moments.as_ref().map(|m| variance_log2(&m.variance)),
            error_bits = value.bound.error.bits(),
            "noise operation"
        );
    }
}

/// The canonical lane index of a family over `range` whose body runs at
/// lane depth `depth` (0 for a top-level family): its own loop symbol, so
/// the lanes of nested families stay distinct in index expressions.
fn family_lane(range: &FamilyRange, depth: usize) -> IntExpr {
    let symbol = IntExpr::LoopIndex(u32::try_from(depth).expect("family depth fits u32"));
    IntExpr::Add(Box::new(range.start().clone()), Box::new(symbol)).canonicalize()
}

/// Whether `index` reads, across the lanes `0..count` of the innermost lane
/// `lane` (symbol `LoopIndex(depth)`), members of one aligned row of length
/// `row`: `index - lane = row * f(outer lanes)` with `f` affine and
/// `count <= row`. The check is exact for the affine index expressions it
/// accepts: every multiplication has a constant factor, so the difference is
/// determined by its values at the origin and the unit points.
fn row_injective(
    index: &IntExpr,
    lane: &IntExpr,
    depth: usize,
    count: &BigUint,
    row: &BigUint,
) -> bool {
    fn affine(expression: &IntExpr) -> bool {
        match expression {
            IntExpr::Const(_) | IntExpr::LoopIndex(_) => true,
            IntExpr::Add(lhs, rhs) | IntExpr::Sub(lhs, rhs) => affine(lhs) && affine(rhs),
            IntExpr::Mul(lhs, rhs) => {
                (lhs.evaluate(&ParamEnv::default()).is_ok() && affine(rhs)) ||
                    (rhs.evaluate(&ParamEnv::default()).is_ok() && affine(lhs))
            }
            _ => false,
        }
    }
    fn symbols(expression: &IntExpr, into: &mut std::collections::BTreeSet<u32>) {
        match expression {
            IntExpr::LoopIndex(slot) => {
                into.insert(*slot);
            }
            IntExpr::Add(lhs, rhs) | IntExpr::Sub(lhs, rhs) | IntExpr::Mul(lhs, rhs) => {
                symbols(lhs, into);
                symbols(rhs, into);
            }
            _ => {}
        }
    }
    if count > row || row.is_zero() {
        return false;
    }
    let difference = IntExpr::Sub(Box::new(index.clone()), Box::new(lane.clone())).canonicalize();
    if !affine(&difference) {
        return false;
    }
    let inner = u32::try_from(depth).expect("family depth fits u32");
    let mut slots = std::collections::BTreeSet::new();
    symbols(&difference, &mut slots);
    slots.insert(inner);
    let at = |point: &BTreeMap<u32, BigInt>| {
        difference.evaluate(&ParamEnv { loop_indices: point.clone(), ..ParamEnv::default() }).ok()
    };
    let origin = slots.iter().map(|slot| (*slot, BigInt::zero())).collect::<BTreeMap<_, _>>();
    let row = BigInt::from(row.clone());
    let Some(base) = at(&origin) else { return false };
    if !(&base % &row).is_zero() {
        return false;
    }
    slots.iter().all(|slot| {
        let mut point = origin.clone();
        point.insert(*slot, BigInt::one());
        at(&point).is_some_and(|value| {
            let step = value - &base;
            if *slot == inner { step.is_zero() } else { (&step % &row).is_zero() }
        })
    })
}

/// Whether distinct lanes of a family read distinct members through
/// `index`: the lane index itself (up to a constant offset), or a selection
/// by the lane among pairwise distinct constants.
pub(super) fn lane_injective(index: &IntExpr, lane: &IntExpr) -> bool {
    let lane = lane.clone().canonicalize();
    if let IntExpr::Select { selector, branches } = index {
        let mut seen = std::collections::BTreeSet::new();
        return selector.clone().canonicalize() == lane &&
            branches.iter().all(|branch| {
                branch.evaluate(&ParamEnv::default()).is_ok_and(|value| seen.insert(value))
            });
    }
    IntExpr::Sub(Box::new(index.clone()), Box::new(lane))
        .canonicalize()
        .evaluate(&ParamEnv::default())
        .is_ok()
}

impl<'a> RnsBggNoiseEvaluator<'a> {
    pub(super) fn new(
        parameters: &'a RnsBggParameters,
        program: &'a ExponentLutProgram,
        workload: &'a RnsBggDerivedWorkload,
        bindings: &'a RnsBggNoiseBindings,
    ) -> Self {
        Self {
            parameters,
            program,
            workload,
            bindings,
            trace: RnsBggNoiseTrace::default(),
            current_site: None,
            event_multiplier: BigUint::one(),
            finalized_outputs: BTreeMap::new(),
            decoder_smudge: None,
            unmask_trapdoor_sigma: None,
            lane_indices: Vec::new(),
            lane_counts: Vec::new(),
            payload_support: BTreeMap::new(),
            reduction_counts: ReductionCounts::default(),
        }
    }

    /// Sparse payload support of `wire` in the innermost enclosing family.
    fn lane_support(&self, wire: WireId) -> Option<&BigUint> {
        self.payload_support
            .get(&wire)
            .filter(|support| {
                support.depth == self.lane_indices.len() && support.row_length.is_none()
            })
            .map(|support| &support.count)
    }

    pub(super) fn set_lane_support(&mut self, wire: WireId, count: BigUint) {
        self.payload_support.insert(
            wire,
            PayloadSupport { depth: self.lane_indices.len(), count, row_length: None },
        );
    }

    /// The row contract `(count, row length)` of `wire` in the innermost
    /// enclosing family, when its lanes are row-wise sparse.
    fn lane_row_support(&self, wire: WireId) -> Option<(&BigUint, &BigUint)> {
        self.payload_support
            .get(&wire)
            .filter(|support| support.depth == self.lane_indices.len())
            .and_then(|support| support.row_length.as_ref().map(|row| (&support.count, row)))
    }

    /// Record that a member of the family `wire` with row contract `(count,
    /// row)` read at `index` has at most `count` nonzero payloads among the
    /// innermost lanes when those lanes stay inside one row, and drop any
    /// earlier record otherwise.
    pub(super) fn observe_family_read(
        &mut self,
        wire: WireId,
        index: &IntExpr,
        contract: Option<(&BigUint, &BigUint)>,
    ) {
        let depth = self.lane_indices.len();
        let supported = match (contract, self.lane_indices.last(), self.lane_counts.last()) {
            (Some((count, row)), Some(lane), Some(lanes)) => {
                row_injective(index, lane, depth - 1, lanes, row).then(|| count.clone())
            }
            _ => None,
        };
        match supported {
            Some(count) => self.set_lane_support(wire, count),
            None => {
                self.payload_support.remove(&wire);
            }
        }
    }

    pub(super) fn site(&self) -> Result<&crate::program::LoweringSite, RnsBggError> {
        self.current_site
            .as_ref()
            .ok_or(RnsBggError::Graph("noise lowering site is missing".into()))
    }

    pub(super) fn level(&self, wire: WireId) -> Result<usize, RnsBggError> {
        self.workload.level(wire)
    }

    pub(super) fn carry_provenance(
        &self,
        mut value: RnsBggNoiseValue,
        parents: impl IntoIterator<Item = BTreeMap<[u8; 32], NoiseSourceProvenance>>,
    ) -> RnsBggNoiseValue {
        let suffix = self
            .current_site
            .as_ref()
            .map(|site| {
                site.indices
                    .iter()
                    .filter_map(|index| index.evaluate(&ParamEnv::default()).ok())
                    .filter_map(|index| index.to_usize())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut provenance: BTreeMap<[u8; 32], NoiseSourceProvenance> = BTreeMap::new();
        for parent in parents {
            for (source, mut identity) in parent {
                // A source already present keeps its first route (see
                // `RnsBggNoiseValue::combine`).
                provenance.entry(source).or_insert_with(|| {
                    identity.route.extend_from_slice(&suffix);
                    identity
                });
            }
        }
        value.provenance = provenance;
        value
    }

    fn add_events(&mut self, count: &BigUint) {
        self.trace.joint_event_count += count * &self.event_multiplier;
    }

    pub(super) fn node_for_wire(
        &self,
        wire: WireId,
    ) -> Result<crate::noise_utils::NoiseNodeId, RnsBggError> {
        self.trace.dependencies.wires.get(&wire).copied().ok_or_else(|| {
            RnsBggError::Graph(format!("noise dependency parent is missing for wire {wire:?}"))
        })
    }

    pub(super) fn rhs_node(&mut self, rhs: &RhsRef<IntExpr>) -> crate::noise_utils::NoiseNodeId {
        let node = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        let site = self.current_site.as_ref();
        let mut indices = site.map(|site| site.indices.clone()).unwrap_or_default();
        indices.push(rhs.index.clone().unwrap_or_else(|| IntExpr::constant(i64::from(rhs.id.0))));
        self.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Binding(
            crate::noise_utils::NoiseBinding::RhsSite {
                gate: site.map(|site| site.output),
                indices,
            },
        ));
        node
    }

    /// Derive the material source set for one LUT from the validated Program
    /// contract.  A/F are automorphism helpers shared by evaluation level and
    /// sigma across every LUT, layout, and site.  B is LUT-specific and also
    /// commits to the evaluation/decomposition levels, its explicit input
    /// layout, and sigma.  No site, family scope, input wire, or program
    /// identity is part of these keys.
    /// The provenance of a Static LUT's material at `evaluation_level`: per
    /// automorphism, the public matrix and helper (`A`, `F`) and, if
    /// `switching` (a private payload), the switching ciphertext (`B`).
    fn lut_material_provenance(
        &self,
        evaluation_level: usize,
        lut: LutId,
        table: &LutTable,
        switching: bool,
    ) -> Result<BTreeMap<[u8; 32], NoiseSourceProvenance>, RnsBggError> {
        let decomposition_level = evaluation_level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("LUT decomposition level overflow"))?;
        self.parameters.level(decomposition_level)?;
        let ring_dimension = self.parameters.level(evaluation_level)?.ring_dimension() as usize;
        let sigmas = table.branch_automorphisms(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let mut sources = BTreeMap::new();
        for sigma in sigmas {
            let a = stable_source_id(&(b"rns-bgg/lut/A", evaluation_level, sigma));
            sources.entry(a).or_insert_with(|| NoiseSourceProvenance {
                source: a,
                route: vec![evaluation_level, sigma],
            });
            let f = stable_source_id(&(b"rns-bgg/lut/F", evaluation_level, sigma));
            sources.entry(f).or_insert_with(|| NoiseSourceProvenance {
                source: f,
                route: vec![evaluation_level, sigma],
            });
            if !switching {
                continue;
            }
            let b = stable_source_id(&(
                b"rns-bgg/lut/B",
                evaluation_level,
                decomposition_level,
                lut,
                table.input_layout(),
                sigma,
            ));
            sources.entry(b).or_insert_with(|| NoiseSourceProvenance {
                source: b,
                route: vec![evaluation_level, decomposition_level, sigma],
            });
        }
        Ok(sources)
    }

    /// Bound the fixed public action used to align one source ring with the
    /// target ring.  The action is setup material, so this is a deterministic
    /// norm envelope rather than a newly sampled noise source.
    ///
    /// `H = Dec_(G_source)(alpha * G_target)` decomposes scalar multiples of
    /// the gadget, so its entries are constant polynomials: an output
    /// coefficient of `e * H` combines the same coefficient of the `D_s`
    /// source columns, without a ring convolution. Each output column has at
    /// most `D_s` digits bounded by `B / 2`; counting `2 D_s` of them keeps a
    /// factor-two margin, giving `l1 = D_s B` and energy `D_s B^2 / 2`.
    fn public_alignment_action_gain(
        &self,
        source_level: usize,
        target_level: usize,
    ) -> Result<RightActionNoiseGain, RnsBggError> {
        let alignment = self.parameters.public_alignment(source_level, target_level)?;
        if alignment.is_noop() {
            return Ok(RightActionNoiseGain::unit());
        }
        let source = self.parameters.level(source_level)?;
        let base = BigUint::one() << source.base_bits();
        let digits = BigUint::from(source.modulus_digits());
        let l1 = &digits * &base;
        Ok(RightActionNoiseGain { l2_squared: digits * base.pow(2u32) / 2u8, l1 })
    }

    /// Construct one shared rounding carrier for a source/key/target route.
    /// The identity intentionally excludes the lowering site: repeated uses
    /// of the same conversion share the same physical rounding sample.
    fn public_alignment_rounding(
        &self,
        source_level: usize,
        key_level: usize,
        target_level: usize,
        alpha: &BigUint,
        role: &'static [u8],
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        if alpha.is_zero() {
            return Err(RnsBggError::Parameters("invalid public alignment ratio"));
        }
        let source = stable_source_id(&(
            b"rns-bgg/public-alignment/rounding".as_slice(),
            role,
            source_level,
            key_level,
            target_level,
        ));
        let moments = NoiseMoments::discrete_rounding_residue(alpha)
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let mut heuristics = NoiseHeuristicLedger::new();
        heuristics.insert(NoiseHeuristicId::ExactDistribution);
        let error = if *alpha == BigUint::one() { BigUint::zero() } else { BigUint::one() };
        Ok(RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error, payload_l1: BigUint::zero() },
            payload_l2_squared: BigUint::zero(),
            moments: Some(moments),
            provenance: BTreeMap::from([(
                source,
                NoiseSourceProvenance {
                    source,
                    route: vec![source_level, key_level, target_level],
                },
            )]),
            heuristics,
            breakdown: None,
        })
    }

    /// Apply the exact public alignment selected by the parameter contract.
    /// The alpha-one route returns the original carrier before constructing
    /// any action, rounding source, or provenance entry.
    fn align_public_operand(
        &self,
        value: &RnsBggNoiseValue,
        source_level: usize,
        target_level: usize,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        let alignment = self.parameters.public_alignment(source_level, target_level)?;
        if alignment.is_noop() {
            return Ok(value.clone());
        }
        let alpha = alignment.ratio().clone();
        let action = self.public_alignment_action_gain(source_level, target_level)?;
        let ciphertext_rounding = self.public_alignment_rounding(
            source_level,
            source_level,
            target_level,
            &alpha,
            b"ciphertext",
        )?;
        let public_key_rounding = self.public_alignment_rounding(
            source_level,
            source_level,
            target_level,
            &alpha,
            b"key",
        )?;
        let ring_dimension = BigUint::from(self.parameters.level(target_level)?.ring_dimension());
        // The public alignment error contains `s * rho_A`, where `s` is the
        // two-component mask row `(s_bar, 1)` with uniform ternary `s_bar`
        // (`sample_secret_row`). Account for both components explicitly;
        // omitting the second row underestimates both hard bounds and moments.
        // The hard bound uses `||s_bar||_1 <= N`, the moments the exact
        // second moment `E||s_bar||_2^2 = 2N/3`; the constant `1` is exact.
        let mask_secret = [
            RnsBggMaskSecretComponent {
                coefficient_l1: ring_dimension.clone(),
                coefficient_l2_squared: NonnegativeRational::new(
                    ring_dimension * 2u8,
                    BigUint::from(3u8),
                )
                .expect("positive ternary second-moment denominator"),
            },
            RnsBggMaskSecretComponent::deterministic(BigUint::one(), BigUint::one()),
        ];
        let policy = NoiseMomentPolicy::Discrete { alpha: alpha.clone() };
        let aligned = value.align_public(
            self.parameters,
            &action,
            &alpha,
            &ciphertext_rounding,
            &public_key_rounding,
            &mask_secret,
            &policy,
        )?;
        // The source contributions keep their identities through the fixed
        // action h/alpha. Under H2 the rounding residue of the aligned
        // ciphertext is a centered uniform per coefficient, independent of
        // the error it rounds: the residue of one freshly sampled ciphertext
        // is its own object, with that ciphertext's member coordinates, and
        // any other ciphertext's residue keeps the route's shared identity.
        // The mask term s * rho_A convolves the key residue with the
        // independent secret; its coefficients are uncorrelated, and every
        // ciphertext aligned along this route shares it.
        let breakdown = match value.contributions("alignment-source") {
            Some(terms) => {
                let policy_rounding = policy
                    .rounding_moments()
                    .map_err(|_| RnsBggError::Parameters("invalid alignment moment policy"))?;
                let zero = zero_mean_moments(NonnegativeRational::zero());
                let mask = self.parameters.public_alignment_noise_moments(
                    &zero,
                    &action,
                    &alpha,
                    &zero,
                    public_key_rounding.moments.as_ref().unwrap_or(&policy_rounding),
                    &mask_secret,
                )?;
                let residue = ciphertext_rounding.moments.clone().unwrap_or(policy_rounding);
                let route = (source_level, target_level);
                let ciphertext_residual = match terms.single_sample() {
                    Some((sample_source, sample)) => NoiseBreakdown::object_residual(
                        stable_source_id(&(
                            b"rns-bgg/public-alignment/ciphertext-residue".as_slice(),
                            sample_source,
                            route,
                        )),
                        "alignment-ciphertext-residue",
                        sample.to_vec(),
                        residue,
                        true,
                    ),
                    None => NoiseBreakdown::object_residual(
                        stable_source_id(&(
                            b"rns-bgg/public-alignment/ciphertext-residue".as_slice(),
                            route,
                        )),
                        "alignment-ciphertext-residue",
                        Vec::new(),
                        residue,
                        true,
                    ),
                };
                let mask = NoiseBreakdown::object_residual(
                    stable_source_id(&(b"rns-bgg/public-alignment/mask".as_slice(), route)),
                    "alignment-mask",
                    Vec::new(),
                    mask,
                    true,
                );
                let mean_gain = NonnegativeRational::new(action.l1.clone(), alpha.clone())
                    .expect("positive alignment ratio");
                let alpha_squared = alpha.pow(2u32);
                // Energy only for a coefficient-uncorrelated contribution.
                let l1_squared = action.l1.pow(2u32);
                Some(
                    terms
                        .transform(|term| {
                            (
                                mean_gain.clone(),
                                term.moments.variance.scaled_ratio(
                                    if term.covariance_bounded {
                                        &action.l2_squared
                                    } else {
                                        &l1_squared
                                    },
                                    &alpha_squared,
                                ),
                                false,
                            )
                        })
                        .merge(&ciphertext_residual)
                        .merge(&mask),
                )
            }
            None => None,
        };
        let mut aligned = aligned;
        aligned.heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
        Ok(aligned.with_breakdown(breakdown))
    }

    pub(super) fn event_coordinates(&self, wire: WireId) -> Result<BigUint, RnsBggError> {
        let level = self.level(wire)?;
        Ok(BigUint::from(self.parameters.level(level)?.ring_dimension()) *
            2u8 *
            BigUint::from(self.parameters.component_columns(level)?))
    }

    pub(super) fn operation_node_with_multiplier(
        &mut self,
        wire: WireId,
        value: RnsBggNoiseValue,
        node: crate::noise_utils::NoiseDependencyNode,
        multiplier: BigUint,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let coordinates = self.event_coordinates(wire)?;
        trace_noise_value(wire, self.level(wire).ok(), &node, &value);
        self.trace.wires.insert(wire, value.traced());
        let node_id = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(node);
        self.trace.dependencies.wires.insert(wire, node_id);
        self.trace.node_event_counts.insert(node_id.0, multiplier * coordinates);
        self.add_events(&BigUint::one());
        Ok((wire, value))
    }

    pub(super) fn operation_node(
        &mut self,
        wire: WireId,
        value: RnsBggNoiseValue,
        node: crate::noise_utils::NoiseDependencyNode,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        self.operation_node_with_multiplier(wire, value, node, self.event_multiplier.clone())
    }

    pub(super) fn merge_decoder_trace(&mut self, other: &RnsBggNoiseTrace) {
        self.trace.joint_event_count += &other.joint_event_count;
        for (wire, value) in &other.wires {
            self.trace.wires.entry(*wire).or_insert_with(|| value.clone());
        }
        let offset = self.trace.dependencies.nodes.len();
        let shift = |node: crate::noise_utils::NoiseNodeId| {
            crate::noise_utils::NoiseNodeId(node.0 + offset)
        };
        for node in &other.dependencies.nodes {
            let node = match node {
                crate::noise_utils::NoiseDependencyNode::Binding(binding) => {
                    crate::noise_utils::NoiseDependencyNode::Binding(binding.clone())
                }
                crate::noise_utils::NoiseDependencyNode::Add { gate, lhs, rhs } => {
                    crate::noise_utils::NoiseDependencyNode::Add {
                        gate: *gate,
                        lhs: shift(*lhs),
                        rhs: shift(*rhs),
                    }
                }
                crate::noise_utils::NoiseDependencyNode::MulPublic { gate, lhs, rhs } => {
                    crate::noise_utils::NoiseDependencyNode::MulPublic {
                        gate: *gate,
                        lhs: shift(*lhs),
                        rhs: shift(*rhs),
                    }
                }
                crate::noise_utils::NoiseDependencyNode::MulPrivate {
                    gate,
                    input,
                    rhs,
                    helper,
                } => crate::noise_utils::NoiseDependencyNode::MulPrivate {
                    gate: *gate,
                    input: shift(*input),
                    rhs: shift(*rhs),
                    helper: helper.map(shift),
                },
                crate::noise_utils::NoiseDependencyNode::Lut { gate, table, input, helper } => {
                    crate::noise_utils::NoiseDependencyNode::Lut {
                        gate: *gate,
                        table: *table,
                        input: shift(*input),
                        helper: helper.map(shift),
                    }
                }
                crate::noise_utils::NoiseDependencyNode::Reduction { gate, input, range } => {
                    crate::noise_utils::NoiseDependencyNode::Reduction {
                        gate: *gate,
                        input: shift(*input),
                        range: range.clone(),
                    }
                }
                crate::noise_utils::NoiseDependencyNode::MulSmallConstant { gate, input } => {
                    crate::noise_utils::NoiseDependencyNode::MulSmallConstant {
                        gate: *gate,
                        input: shift(*input),
                    }
                }
            };
            self.trace.dependencies.nodes.push(node);
        }
        for (node, count) in &other.node_event_counts {
            self.trace.node_event_counts.insert(node + offset, count.clone());
        }
        for (wire, node) in &other.dependencies.wires {
            self.trace
                .dependencies
                .wires
                .entry(*wire)
                .or_insert(crate::noise_utils::NoiseNodeId(node.0 + offset));
        }
    }

    fn reduce_add(
        &mut self,
        wire: WireId,
        source_wire: WireId,
        family: &RnsBggNoiseFamily,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let n = family.validate_support()?;
        let k = family.max_nonzero_payloads.clone().min(n.clone());
        // The payload support only bounds the plaintext; every lane's
        // sampled errors and residuals are summed.
        let mut counts = ReductionCounts::default();
        let breakdown = family.element.contributions("family-element").map(|breakdown| {
            breakdown.reduce(
                self.lane_indices.len() + 1,
                &family_lane(&family.range, self.lane_indices.len()),
                &n,
                &mut counts,
            )
        });
        let value = RnsBggNoiseValue {
            bound: RnsBggNoiseBound {
                error: &family.element.bound.error * &n,
                payload_l1: &family.element.bound.payload_l1 * &k,
            },
            payload_l2_squared: &family.element.payload_l2_squared * k.pow(2),
            moments: family.element.moments.clone(),
            provenance: family.element.provenance.clone(),
            heuristics: family.element.heuristics.clone(),
            breakdown: None,
        }
        .with_breakdown(breakdown);
        tracing::debug!(
            wire = wire.0,
            members = %n,
            coherent = counts.coherent,
            independent = counts.independent,
            histogram = counts.histogram,
            unknown_index = counts.unknown_index,
            variance_log2 = value.moments.as_ref().map(|m| variance_log2(&m.variance)),
            contributions = %value.describe_contributions(),
            "noise family sum"
        );
        self.reduction_counts.coherent += counts.coherent;
        self.reduction_counts.independent += counts.independent;
        self.reduction_counts.histogram += counts.histogram;
        self.reduction_counts.unknown_index += counts.unknown_index;
        self.trace.wires.insert(wire, value.traced());
        let input_node = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Binding(
            crate::noise_utils::NoiseBinding::Input(source_wire),
        ));
        let node = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Reduction {
            gate: Some(wire),
            input: input_node,
            range: family.range.clone(),
        });
        self.trace.dependencies.wires.insert(wire, node);
        self.trace.node_event_counts.insert(node.0, n.clone() * self.event_coordinates(wire)?);
        self.add_events(&n);
        Ok((wire, value))
    }

    fn fresh_gain(&self, level: usize) -> Result<RightActionNoiseGain, RnsBggError> {
        if level + 1 < self.parameters.levels().len() {
            self.parameters.level(level + 1)?;
        }
        Ok(RightActionNoiseGain::unit())
    }

    /// Contributions of a non-terminal flat LUT. A coefficient output stays
    /// in the units before its one scale-and-round, which `finish_outputs`
    /// applies; any other output is converted here.
    #[allow(clippy::too_many_arguments)]
    fn lut_contributions(
        &self,
        level: usize,
        lut: LutId,
        table: &LutTable,
        input: &RnsBggNoiseValue,
        input_terms: &NoiseBreakdown,
        coefficient_output: bool,
    ) -> Result<NoiseBreakdown, RnsBggError> {
        let branches = table.input_width();
        let payload_gain = RightActionNoiseGain {
            l1: input.bound.payload_l1.clone(),
            l2_squared: input.payload_l2_squared.clone(),
        };
        let terms = self.parameters.lut_noise_terms(
            level,
            branches,
            &payload_gain,
            &self.fresh_gain(level)?,
        )?;
        // Helper and B-key errors are sampled per automorphism; a repeated
        // automorphism reuses its sample, so only distinct ones add linearly.
        let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let automorphisms =
            table.branch_automorphisms(ring_dimension).map_err(|_| RnsBggError::Lut)?.len();
        let branch_squares = if automorphisms == branches {
            BigUint::from(branches)
        } else {
            BigUint::from(branches).pow(2)
        };
        let divisor = if coefficient_output { BigUint::one() } else { terms.scale.pow(2) };
        let units =
            |variance: &NonnegativeRational| variance.scaled_ratio(&BigUint::one(), &divisor);
        let mean_gain = NonnegativeRational::new(
            terms.input_mean_gain.clone(),
            if coefficient_output { BigUint::one() } else { terms.scale.clone() },
        )
        .expect("positive scale");
        // Energy only for a coefficient-uncorrelated contribution (H1/H4);
        // otherwise the squared L1 action of the coherent branch sum.
        let l1_squared = terms.input_mean_gain.pow(2);
        let inherited = input_terms.transform(|term| {
            let action =
                if term.covariance_bounded { &terms.input_variance_gain } else { &l1_squared };
            (mean_gain.clone(), units(&term.moments.variance.scaled(action)), false)
        });
        let helper = NoiseBreakdown::sampled(
            stable_source_id(&(b"rns-bgg/lut/helper".as_slice(), level)),
            "lut-helper",
            zero_mean_moments(units(&terms.helper.scaled(&branch_squares))),
            false,
        );
        // Each branch's B-key error is a fresh Gaussian sample weighted by
        // the automorphed input plaintext and the scalar `p`: under a signed
        // monomial plaintext its image keeps the sample's covariance bound,
        // as the RHS error of a direct product does, and the branch sum
        // keeps it whatever the branch correlation.
        let monomial = payload_gain.l1.pow(2) == payload_gain.l2_squared;
        let fresh = NoiseBreakdown::sampled(
            stable_source_id(&(b"rns-bgg/lut/B".as_slice(), level, lut)),
            "lut-fresh",
            zero_mean_moments(units(&terms.fresh.scaled(&branch_squares))),
            monomial,
        );
        // The remainders are determined by the LUT and its input: every
        // invocation of the same LUT on the same value shares them. Under H2
        // each branch's internal remainder sigma_k(s) * rho_k, averaged over
        // the ternary secret and a centered uncorrelated residue, has
        // covariance at most V I. Their branch correlation is not assumed:
        // the coherent J^2 V sum keeps the (sum sqrt V)^2 I = J^2 V I bound,
        // so a later fixed action may use its energy. The outer public and
        // encoding remainders of the one scale-and-round are bounded the
        // same way, and H2 makes them uncorrelated with each other.
        let operand = input_terms.operand();
        let residual = |tag: &'static [u8], origin: &'static str, variance: NonnegativeRational| {
            match &operand {
                Some((operand, sample)) => NoiseBreakdown::object_residual(
                    stable_source_id(&(tag, operand, lut, level)),
                    origin,
                    sample.clone(),
                    zero_mean_moments(variance),
                    true,
                ),
                None => NoiseBreakdown::opaque(origin, zero_mean_moments(variance)),
            }
        };
        let mut contributions = inherited.merge(&helper).merge(&fresh).merge(&residual(
            b"rns-bgg/lut/rounding",
            "lut-rounding",
            units(&terms.internal_rounding),
        ));
        if !coefficient_output {
            contributions = contributions.merge(&residual(
                b"rns-bgg/lut/outer-rounding",
                "lut-outer-rounding",
                terms.outer_rounding,
            ));
        }
        Ok(contributions)
    }

    /// Normalize one coefficient-output value: its one scale-and-round by
    /// the source ratio, with the public-key and encoding roundings (or, at
    /// the terminal level, the terminal encoding error). On an equal-basis
    /// level the ratio is one: the ModDown is an identity and adds no
    /// rounding.
    fn normalize_coefficient(
        &mut self,
        wire: WireId,
        mut value: RnsBggNoiseValue,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        let level = self.level(wire)?;
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            // An equal-basis ModDown is an identity: no rounding is added.
            self.parameters.dynamic_route(level + 1)?;
            return Ok(value);
        }
        let ratio = self.parameters.scale(level)?.clone();
        if ratio == BigUint::one() {
            // The terminal level has no scale-and-round.
            value.bound.error += self.parameters.encoding_error_bound();
            let breakdown = value.contributions("coefficient-output").map(|terms| {
                terms.merge(&NoiseBreakdown::opaque(
                    "terminal-encoding",
                    zero_mean_moments(self.parameters.encoding_error_variance()),
                ))
            });
            value = value.with_breakdown(breakdown);
        } else {
            // The aggregated noise is still before the flat LUT's one
            // scale-and-round (spec §8.2); convert it exactly once,
            // with the public-key and encoding roundings.
            value.bound.error = self.parameters.lut_mod_down_bound(level, &value.bound.error)?;
            let rounding = zero_mean_moments(self.parameters.lut_public_rounding_variance(level)?)
                .add_with_covariance(
                    &NoiseMoments::discrete_rounding_residue(&ratio)
                        .map_err(|error| RnsBggError::Graph(error.to_string()))?,
                    NoiseCorrelation::Independent,
                );
            let mean_gain = NonnegativeRational::new(BigUint::one(), ratio.clone())
                .expect("positive coefficient modulus ratio");
            let breakdown = value.contributions("coefficient-output").map(|terms| {
                // The final rounding is determined by the value it
                // rounds.
                let rounding = match terms.operand() {
                    Some((operand, sample)) => NoiseBreakdown::object_residual(
                        stable_source_id(&(
                            b"rns-bgg/coefficient-final-rounding".as_slice(),
                            operand,
                            level,
                        )),
                        "coefficient-final-rounding",
                        sample,
                        rounding.clone(),
                        true,
                    ),
                    None => NoiseBreakdown::opaque("coefficient-final-rounding", rounding.clone()),
                };
                terms
                    .transform(|term| {
                        (
                            mean_gain.clone(),
                            term.moments.variance.scaled_ratio(&BigUint::one(), &ratio.pow(2u32)),
                            false,
                        )
                    })
                    .merge(&rounding)
            });
            value = value.with_breakdown(breakdown);
        }
        if ratio != BigUint::one() {
            let source = stable_source_id(&(
                b"rns-bgg/coefficient-final-mod-down".as_slice(),
                self.program.id(),
                wire,
                level,
            ));
            value.provenance.insert(source, NoiseSourceProvenance { source, route: vec![level] });
        }
        self.add_events(&BigUint::one());
        if let Some(node) = self.trace.dependencies.wires.get(&wire).copied() {
            let coordinates = self.event_coordinates(wire)?;
            *self.trace.node_event_counts.entry(node.0).or_default() += coordinates;
        }
        Ok(value)
    }

    /// The direct-decoding action `G^-1(w)` of a normalized carrier at
    /// `level` (see [`RnsBggParameters::output_carrier`]): a fixed right
    /// action with the balanced digits of `w`, whose energy applies only to a
    /// contribution with an established covariance bound.
    fn output_action(
        &self,
        mut value: RnsBggNoiseValue,
        level: usize,
        weight: &BigUint,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        let (l1, l2_squared) = self.parameters.gadget_inverse_norms(level, weight)?;
        value.bound.error *= &l1;
        let mean_gain =
            NonnegativeRational::new(l1.clone(), BigUint::one()).expect("positive output action");
        let l1_squared = &l1 * &l1;
        let breakdown = value.contributions("output-action").map(|terms| {
            terms.transform(|term| {
                let gain = if term.covariance_bounded { &l2_squared } else { &l1_squared };
                (mean_gain.clone(), term.moments.variance.scaled(gain), false)
            })
        });
        Ok(value.with_breakdown(breakdown))
    }

    pub(super) fn finish_outputs(
        &mut self,
        outputs: Vec<(WireId, RnsBggNoiseValue)>,
    ) -> Result<Vec<RnsBggNoiseValue>, RnsBggError> {
        let mut finalized = Vec::with_capacity(outputs.len());
        for (wire, value) in outputs {
            // Every output is resolved once, after the complete aggregation
            // topology, through the same carrier as the public-key and
            // encoding lowerings: a coefficient output is normalized once
            // (exponent outputs already carry their normal result and must
            // not be rounded again), then the selected direct-decoding action
            // adds its gain.
            if let Some(previous) = self.finalized_outputs.get(&wire) {
                finalized.push(previous.clone());
                continue;
            }
            let level = self.level(wire)?;
            let carrier = self.parameters.output_carrier(
                self.program,
                wire,
                level,
                self.bindings.output_use,
            )?;
            let mut value = if carrier.level != level {
                self.normalize_coefficient(wire, value)?
            } else {
                value
            };
            if let Some(weight) = &carrier.action {
                value = self.output_action(value, carrier.level, weight)?;
            }
            self.finalized_outputs.insert(wire, value.clone());
            finalized.push(value);
        }
        let counts = &self.reduction_counts;
        tracing::info!(
            coherent = counts.coherent,
            independent = counts.independent,
            histogram = counts.histogram,
            unknown_index = counts.unknown_index,
            "noise family reductions"
        );
        for value in &finalized {
            tracing::debug!(contributions = %value.describe_contributions(), "noise output");
        }
        Ok(finalized)
    }
}

impl<'a> FamilyExecution<RnsBggNoiseEvaluator<'a>> for RnsBggNoiseFamilies {
    type WireFamily = (WireId, RnsBggNoiseFamily);
    type RhsFamily = (RhsId, FamilyRange);
    /// A selected member is one the noise model cannot name: its payload
    /// support is unknown (no row contract holds) and its per-member samples
    /// take the unknown coordinate, so reductions over it stay coherent.
    fn wire_selected(
        &mut self,
        evaluator: &mut RnsBggNoiseEvaluator<'a>,
        family: &Self::WireFamily,
        _data: crate::program::DataId,
        position: &IntExpr,
    ) -> Result<(WireId, RnsBggNoiseValue), ProgramLowerError<RnsBggError>> {
        evaluator.observe_family_read(family.0, position, None);
        let element = &family.1.element;
        let member =
            element.breakdown.as_ref().map(|breakdown| breakdown.member_unknown(self.depth));
        Ok((family.0, element.clone().with_breakdown(member)))
    }
    fn wire_at(
        &mut self,
        evaluator: &mut RnsBggNoiseEvaluator<'a>,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<(WireId, RnsBggNoiseValue), ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "noise family index is not canonical".into(),
                ),
            ));
        }
        evaluator.observe_family_read(
            family.0,
            index,
            family.1.row_length.as_ref().map(|row| (&family.1.max_nonzero_payloads, row)),
        );
        let element = &family.1.element;
        let member = element.breakdown.as_ref().map(|breakdown| {
            let identity = self
                .depth
                .checked_sub(1)
                .is_some_and(|depth| *index == family_lane(&family.1.range, depth));
            breakdown.member(self.depth, index, identity)
        });
        Ok((family.0, element.clone().with_breakdown(member)))
    }
    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<RhsRef<IntExpr>, ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "noise RHS index is not canonical".into(),
                ),
            ));
        }
        Ok(RhsRef { id: family.0, index: Some(index.clone()) })
    }
    /// A member selected at run time is analyzed as member `0` at every selection: all
    /// selected products then read one member, whose errors add coherently, the worst case of
    /// the correlation bookkeeping.
    fn rhs_selected(
        &mut self,
        family: &Self::RhsFamily,
        _: crate::program::DataId,
        _: &IntExpr,
    ) -> Result<RhsRef<IntExpr>, ProgramLowerError<RnsBggError>> {
        Ok(RhsRef { id: family.0, index: Some(IntExpr::constant(0)) })
    }
    fn parallel(
        &mut self,
        evaluator: &mut RnsBggNoiseEvaluator<'a>,
        range: &FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut RnsBggNoiseEvaluator<'a>,
            IntExpr,
        ) -> Result<(WireId, RnsBggNoiseValue), ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, ProgramLowerError<RnsBggError>> {
        let index = family_lane(range, evaluator.lane_indices.len());
        let count = range
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_biguint())
            .ok_or_else(|| {
                ProgramLowerError::Backend(RnsBggError::Binding("family count is not concrete"))
            })?;
        let previous_multiplier = evaluator.event_multiplier.clone();
        evaluator.event_multiplier *= &count;
        evaluator.lane_indices.push(index.clone());
        evaluator.lane_counts.push(count.clone());
        self.depth += 1;
        let result = body(self, evaluator, index);
        let body_depth = self.depth;
        self.depth -= 1;
        let (wire, mut element) = match result {
            Ok(value) => value,
            Err(error) => {
                evaluator.lane_indices.pop();
                evaluator.lane_counts.pop();
                evaluator.event_multiplier = previous_multiplier;
                return Err(error);
            }
        };
        element.breakdown = element
            .breakdown
            .map(|breakdown| Arc::new(breakdown.as_ref().clone().with_lane_depth(body_depth)));
        // Members weighted by distinct members of a sparse RHS family have
        // nonzero payloads in at most that family's sparse count of lanes.
        let row_length = evaluator.lane_row_support(wire).map(|(_, row)| row.clone());
        let max_nonzero_payloads = match evaluator.lane_row_support(wire) {
            Some((per_row, _)) => per_row.clone(),
            None => evaluator
                .lane_support(wire)
                .map_or(count.clone(), |support| support.min(&count).clone()),
        };
        evaluator.lane_indices.pop();
        evaluator.lane_counts.pop();
        evaluator.event_multiplier = previous_multiplier;
        Ok((
            wire,
            RnsBggNoiseFamily { element, range: range.clone(), max_nonzero_payloads, row_length },
        ))
    }
    fn sum(
        &mut self,
        evaluator: &mut RnsBggNoiseEvaluator<'a>,
        family: &Self::WireFamily,
        columns: usize,
    ) -> Result<(WireId, RnsBggNoiseValue), ProgramLowerError<RnsBggError>> {
        let output = evaluator.site().map_err(ProgramLowerError::Backend)?.output;
        // Every entry of a row Sum is the sum of one column's block: the
        // first block stands for every column, whose members have the same
        // noise.
        let mut block = family.1.clone();
        if columns > 1 {
            let count = block
                .range
                .count()
                .evaluate(&ParamEnv::default())
                .ok()
                .and_then(|count| count.to_usize())
                .ok_or(ProgramLowerError::Backend(RnsBggError::Parameters(
                    "a row Sum needs a constant member count",
                )))?;
            let per_column = IntExpr::constant(count / columns);
            block.range =
                FamilyRange::bounded(block.range.start().clone(), per_column.clone(), per_column)?;
            block.max_nonzero_payloads =
                block.max_nonzero_payloads.min(BigUint::from(count / columns));
        }
        evaluator.reduce_add(output, family.0, &block).map_err(ProgramLowerError::Backend)
    }
}

impl ExponentLUTProgramLowering for RnsBggNoiseEvaluator<'_> {
    type Wire = (WireId, RnsBggNoiseValue);
    type Rhs = RhsRef<IntExpr>;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        let level = self.level(lhs.0)?;
        if level != self.level(rhs.0)? {
            return Err(RnsBggError::Parameters("add operands have different logical levels"));
        }
        let wire = self.site()?.output;
        let correlation =
            if lhs.1.provenance.keys().any(|source| rhs.1.provenance.contains_key(source)) {
                NoiseCorrelation::SharedSource
            } else if !lhs.1.provenance.is_empty() && !rhs.1.provenance.is_empty() {
                NoiseCorrelation::Independent
            } else {
                NoiseCorrelation::Unknown
            };
        let breakdown = match (lhs.1.contributions("add-lhs"), rhs.1.contributions("add-rhs")) {
            (Some(lhs), Some(rhs)) => Some(lhs.merge(&rhs)),
            _ => None,
        };
        let mut value = lhs.1.combine(&rhs.1, correlation).with_breakdown(breakdown);
        value.bound = lhs.1.bound.add(&rhs.1.bound);
        value.payload_l2_squared =
            RnsBggNoiseValue::l2_add_bound(&lhs.1.payload_l2_squared, &rhs.1.payload_l2_squared);
        let node = crate::noise_utils::NoiseDependencyNode::Add {
            gate: Some(wire),
            lhs: self.node_for_wire(lhs.0)?,
            rhs: self.node_for_wire(rhs.0)?,
        };
        // A lane's sum is nonzero only where one of its summands is.
        if let (Some(lhs_support), Some(rhs_support)) =
            (self.lane_support(lhs.0), self.lane_support(rhs.0))
        {
            let count = lhs_support + rhs_support;
            self.set_lane_support(wire, count);
        }
        self.operation_node(wire, value, node)
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let lhs_level = self.level(lhs.0)?;
        let rhs_level = self.level(rhs.0)?;
        let lhs_knowledge = self
            .program
            .wire_knowledge(lhs.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPublic lhs".into()))?;
        let rhs_knowledge = self
            .program
            .wire_knowledge(rhs.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPublic rhs".into()))?;
        if lhs_knowledge != crate::program::AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Parameters("MulPublic LHS must be public"));
        }
        let target_level =
            self.parameters.mul_public_target_level(lhs_level, rhs_level, rhs_knowledge)?;
        let lhs_value = self.align_public_operand(&lhs.1, lhs_level, target_level)?;
        let rhs_value = if rhs_knowledge == crate::program::AttributeKnowledge::PublicPayload {
            self.align_public_operand(&rhs.1, rhs_level, target_level)?
        } else {
            rhs.1.clone()
        };
        let gain = self.parameters.mul_public_action_gain(target_level)?;
        let wire = self.site()?.output;
        let plaintext_gain = lhs_value.bound.payload_l1.clone();
        // E_out = E_left * decomposition(A_right) + mu_left * E_right. Each
        // side is a fixed public action on its own contributions; the energy
        // cap applies only to a covariance-bounded left contribution (H1/H4),
        // any other keeps the squared L1 bound. A signed monomial left
        // plaintext keeps the right contributions' covariance bound.
        let mut energy_cap_used = false;
        let breakdown = match (
            lhs_value.contributions("mul-public-lhs"),
            rhs_value.contributions("mul-public-rhs"),
        ) {
            (Some(lhs_terms), Some(rhs_terms)) => {
                let decomposition_l1 = NonnegativeRational::new(gain.l1.clone(), BigUint::one())
                    .expect("positive decomposition gain");
                let l1_squared = gain.l1.pow(2);
                let lhs_terms = lhs_terms.transform(|term| {
                    energy_cap_used |= term.covariance_bounded;
                    let action =
                        if term.covariance_bounded { &gain.l2_squared } else { &l1_squared };
                    (decomposition_l1.clone(), term.moments.variance.scaled(action), false)
                });
                let monomial = plaintext_gain.pow(2) == lhs_value.payload_l2_squared;
                let plaintext_l1 = NonnegativeRational::new(plaintext_gain.clone(), BigUint::one())
                    .expect("positive plaintext gain");
                let rhs_terms = rhs_terms.transform(|term| {
                    (
                        plaintext_l1.clone(),
                        term.moments.variance.scaled(&plaintext_gain.pow(2)),
                        term.covariance_bounded && monomial,
                    )
                });
                Some(lhs_terms.merge(&rhs_terms))
            }
            _ => None,
        };
        let mut heuristics = lhs_value.heuristics.clone();
        if energy_cap_used {
            heuristics.insert(NoiseHeuristicId::H1StateUncorrelated);
            heuristics.insert(NoiseHeuristicId::H4RhsIndependence);
        }
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound: rhs_value.bound.mul_public_with_action(
                    &lhs_value.bound,
                    &plaintext_gain,
                    &gain,
                ),
                payload_l2_squared: &lhs_value.payload_l2_squared * &rhs_value.payload_l2_squared,
                moments: None,
                provenance: BTreeMap::new(),
                heuristics,
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [lhs_value.provenance.clone(), rhs_value.provenance.clone()],
        );
        let node = crate::noise_utils::NoiseDependencyNode::MulPublic {
            gate: Some(wire),
            lhs: self.node_for_wire(lhs.0)?,
            rhs: self.node_for_wire(rhs.0)?,
        };
        // A public factor keeps the zero payloads of the right operand.
        if let Some(count) = self.lane_support(rhs.0).cloned() {
            self.set_lane_support(wire, count);
        }
        self.operation_node(wire, value, node)
    }
    fn mul_small_constant(
        &mut self,
        input: &Self::Wire,
        coefficients: &[i64],
        _: &IntExpr,
    ) -> Result<Self::Wire, Self::Error> {
        let wire = self.site()?.output;
        let l1 = BigUint::from(
            crate::program::small_constant_l1(coefficients)
                .ok_or(RnsBggError::Parameters("small constant l1 overflow"))?,
        );
        // E_out = E c: every error coefficient is a signed sum of `||c||_1`
        // weighted coefficients of E, so the bound and the mean gain scale
        // by `||c||_1` and the variance by its square, as for a public
        // plaintext multiplier. A single term `c X^k` permutes and scales the
        // coefficients uniformly, so it keeps a contribution's covariance
        // bound; a monomial (`|c| = 1`) leaves the noise unchanged.
        let single_term = crate::program::small_constant_is_single_term(coefficients);
        let gain = NonnegativeRational::new(l1.clone(), BigUint::one())
            .expect("a nonzero small constant has a positive l1 norm");
        let l1_squared = l1.pow(2);
        let breakdown = input.1.contributions("mul-small-constant").map(|terms| {
            terms.transform(|term| {
                (
                    gain.clone(),
                    term.moments.variance.scaled(&l1_squared),
                    term.covariance_bounded && single_term,
                )
            })
        });
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound: input.1.bound.mul_public(&l1),
                payload_l2_squared: &input.1.payload_l2_squared * &l1_squared,
                moments: None,
                provenance: BTreeMap::new(),
                heuristics: input.1.heuristics.clone(),
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone()],
        );
        let node = crate::noise_utils::NoiseDependencyNode::MulSmallConstant {
            gate: Some(wire),
            input: self.node_for_wire(input.0)?,
        };
        // A constant factor keeps the zero payloads of its operand.
        if let Some(count) = self.lane_support(input.0).cloned() {
            self.set_lane_support(wire, count);
        }
        self.operation_node(wire, value, node)
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid MulPrivate input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Parameters("RNS-BGG MulPrivate requires a private input"));
        }
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            return self.dynamic_mul(input, rhs);
        }
        let logical_level = self.level(input.0)?;
        let level = logical_level;
        let (contract, sparse_count, row_length) = match self
            .bindings
            .rhs
            .get(&rhs.id)
            .ok_or(RnsBggError::Binding("missing RHS noise contract"))?
        {
            InputBinding::Scalar(value) => (value.clone(), None, None),
            InputBinding::Family(family) => (
                family.element.clone(),
                Some(family.max_nonzero_payloads.clone()),
                family.row_length.clone(),
            ),
        };
        let fresh_error_gain = self.fresh_gain(level)?;
        let wire = self.site()?.output;
        let output_l1 = &input.1.bound.payload_l1 * &contract.payload_l1;
        let bound = self.parameters.private_mul_noise_bound(
            level,
            &input.1.bound,
            &fresh_error_gain.l1,
            output_l1.clone(),
        )?;
        let payload_gain = RightActionNoiseGain {
            l1: input.1.bound.payload_l1.clone(),
            l2_squared: input.1.payload_l2_squared.clone(),
        };
        let terms =
            self.parameters.private_mul_noise_terms(level, &payload_gain, &fresh_error_gain)?;
        // e*U/p keeps each input contribution's identity. -mu*F is the RHS
        // artifact's own sample: every gate reading the same RHS member
        // shares it, and distinct members of an RHS family are distinct
        // preparations. The remainders are this gate's residual.
        let breakdown = input.1.contributions("mul-private-input").map(|input_terms| {
            // The energy gain needs coefficient-uncorrelated input (H1) under
            // the fixed prepared action (H4); any other contribution keeps
            // the squared L1 action.
            let l1 = &terms.inherited_mean_gain;
            let inherited = input_terms.transform(|term| {
                let variance = if term.covariance_bounded {
                    terms.inherited_variance(&term.moments.variance)
                } else {
                    term.moments.variance.scaled_ratio(&l1.numerator.pow(2), &l1.denominator.pow(2))
                };
                (l1.clone(), variance, false)
            });
            let rhs_source = stable_source_id(&(b"rns-bgg/rhs-error".as_slice(), rhs.id, level));
            let rhs_moments = zero_mean_moments(terms.rhs_error.clone());
            let rhs_bounded = payload_gain.l1.pow(2) == payload_gain.l2_squared;
            let rhs_error = match rhs.index.as_ref() {
                Some(index) => NoiseBreakdown::sampled_member(
                    rhs_source,
                    "rhs-error",
                    self.lane_indices.len(),
                    index.clone(),
                    rhs_moments,
                    rhs_bounded,
                ),
                None => NoiseBreakdown::sampled(rhs_source, "rhs-error", rhs_moments, rhs_bounded),
            };
            // Under H2 both remainders are centered uniforms independent per
            // coefficient and of the independent secret, so their images
            // are coefficient-uncorrelated. The remainders are determined by
            // the operation and its operands: every gate that multiplies the
            // same input by the same RHS member at this level shares them.
            // The RHS member and the input's members are coordinates, not
            // part of the source, so aliasing reads stay coherent.
            let rounding = match input_terms.operand() {
                Some((operand, input_sample)) => NoiseBreakdown::object_residual(
                    stable_source_id(&(
                        b"rns-bgg/private-mul/rounding".as_slice(),
                        operand,
                        rhs.id,
                        level,
                    )),
                    "private-mul-rounding",
                    rhs.index
                        .as_ref()
                        .map(|index| SampleCoordinate::Index {
                            depth: self.lane_indices.len(),
                            index: Arc::new(index.clone()),
                        })
                        .into_iter()
                        .chain(input_sample)
                        .collect(),
                    zero_mean_moments(terms.rounding.clone()),
                    true,
                ),
                None => NoiseBreakdown::opaque(
                    "private-mul-rounding",
                    zero_mean_moments(terms.rounding.clone()),
                ),
            };
            inherited.merge(&rhs_error).merge(&rounding)
        });
        tracing::trace!(
            wire = wire.0,
            level,
            inherited_gain_log2 = variance_log2(&terms.inherited_variance_gain),
            rhs_error_log2 = variance_log2(&terms.rhs_error),
            payload_l1_bits = payload_gain.l1.bits(),
            payload_l2_squared_bits = payload_gain.l2_squared.bits(),
            rounding_log2 = variance_log2(&terms.rounding),
            "private multiplication variance terms"
        );
        let mut heuristics = input.1.heuristics.clone();
        heuristics.insert(NoiseHeuristicId::H1StateUncorrelated);
        heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
        heuristics.insert(NoiseHeuristicId::H4RhsIndependence);
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound,
                payload_l2_squared: &input.1.payload_l2_squared * &contract.payload_l2_squared,
                moments: None,
                provenance: BTreeMap::new(),
                heuristics,
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone()],
        );
        let node = crate::noise_utils::NoiseDependencyNode::MulPrivate {
            gate: Some(wire),
            input: self.node_for_wire(input.0)?,
            rhs: self.rhs_node(rhs),
            helper: None,
        };
        if let (Some(count), Some(index), Some(lane), Some(lanes)) =
            (sparse_count, rhs.index.as_ref(), self.lane_indices.last(), self.lane_counts.last())
        {
            // Without a row contract the count bounds the whole family, so the
            // lanes must read distinct members; with one it bounds each aligned
            // row, so the lanes must stay inside one row.
            match &row_length {
                None if lane_injective(index, lane) => self.set_lane_support(wire, count),
                Some(row)
                    if row_injective(index, lane, self.lane_indices.len() - 1, lanes, row) =>
                {
                    self.set_lane_support(wire, count)
                }
                // The lanes read the whole family member by member: the
                // family they form keeps the row contract.
                Some(row) if index.clone().canonicalize() == lane.clone().canonicalize() => {
                    self.payload_support.insert(
                        wire,
                        PayloadSupport {
                            depth: self.lane_indices.len(),
                            count,
                            row_length: Some(row.clone()),
                        },
                    );
                }
                _ => {}
            }
        }
        self.operation_node(wire, value, node)
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        let public = self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid LUT input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload;
        let lut = self.program.lut_id_at(self.site()?.output).map_err(|_| RnsBggError::Lut)?;
        if public &&
            self.decoder_smudge
                .as_ref()
                .is_some_and(|declaration| declaration.smudge().lut() == lut)
        {
            return Err(RnsBggError::Parameters("a decoder smudge LUT needs a private input"));
        }
        let level = self.level(input.0)?;
        if table.automorphism_index().is_some() &&
            (public || self.parameters.rhs_mode() == RhsMode::Dynamic)
        {
            return Err(RnsBggError::Parameters(
                "an automorphism table needs a Static private input",
            ));
        }
        if self.parameters.rhs_mode() == RhsMode::Dynamic {
            let value = if public {
                self.dynamic_public_lut(input, table)?
            } else {
                self.dynamic_lut(input, table)?
            };
            return self.finish_lut(lut, level, value);
        }
        if public {
            return self.public_lut(input, table, lut, level);
        }
        let fresh_error_gain = self.fresh_gain(level)?;
        let wire = self.site()?.output;
        // Exponent outputs are signed monomials, irrespective of the
        // numeric exponent stored in the table.  Coefficient outputs instead
        // carry the table's integer coefficient bound. `Aut_k` returns `sigma_k` of its input,
        // whose coefficients are a signed permutation of the input's, so it keeps the input's
        // payload norm.
        let output_payload_l1 = match table.output_representation() {
            _ if table.automorphism_index().is_some() => input.1.bound.payload_l1.clone(),
            crate::program::LutOutputRepresentation::Exponent(_) => BigUint::one(),
            crate::program::LutOutputRepresentation::Coefficient => {
                BigUint::from(table.max_output())
            }
        };
        let output_payload_l2_squared = output_payload_l1.pow(2u32);
        let coefficient_output = table.output_representation().is_coefficient();
        let terminal_coefficient =
            coefficient_output && level + 1 >= self.parameters.levels().len();
        // A coefficient output keeps its noise before the scale-and-round
        // through its aggregation; `finish_outputs` converts it exactly once.
        let bound = if terminal_coefficient {
            RnsBggNoiseBound {
                error: input.1.bound.error.clone(),
                payload_l1: output_payload_l1.clone(),
            }
        } else if coefficient_output {
            self.parameters.lut_noise_bound_before_mod_down(
                level,
                &input.1.bound,
                table.input_width(),
                &fresh_error_gain.l1,
                output_payload_l1,
            )?
        } else {
            self.parameters.lut_noise_bound(
                level,
                &input.1.bound,
                table.input_width(),
                &fresh_error_gain.l1,
                output_payload_l1,
            )?
        };
        let breakdown = input
            .1
            .contributions("lut-input")
            .map(|input_terms| {
                if terminal_coefficient {
                    return Ok(input_terms.into_owned());
                }
                self.lut_contributions(
                    level,
                    lut,
                    table,
                    &input.1,
                    &input_terms,
                    coefficient_output,
                )
            })
            .transpose()?;
        let material = self.lut_material_provenance(level, lut, table, true)?;
        let mut heuristics = input.1.heuristics.clone();
        if !terminal_coefficient {
            heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
        }
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound,
                payload_l2_squared: output_payload_l2_squared,
                moments: None,
                provenance: BTreeMap::new(),
                heuristics,
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone(), material],
        );
        let node = crate::noise_utils::NoiseDependencyNode::Lut {
            gate: Some(wire),
            table: lut,
            input: self.node_for_wire(input.0)?,
            helper: None,
        };
        let value = self.operation_node_with_multiplier(
            wire,
            value,
            node,
            self.event_multiplier.clone() * BigUint::from(table.input_width()),
        )?;
        self.finish_lut(lut, level, value)
    }
}

impl RnsBggNoiseEvaluator<'_> {
    /// A Static LUT on a public payload at `level` (see
    /// [`RnsBggParameters::public_lut_noise_bound`]), term by term with the
    /// encoding lowering: the input decomposed once per branch, the helpers
    /// (one coherent source per level, shared with private LUTs, which read
    /// the same helpers), and the mask and ModDown remainders, all in
    /// level-`l + 1` units after the one scale-and-round.
    fn public_lut(
        &mut self,
        input: &(WireId, RnsBggNoiseValue),
        table: &LutTable,
        lut: LutId,
        level: usize,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let wire = self.site()?.output;
        let branches = table.input_width();
        // The bridge length of every nonidentity branch, as the lowering
        // builds it.
        let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let sigmas = table
            .input_layout()
            .automorphism_indices(ring_dimension)
            .map_err(|_| RnsBggError::Lut)?;
        let hops = crate::rns_bgg::public_lut::inverse_words(&sigmas, ring_dimension)?
            .iter()
            .skip(1)
            .map(Vec::len)
            .collect::<Vec<_>>();
        let output_payload_l1 = match table.output_representation() {
            crate::program::LutOutputRepresentation::Exponent(_) => BigUint::one(),
            crate::program::LutOutputRepresentation::Coefficient => {
                BigUint::from(table.max_output())
            }
        };
        let bound = self.parameters.public_lut_noise_bound(
            level,
            &input.1.bound,
            &hops,
            output_payload_l1.clone(),
        )?;
        let (input_gain, helper_gain, _) =
            self.parameters.public_lut_gains(level, &hops, &input.1.bound.payload_l1)?;
        // Every bridge and final mask switch leaves one remainder; their
        // coherent sum is counted as that many branches.
        let rounding = self.parameters.lut_noise_terms(
            level,
            hops.iter().sum::<usize>().max(1),
            &RightActionNoiseGain {
                l1: input.1.bound.payload_l1.clone(),
                l2_squared: input.1.payload_l2_squared.clone(),
            },
            &self.fresh_gain(level)?,
        )?;
        let scale = self.parameters.scale(level)?.clone();
        let units = |variance: &NonnegativeRational| {
            variance.scaled_ratio(&BigUint::one(), &scale.pow(2u32))
        };
        let breakdown = input.1.contributions("public-lut-input").map(|input_terms| {
            // The branches add coherently: the squared `l1` gains.
            let mean_gain =
                NonnegativeRational::new(input_gain.clone(), scale.clone()).expect("scale");
            let input_variance_gain = input_gain.pow(2u32);
            let inherited = input_terms.transform(|term| {
                (
                    mean_gain.clone(),
                    units(&term.moments.variance.scaled(&input_variance_gain)),
                    false,
                )
            });
            let helper = NoiseBreakdown::sampled(
                stable_source_id(&(b"rns-bgg/lut/helper".as_slice(), level)),
                "lut-helper",
                zero_mean_moments(units(
                    &self.parameters.encoding_error_variance().scaled(&helper_gain.pow(2u32)),
                )),
                false,
            );
            // The mask remainders and the one ModDown's public and
            // encoding remainders, as for a private LUT.
            let operand = input_terms.operand();
            let residual =
                |tag: &'static [u8], origin: &'static str, variance: NonnegativeRational| {
                    match &operand {
                        Some((operand, sample)) => NoiseBreakdown::object_residual(
                            stable_source_id(&(tag, operand, lut, level)),
                            origin,
                            sample.clone(),
                            zero_mean_moments(variance),
                            true,
                        ),
                        None => NoiseBreakdown::opaque(origin, zero_mean_moments(variance)),
                    }
                };
            inherited
                .merge(&helper)
                .merge(&residual(
                    b"rns-bgg/lut/public-rounding",
                    "public-lut-rounding",
                    units(&rounding.internal_rounding),
                ))
                .merge(&residual(
                    b"rns-bgg/lut/public-outer-rounding",
                    "public-lut-outer-rounding",
                    rounding.outer_rounding.clone(),
                ))
        });
        let material = self.lut_material_provenance(level, lut, table, false)?;
        let mut heuristics = input.1.heuristics.clone();
        heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
        let value = self.carry_provenance(
            RnsBggNoiseValue {
                bound,
                payload_l2_squared: output_payload_l1.pow(2u32),
                moments: None,
                provenance: BTreeMap::new(),
                heuristics,
                breakdown: None,
            }
            .with_breakdown(breakdown),
            [input.1.provenance.clone(), material],
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

    /// The common tail of a LUT at `level`: a decoder smudge member is
    /// normalized once in its root body, then weighted and placed.
    fn finish_lut(
        &mut self,
        lut: LutId,
        level: usize,
        (wire, value): (WireId, RnsBggNoiseValue),
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        if let Some(declaration) = self.decoder_smudge.clone() &&
            declaration.smudge().lut() == lut
        {
            // A decoder smudge member: moved down once in its root body, then
            // weighted by `G^-1(w)` for its halved weight `w = 2^-1 P^digit mod
            // q`, and placed by a monomial, as the encoding does. The error
            // gain is the largest `l1(G^-1(w))` over the halved weights; the
            // payload `P^digit delta / 2` is bounded by the largest numerator.
            let value = self.normalize_coefficient(wire, value)?;
            let numerators = declaration.member_weights();
            let weights = self.parameters.halved_weights(level + 1, &numerators)?;
            let l1 = self.parameters.gadget_inverse_l1(level + 1, &weights)?;
            let weight = numerators.iter().max().cloned().unwrap_or_default();
            return Ok((wire, value.constant_digit_action(&l1, &weight)));
        }
        Ok((wire, value))
    }
}

impl ProgramLoweringContext for RnsBggNoiseEvaluator<'_> {
    fn replace_site(
        &mut self,
        site: Option<crate::program::LoweringSite>,
    ) -> Option<crate::program::LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

#[cfg(test)]
mod test {
    use super::{
        super::{
            audit::trace_program,
            test::{nested_alignment_parameters, value},
        },
        *,
    };
    use crate::program::ExpLayout;

    #[test]
    fn lut_payload_norms_follow_output_representation_through_private_mul() {
        fn trace_lut(table: LutTable, follow_private_mul: bool) -> RnsBggNoiseTrace {
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
            let lut = builder.register_lut(table).unwrap();
            let lut_output = builder.lut(crate::program::Operand::Scalar(input), lut).unwrap();
            let output = if follow_private_mul {
                let rhs = builder.rhs_input().unwrap();
                builder
                    .mul_private(
                        crate::program::Operand::Scalar(lut_output),
                        crate::program::Operand::Scalar(rhs),
                    )
                    .unwrap()
            } else {
                lut_output
            };
            let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
            let source = DCRTPolyParams::new(16, 3, 17, 4, None, None);
            let moduli = source.to_crt().0;
            let middle = source
                .select_modulus(
                    &moduli[..2]
                        .iter()
                        .fold(BigUint::one(), |product, prime| product * BigUint::from(*prime)),
                )
                .unwrap();
            let terminal = source.select_modulus(&BigUint::from(moduli[0])).unwrap();
            let parameters = RnsBggParameters::new(vec![source, middle, terminal], 1, 1).unwrap();
            let rhs = follow_private_mul
                .then_some(RnsBggNoiseRhs {
                    payload_l1: BigUint::one(),
                    payload_l2_squared: BigUint::one(),
                })
                .into_iter()
                .collect::<Vec<_>>();
            let bindings =
                RnsBggNoiseBindings::from_inputs(&program, &[value(&[1])], &rhs).unwrap();
            trace_program(&parameters, &program, &bindings).unwrap()
        }

        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let exponent = trace_lut(
            LutTable::unary(
                2,
                layout,
                2,
                vec![0, 0],
                crate::program::LutOutputRepresentation::Exponent(layout),
            )
            .unwrap(),
            true,
        );
        let coefficient = trace_lut(
            LutTable::unary(
                2,
                layout,
                3,
                vec![0, 2],
                crate::program::LutOutputRepresentation::Coefficient,
            )
            .unwrap(),
            false,
        );
        assert_eq!(exponent.outputs[0].bound.payload_l1, BigUint::one());
        assert_eq!(exponent.outputs[0].payload_l2_squared, BigUint::one());
        assert_eq!(coefficient.outputs[0].bound.payload_l1, BigUint::from(2u8));
        assert_eq!(coefficient.outputs[0].payload_l2_squared, BigUint::from(4u8));
    }

    fn output_variance(
        program: &ExponentLutProgram,
        parameters: &RnsBggParameters,
        inputs: BTreeMap<WireId, InputBinding<RnsBggNoiseValue, RnsBggNoiseFamily>>,
        rhs: BTreeMap<RhsId, InputBinding<RnsBggNoiseRhs, RnsBggNoiseFamily<RnsBggNoiseRhs>>>,
    ) -> NonnegativeRational {
        let bindings = RnsBggNoiseBindings::new(inputs, rhs).unwrap();
        let trace = trace_program(parameters, program, &bindings).unwrap();
        trace.outputs[0].moments.as_ref().unwrap().variance.clone()
    }

    fn unit_rhs() -> RnsBggNoiseRhs {
        RnsBggNoiseRhs { payload_l1: BigUint::one(), payload_l2_squared: BigUint::one() }
    }

    #[test]
    fn family_sum_keeps_rhs_errors_independent_and_the_shared_input_coherent() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(4)).unwrap();
        let products = builder
            .parallel(FamilyRange::full(IntExpr::constant(4)).unwrap(), |builder, index| {
                builder
                    .mul_private(
                        crate::program::Operand::Scalar(input),
                        crate::program::Operand::Member { family: rhs, index },
                    )
                    .map(crate::program::Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(products).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let parameters = nested_alignment_parameters();
        let encoding = RnsBggNoiseValue::sampled_encoding(&parameters, b"state", &[BigInt::one()]);
        let variance = output_variance(
            &program,
            &parameters,
            BTreeMap::from([(input, InputBinding::Scalar(encoding))]),
            BTreeMap::from([(
                rhs,
                InputBinding::Family(RnsBggNoiseFamily {
                    element: unit_rhs(),
                    range: FamilyRange::full(IntExpr::constant(4)).unwrap(),
                    // One-hot payloads do not remove the other members' errors.
                    max_nonzero_payloads: BigUint::one(),
                    row_length: None,
                }),
            )]),
        );
        let terms = parameters
            .private_mul_noise_terms(
                0,
                &RightActionNoiseGain::unit(),
                &RightActionNoiseGain::unit(),
            )
            .unwrap();
        let inherited = terms.inherited_variance(&parameters.encoding_error_variance());
        // 4² for the shared input and the gate's residual, 4 for the RHS
        // family's distinct samples.
        let expected = inherited
            .scaled(&16u8.into())
            .add(&terms.rhs_error.scaled(&4u8.into()))
            .add(&terms.rounding.scaled(&16u8.into()));
        assert_eq!(variance, expected);
    }

    #[test]
    fn repeated_private_products_share_their_residuals_and_opaque_input_keeps_l1() {
        fn program(repeat: bool) -> (ExponentLutProgram, WireId, RhsId) {
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
            let rhs = builder.rhs_input().unwrap();
            let product = |builder: &mut crate::program::ExponentLutProgramBuilder| {
                builder
                    .mul_private(
                        crate::program::Operand::Scalar(input),
                        crate::program::Operand::Scalar(rhs),
                    )
                    .unwrap()
            };
            let first = product(&mut builder);
            let output = if repeat {
                let second = product(&mut builder);
                builder
                    .add(
                        crate::program::Operand::Scalar(first),
                        crate::program::Operand::Scalar(second),
                    )
                    .unwrap()
            } else {
                first
            };
            let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
            (program, input, rhs)
        }
        let parameters = nested_alignment_parameters();
        let run = |repeat: bool, value: RnsBggNoiseValue| {
            let (program, input, rhs) = program(repeat);
            output_variance(
                &program,
                &parameters,
                BTreeMap::from([(input, InputBinding::Scalar(value))]),
                BTreeMap::from([(rhs, InputBinding::Scalar(unit_rhs()))]),
            )
        };
        let encoding = RnsBggNoiseValue::sampled_encoding(&parameters, b"state", &[BigInt::one()]);
        // Two gates perform the same deterministic product: all of its
        // contributions, including the ModDown remainders, repeat.
        let single = run(false, encoding.clone());
        assert_eq!(run(true, encoding.clone()), single.scaled(&4u8.into()));

        // An envelope without a breakdown is not coefficient-uncorrelated:
        // the inherited error keeps the squared L1 action.
        let mut opaque = encoding;
        opaque.breakdown = None;
        let terms = parameters
            .private_mul_noise_terms(
                0,
                &RightActionNoiseGain::unit(),
                &RightActionNoiseGain::unit(),
            )
            .unwrap();
        let l1 = &terms.inherited_mean_gain;
        let inherited = parameters
            .encoding_error_variance()
            .scaled_ratio(&l1.numerator.pow(2), &l1.denominator.pow(2));
        let fallback = run(false, opaque);
        let below = |left: &NonnegativeRational, right: &NonnegativeRational| {
            &left.numerator * &right.denominator < &right.numerator * &left.denominator
        };
        assert!(below(
            &terms.inherited_variance(&parameters.encoding_error_variance()),
            &inherited
        ));
        assert!(!below(&fallback, &inherited.add(&terms.rhs_error).add(&terms.rounding)));
        assert!(below(&single, &fallback));
    }

    #[test]
    fn aliasing_rhs_reads_share_their_residuals() {
        // `rhs[0]` and `rhs[Select(i, [0, 0])]` read the same RHS member in
        // every lane, so both products are the same deterministic product.
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(2)).unwrap();
        let wires = std::cell::Cell::new(None);
        let family = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, lane| {
                let product = |builder: &mut crate::program::ExponentLutProgramBuilder,
                               index: IntExpr| {
                    builder.mul_private(
                        crate::program::Operand::Scalar(input),
                        crate::program::Operand::Member { family: rhs, index },
                    )
                };
                let direct = product(builder, IntExpr::constant(0))?;
                let aliased = product(
                    builder,
                    IntExpr::Select {
                        selector: Box::new(lane),
                        branches: vec![IntExpr::constant(0), IntExpr::constant(0)],
                    },
                )?;
                let sum = builder.add(
                    crate::program::Operand::Scalar(direct),
                    crate::program::Operand::Scalar(aliased),
                )?;
                wires.set(Some((direct, sum)));
                Ok(crate::program::Operand::Scalar(sum))
            })
            .unwrap();
        let output = builder.sum(family).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let parameters = nested_alignment_parameters();
        let bindings = RnsBggNoiseBindings::new(
            BTreeMap::from([(
                input,
                InputBinding::Scalar(RnsBggNoiseValue::sampled_encoding(
                    &parameters,
                    b"state",
                    &[BigInt::one()],
                )),
            )]),
            BTreeMap::from([(
                rhs,
                InputBinding::Family(RnsBggNoiseFamily {
                    element: unit_rhs(),
                    range: FamilyRange::full(IntExpr::constant(2)).unwrap(),
                    max_nonzero_payloads: BigUint::one(),
                    row_length: None,
                }),
            )]),
        )
        .unwrap();
        let trace = trace_program(&parameters, &program, &bindings).unwrap();
        let (direct, sum) = wires.get().unwrap();
        let variance = |wire| trace.wires[&wire].moments.as_ref().unwrap().variance.clone();
        assert_eq!(variance(sum), variance(direct).scaled(&4u8.into()));
    }

    #[test]
    fn mul_public_energy_cap_applies_to_white_errors_and_independent_members() {
        fn program(family: bool) -> (ExponentLutProgram, WireId, WireId) {
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let state = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
            let public = if family {
                builder
                    .input_family(
                        IntExpr::constant(4),
                        crate::program::AttributeKnowledge::PublicPayload,
                    )
                    .unwrap()
            } else {
                builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap()
            };
            let output = if family {
                let products = builder
                    .parallel(FamilyRange::full(IntExpr::constant(4)).unwrap(), |builder, index| {
                        builder
                            .mul_public(
                                crate::program::Operand::Member { family: public, index },
                                crate::program::Operand::Scalar(state),
                            )
                            .map(crate::program::Operand::Scalar)
                    })
                    .unwrap();
                builder.sum(products).unwrap()
            } else {
                builder
                    .mul_public(
                        crate::program::Operand::Scalar(public),
                        crate::program::Operand::Scalar(state),
                    )
                    .unwrap()
            };
            let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
            (program, state, public)
        }
        let parameters = nested_alignment_parameters();
        let encoding =
            |label: &[u8]| RnsBggNoiseValue::sampled_encoding(&parameters, label, &[BigInt::one()]);
        let v = parameters.encoding_error_variance();
        let gain = parameters.mul_public_action_gain(0).unwrap();
        assert!(gain.l2_squared < gain.l1.pow(2));

        let (scalar, state, public) = program(false);
        let white = output_variance(
            &scalar,
            &parameters,
            BTreeMap::from([
                (state, InputBinding::Scalar(encoding(b"state"))),
                (public, InputBinding::Scalar(encoding(b"a"))),
            ]),
            BTreeMap::new(),
        );
        assert_eq!(white, v.scaled(&gain.l2_squared).add(&v));
        // An envelope without a source-aware breakdown is not white; it keeps
        // the squared L1 action and its unknown covariance with the rest.
        let mut opaque = encoding(b"a");
        opaque.breakdown = None;
        let fallback = output_variance(
            &scalar,
            &parameters,
            BTreeMap::from([
                (state, InputBinding::Scalar(encoding(b"state"))),
                (public, InputBinding::Scalar(opaque)),
            ]),
            BTreeMap::new(),
        );
        assert_eq!(fallback, v.scaled(&gain.l1.pow(2)).sum_variance_bound(&v));

        let (summed, state, public) = program(true);
        let family = |element: RnsBggNoiseValue| {
            InputBinding::Family(RnsBggNoiseFamily {
                element,
                range: FamilyRange::full(IntExpr::constant(4)).unwrap(),
                max_nonzero_payloads: 4u8.into(),
                row_length: None,
            })
        };
        let independent = output_variance(
            &summed,
            &parameters,
            BTreeMap::from([
                (state, InputBinding::Scalar(encoding(b"state"))),
                (public, family(encoding(b"a").independent_family_members())),
            ]),
            BTreeMap::new(),
        );
        assert_eq!(independent, v.scaled(&(&gain.l2_squared * 4u8)).add(&v.scaled(&16u8.into())));
        // Without the binding contract every member is the same sample.
        let shared = output_variance(
            &summed,
            &parameters,
            BTreeMap::from([
                (state, InputBinding::Scalar(encoding(b"state"))),
                (public, family(encoding(b"a"))),
            ]),
            BTreeMap::new(),
        );
        assert_eq!(shared, v.scaled(&(&gain.l2_squared * 16u8)).add(&v.scaled(&16u8.into())));
    }

    /// A wire selection reads an unknown member: the sum of products over
    /// independent members stays coherent, as for a family whose members are
    /// one sample, where member reads by the lane are independent.
    #[test]
    fn wire_selection_reads_an_unknown_member() {
        fn program(lookup: bool) -> (ExponentLutProgram, WireId, WireId) {
            use crate::program::Operand;
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let state = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
            let public = builder
                .input_family(
                    IntExpr::constant(4),
                    crate::program::AttributeKnowledge::PublicPayload,
                )
                .unwrap();
            let data = builder.data_family(4, Some(4)).unwrap();
            let products = builder
                .parallel(FamilyRange::full(IntExpr::constant(4)).unwrap(), |builder, index| {
                    let read = if lookup {
                        Operand::Selected { family: public, data, position: index }
                    } else {
                        Operand::Member { family: public, index }
                    };
                    builder.mul_public(read, Operand::Scalar(state)).map(Operand::Scalar)
                })
                .unwrap();
            let output = builder.sum(products).unwrap();
            let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
            (program, state, public)
        }
        let parameters = nested_alignment_parameters();
        let encoding =
            |label: &[u8]| RnsBggNoiseValue::sampled_encoding(&parameters, label, &[BigInt::one()]);
        let v = parameters.encoding_error_variance();
        let gain = parameters.mul_public_action_gain(0).unwrap();
        let variance = |lookup: bool| {
            let (program, state, public) = program(lookup);
            output_variance(
                &program,
                &parameters,
                BTreeMap::from([
                    (state, InputBinding::Scalar(encoding(b"state"))),
                    (
                        public,
                        InputBinding::Family(RnsBggNoiseFamily {
                            element: encoding(b"a").independent_family_members(),
                            range: FamilyRange::full(IntExpr::constant(4)).unwrap(),
                            max_nonzero_payloads: 4u8.into(),
                            row_length: None,
                        }),
                    ),
                ]),
                BTreeMap::new(),
            )
        };
        assert_eq!(
            variance(false),
            v.scaled(&(&gain.l2_squared * 4u8)).add(&v.scaled(&16u8.into()))
        );
        assert_eq!(
            variance(true),
            v.scaled(&(&gain.l2_squared * 16u8)).add(&v.scaled(&16u8.into()))
        );
    }

    #[test]
    fn lut_material_sources_reuse_a_and_f_but_split_lut_specific_b() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let first_table = builder
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let second_table = builder
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![1, 0],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let output = builder
            .add(crate::program::Operand::Scalar(input), crate::program::Operand::Scalar(input))
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        // Material at evaluation level 1 also needs its decomposition level 2;
        // keep that level in the fixture so the test exercises level-keyed
        // sharing rather than failing parameter validation first.
        let parameters =
            RnsBggParameters::new(vec![level.clone(), level.clone(), level], 1, 1).unwrap();
        let bindings = RnsBggNoiseBindings::default();
        let evaluator = RnsBggNoiseEvaluator::new(&parameters, &program, &workload, &bindings);
        let projection = program.encryption_inputs();
        let first = evaluator
            .lut_material_provenance(
                0,
                first_table,
                &projection.luts()[first_table.0 as usize],
                true,
            )
            .unwrap();
        let same = evaluator
            .lut_material_provenance(
                0,
                first_table,
                &projection.luts()[first_table.0 as usize],
                true,
            )
            .unwrap();
        let different = evaluator
            .lut_material_provenance(
                0,
                second_table,
                &projection.luts()[second_table.0 as usize],
                true,
            )
            .unwrap();
        let different_level = evaluator
            .lut_material_provenance(
                1,
                first_table,
                &projection.luts()[first_table.0 as usize],
                true,
            )
            .unwrap();

        assert_eq!(first, same);
        assert!(!first.is_empty());
        assert!(first.keys().any(|source| !different.contains_key(source)));
        assert!(
            first.keys().filter(|source| different.contains_key(*source)).count() < first.len()
        );
        assert!(first.keys().all(|source| !different_level.contains_key(source)));
    }

    #[test]
    fn coefficient_output_finalizes_once_and_reuses_duplicate_declarations() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let table = builder
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let output = builder.lut(crate::program::Operand::Scalar(input), table).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 2).unwrap();
        let bindings = RnsBggNoiseBindings::default();
        let mut evaluator = RnsBggNoiseEvaluator::new(&parameters, &program, &workload, &bindings);
        let outputs =
            evaluator.finish_outputs(vec![(output, value(&[1])), (output, value(&[1]))]).unwrap();
        // The aggregated pre-rounding noise is scaled and rounded once, then
        // scaled up by the `G^-1(k)` action, whose gain is its digit l1 norm.
        let (_, k, _) = parameters.coefficient_output_decoding(&program, output, 1).unwrap();
        let (l1, _) = parameters.gadget_inverse_norms(1, &k).unwrap();
        let expected = parameters.lut_mod_down_bound(0, &value(&[1]).bound.error).unwrap() * l1;

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].bound.error, expected);
        assert_eq!(outputs[0].bound, outputs[1].bound);
        assert_eq!(evaluator.trace.joint_event_count, BigUint::one());
        assert_eq!(outputs[0].provenance.len(), 1);
        assert_eq!(outputs[0].provenance, outputs[1].provenance);
    }
}
