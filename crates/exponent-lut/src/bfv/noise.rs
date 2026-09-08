//! Deterministic bounds and conditional moments for the common Program.
use super::{BfvError, BfvNoiseBound, BfvParameterManifest, BfvParameters, BfvPlanRequest};
use crate::{
    noise_refresh::{
        refresh::{DigitPlanError, RefreshDigitPlan},
        simulation::{AverageCaseConfig, NoiseModelKind},
    },
    noise_utils::{NoiseMoments, NonnegativeRational, RightActionNoiseGain},
    program::{
        ExponentLutProgram, ExponentProgramLowering, FamilyRange, LutId, ProgramBindings,
        ProgramFamilyBindings, ProgramGateInstance, ProgramLowerError, ProgramLoweringTypes,
        WireId,
    },
};
use mxx_ir_core::ParamEnv;
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

impl BfvParameters {
    /// Physical-coordinate variance transfer for the direct product's four
    /// error terms. Conditional model: deviations from the coefficient means
    /// are uncorrelated with a common variance bound, fixed RHS errors are CBD(eta)
    /// and independent, and both ModDown remainders are independent centered
    /// uniforms, independent of inherited/fresh errors and the ternary secret.
    /// The prepared right action is fixed under H4's conditional model; its
    /// action on a nonzero input mean is accounted for separately as bias.
    /// These are average-case heuristics, not deterministic independence claims.
    /// Reused error sources must have their linear actions combined first.
    /// `payload_gain` bounds the actual reduced input plaintext norms.
    pub fn private_mul_noise_variance(
        &self,
        level: usize,
        input_variance: &NonnegativeRational,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<NonnegativeRational, BfvError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let p_squared = p * p;
        // e*C/p: two centered polynomial entries per output column, each
        // with N coefficients bounded by (q-1)/2. No ciphertext scan is needed.
        let inherited = NonnegativeRational::new(
            &input_variance.numerator * &n * (next.modulus().as_ref() - 1u8).pow(2),
            &input_variance.denominator * 2u8 * &p_squared,
        )
        .expect("positive scale and variance denominator");
        // -mu*sum(a_j*F_j): combine both deterministic actions on each F_j.
        // Weighted F coefficients are correlated, so multiplying squared l2
        // norms is not valid. Young's two convolution bounds give a cap using
        // the retained norms. The sum of squared l1 gains is bounded by the
        // square of their sum. Monomial payloads preserve the exact fresh gain.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        let fresh = NonnegativeRational::new(BigUint::from(self.eta()) * fresh_gain, 2u8.into())
            .expect("positive constant variance denominator");
        // s*rho(A*C)/p - rho(c*C)/p: E||[s_bar,1]||_2^2 = 2N/3+1.
        // Adding the online remainder gives (2N/3+2)*(p^2-1)/(12*p^2).
        let rounding = NonnegativeRational::new((&n + 3u8) * (&p_squared - 1u8), 18u8 * &p_squared)
            .expect("positive scale");
        Ok(inherited.add(&fresh).add(&rounding))
    }

    /// Physical variance of the explicit LUT equation (6.6). In addition to
    /// the direct-product coefficient/fresh/remainder model above, this uses
    /// the flat-branch decorrelation heuristic (H3). It does not apply to an
    /// explicit Add, a repeated invocation sharing helpers, or a raw terminal
    /// reduction: those must retain their shared sources and outer rounding.
    /// Each mask helper is sampled with CBD(eta) at the input level. Branch
    /// fresh gains describe ciphertext weighting, not fresh plaintext targets.
    pub fn lut_noise_variance(
        &self,
        level: usize,
        input_variance: &NonnegativeRational,
        branches: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<NonnegativeRational, BfvError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let j = BigUint::from(branches);
        let p_squared = p * p;
        let cbd = NonnegativeRational::new(self.eta().into(), 2u8.into())
            .expect("positive constant variance denominator");
        // sum_k (sigma_k(e)*C_k + ed_k*B_k)/p. C_k and B_k are
        // centered two-by-two blocks in q, with the same coefficient cap.
        let input_and_helper = input_variance.add(&cbd);
        let inherited_and_helper = NonnegativeRational::new(
            &input_and_helper.numerator * &j * &n * (next.modulus().as_ref() - 1u8).pow(2),
            &input_and_helper.denominator * 2u8 * &p_squared,
        )
        .expect("positive scale and variance denominator");
        // Automorphisms preserve both norms. Combine deterministic actions on
        // each F_k before estimating convolution, just as for the direct gate.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        let fresh = cbd.scaled(&(&j * fresh_gain));
        // J internal public remainders, one outer public remainder, and one
        // outer encoding remainder. E||[s_bar,1]||_2^2 = (2N+3)/3.
        // The two outer terms occur once, after the branch sum.
        let rounding = NonnegativeRational::new(
            ((&j + 1u8) * (&n * 2u8 + 3u8) + 3u8) * (&p_squared - 1u8),
            36u8 * &p_squared,
        )
        .expect("positive scale");
        Ok(inherited_and_helper.add(&fresh).add(&rounding))
    }

    /// Direct product: e*C/p - mu*F + s*rho(A*C)/p - rho(c*C)/p.
    /// K(C) <= N*(q-1) for a centered two-by-two polynomial block. A composed
    /// fixed RHS contributes all weighted fresh errors, even for one-hot payloads.
    pub fn private_mul_noise_bound(
        &self,
        level: usize,
        input: &BfvNoiseBound,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<BfvNoiseBound, BfvError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let twice_p = p * 2u8;
        let inherited = &input.error * &n * (next.modulus().as_ref() - 1u8) * 2u8;
        let fresh = &input.payload_l1 * fresh_error_gain * self.eta() * &twice_p;
        // S = ||s_bar||_1 + 1 <= N + 1 for the ternary secret.
        let rounding = (n + 2u8) * (p - 1u8);
        Ok(BfvNoiseBound {
            error: (inherited + fresh + rounding + &twice_p - 1u8) / twice_p,
            payload_l1: output_payload_l1,
        })
    }
}

/// A family envelope applies to each active member, never to its sum.
/// `one_hot` is an explicit trusted payload contract; it never suppresses noise.
#[derive(Clone, Debug)]
pub struct BfvNoiseValue {
    pub level: usize,
    pub bound: BfvNoiseBound,
    /// Squared coefficient norm of the reduced plaintext, bounded separately
    /// from l1 so public polynomial weights do not become monomial estimates.
    pub payload_l2_squared: BigUint,
    /// Physical moments under the stated average-case model. None means that
    /// no moment model is available, not zero variance. Nonzero coefficient
    /// means propagate as intervals through prepared linear actions. The
    /// centered coefficient/RHS/branch/remainder variance heuristics apply to
    /// deviations from these means, not to their deterministic bias.
    pub moments: Option<NoiseMoments>,
    pub public: bool,
    pub range: Option<FamilyRange>,
    pub one_hot: bool,
}

impl BfvNoiseValue {
    /// Known public coefficients provide the convolution gain directly. The
    /// coefficients are the chosen integer lift, with negacyclic terms reduced
    /// and combined before constructing this value.
    pub fn public(level: usize, error_bound: BigUint, coefficients: &[BigInt]) -> Self {
        Self {
            level,
            moments: error_bound.is_zero().then(|| NoiseMoments {
                mean_numerator: BigInt::zero(),
                mean_denominator: BigUint::one(),
                mean_radius: NonnegativeRational::zero(),
                variance: NonnegativeRational::zero(),
            }),
            payload_l2_squared: coefficients.iter().map(|c| c.magnitude().pow(2)).sum(),
            bound: BfvNoiseBound {
                error: error_bound,
                payload_l1: coefficients.iter().map(|c| c.magnitude()).sum(),
            },
            public: true,
            range: None,
            one_hot: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BfvNoiseRhs {
    pub payload_l1: BigUint,
    pub payload_l2_squared: BigUint,
    /// The setup contract guarantees at most one nonzero coefficient-selector
    /// operand across precisely this active family range.
    pub one_hot: Option<FamilyRange>,
}

#[derive(Clone, Debug)]
pub struct BfvGateNoiseMaterial {
    pub level: usize,
    pub table: Option<[u8; 32]>,
    /// Sum of active public multiplier l1 bounds in each composed branch.
    /// Zero selectors still contribute; zero public multipliers do not.
    /// The squared gain sums coefficient squares across separately sampled
    /// RHS errors; it is not the square of the total l1 gain.
    pub fresh_error_gain: RightActionNoiseGain,
}

impl BfvGateNoiseMaterial {
    /// Setup-time contract for coefficient selectors with at most one nonzero
    /// active member. Bounds refer to reduced public multiplier polynomials.
    /// The selected payload uses the maximum gain; all ciphertext errors use
    /// their sum. The construction contract uses distinct fresh RHS errors;
    /// it must combine weights of a reused error source before forming gains.
    /// No secret selector is inspected and no runtime check is added.
    pub fn from_one_hot_labels(
        level: usize,
        table: Option<[u8; 32]>,
        multipliers: &[RightActionNoiseGain],
    ) -> Result<(Self, BfvNoiseRhs), BfvError> {
        let payload_l1 = multipliers
            .iter()
            .map(|gain| &gain.l1)
            .max()
            .ok_or(BfvError::Parameters("empty RHS composition"))?
            .clone();
        Ok((
            Self {
                level,
                table,
                fresh_error_gain: RightActionNoiseGain {
                    l1: multipliers.iter().map(|gain| &gain.l1).sum(),
                    l2_squared: multipliers.iter().map(|gain| &gain.l2_squared).sum(),
                },
            },
            BfvNoiseRhs {
                payload_l1,
                payload_l2_squared: multipliers
                    .iter()
                    .map(|gain| &gain.l2_squared)
                    .max()
                    .unwrap()
                    .clone(),
                one_hot: None,
            },
        ))
    }
}

pub struct BfvNoiseTrace {
    /// None for a raw trace; set only by a successful full Program audit.
    pub acceptance: Option<NoiseModelKind>,
    pub joint_event_count: BigUint,
    /// Includes the configured tail correction, in physical coordinates.
    pub z_squared: Option<NonnegativeRational>,
    pub dependencies: crate::noise_utils::NoiseDependencyGraph,
    pub wires: BTreeMap<WireId, BfvNoiseValue>,
    pub gates: BTreeMap<ProgramGateInstance, BfvNoiseValue>,
}

pub fn trace_program(
    parameters: &BfvParameters,
    program: &ExponentLutProgram,
    bindings: &ProgramBindings<'_, BfvNoiseValue, BfvNoiseRhs, BfvGateNoiseMaterial>,
    ranges: &ProgramFamilyBindings,
) -> Result<BfvNoiseTrace, ProgramLowerError<BfvError>> {
    let (wires, evaluator, dependencies) =
        crate::scheme::trace_noise::<super::BfvScheme>(program, parameters, bindings, ranges)?;
    Ok(BfvNoiseTrace {
        dependencies,
        wires,
        gates: evaluator.gates,
        acceptance: None,
        joint_event_count: BigUint::zero(),
        z_squared: None,
    })
}

impl BfvParameters {
    /// Search public prime candidates using the full Program audit. The
    /// request's terminal depth is the lower bound; its CRT/memory budgets
    /// remain hard limits on the declared allocation profile. Width/slot
    /// counts must cover the workload's storage contract; this is not yet a
    /// liveness-based peak-memory scheduler. The returned manifest describes allocation,
    /// while the returned trace is the actual Program acceptance evidence.
    /// This conservative search does not certify globally minimal parameters.
    pub fn search_program(
        request: &super::BfvPlanRequest,
        program: &ExponentLutProgram,
        bindings: &ProgramBindings<'_, BfvNoiseValue, BfvNoiseRhs, BfvGateNoiseMaterial>,
        ranges: &ProgramFamilyBindings,
        terminal_value_bounds: &BTreeMap<WireId, BigUint>,
        average: Option<&AverageCaseConfig>,
    ) -> Result<(Self, super::BfvParameterManifest, BfvNoiseTrace), BfvError> {
        let mut candidate = request.clone();
        loop {
            let attempt = Self::plan(&candidate).and_then(|(parameters, manifest)| {
                let trace = parameters.audit_program(
                    program,
                    bindings,
                    ranges,
                    terminal_value_bounds,
                    average,
                )?;
                Ok((parameters, manifest, trace))
            });
            match attempt {
                Ok(accepted) => return Ok(accepted),
                Err(BfvError::Budget("evaluation noise" | "terminal range")) => {
                    if candidate.terminal_depth >= candidate.max_crt_depth {
                        return Err(BfvError::Budget("CRT depth"));
                    }
                    candidate.terminal_depth += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Setup-time arithmetic audit of the actual Program DAG. Bounds describe
    /// ideal terminal integer outputs, including their scale/row target where
    /// applicable; this does not assert a decoder or a security certificate.
    /// Every declared output needs exactly one bound, and its own level is
    /// used. This pass reads public contracts only, never runtime ciphertexts.
    /// Average-only acceptance requires the explicit config opt-in and all
    /// input moment models. It uses H1/H3/H4, centered remainder models and
    /// H6 Gaussian-tail closure, with one union budget for the audited Program
    /// coordinates. This is not a budget for a complete refresh protocol.
    pub fn audit_program(
        &self,
        program: &ExponentLutProgram,
        bindings: &ProgramBindings<'_, BfvNoiseValue, BfvNoiseRhs, BfvGateNoiseMaterial>,
        ranges: &ProgramFamilyBindings,
        terminal_value_bounds: &BTreeMap<WireId, BigUint>,
        average: Option<&AverageCaseConfig>,
    ) -> Result<BfvNoiseTrace, BfvError> {
        let outputs = program.outputs().iter().copied().collect::<std::collections::BTreeSet<_>>();
        if terminal_value_bounds.keys().copied().collect::<std::collections::BTreeSet<_>>() !=
            outputs
        {
            return Err(BfvError::Parameters("terminal bounds must match declared Program outputs"));
        }
        let mut trace = trace_program(self, program, bindings, ranges).map_err(|e| match e {
            ProgramLowerError::Operation { source, .. } => source,
            ProgramLowerError::Validation(error) => BfvError::Graph(error.to_string()),
            ProgramLowerError::Graph(error) => BfvError::Graph(error.to_string()),
        })?;
        // Deduplicate scalar wires and gate results by their common dependency
        // node. A captured wire is one source; family gate coordinates still
        // contribute one event per active member and polynomial coefficient.
        let mut values = BTreeMap::new();
        for (wire, value) in &trace.wires {
            values.insert(trace.dependencies.wires[wire].0, value);
        }
        for (index, node) in trace.dependencies.nodes.iter().enumerate() {
            use crate::noise_utils::NoiseDependencyNode as Node;
            let gate = match node {
                Node::Binding(_) => continue,
                Node::Add { gate, .. } |
                Node::MulPublic { gate, .. } |
                Node::MulPrivate { gate, .. } |
                Node::Lut { gate, .. } |
                Node::Reduction { gate, .. } => gate,
            };
            values.insert(index, &trace.gates[gate]);
        }
        let mut events = BigUint::zero();
        for value in values.values() {
            let count = match &value.range {
                None => BigUint::one(),
                Some(range) => range
                    .count()
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|count| count.to_biguint())
                    .ok_or(BfvError::Parameters("noise audit requires concrete active ranges"))?,
            };
            events += count * self.level(value.level)?.ring_dimension() * 2u8;
        }
        let average = average.filter(|config| config.allow_average_acceptance);
        let z_squared = average
            .map(|config| {
                let z = config
                    .z_squared_for_events(&events)
                    .map_err(|_| BfvError::Parameters("invalid average-case event budget"))?;
                let correction = config
                    .tail_correction_bits
                    .checked_mul(2)
                    .ok_or(BfvError::Parameters("invalid tail correction"))?;
                Ok::<_, BfvError>(z.scaled(&(BigUint::one() << correction)))
            })
            .transpose()?;
        for value in values.values() {
            let fits = match &z_squared {
                None => self.noise_fits(value.level, &value.bound)?,
                Some(z) => {
                    let moments = value.moments.as_ref().ok_or(BfvError::Parameters(
                        "average-case input moment model is missing",
                    ))?;
                    let modulus_limit = NonnegativeRational::new(
                        self.level(value.level)?.modulus().as_ref().clone(),
                        8u8.into(),
                    )
                    .expect("positive constant denominator");
                    moments.fits(&modulus_limit, z) &&
                        (value.level + 1 == self.levels.len() ||
                            moments.fits(
                                &NonnegativeRational::new(
                                    self.scale(value.level)?.clone(),
                                    2u8.into(),
                                )
                                .expect("positive constant denominator"),
                                z,
                            ))
                }
            };
            if !fits {
                return Err(BfvError::Budget("evaluation noise"));
            }
        }
        for (wire, value_bound) in terminal_value_bounds {
            let value = trace.wires.get(wire).ok_or(BfvError::Context)?;
            let modulus = self.level(value.level)?.modulus();
            let modulus = modulus.as_ref();
            let payload = value_bound * 2u8;
            let fits = match &z_squared {
                None => modulus > &(&payload + &value.bound.error * 2u8),
                Some(z) => {
                    modulus > &payload &&
                        value.moments.as_ref().expect("audited moments").fits(
                            &NonnegativeRational::new(modulus - payload, 2u8.into())
                                .expect("positive constant denominator"),
                            z,
                        )
                }
            };
            if !fits {
                return Err(BfvError::Budget("terminal range"));
            }
        }
        trace.acceptance = Some(if average.is_some() {
            NoiseModelKind::AverageCase
        } else {
            NoiseModelKind::WorstCase
        });
        trace.joint_event_count = events;
        trace.z_squared = z_squared;
        Ok(trace)
    }
}

pub struct BfvNoiseEvaluator<'a> {
    pub(super) parameters: &'a BfvParameters,
    pub(super) ranges: &'a ProgramFamilyBindings,
    pub(super) gates: BTreeMap<ProgramGateInstance, BfvNoiseValue>,
}

impl BfvNoiseEvaluator<'_> {
    // Captured scalars keep their dependency identity. Only the transfer's
    // output acquires the current member range; no new source is introduced.
    fn gate_range(
        &self,
        gate: ProgramGateInstance,
        inputs: &[&BfvNoiseValue],
    ) -> Result<Option<FamilyRange>, BfvError> {
        let range = match gate {
            ProgramGateInstance::FamilyGate { family, .. } => {
                Some(self.ranges.range(family).ok_or(BfvError::Context)?.clone())
            }
            _ => None,
        };
        if inputs.iter().any(|input| input.range.is_some() && input.range != range) {
            return Err(BfvError::Context);
        }
        Ok(range)
    }

    fn record(
        &mut self,
        gate: ProgramGateInstance,
        value: BfvNoiseValue,
    ) -> Result<BfvNoiseValue, BfvError> {
        self.gates.insert(gate, value.clone());
        Ok(value)
    }
}

impl ProgramLoweringTypes for BfvNoiseEvaluator<'_> {
    type Wire = BfvNoiseValue;
    type Rhs = BfvNoiseRhs;
    type HelperSet = BfvGateNoiseMaterial;
    type Error = BfvError;
}

impl ExponentProgramLowering for BfvNoiseEvaluator<'_> {
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        material: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[input])?;
        let material = material.ok_or(BfvError::Context)?;
        if material.level != input.level || material.table.is_some() {
            return Err(BfvError::Context);
        }
        if let Some(rhs_range) = &rhs.one_hot {
            if range.as_ref() != Some(rhs_range) || rhs.payload_l1 > BigUint::one() {
                return Err(BfvError::Context);
            }
        }
        let moments = input
            .moments
            .as_ref()
            .map(|m| {
                let variance = self.parameters.private_mul_noise_variance(
                    input.level,
                    &m.variance,
                    &RightActionNoiseGain {
                        l1: input.bound.payload_l1.clone(),
                        l2_squared: input.payload_l2_squared.clone(),
                    },
                    &material.fresh_error_gain,
                )?;
                let next = self.parameters.level(input.level + 1)?;
                let l1 = NonnegativeRational::new(
                    BigUint::from(next.ring_dimension()) * (next.modulus().as_ref() - 1u8),
                    self.parameters.scale(input.level)?.clone(),
                )
                .expect("positive scale");
                Ok::<_, BfvError>(m.linear_image(&l1, variance))
            })
            .transpose()?;
        let bound = self.parameters.private_mul_noise_bound(
            input.level,
            &input.bound,
            &material.fresh_error_gain.l1,
            &input.bound.payload_l1 * &rhs.payload_l1,
        )?;
        self.record(
            gate,
            BfvNoiseValue {
                level: input.level + 1,
                payload_l2_squared: (input.bound.payload_l1.pow(2) * &rhs.payload_l2_squared)
                    .min(&input.payload_l2_squared * rhs.payload_l1.pow(2)),
                moments,
                bound,
                public: false,
                range,
                one_hot: (input.range.is_some() && input.one_hot) || rhs.one_hot.is_some(),
            },
        )
    }

    fn mul_public(
        &mut self,
        private: &Self::Wire,
        public: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[private, public])?;
        if private.level != public.level || !public.public {
            return Err(BfvError::Context);
        }
        self.record(
            gate,
            BfvNoiseValue {
                level: private.level,
                payload_l2_squared: (private.bound.payload_l1.pow(2) * &public.payload_l2_squared)
                    .min(&private.payload_l2_squared * public.bound.payload_l1.pow(2)),
                moments: private.moments.as_ref().map(|m| {
                    // A public convolution can correlate coefficients. l1^2
                    // remains valid for repeated actions without resetting H1.
                    m.linear_image(
                        &NonnegativeRational::new(public.bound.payload_l1.clone(), BigUint::one())
                            .expect("positive constant denominator"),
                        m.variance.scaled(&public.bound.payload_l1.pow(2)),
                    )
                }),
                bound: private.bound.mul_public(&public.bound.payload_l1),
                public: private.public,
                range,
                one_hot: private.range.is_some() && private.one_hot,
            },
        )
    }

    fn add(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[lhs, rhs])?;
        if lhs.level != rhs.level {
            return Err(BfvError::Context);
        }
        self.record(
            gate,
            BfvNoiseValue {
                level: lhs.level,
                payload_l2_squared: (&lhs.bound.payload_l1 + &rhs.bound.payload_l1)
                    .pow(2)
                    .min((&lhs.payload_l2_squared + &rhs.payload_l2_squared) * 2u8),
                moments: lhs
                    .moments
                    .as_ref()
                    .zip(rhs.moments.as_ref())
                    .map(|(left, right)| left.add(right)),
                bound: lhs.bound.add(&rhs.bound),
                public: lhs.public && rhs.public,
                range,
                // Two independent one-hot families need not select the same slot.
                one_hot: false,
            },
        )
    }

    fn lut(
        &mut self,
        program: &ExponentLutProgram,
        lut: LutId,
        input: &Self::Wire,
        helpers: Option<&Self::HelperSet>,
        rhs: Option<&Self::Rhs>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[input])?;
        let material = helpers.ok_or(BfvError::Lut)?;
        use mxx_primitives::poly::PolyParams;
        let layout = program.layout().ok_or(BfvError::Lut)?;
        if layout.exponent.stride_log2 != 0 ||
            layout.ring_dimension_log2 !=
                self.parameters.level(input.level)?.ring_dimension().trailing_zeros()
        {
            return Err(BfvError::Parameters("BFV requires an unstrided exponent layout"));
        }
        let table = program.lut(lut).ok_or(BfvError::Lut)?;
        if material.level != input.level || material.table != Some(table.commitment()) {
            return Err(BfvError::Context);
        }
        if rhs.is_some_and(|rhs| rhs.one_hot.is_some()) {
            return Err(BfvError::Context);
        }
        let rhs_l1 = rhs.map_or_else(BigUint::one, |rhs| rhs.payload_l1.clone());
        // A linear table map sends each supported input monomial to a unit
        // monomial. Triangle inequality preserves this product l1 bound even
        // when table entries collide. Identity works for the entire ring.
        let output_l1 = &input.bound.payload_l1 * rhs_l1;
        let moments = input
            .moments
            .as_ref()
            .map(|m| {
                let variance = self.parameters.lut_noise_variance(
                    input.level,
                    &m.variance,
                    table.values().len(),
                    &RightActionNoiseGain {
                        l1: input.bound.payload_l1.clone(),
                        l2_squared: input.payload_l2_squared.clone(),
                    },
                    &material.fresh_error_gain,
                )?;
                let next = self.parameters.level(input.level + 1)?;
                let l1 = NonnegativeRational::new(
                    BigUint::from(table.values().len()) *
                        BigUint::from(next.ring_dimension()) *
                        (next.modulus().as_ref() - 1u8),
                    self.parameters.scale(input.level)?.clone(),
                )
                .expect("positive scale");
                Ok::<_, BfvError>(m.linear_image(&l1, variance))
            })
            .transpose()?;
        let bound = self.parameters.lut_noise_bound(
            input.level,
            &input.bound,
            table.values().len(),
            &material.fresh_error_gain.l1,
            output_l1,
        )?;
        self.record(
            gate,
            BfvNoiseValue {
                level: input.level + 1,
                // Different table entries may collide: retain the l1-based
                // cap until the table domain proves a tighter squared norm.
                payload_l2_squared: bound.payload_l1.pow(2),
                moments,
                bound,
                public: input.public && rhs.is_none(),
                range,
                one_hot: (input.range.is_some() && input.one_hot) ||
                    rhs.is_some_and(|rhs| rhs.one_hot.is_some()),
            },
        )
    }
}

impl BfvNoiseEvaluator<'_> {
    pub(super) fn reduce_add(
        &mut self,
        value: &BfvNoiseValue,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<BfvNoiseValue, BfvError> {
        if value.range.as_ref().is_some_and(|bound_range| bound_range != range) {
            return Err(BfvError::Context);
        }
        let count = range
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|count| count.to_biguint())
            .filter(|count| *count > 0u8.into())
            .ok_or(BfvError::Parameters("noise trace needs a concrete active family count"))?;
        self.record(
            gate,
            BfvNoiseValue {
                level: value.level,
                payload_l2_squared: if value.range.is_some() && value.one_hot {
                    value.payload_l2_squared.clone()
                } else {
                    &value.payload_l2_squared * count.pow(2)
                },
                moments: value.moments.as_ref().map(|m| m.scaled(&count)),
                bound: BfvNoiseBound {
                    error: &value.bound.error * &count,
                    payload_l1: if value.range.is_some() && value.one_hot {
                        value.bound.payload_l1.clone()
                    } else {
                        &value.bound.payload_l1 * count
                    },
                },
                public: value.public,
                range: None,
                one_hot: false,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bfv::BfvPlanRequest,
        layout::ExpLayout,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, ProgramFamilyId, ProgramWireType,
        },
    };

    #[test]
    fn test_bfv_composed_noise_uses_sum_for_errors_and_max_for_payload() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = BfvParameters::plan(&BfvPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![1],
            eta: 2,
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 3u8.into(),
        })
        .unwrap();
        let gains = [
            RightActionNoiseGain::from_polynomial_coefficients([2.into(), (-1).into()]),
            RightActionNoiseGain::from_polynomial_coefficients([2.into()]),
            RightActionNoiseGain::from_polynomial_coefficients([0.into()]),
        ];
        let (material, rhs) = BfvGateNoiseMaterial::from_one_hot_labels(0, None, &gains).unwrap();
        assert_eq!(material.fresh_error_gain.l1, BigUint::from(5u8));
        assert_eq!(material.fresh_error_gain.l2_squared, BigUint::from(9u8));
        assert_eq!(rhs.payload_l1, BigUint::from(3u8));
        // Enumerate eta=2 CBD errors (four independent bits each), using
        // three independent coefficients weighted by 2, -1, and 2. This is
        // the fresh contribution of the two nonzero multiplier polynomials.
        let cbd = |bits: u32| -> i64 {
            (bits & 3).count_ones() as i64 - ((bits >> 2) & 3).count_ones() as i64
        };
        let mut square_sum = 0u64;
        let mut mean_sum = 0i64;
        for bits in 0u32..4096 {
            let error = 2 * cbd(bits) - cbd(bits >> 4) + 2 * cbd(bits >> 8);
            mean_sum += error;
            square_sum += (error * error) as u64;
        }
        assert_eq!(mean_sum, 0);
        let fresh_variance = NonnegativeRational::new(square_sum.into(), 4096u32.into()).unwrap();
        let inherited = NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap();
        let without_fresh = parameters
            .private_mul_noise_variance(
                0,
                &inherited,
                &RightActionNoiseGain::unit(),
                &RightActionNoiseGain::from_polynomial_coefficients([0.into()]),
            )
            .unwrap();
        let with_fresh = parameters
            .private_mul_noise_variance(
                0,
                &inherited,
                &RightActionNoiseGain::unit(),
                &material.fresh_error_gain,
            )
            .unwrap();
        assert_eq!(with_fresh, without_fresh.add(&fresh_variance));
        // The same independently enumerated CBD contribution appears once
        // per explicit LUT branch; one-hot payloads do not erase RHS errors.
        for branches in [1usize, 2, 4] {
            let zero = RightActionNoiseGain::from_polynomial_coefficients([0.into()]);
            let without = parameters
                .lut_noise_variance(0, &inherited, branches, &RightActionNoiseGain::unit(), &zero)
                .unwrap();
            let with = parameters
                .lut_noise_variance(
                    0,
                    &inherited,
                    branches,
                    &RightActionNoiseGain::unit(),
                    &material.fresh_error_gain,
                )
                .unwrap();
            assert_eq!(with, without.add(&fresh_variance.scaled(&branches.into())));
        }
        // Repeated polynomial actions reuse coefficient errors within a
        // source: (1+X)^2 has coefficients 1,2,1 and variance gain 6, not 4.
        let polynomial = RightActionNoiseGain::from_polynomial_coefficients([1.into(), 1.into()]);
        let mut composed_squares = 0u64;
        for bits in 0u32..4096 {
            let error = cbd(bits) + 2 * cbd(bits >> 4) + cbd(bits >> 8);
            composed_squares += (error * error) as u64;
        }
        let composed_variance =
            NonnegativeRational::new(composed_squares.into(), 4096u32.into()).unwrap();
        assert_eq!(composed_variance, NonnegativeRational::new(6u8.into(), 1u8.into()).unwrap());
        let combined =
            parameters.private_mul_noise_variance(0, &inherited, &polynomial, &polynomial).unwrap();
        let actual = without_fresh.add(&composed_variance);
        assert!(
            &combined.numerator * &actual.denominator >= &actual.numerator * &combined.denominator
        );
        // With an unweighted fresh error, the other Young bound preserves
        // the exact squared-l2 payload gain instead of using its squared l1.
        let unweighted = parameters
            .private_mul_noise_variance(0, &inherited, &polynomial, &RightActionNoiseGain::unit())
            .unwrap();
        assert_eq!(
            unweighted,
            without_fresh.add(&NonnegativeRational::new(2u8.into(), 1u8.into()).unwrap())
        );
        assert!(rhs.one_hot.is_none(), "the composed RHS is a scalar payload");
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let input_id = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let family = builder.rhs_family(2).unwrap();
        let rhs_id = builder.rhs_input(family, 2, 2).unwrap();
        let output = builder.private_mul(builder.input_wire(input_id).unwrap(), rhs_id).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let input = BfvNoiseValue {
            level: 0,
            bound: BfvNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
            payload_l2_squared: 1u8.into(),
            moments: Some(NoiseMoments {
                mean_numerator: BigInt::zero(),
                mean_denominator: BigUint::one(),
                mean_radius: NonnegativeRational::zero(),
                variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
            }),
            public: false,
            range: None,
            one_hot: false,
        };
        let gate = ProgramGateInstance::Gate { step: 0 };
        let inputs = BTreeMap::from([(input_id, input.clone())]);
        let rhs = BTreeMap::from([(rhs_id, rhs)]);
        let helpers = BTreeMap::from([(gate, material)]);
        let trace = trace_program(
            &parameters,
            &program,
            &ProgramBindings::new(
                &inputs,
                &rhs,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                Some(&helpers),
            ),
            &ProgramFamilyBindings::new(),
        )
        .unwrap();
        let result = &trace.wires[&output].bound;
        assert_eq!(result.payload_l1, BigUint::from(3u8));
        assert_eq!(trace.wires[&output].payload_l2_squared, BigUint::from(5u8));
        assert_eq!(trace.wires[&output].moments.as_ref().unwrap().variance, with_fresh);
        let single =
            parameters.private_mul_noise_bound(0, &input.bound, &1u8.into(), 3u8.into()).unwrap();
        // Only the fresh term changes: J * ||mu||_1 * (5 - 1) * eta = 8.
        // The exact rounding terms and inherited/mask error remain unchanged.
        assert_eq!(result.error, single.error + 8u8);
        assert!(BfvGateNoiseMaterial::from_one_hot_labels(0, None, &[]).is_err());
    }

    #[test]
    fn test_bfv_noise_trace_tracks_public_coefficients_and_one_hot_payload() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = BfvParameters::plan(&BfvPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![1],
            eta: 2,
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 1u8.into(),
        })
        .unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let input = builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let selectors = builder.rhs_family(2).unwrap();
        let public_members = builder.wire_family(2).unwrap();
        let selected = builder
            .select(builder.input_wire(input).unwrap(), selectors, public_members, 4)
            .unwrap();
        let output = builder.add(selected, selected).unwrap();
        builder.output(output).unwrap();
        let program = builder.build().unwrap();
        let range = FamilyRange::bounded(1, 3, 4).unwrap();
        let mut ranges = ProgramFamilyBindings::new();
        ranges.bind(ProgramFamilyId::from_index(0), range.clone());
        let inputs = BTreeMap::from([(
            input,
            BfvNoiseValue {
                level: 0,
                bound: BfvNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                payload_l2_squared: 1u8.into(),
                moments: Some(NoiseMoments {
                    mean_numerator: BigInt::zero(),
                    mean_denominator: BigUint::one(),
                    mean_radius: NonnegativeRational::zero(),
                    variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
                }),
                public: false,
                range: None,
                one_hot: false,
            },
        )]);
        let mut member = BfvNoiseValue::public(1, 2u8.into(), &[2.into(), (-1).into()]);
        assert_eq!(member.bound.payload_l1, BigUint::from(3u8));
        member.range = Some(range.clone());
        let members = BTreeMap::from([(public_members, member)]);
        let rhs = BTreeMap::from([(
            selectors,
            BfvNoiseRhs {
                payload_l1: 1u8.into(),
                payload_l2_squared: 1u8.into(),
                one_hot: Some(range.clone()),
            },
        )]);
        let crate::program::ProgramStep::Family { id: family, .. } = &program.steps()[0] else {
            panic!("family selection step")
        };
        let gate = ProgramGateInstance::FamilyGate { step: 0, gate: 0, family: *family };
        let helpers = BTreeMap::from([(
            gate,
            BfvGateNoiseMaterial {
                level: 0,
                table: None,
                fresh_error_gain: RightActionNoiseGain::unit(),
            },
        )]);
        let evaluate = |rhs: &BTreeMap<crate::program::RhsFamilyId, BfvNoiseRhs>,
                        terminal_bounds: &BTreeMap<WireId, BigUint>| {
            parameters.audit_program(
                &program,
                &ProgramBindings::new(
                    &inputs,
                    &BTreeMap::new(),
                    rhs,
                    &members,
                    &BTreeMap::new(),
                    Some(&helpers),
                ),
                &ranges,
                terminal_bounds,
                None,
            )
        };
        let terminal_bounds = BTreeMap::from([(output, BigUint::from(6u8))]);
        let trace = evaluate(&rhs, &terminal_bounds).unwrap();
        assert_eq!(trace.acceptance, Some(NoiseModelKind::WorstCase));
        // One scalar input, two three-member gates, one reduction, one Add.
        assert_eq!(trace.joint_event_count, BigUint::from(n) * 18u8);
        let dependencies = &trace.dependencies;
        let crate::noise_utils::NoiseDependencyNode::Add { lhs, rhs: same_rhs, .. } =
            &dependencies.nodes[dependencies.wires[&output].0]
        else {
            panic!("the final primitive is Add")
        };
        assert_eq!(lhs, same_rhs, "cloning a wire must preserve its noise ancestor");
        assert_eq!(*lhs, dependencies.wires[&selected]);
        assert!(matches!(
            &dependencies.nodes[lhs.0],
            crate::noise_utils::NoiseDependencyNode::Reduction { range: active, .. } if active == &range
        ));
        assert!(dependencies.nodes.len() < 16, "families remain symbolic in the dependency DAG");
        assert!(evaluate(&rhs, &BTreeMap::new()).is_err());
        assert!(evaluate(&rhs, &BTreeMap::from([(selected, 6u8.into())])).is_err());
        assert!(matches!(
            evaluate(&rhs, &BTreeMap::from([(output, BigUint::from(1u8) << 128)])),
            Err(BfvError::Budget("terminal range"))
        ));
        let captured =
            dependencies
                .nodes
                .iter()
                .find_map(|node| match node {
                    crate::noise_utils::NoiseDependencyNode::MulPrivate {
                        gate: at, input, ..
                    } if *at == gate => Some(*input),
                    _ => None,
                })
                .expect("selection starts with private multiplication");
        assert_eq!(
            captured,
            dependencies.wires[&program.input_wires()[&input]],
            "family capture preserves the original scalar source"
        );
        let mut scalar_evaluator =
            BfvNoiseEvaluator { parameters: &parameters, ranges: &ranges, gates: BTreeMap::new() };
        let repeated_capture = scalar_evaluator.reduce_add(&inputs[&input], &range, gate).unwrap();
        assert_eq!(repeated_capture.bound.error, BigUint::from(6u8));
        assert_eq!(repeated_capture.bound.payload_l1, BigUint::from(3u8));
        assert_eq!(
            repeated_capture.moments.as_ref().unwrap().variance,
            inputs[&input].moments.as_ref().unwrap().variance.scaled(&9u8.into())
        );
        let private_product = &trace.gates[&gate];
        assert_eq!(private_product.bound.payload_l1, BigUint::one());
        let public_product =
            &trace.gates[&ProgramGateInstance::FamilyGate { step: 0, gate: 1, family: *family }];
        assert_eq!(public_product.bound.payload_l1, BigUint::from(3u8));
        assert_eq!(public_product.bound.error, &private_product.bound.error * 3u8);
        assert_eq!(public_product.payload_l2_squared, BigUint::from(5u8));
        assert_eq!(
            public_product.moments.as_ref().unwrap().variance,
            private_product.moments.as_ref().unwrap().variance.scaled(&9u8.into())
        );
        // One-hot selection saves the payload factor, but all three sampled
        // selector ciphertext errors survive into the sum.
        assert_eq!(trace.wires[&selected].bound.payload_l1, BigUint::from(3u8));
        assert_eq!(trace.wires[&selected].bound.error, &public_product.bound.error * 3u8);
        assert_eq!(trace.wires[&output].bound.payload_l1, BigUint::from(6u8));
        assert_eq!(trace.wires[&output].bound.error, &public_product.bound.error * 6u8);
        assert_eq!(trace.wires[&selected].payload_l2_squared, BigUint::from(5u8));
        assert_eq!(trace.wires[&output].payload_l2_squared, BigUint::from(20u8));
        assert_eq!(
            trace.wires[&selected].moments.as_ref().unwrap().variance,
            public_product.moments.as_ref().unwrap().variance.scaled(&9u8.into())
        );
        assert_eq!(
            trace.wires[&output].moments.as_ref().unwrap().variance,
            trace.wires[&selected].moments.as_ref().unwrap().variance.scaled(&4u8.into())
        );

        // Exercise the other nonlinear primitive through the same Program
        // interpreter, rather than only calling its variance formula directly.
        let mut lut_builder = ExponentLutProgramBuilder::new();
        lut_builder.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let lut_input = lut_builder
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let table = crate::program::LutTable::unary(2, 2, vec![1, 0]).unwrap();
        let lut_material = BfvGateNoiseMaterial {
            level: 0,
            table: Some(table.commitment()),
            fresh_error_gain: RightActionNoiseGain::unit(),
        };
        let lut_output =
            lut_builder.lut(lut_builder.input_wire(lut_input).unwrap(), table, None).unwrap();
        lut_builder.output(lut_output).unwrap();
        let lut_program = lut_builder.build().unwrap();
        let mut lut_inputs = BTreeMap::from([(lut_input, inputs[&input].clone())]);
        // Native base-four digits have mean 1/2, not a centered CBD mean.
        let native = NoiseMoments::uniform_interval((-1).into(), 4u8.into());
        lut_inputs.get_mut(&lut_input).unwrap().moments = Some(native.clone());
        let lut_materials = BTreeMap::from([(ProgramGateInstance::Gate { step: 0 }, lut_material)]);
        let lut_trace = trace_program(
            &parameters,
            &lut_program,
            &ProgramBindings::new(
                &lut_inputs,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                Some(&lut_materials),
            ),
            &ProgramFamilyBindings::new(),
        )
        .unwrap();
        assert_eq!(
            lut_trace.wires[&lut_output].moments.as_ref().unwrap().variance,
            parameters
                .lut_noise_variance(
                    0,
                    &native.variance,
                    2,
                    &RightActionNoiseGain::unit(),
                    &RightActionNoiseGain::unit()
                )
                .unwrap()
        );

        assert!(
            !lut_trace.wires[&lut_output].moments.as_ref().unwrap().mean_radius.numerator.is_zero()
        );
        let mut biased = private_product.clone();
        biased.moments = Some(native.clone());
        let weighted = scalar_evaluator
            .mul_public(
                &biased,
                &members[&public_members],
                ProgramGateInstance::FamilyGate { step: 0, gate: 1, family: *family },
            )
            .unwrap();
        let weighted_moments = weighted.moments.as_ref().unwrap();
        assert_eq!(
            weighted_moments.mean_radius,
            NonnegativeRational::new(3u8.into(), 2u8.into()).unwrap()
        );
        assert_eq!(weighted_moments.variance, native.variance.scaled(&9u8.into()));
        let doubled = scalar_evaluator
            .add(
                &weighted,
                &weighted,
                ProgramGateInstance::FamilyGate { step: 0, gate: 2, family: *family },
            )
            .unwrap();
        assert_eq!(
            doubled.moments.as_ref().unwrap().mean_radius,
            NonnegativeRational::new(3u8.into(), 1u8.into()).unwrap()
        );

        // Without the trusted selector contract, the same shape alone must
        // not cause the analyzer to assume a one-hot plaintext sum.
        let mut untrusted_rhs = rhs.clone();
        untrusted_rhs.get_mut(&selectors).unwrap().one_hot = None;
        let untrusted = evaluate(&untrusted_rhs, &terminal_bounds).unwrap();
        assert_eq!(untrusted.wires[&selected].bound.payload_l1, BigUint::from(9u8));
        assert_eq!(untrusted.wires[&selected].bound.error, trace.wires[&selected].bound.error);

        // The chain envelope alone accepts one terminal prime, but the
        // actual Program output range requires two. Candidate acceptance
        // must rerun the complete trace, including selection and final Add.
        let request = BfvPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![4],
            eta: 2,
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 1u8.into(),
        };
        let terminal = BTreeMap::from([(output, BigUint::from(1u8) << 35)]);
        let search = |request: &BfvPlanRequest| {
            BfvParameters::search_program(
                request,
                &program,
                &ProgramBindings::new(
                    &inputs,
                    &BTreeMap::new(),
                    &rhs,
                    &members,
                    &BTreeMap::new(),
                    Some(&helpers),
                ),
                &ranges,
                &terminal,
                None,
            )
        };
        let (accepted, manifest, accepted_trace) = search(&request).unwrap();
        assert_eq!(accepted.level(1).unwrap().crt_depth(), 2);
        assert_eq!(manifest.workload.terminal_depth, 2);
        assert_eq!(accepted_trace.wires[&output].bound.payload_l1, BigUint::from(6u8));
        assert!(manifest.validate().is_ok());
        let mut insufficient = request;
        insufficient.max_crt_depth = 4;
        assert!(matches!(search(&insufficient), Err(BfvError::Budget("CRT depth"))));

        // A gate-free output must still satisfy noise acceptance. Its level
        // is supplied by the input contract, not inferred from the last LUT.
        let mut direct = ExponentLutProgramBuilder::new();
        direct.set_layout(ExpLayout { stride_log2: 0, width_log2: 1 }, n as usize).unwrap();
        let direct_input = direct
            .input(2, ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private })
            .unwrap();
        let direct_output = direct.input_wire(direct_input).unwrap();
        direct.output(direct_output).unwrap();
        let direct = direct.build().unwrap();
        let mut direct_inputs = BTreeMap::from([(
            direct_input,
            BfvNoiseValue {
                level: 1,
                bound: BfvNoiseBound { error: 1u8.into(), payload_l1: 1u8.into() },
                payload_l2_squared: 1u8.into(),
                moments: Some(NoiseMoments {
                    mean_numerator: BigInt::zero(),
                    mean_denominator: BigUint::one(),
                    mean_radius: NonnegativeRational::zero(),
                    variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
                }),
                public: false,
                range: None,
                one_hot: false,
            },
        )]);
        let direct_bounds = BTreeMap::from([(direct_output, 1u8.into())]);
        let check_direct = |inputs: &BTreeMap<crate::program::ProgramInputId, BfvNoiseValue>,
                            average: Option<&AverageCaseConfig>| {
            parameters.audit_program(
                &direct,
                &ProgramBindings::new(
                    inputs,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    None,
                ),
                &ProgramFamilyBindings::new(),
                &direct_bounds,
                average,
            )
        };
        assert!(check_direct(&direct_inputs, None).is_ok());
        direct_inputs.get_mut(&direct_input).unwrap().bound.error =
            parameters.level(1).unwrap().modulus().as_ref().clone();
        assert!(matches!(
            check_direct(&direct_inputs, None),
            Err(BfvError::Budget("evaluation noise"))
        ));
        let average = AverageCaseConfig { allow_average_acceptance: true, ..Default::default() };
        let accepted = check_direct(&direct_inputs, Some(&average)).unwrap();
        assert_eq!(accepted.acceptance, Some(NoiseModelKind::AverageCase));
        assert_eq!(accepted.joint_event_count, BigUint::from(n) * 2u8);
        assert_eq!(
            accepted.z_squared.as_ref().unwrap(),
            &average.z_squared_for_events(&accepted.joint_event_count).unwrap()
        );
        assert!(check_direct(&direct_inputs, Some(&AverageCaseConfig::default())).is_err());
        // A zero center does not erase a nonzero mean interval. At exact
        // modulus/8 clearance even zero centered variance must be rejected.
        let moments = direct_inputs.get_mut(&direct_input).unwrap().moments.as_mut().unwrap();
        moments.mean_radius = NonnegativeRational::new(
            parameters.level(1).unwrap().modulus().as_ref().clone(),
            8u8.into(),
        )
        .unwrap();
        moments.variance = NonnegativeRational::zero();
        assert!(matches!(
            check_direct(&direct_inputs, Some(&average)),
            Err(BfvError::Budget("evaluation noise"))
        ));
        direct_inputs.get_mut(&direct_input).unwrap().moments = None;
        assert!(matches!(
            check_direct(&direct_inputs, Some(&average)),
            Err(BfvError::Parameters("average-case input moment model is missing"))
        ));
    }
}

impl BfvParameters {
    /// Double-scale note (6.6), with arithmetic on exact products. The
    /// numerator includes internal public-matrix rounding and outer rounding.
    /// The caller supplies the output payload bound from its domain analysis:
    /// an identity LUT on a polynomial must not be assigned a monomial bound.
    /// `fresh_error_gain` is the sum of active public multiplier l1 bounds
    /// when composing RHS ciphertexts (the slot count for monomial labels).
    /// It scales only fresh RHS error, not inherited or mask-helper error.
    pub fn lut_noise_bound(
        &self,
        level: usize,
        input: &BfvNoiseBound,
        branches: usize,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<BfvNoiseBound, BfvError> {
        if branches == 0 {
            return Err(BfvError::Parameters("empty branch family"));
        }
        let source = self.level(level)?;
        let next = self.level(level + 1)?;
        let p = self.scale(level)?;
        let n = BigUint::from(source.ring_dimension());
        let j = BigUint::from(branches);
        let s = &n + 1u8;
        let twice_p = p * 2u8;
        let inherited = &j * &n * (next.modulus().as_ref() - 1u8) * (&input.error + self.eta) * 2u8;
        let fresh = &j * &input.payload_l1 * fresh_error_gain * BigUint::from(self.eta);
        let rounding = ((&j + 1u8) * s + 1u8) * (p - 1u8);
        let numerator = inherited + fresh * &twice_p + rounding;
        Ok(BfvNoiseBound {
            error: (numerator + &twice_p - 1u8) / &twice_p,
            payload_l1: output_payload_l1,
        })
    }

    pub fn noise_fits(&self, level: usize, noise: &BfvNoiseBound) -> Result<bool, BfvError> {
        let range_fits = self.level(level)?.modulus().as_ref() > &(&noise.error * 8u8);
        // Nonterminal states must resolve adjacent p_i-spaced payloads. The
        // last level is a terminal carrier with a separately audited range.
        Ok(range_fits &&
            (level + 1 == self.levels.len() || self.scale(level)? > &(&noise.error * 2u8)))
    }

    /// Conservative backward prime allocation followed by a forward exact
    /// audit. Rejects finite depth/memory budgets instead of inflating them.
    pub fn plan(request: &BfvPlanRequest) -> Result<(Self, BfvParameterManifest), BfvError> {
        request.validate()?;
        let pool = DCRTPolyParams::try_new(
            request.ring_dimension,
            request.max_crt_depth,
            request.crt_bits,
            1,
            None,
            None,
        )
        .map_err(|_| BfvError::Parameters("unsupported prime pool"))?;
        let primes = pool.to_crt().0;
        let mut used = request.terminal_depth;
        let mut modulus = primes[..used].iter().map(|p| BigUint::from(*p)).product::<BigUint>();
        let mut reverse = vec![
            pool.select_modulus(&modulus).ok_or(BfvError::Parameters("invalid terminal base"))?,
        ];
        for &width in request.widths.iter().rev() {
            let threshold = &modulus * BigUint::from(width) * request.ring_dimension * 2u8;
            let mut drop = BigUint::one();
            while drop < threshold {
                let prime = primes.get(used).ok_or(BfvError::Budget("CRT depth"))?;
                drop *= *prime;
                used += 1;
            }
            modulus *= drop;
            reverse.push(
                pool.select_modulus(&modulus)
                    .ok_or(BfvError::Parameters("invalid planned base"))?,
            );
        }
        reverse.reverse();
        let parameters = Self::new(reverse, request.eta)?;
        let manifest = parameters.audit(request)?;
        Ok((parameters, manifest))
    }

    /// Recompute the certificate from public workload data and exact bases.
    /// This chain planner covers monomial LUT stages; arbitrary Program DAGs
    /// must use their full noise trace before claiming parameter acceptance.
    pub(super) fn audit(&self, request: &BfvPlanRequest) -> Result<BfvParameterManifest, BfvError> {
        request.validate()?;
        let stages = request.widths.len();
        if self.levels.len() != stages + 1 ||
            self.levels[0].ring_dimension() != request.ring_dimension ||
            self.eta != request.eta ||
            self.levels[0].crt_depth() > request.max_crt_depth ||
            self.levels[stages].crt_depth() != request.terminal_depth ||
            self.levels.iter().any(|level| level.crt_bits() > request.crt_bits)
        {
            return Err(BfvError::Parameters("modulus chain does not match workload"));
        }
        let mut bounds =
            vec![BfvNoiseBound { error: request.eta.into(), payload_l1: BigUint::one() }];
        let mut bytes = BigUint::zero();
        for i in 0..stages {
            let bound = self.lut_noise_bound(
                i,
                &bounds[i],
                request.widths[i],
                &BigUint::from(request.slots[i]),
                BigUint::one(),
            )?;
            if !self.noise_fits(i + 1, &bound)? {
                return Err(BfvError::Budget("evaluation noise"));
            }
            bounds.push(bound);
            let high = BigUint::from(self.levels[i].crt_depth());
            let low = BigUint::from(self.levels[i + 1].crt_depth());
            let branches = BigUint::from(request.widths[i]);
            // RHS plus helper rows/public matrices; include both public/state
            // values, C/b lift scratch, and raw accumulators for one branch wave.
            bytes += (4u8 * &branches * request.slots[i] * low +
                6u8 * &branches * &high +
                20u8 * high) *
                request.ring_dimension *
                8u8;
        }
        if bytes > BigUint::from(request.memory_budget_bytes) {
            return Err(BfvError::Budget("memory"));
        }
        let terminal = self.level(stages)?;
        let range = (&request.terminal_value_bound + &bounds[stages].error) * 2u8;
        if terminal.modulus().as_ref() <= &range {
            return Err(BfvError::Budget("terminal range"));
        }
        let manifest = BfvParameterManifest {
            workload: request.clone(),
            ring_dimension: request.ring_dimension,
            eta: request.eta,
            levels: self.levels.iter().map(|level| level.to_crt().0).collect(),
            drops: self.drops[..stages].to_vec(),
            payload_scales: self.drops.clone(),
            noise_bounds: bounds,
            estimated_bytes: bytes,
            security_certified: false,
        };
        Ok(manifest)
    }
}

/// Unresolved refresh workload. Input levels and noise come from the Program
/// values passed to `plan_refresh`, rather than independently copied bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BfvRefreshPlanRequest {
    pub data_branches: usize,
    pub prf_branches: usize,
    pub base: usize,
    pub fresh_support: BigUint,
    pub fresh_bound: BigUint,
    pub decoder_eta: usize,
    pub observations: BigUint,
    pub statistical_bits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BfvRefreshPlan {
    pub model: NoiseModelKind,
    pub z_squared: Option<NonnegativeRational>,
    pub digits: RefreshDigitPlan,
    /// Physical moments under the independent uniform native-digit model.
    /// One fresh source is reused across all CRT channels. These conditional
    /// moments do not participate in deterministic spacing acceptance.
    pub fresh_uniform_moments: crate::noise_utils::NoiseMoments,
    /// Distinct centered mask distributions, in canonical CRT slot order.
    pub mask_uniform_moments: Vec<crate::noise_utils::NoiseMoments>,
    pub exposed_errors: Vec<BigUint>,
    pub spacings: Vec<BigUint>,
    pub distinct_prf_outputs: BigUint,
    pub terminal_digit_emissions: BigUint,
    /// Persistent C, b, input-key and raw-public polynomial towers. This is
    /// total artifact storage, not peak workspace or a measured runtime.
    pub terminal_artifact_bytes: BigUint,
}

impl BfvParameters {
    /// Solve counts against the same aggregate-before-ModDown recurrence used
    /// by evaluation, including decoder noise exactly once. Large target
    /// weights do not multiply these errors: RHS errors are sampled afterwards.
    /// Each input is a traced Program value (or an input sampling contract)
    /// under these parameters, supplying one terminal member envelope. Data and
    /// PRF may come from different Programs. Setup establishes this binding;
    /// the planner does not rerun either Program or inspect ciphertexts.
    pub fn plan_refresh(
        &self,
        request: &BfvRefreshPlanRequest,
        data: &BfvNoiseValue,
        prf: &BfvNoiseValue,
        average: Option<&AverageCaseConfig>,
    ) -> Result<BfvRefreshPlan, BfvError> {
        if data.level != prf.level {
            return Err(BfvError::Parameters("refresh inputs need a common terminal level"));
        }
        let input_level = data.level;
        let source = self.level(input_level)?;
        let output = self.level(input_level + 1)?;
        if output.crt_depth() < 2 || request.decoder_eta == 0 {
            return Err(BfvError::Parameters("refresh needs two CRT primes and decoder error"));
        }
        let q = output.modulus();
        let spacings =
            output.to_crt().0.iter().map(|prime| q.as_ref() / *prime).collect::<Vec<_>>();
        let mut digits = RefreshDigitPlan::new(
            request.base,
            &vec![1u8.into(); spacings.len()],
            &request.fresh_support,
        )
        .map_err(|_| BfvError::Parameters("refresh digit support"))?;
        let coordinates = 2usize * source.ring_dimension() as usize;
        let average = average.filter(|config| config.allow_average_acceptance);
        let tail =
            |plan: &RefreshDigitPlan| -> Result<Option<NonnegativeRational>, DigitPlanError> {
                average
                    .map(|config| {
                        // Final exposed coordinates and every PRF state coordinate.
                        // The count follows the current fixed-point digit schedule.
                        let outputs = (plan.mask_counts.iter().sum::<usize>() + plan.fresh_count) *
                            coordinates;
                        let events = &request.observations +
                            BigUint::from(outputs) * coordinates * (input_level + 1);
                        config
                            .z_squared_for_events(&events)
                            .map(|z| {
                                z.scaled(&(BigUint::one() << (2 * config.tail_correction_bits)))
                            })
                            .map_err(|_| DigitPlanError::Parameters)
                    })
                    .transpose()
            };
        let exposed_errors = digits
            .fit_masks(
                &spacings,
                &request.observations,
                request.statistical_bits,
                &request.fresh_bound,
                |plan| {
                    plan.mask_counts
                        .iter()
                        .map(|mask_count| {
                            let count = mask_count
                                .checked_add(plan.fresh_count)
                                .and_then(|count| count.checked_mul(coordinates))
                                .and_then(|count| count.checked_mul(request.prf_branches))
                                .ok_or(DigitPlanError::Parameters)?;
                            if let Some(z) = tail(plan)? {
                                let mut moments = self
                                    .terminal_noise_moments(
                                        input_level,
                                        [(data, request.data_branches), (prf, count)],
                                    )
                                    .map_err(|_| DigitPlanError::Parameters)?;
                                // Online decoder CBD is fresh under the same
                                // conditional residual-independence model.
                                moments.variance = moments.variance.add(
                                    &NonnegativeRational::new(
                                        request.decoder_eta.into(),
                                        2u8.into(),
                                    )
                                    .unwrap(),
                                );
                                return Ok(moments.tail_bound(&z));
                            }
                            self.terminal_noise_bound(
                                input_level,
                                [
                                    (&data.bound, request.data_branches, &1u8.into()),
                                    (&prf.bound, count, &1u8.into()),
                                ],
                                0u8.into(),
                            )
                            .map(|bound| bound.error + request.decoder_eta)
                            .map_err(|_| DigitPlanError::Parameters)
                        })
                        .collect()
                },
            )
            .map_err(|error| match error {
                DigitPlanError::MaskRange => BfvError::Budget("refresh mask spacing"),
                DigitPlanError::FreshRange => BfvError::Budget("refresh fresh-error support"),
                DigitPlanError::Parameters => BfvError::Parameters("refresh digit/error workload"),
            })?;
        let fresh_error = digits.fresh_interval().1.magnitude().clone();
        if !self.noise_fits(
            input_level + 1,
            &BfvNoiseBound { error: fresh_error, payload_l1: data.bound.payload_l1.clone() },
        )? {
            return Err(BfvError::Budget("refreshed state scale"));
        }
        let masks = digits.mask_counts.iter().map(|count| BigUint::from(*count)).sum::<BigUint>();
        let distinct_prf_outputs = (&masks + digits.fresh_count) * coordinates;
        let terminal_digit_emissions =
            (&masks + BigUint::from(digits.fresh_count) * spacings.len()) * coordinates;
        let high = BigUint::from(source.crt_depth());
        let low = BigUint::from(output.crt_depth());
        // Indexed digit terms store fresh C and b for every branch. Input
        // matrices have four high-base polynomials; a raw aggregate is one
        // four-polynomial high-base matrix per group (mask/fresh/data).
        let digit_towers =
            &terminal_digit_emissions * (&low * request.prf_branches * 8u8 + 4u8 * &high);
        let data_towers =
            BigUint::from(spacings.len()) * (&low * request.data_branches * 8u8 + 4u8 * &high);
        let shared_towers = BigUint::from(spacings.len()) * (12u8 * &high + 6u8 * &low) +
            4u8 * &low +
            (BigUint::from(request.prf_branches) * 2u8 + request.data_branches) *
                4u8 *
                &high *
                spacings.len();
        let terminal_artifact_bytes =
            (digit_towers + data_towers + shared_towers) * source.ring_dimension() * 8u8;
        Ok(BfvRefreshPlan {
            model: if average.is_some() {
                NoiseModelKind::AverageCase
            } else {
                NoiseModelKind::WorstCase
            },
            z_squared: tail(&digits).map_err(|_| BfvError::Parameters("refresh event budget"))?,
            fresh_uniform_moments: digits.fresh_uniform_moments(),
            mask_uniform_moments: (0..digits.mask_counts.len())
                .map(|slot| digits.mask_uniform_moments(slot))
                .collect(),
            digits,
            exposed_errors,
            spacings,
            distinct_prf_outputs,
            terminal_digit_emissions,
            terminal_artifact_bytes,
        })
    }
}

#[cfg(test)]
mod refresh_plan_tests {
    use super::*;
    #[test]
    fn test_bfv_refresh_plan_uses_emitted_digits_in_noise_fixed_point() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(4);
        let (parameters, _) = BfvParameters::plan(&super::super::BfvPlanRequest {
            ring_dimension: n,
            // One composed PRF product precedes refresh and continuation.
            widths: vec![1, n as usize, 1],
            slots: vec![2, 1, 1],
            eta: 2,
            crt_bits: 17,
            terminal_depth: 2,
            max_crt_depth: 24,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 32u8.into(),
        })
        .unwrap();
        let request = BfvRefreshPlanRequest {
            data_branches: n as usize,
            prf_branches: n as usize,
            base: 4,
            fresh_support: 5u8.into(),
            fresh_bound: 10u8.into(),
            decoder_eta: 2,
            observations: 8u8.into(),
            statistical_bits: 16,
        };
        use crate::{
            layout::ExpLayout,
            program::{AttributeKnowledge, ExponentLutProgramBuilder, ProgramWireType},
        };
        let mut builder = ExponentLutProgramBuilder::new();
        builder
            .set_layout(ExpLayout { stride_log2: 0, width_log2: n.trailing_zeros() }, n as usize)
            .unwrap();
        let private = ProgramWireType { stride_log2: 0, knowledge: AttributeKnowledge::Private };
        let initial = builder.input(n as usize, private).unwrap();
        let data = builder.input(n as usize, private).unwrap();
        let family = builder.rhs_family(n as usize).unwrap();
        let rhs = builder.rhs_input(family, n as usize, n as usize).unwrap();
        let prf = builder.private_mul(builder.input_wire(initial).unwrap(), rhs).unwrap();
        let data_wire = builder.input_wire(data).unwrap();
        builder.output(data_wire).unwrap();
        builder.output(prf).unwrap();
        let program = builder.build().unwrap();
        let inputs = BTreeMap::from([(initial, 0), (data, 1)].map(|(id, level)| {
            (
                id,
                BfvNoiseValue {
                    level,
                    bound: BfvNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
                    payload_l2_squared: 1u8.into(),
                    moments: Some(NoiseMoments {
                        mean_numerator: BigInt::zero(),
                        mean_denominator: BigUint::one(),
                        mean_radius: NonnegativeRational::zero(),
                        variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
                    }),
                    public: false,
                    range: None,
                    one_hot: false,
                },
            )
        }));
        // The composed one-hot RHS retains both fresh ciphertext errors.
        let (material, rhs_bound) = BfvGateNoiseMaterial::from_one_hot_labels(
            0,
            None,
            &[RightActionNoiseGain::unit(), RightActionNoiseGain::unit()],
        )
        .unwrap();
        let rhs_bounds = BTreeMap::from([(rhs, rhs_bound)]);
        let materials = BTreeMap::from([(ProgramGateInstance::Gate { step: 0 }, material)]);
        let trace = trace_program(
            &parameters,
            &program,
            &ProgramBindings::new(
                &inputs,
                &rhs_bounds,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                Some(&materials),
            ),
            &ProgramFamilyBindings::new(),
        )
        .unwrap();
        let plan = parameters
            .plan_refresh(&request, &trace.wires[&data_wire], &trace.wires[&prf], None)
            .unwrap();
        assert_eq!(plan.model, NoiseModelKind::WorstCase);
        assert!(plan.z_squared.is_none());
        let data = &trace.wires[&data_wire];
        let prf = &trace.wires[&prf];
        let policy = AverageCaseConfig {
            failure_exponent: 16,
            allow_average_acceptance: true,
            ..Default::default()
        };
        let average = parameters.plan_refresh(&request, data, prf, Some(&policy)).unwrap();
        assert_eq!(average.model, NoiseModelKind::AverageCase);
        assert!(average.z_squared.is_some());
        assert_ne!(average.exposed_errors, plan.exposed_errors);
        let combined = parameters.terminal_noise_moments(data.level, [(data, 3)]).unwrap();
        let split = parameters.terminal_noise_moments(data.level, [(data, 1), (data, 2)]).unwrap();
        assert_eq!(combined, split, "grouping cannot duplicate outer rounding");
        let primitive = parameters
            .lut_noise_variance(
                data.level,
                &data.moments.as_ref().unwrap().variance,
                3,
                &RightActionNoiseGain::unit(),
                &RightActionNoiseGain::unit(),
            )
            .unwrap();
        assert_eq!(
            combined.variance, primitive,
            "one terminal group has the same signed terms as one LUT"
        );
        let mut missing = data.clone();
        missing.moments = None;
        assert!(parameters.plan_refresh(&request, &missing, prf, Some(&policy)).is_err());
        assert_eq!(plan.digits.fresh_count, 2);
        assert!(plan.digits.mask_counts.iter().all(|count| *count > plan.digits.fresh_count));
        assert!(plan.terminal_digit_emissions > plan.distinct_prf_outputs);
        for (index, error) in plan.exposed_errors.iter().enumerate() {
            let support = plan.digits.support(plan.digits.mask_counts[index]);
            assert!(&support + error * 2u8 < plan.spacings[index]);
            assert!(support >= (error * &request.observations * 2u8) << request.statistical_bits);
        }
    }
}

/// The terminal numerator keeps inherited/fresh/helper/internal-rounding terms
/// separate until all stages have been added. Outer rounding occurs only once.
pub struct BfvTerminalNoise {
    pub error: BigUint,
    pub coefficient_bound: BigUint,
    pub branches: usize,
}

impl BfvParameters {
    /// Setup-time integer recovery and evaluation-noise checks at the terminal
    /// output level. Weights affect the required range, never the error gain.
    pub fn terminal_noise_fits(
        &self,
        output_level: usize,
        bound: &BfvTerminalNoise,
    ) -> Result<bool, BfvError> {
        let q = self.level(output_level)?.modulus();
        let error = &bound.error;
        Ok(q.as_ref() > &((&bound.coefficient_bound + error) * 2u8) && q.as_ref() > &(error * 8u8))
    }

    /// Conditional terminal moments from the same raw aggregate as
    /// `terminal_noise_bound`. H1/H3/H4 decorrelate centered right actions
    /// across freshly prepared targets, including reused input/helper errors;
    /// this is an explicit heuristic, not independence inferred from loop indices.
    /// Fresh target CBD errors have distinct sampling identities. All internal
    /// remainders are counted, while the two outer remainders occur only once.
    pub fn terminal_noise_moments<'a>(
        &self,
        level: usize,
        stages: impl IntoIterator<Item = (&'a BfvNoiseValue, usize)>,
    ) -> Result<NoiseMoments, BfvError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let q = self.level(level + 1)?.modulus();
        let p_squared = p * p;
        let cbd = NonnegativeRational::new(self.eta().into(), 2u8.into()).unwrap();
        let mut result = NoiseMoments {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new((&n + 3u8) * (&p_squared - 1u8), 18u8 * &p_squared)
                .unwrap(),
        };
        for (input, count) in stages {
            let m = input
                .moments
                .as_ref()
                .ok_or(BfvError::Parameters("terminal input moment model is missing"))?;
            let j = BigUint::from(count);
            let input_helper = m.variance.add(&cbd);
            let inherited_helper = NonnegativeRational::new(
                &input_helper.numerator * &j * &n * (q.as_ref() - 1u8).pow(2),
                &input_helper.denominator * 2u8 * &p_squared,
            )
            .unwrap();
            let fresh = cbd.scaled(&(&j * &input.payload_l2_squared));
            let internal_rounding = NonnegativeRational::new(
                &j * (&n * 2u8 + 3u8) * (&p_squared - 1u8),
                36u8 * &p_squared,
            )
            .unwrap();
            let mean = m.linear_image(
                &NonnegativeRational::new(&j * &n * (q.as_ref() - 1u8), p.clone()).unwrap(),
                NonnegativeRational::zero(),
            );
            result.mean_radius = result.mean_radius.add(&mean.mean_radius);
            result.variance =
                result.variance.add(&inherited_helper).add(&fresh).add(&internal_rounding);
        }
        Ok(result)
    }

    /// Each group is (input bound, total branches, RHS l1 gain). Identical
    /// member bounds can be grouped by adding their branch counts, without
    /// enumerating PRF draws. All inner errors scale by total branches; outer
    /// rounding is counted once across all groups.
    pub fn terminal_noise_bound<'a>(
        &self,
        level: usize,
        stages: impl IntoIterator<Item = (&'a BfvNoiseBound, usize, &'a BigUint)>,
        coefficient_bound: BigUint,
    ) -> Result<BfvTerminalNoise, BfvError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let q = self.level(level + 1)?.modulus();
        let twice_p = p * 2u8;
        let mut numerator = (&n + 2u8) * (p - 1u8);
        let mut branches = 0usize;
        for (input, count, gain) in stages {
            if count == 0 {
                return Err(BfvError::Parameters("empty terminal stage"));
            }
            branches = branches.checked_add(count).ok_or(BfvError::Budget("terminal branches"))?;
            let j = BigUint::from(count);
            numerator += &j * &n * (q.as_ref() - 1u8) * (&input.error + self.eta()) * 2u8;
            numerator += &twice_p * &j * &input.payload_l1 * gain * self.eta();
            numerator += &j * (&n + 1u8) * (p - 1u8);
        }
        if branches == 0 {
            return Err(BfvError::Parameters("empty terminal batch"));
        }
        Ok(BfvTerminalNoise {
            error: (numerator + &twice_p - 1u8) / twice_p,
            coefficient_bound,
            branches,
        })
    }
}
