//! Deterministic bounds and conditional moments for the common Program.
use super::{
    RnsBggError, RnsBggNoiseBound, RnsBggParameterManifest, RnsBggParameters, RnsBggPlanRequest,
    canonical_encoding_sigma, encoding_error_bound,
};
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

impl RnsBggParameters {
    pub fn encoding_error_variance(&self) -> NonnegativeRational {
        let (mut numerator, exponent) = self.encoding_error_sigma.as_bigint_and_exponent();
        let denominator = if exponent >= 0 {
            BigUint::from(10u8).pow(exponent as u32)
        } else {
            numerator *= BigInt::from(10u8).pow((-exponent) as u32);
            BigUint::one()
        };
        NonnegativeRational::new(
            numerator.to_biguint().expect("positive encoding sigma").pow(2),
            denominator.pow(2),
        )
        .expect("positive encoding sigma variance")
    }

    /// Physical-coordinate variance transfer for the direct product's four
    /// error terms. Conditional model: deviations from the coefficient means
    /// are uncorrelated with a common variance bound, fixed RHS errors are the
    /// encoding Gaussian with variance sigma²
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
    ) -> Result<NonnegativeRational, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let p_squared = p * p;
        // e*U/p: 2D bounded digit polynomials per output column, each with
        // N coefficients bounded by b/2. H4 conditions on the prepared U;
        // this is an energy cap, not a uniform-digit independence assumption.
        let inherited = NonnegativeRational::new(
            &input_variance.numerator *
                &n *
                next.modulus_digits() *
                (BigUint::one() << (2 * next.base_bits())),
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
        let fresh = self.encoding_error_variance().scaled(&fresh_gain);
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
    /// Each mask helper is sampled with the encoding Gaussian at the input level. Branch
    /// fresh gains describe ciphertext weighting, not fresh plaintext targets.
    pub fn lut_noise_variance(
        &self,
        level: usize,
        input_variance: &NonnegativeRational,
        branches: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let j = BigUint::from(branches);
        let p_squared = p * p;
        let gaussian = self.encoding_error_variance();
        // sum_k (sigma_k(e)*U_k + ed_k*V_k)/p. Both low-ring
        // decompositions have 2D entries per column with coefficients <= b/2.
        let input_and_helper = input_variance.add(&gaussian);
        let inherited_and_helper = NonnegativeRational::new(
            &input_and_helper.numerator *
                &j *
                &n *
                (BigUint::one() << (2 * next.base_bits())) *
                next.modulus_digits(),
            &input_and_helper.denominator * 2u8 * &p_squared,
        )
        .expect("positive scale and variance denominator");
        // Automorphisms preserve both norms. Combine deterministic actions on
        // each F_k before estimating convolution, just as for the direct gate.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        let fresh = gaussian.scaled(&(&j * fresh_gain));
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

    /// Direct product: e*U/p - mu*F + s*rho(A*U)/p - rho(c*U)/p.
    /// U = Dec(C), with 2D polynomials bounded by b/2 per output column:
    /// K(U) <= N*D*b. A composed
    /// fixed RHS contributes all weighted fresh errors, even for one-hot payloads.
    pub fn private_mul_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let twice_p = p * 2u8;
        let inherited =
            &input.error * &n * next.modulus_digits() * (BigUint::one() << next.base_bits()) * 2u8;
        let fresh = &input.payload_l1 * fresh_error_gain * &self.encoding_error_bound * &twice_p;
        // S = ||s_bar||_1 + 1 <= N + 1 for the ternary secret.
        let rounding = (n + 2u8) * (p - 1u8);
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + &twice_p - 1u8) / twice_p,
            payload_l1: output_payload_l1,
        })
    }
}

/// A family envelope applies to each active member, never to its sum.
/// `one_hot` is an explicit trusted payload contract; it never suppresses noise.
#[derive(Clone, Debug)]
pub struct RnsBggNoiseValue {
    pub level: usize,
    pub bound: RnsBggNoiseBound,
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

impl RnsBggNoiseValue {
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
            bound: RnsBggNoiseBound {
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
pub struct RnsBggNoiseRhs {
    pub payload_l1: BigUint,
    pub payload_l2_squared: BigUint,
    /// The setup contract guarantees at most one nonzero coefficient-selector
    /// operand across precisely this active family range.
    pub one_hot: Option<FamilyRange>,
}

#[derive(Clone, Debug)]
pub struct RnsBggGateNoiseMaterial {
    pub level: usize,
    pub table: Option<[u8; 32]>,
    /// Sum of active public multiplier l1 bounds in each composed branch.
    /// Zero selectors still contribute; zero public multipliers do not.
    /// The squared gain sums coefficient squares across separately sampled
    /// RHS errors; it is not the square of the total l1 gain.
    pub fresh_error_gain: RightActionNoiseGain,
}

impl RnsBggGateNoiseMaterial {
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
    ) -> Result<(Self, RnsBggNoiseRhs), RnsBggError> {
        let payload_l1 = multipliers
            .iter()
            .map(|gain| &gain.l1)
            .max()
            .ok_or(RnsBggError::Parameters("empty RHS composition"))?
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
            RnsBggNoiseRhs {
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

pub struct RnsBggNoiseTrace {
    /// None for a raw trace; set only by a successful full Program audit.
    pub acceptance: Option<NoiseModelKind>,
    pub joint_event_count: BigUint,
    /// Includes the configured tail correction, in physical coordinates.
    pub z_squared: Option<NonnegativeRational>,
    pub dependencies: crate::noise_utils::NoiseDependencyGraph,
    pub wires: BTreeMap<WireId, RnsBggNoiseValue>,
    pub gates: BTreeMap<ProgramGateInstance, RnsBggNoiseValue>,
}

pub fn trace_program(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    bindings: &ProgramBindings<'_, RnsBggNoiseValue, RnsBggNoiseRhs, RnsBggGateNoiseMaterial>,
    ranges: &ProgramFamilyBindings,
) -> Result<RnsBggNoiseTrace, ProgramLowerError<RnsBggError>> {
    let (wires, evaluator, dependencies) =
        crate::scheme::trace_noise::<super::RnsBggScheme>(program, parameters, bindings, ranges)?;
    Ok(RnsBggNoiseTrace {
        dependencies,
        wires,
        gates: evaluator.gates,
        acceptance: None,
        joint_event_count: BigUint::zero(),
        z_squared: None,
    })
}

impl RnsBggParameters {
    /// Search public prime candidates using the full Program audit. The
    /// request's terminal depth is the lower bound; its CRT/memory budgets
    /// remain hard limits on the declared allocation profile. Width/slot
    /// counts must cover the workload's storage contract; this is not yet a
    /// liveness-based peak-memory scheduler. The returned manifest describes allocation,
    /// while the returned trace is the actual Program acceptance evidence.
    /// This conservative search does not certify globally minimal parameters.
    pub fn search_program(
        request: &super::RnsBggPlanRequest,
        program: &ExponentLutProgram,
        bindings: &ProgramBindings<'_, RnsBggNoiseValue, RnsBggNoiseRhs, RnsBggGateNoiseMaterial>,
        ranges: &ProgramFamilyBindings,
        terminal_value_bounds: &BTreeMap<WireId, BigUint>,
        average: Option<&AverageCaseConfig>,
    ) -> Result<(Self, super::RnsBggParameterManifest, RnsBggNoiseTrace), RnsBggError> {
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
                Err(RnsBggError::Budget("evaluation noise" | "terminal range")) => {
                    if candidate.terminal_depth >= candidate.max_crt_depth {
                        return Err(RnsBggError::Budget("CRT depth"));
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
        bindings: &ProgramBindings<'_, RnsBggNoiseValue, RnsBggNoiseRhs, RnsBggGateNoiseMaterial>,
        ranges: &ProgramFamilyBindings,
        terminal_value_bounds: &BTreeMap<WireId, BigUint>,
        average: Option<&AverageCaseConfig>,
    ) -> Result<RnsBggNoiseTrace, RnsBggError> {
        let outputs = program.outputs().iter().copied().collect::<std::collections::BTreeSet<_>>();
        if terminal_value_bounds.keys().copied().collect::<std::collections::BTreeSet<_>>() !=
            outputs
        {
            return Err(RnsBggError::Parameters(
                "terminal bounds must match declared Program outputs",
            ));
        }
        let mut trace = trace_program(self, program, bindings, ranges).map_err(|e| match e {
            ProgramLowerError::Operation { source, .. } => source,
            ProgramLowerError::Validation(error) => RnsBggError::Graph(error.to_string()),
            ProgramLowerError::Graph(error) => RnsBggError::Graph(error.to_string()),
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
                    .ok_or(RnsBggError::Parameters(
                        "noise audit requires concrete active ranges",
                    ))?,
            };
            events += count *
                self.level(value.level)?.ring_dimension() *
                2u8 *
                self.component_columns(value.level)?;
        }
        let average = average.filter(|config| config.allow_average_acceptance);
        let z_squared = average
            .map(|config| {
                let z = config
                    .z_squared_for_events(&events)
                    .map_err(|_| RnsBggError::Parameters("invalid average-case event budget"))?;
                let correction = config
                    .tail_correction_bits
                    .checked_mul(2)
                    .ok_or(RnsBggError::Parameters("invalid tail correction"))?;
                Ok::<_, RnsBggError>(z.scaled(&(BigUint::one() << correction)))
            })
            .transpose()?;
        for value in values.values() {
            let fits = match &z_squared {
                None => self.noise_fits(value.level, &value.bound)?,
                Some(z) => {
                    let moments = value.moments.as_ref().ok_or(RnsBggError::Parameters(
                        "average-case input moment model is missing",
                    ))?;
                    let modulus_limit = NonnegativeRational::new(
                        self.level(value.level)?.modulus().as_ref().clone(),
                        8u8.into(),
                    )
                    .expect("positive constant denominator");
                    moments.fits(&modulus_limit, z) &&
                        (value.level + 1 == self.levels.len() ||
                            self.scale(value.level)? == &BigUint::one() ||
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
                return Err(RnsBggError::Budget("evaluation noise"));
            }
        }
        for (wire, value_bound) in terminal_value_bounds {
            let value = trace.wires.get(wire).ok_or(RnsBggError::Context)?;
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
                return Err(RnsBggError::Budget("terminal range"));
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

pub struct RnsBggNoiseEvaluator<'a> {
    pub(super) parameters: &'a RnsBggParameters,
    pub(super) ranges: &'a ProgramFamilyBindings,
    pub(super) gates: BTreeMap<ProgramGateInstance, RnsBggNoiseValue>,
}

impl RnsBggNoiseEvaluator<'_> {
    // Captured scalars keep their dependency identity. Only the transfer's
    // output acquires the current member range; no new source is introduced.
    fn gate_range(
        &self,
        gate: ProgramGateInstance,
        inputs: &[&RnsBggNoiseValue],
    ) -> Result<Option<FamilyRange>, RnsBggError> {
        let range = match gate {
            ProgramGateInstance::FamilyGate { family, .. } => {
                Some(self.ranges.range(family).ok_or(RnsBggError::Context)?.clone())
            }
            _ => None,
        };
        if inputs.iter().any(|input| input.range.is_some() && input.range != range) {
            return Err(RnsBggError::Context);
        }
        Ok(range)
    }

    fn record(
        &mut self,
        gate: ProgramGateInstance,
        value: RnsBggNoiseValue,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        self.gates.insert(gate, value.clone());
        Ok(value)
    }
}

impl ProgramLoweringTypes for RnsBggNoiseEvaluator<'_> {
    type Wire = RnsBggNoiseValue;
    type Rhs = RnsBggNoiseRhs;
    type HelperSet = RnsBggGateNoiseMaterial;
    type Error = RnsBggError;
}

impl ExponentProgramLowering for RnsBggNoiseEvaluator<'_> {
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
        material: Option<&Self::HelperSet>,
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[input])?;
        let material = material.ok_or(RnsBggError::Context)?;
        if material.level != input.level || material.table.is_some() {
            return Err(RnsBggError::Context);
        }
        if let Some(rhs_range) = &rhs.one_hot {
            if range.as_ref() != Some(rhs_range) || rhs.payload_l1 > BigUint::one() {
                return Err(RnsBggError::Context);
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
                    BigUint::from(next.ring_dimension()) *
                        next.modulus_digits() *
                        (BigUint::one() << next.base_bits()),
                    self.parameters.scale(input.level)?.clone(),
                )
                .expect("positive scale");
                Ok::<_, RnsBggError>(m.linear_image(&l1, variance))
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
            RnsBggNoiseValue {
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
            return Err(RnsBggError::Context);
        }
        self.record(
            gate,
            RnsBggNoiseValue {
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
            return Err(RnsBggError::Context);
        }
        self.record(
            gate,
            RnsBggNoiseValue {
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
        gate: ProgramGateInstance,
    ) -> Result<Self::Wire, Self::Error> {
        let range = self.gate_range(gate, &[input])?;
        let material = helpers.ok_or(RnsBggError::Lut)?;
        use mxx_primitives::poly::PolyParams;
        let layout = program.layout().ok_or(RnsBggError::Lut)?;
        if layout.exponent.stride_log2 != 0 ||
            layout.ring_dimension_log2 !=
                self.parameters.level(input.level)?.ring_dimension().trailing_zeros()
        {
            return Err(RnsBggError::Parameters("RNS-BGG requires an unstrided exponent layout"));
        }
        let table = program.lut(lut).ok_or(RnsBggError::Lut)?;
        if material.level != input.level || material.table != Some(table.commitment()) {
            return Err(RnsBggError::Context);
        }
        // The input already includes any preceding private product. A unary
        // table preserves its l1 bound even when output monomials collide.
        let output_l1 = input.bound.payload_l1.clone();
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
                        (BigUint::one() << next.base_bits()) *
                        next.modulus_digits(),
                    self.parameters.scale(input.level)?.clone(),
                )
                .expect("positive scale");
                Ok::<_, RnsBggError>(m.linear_image(&l1, variance))
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
            RnsBggNoiseValue {
                level: input.level + 1,
                // Different table entries may collide: retain the l1-based
                // cap until the table domain proves a tighter squared norm.
                payload_l2_squared: bound.payload_l1.pow(2),
                moments,
                bound,
                public: input.public,
                range,
                one_hot: input.range.is_some() && input.one_hot,
            },
        )
    }
}

impl RnsBggNoiseEvaluator<'_> {
    pub(super) fn reduce_add(
        &mut self,
        value: &RnsBggNoiseValue,
        range: &FamilyRange,
        gate: ProgramGateInstance,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        if value.range.as_ref().is_some_and(|bound_range| bound_range != range) {
            return Err(RnsBggError::Context);
        }
        let count = range
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|count| count.to_biguint())
            .filter(|count| *count > 0u8.into())
            .ok_or(RnsBggError::Parameters("noise trace needs a concrete active family count"))?;
        self.record(
            gate,
            RnsBggNoiseValue {
                level: value.level,
                payload_l2_squared: if value.range.is_some() && value.one_hot {
                    value.payload_l2_squared.clone()
                } else {
                    &value.payload_l2_squared * count.pow(2)
                },
                moments: value.moments.as_ref().map(|m| m.scaled(&count)),
                bound: RnsBggNoiseBound {
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
        layout::ExpLayout,
        program::{
            AttributeKnowledge, ExponentLutProgramBuilder, ProgramFamilyId, ProgramWireType,
        },
        rns_bgg::RnsBggPlanRequest,
    };

    #[test]
    fn test_rns_bgg_binary_expansion_noise_and_shared_ancestry() {
        use crate::{
            noise_utils::NoiseDependencyNode,
            program::{
                AttributeKnowledge, ExponentLutProgramBuilder, Gate, LutTable, ProgramWireType,
            },
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let request = RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1, 4],
            slots: vec![1, 1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 6,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 3u8.into(),
            operations_per_drop: 1,
        };
        let (parameters, _) = RnsBggParameters::plan(&request).unwrap();
        let build = |automatic| {
            let mut builder = ExponentLutProgramBuilder::new();
            builder
                .set_layout(crate::layout::ExpLayout { stride_log2: 0, width_log2: 2 }, n as usize)
                .unwrap();
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
            let unary = if automatic {
                None
            } else {
                Some(
                    builder.register_lut(LutTable::unary(4, 2, vec![0, 1, 1, 0]).unwrap()).unwrap(),
                )
            };
            let input_wire = builder.input_wire(input).unwrap();
            for _ in 0..2 {
                let output = if automatic {
                    builder.lut(input_wire, binary, Some(rhs)).unwrap()
                } else {
                    let product = builder.private_mul(input_wire, rhs).unwrap();
                    builder.lut(product, unary.unwrap(), None).unwrap()
                };
                builder.output(output).unwrap();
            }
            (builder.build().unwrap(), input, rhs)
        };
        let (automatic, input_id, rhs_id) = build(true);
        let (explicit, _, _) = build(false);
        assert_eq!(automatic.id(), explicit.id());
        let Gate::Lut { lut_id, output, .. } = automatic.gates()[1] else { panic!("LUT") };
        let input = RnsBggNoiseValue {
            level: 0,
            bound: RnsBggNoiseBound { error: 5u8.into(), payload_l1: 1u8.into() },
            payload_l2_squared: 1u8.into(),
            public: false,
            range: None,
            one_hot: false,
            moments: Some(NoiseMoments {
                mean_numerator: 1.into(),
                mean_denominator: 1u8.into(),
                mean_radius: NonnegativeRational::zero(),
                variance: NonnegativeRational::new(3u8.into(), 2u8.into()).unwrap(),
            }),
        };
        let inputs = BTreeMap::from([(input_id, input.clone())]);
        let rhs = BTreeMap::from([(
            rhs_id,
            RnsBggNoiseRhs {
                payload_l1: 3u8.into(),
                payload_l2_squared: 5u8.into(),
                one_hot: None,
            },
        )]);
        let material = |level, table| RnsBggGateNoiseMaterial {
            level,
            table,
            fresh_error_gain: RightActionNoiseGain::unit(),
        };
        let helpers = [0, 2]
            .into_iter()
            .map(|step| (ProgramGateInstance::Gate { step }, material(0, None)))
            .collect::<BTreeMap<_, _>>();
        let lut_helpers = BTreeMap::from([(
            lut_id,
            material(1, Some(automatic.lut(lut_id).unwrap().commitment())),
        )]);
        let trace = |parameters: &RnsBggParameters, program: &ExponentLutProgram| {
            trace_program(
                parameters,
                program,
                &ProgramBindings::new(
                    &inputs,
                    &rhs,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &lut_helpers,
                    Some(&helpers),
                ),
                &ProgramFamilyBindings::new(),
            )
        };
        let automatic_trace = trace(&parameters, &automatic).unwrap();
        let explicit_trace = trace(&parameters, &explicit).unwrap();
        assert_eq!(automatic_trace.dependencies, explicit_trace.dependencies);
        for (gate, state) in &automatic_trace.gates {
            let expected = &explicit_trace.gates[gate];
            assert_eq!(state.bound.error, expected.bound.error);
            assert_eq!(state.bound.payload_l1, expected.bound.payload_l1);
            assert_eq!(state.moments, expected.moments);
            assert_eq!(
                state.level,
                if matches!(gate, ProgramGateInstance::Gate { step: 0 | 2 }) { 1 } else { 2 }
            );
        }
        let product =
            parameters.private_mul_noise_bound(0, &input.bound, &1u8.into(), 3u8.into()).unwrap();
        let expected = parameters.lut_noise_bound(1, &product, 4, &1u8.into(), 3u8.into()).unwrap();
        assert_eq!(automatic_trace.wires[&output].bound.error, expected.error);
        assert_eq!(automatic_trace.wires[&output].bound.payload_l1, BigUint::from(3u8));
        let products = automatic_trace
            .dependencies
            .nodes
            .iter()
            .filter_map(|node| match node {
                NoiseDependencyNode::MulPrivate { rhs, .. } => Some(rhs),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(products.len(), 2);
        assert_eq!(products[0], products[1]);
        let lut_sources = automatic_trace
            .dependencies
            .nodes
            .iter()
            .filter_map(|node| match node {
                NoiseDependencyNode::Lut { helper, .. } => Some(helper),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lut_sources.len(), 2);
        assert!(lut_sources[0].is_some());
        assert_eq!(lut_sources[0], lut_sources[1]);
        let short = RnsBggParameters::new(
            parameters.levels()[..2].to_vec(),
            parameters.encoding_error_sigma.clone(),
            parameters.operations_per_drop(),
        )
        .unwrap();
        assert!(trace(&short, &automatic).is_err());
    }

    #[test]
    fn test_rns_bgg_noise_fits_hold_and_drop_acceptance_contract() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let source = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let target = source
            .select_modulus(&source.to_crt().0[..2].iter().copied().map(BigUint::from).product())
            .unwrap();
        let parameters =
            RnsBggParameters::new(vec![source.clone(), source.clone(), target], 1, 2).unwrap();
        let hold_modulus = parameters.level(0).unwrap().modulus().as_ref().clone();
        let drop_scale = parameters.scale(1).unwrap().clone();
        let hold_error = &drop_scale / 2u8 + 1u8;
        assert!(
            parameters
                .noise_fits(0, &RnsBggNoiseBound { error: hold_error, payload_l1: BigUint::one() })
                .unwrap()
        );
        assert!(
            !parameters
                .noise_fits(
                    1,
                    &RnsBggNoiseBound {
                        error: &drop_scale / 2u8 + 1u8,
                        payload_l1: BigUint::one(),
                    },
                )
                .unwrap()
        );
        assert!(
            !parameters
                .noise_fits(
                    0,
                    &RnsBggNoiseBound {
                        error: &hold_modulus / 8u8 + 1u8,
                        payload_l1: BigUint::one(),
                    },
                )
                .unwrap()
        );
    }

    #[test]
    fn test_rns_bgg_encoding_gaussian_uses_sigma_squared_variance() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(4u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: BigUint::one(),
            operations_per_drop: 1,
        })
        .unwrap();
        let input = NonnegativeRational::zero();
        let zero = RightActionNoiseGain::from_polynomial_coefficients([0.into()]);
        let without = parameters
            .private_mul_noise_variance(0, &input, &RightActionNoiseGain::unit(), &zero)
            .unwrap();
        let with = parameters
            .private_mul_noise_variance(
                0,
                &input,
                &RightActionNoiseGain::unit(),
                &RightActionNoiseGain::unit(),
            )
            .unwrap();
        assert_eq!(with, without.add(&NonnegativeRational::new(16u8.into(), 1u8.into()).unwrap()));
    }

    #[test]
    fn test_rns_bgg_noise_trace_tracks_public_coefficients_and_one_hot_payload() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|value| value.parse().unwrap())
            .unwrap_or(16);
        let (parameters, _) = RnsBggParameters::plan(&RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 4,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
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
            RnsBggNoiseValue {
                level: 0,
                bound: RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
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
        let mut member = RnsBggNoiseValue::public(1, 2u8.into(), &[2.into(), (-1).into()]);
        assert_eq!(member.bound.payload_l1, BigUint::from(3u8));
        member.range = Some(range.clone());
        let members = BTreeMap::from([(public_members, member)]);
        let rhs = BTreeMap::from([(
            selectors,
            RnsBggNoiseRhs {
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
            RnsBggGateNoiseMaterial {
                level: 0,
                table: None,
                fresh_error_gain: RightActionNoiseGain::unit(),
            },
        )]);
        let evaluate = |rhs: &BTreeMap<crate::program::RhsFamilyId, RnsBggNoiseRhs>,
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
        assert_eq!(
            trace.joint_event_count,
            BigUint::from(n) *
                2u8 *
                (parameters.component_columns(0).unwrap() +
                    8 * parameters.component_columns(1).unwrap())
        );
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
            Err(RnsBggError::Budget("terminal range"))
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
        let mut scalar_evaluator = RnsBggNoiseEvaluator {
            parameters: &parameters,
            ranges: &ranges,
            gates: BTreeMap::new(),
        };
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
        let lut_material = RnsBggGateNoiseMaterial {
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
        let request = RnsBggPlanRequest {
            ring_dimension: n,
            widths: vec![1],
            slots: vec![4],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 1,
            max_crt_depth: 8,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 1u8.into(),
            operations_per_drop: 1,
        };
        let terminal = BTreeMap::from([(output, BigUint::from(1u8) << 35)]);
        let search = |request: &RnsBggPlanRequest| {
            RnsBggParameters::search_program(
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
        insufficient.max_crt_depth = insufficient.terminal_depth;
        assert!(matches!(search(&insufficient), Err(RnsBggError::Budget("CRT depth"))));

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
            RnsBggNoiseValue {
                level: 1,
                bound: RnsBggNoiseBound { error: 1u8.into(), payload_l1: 1u8.into() },
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
        let check_direct = |inputs: &BTreeMap<crate::program::ProgramInputId, RnsBggNoiseValue>,
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
            Err(RnsBggError::Budget("evaluation noise"))
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
            Err(RnsBggError::Budget("evaluation noise"))
        ));
        direct_inputs.get_mut(&direct_input).unwrap().moments = None;
        assert!(matches!(
            check_direct(&direct_inputs, Some(&average)),
            Err(RnsBggError::Parameters("average-case input moment model is missing"))
        ));
    }
}

impl RnsBggParameters {
    /// Section 9 LUT alignment, with bounded U and V and exact products. The
    /// numerator includes internal public-matrix rounding and outer rounding.
    /// The caller supplies the output payload bound from its domain analysis:
    /// an identity LUT on a polynomial must not be assigned a monomial bound.
    /// `fresh_error_gain` is the sum of active public multiplier l1 bounds
    /// when composing RHS ciphertexts (the slot count for monomial labels).
    /// It scales only fresh RHS error, not inherited or mask-helper error.
    pub fn lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        branches: usize,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        if branches == 0 {
            return Err(RnsBggError::Parameters("empty branch family"));
        }
        let source = self.level(level)?;
        let next = self.level(level + 1)?;
        let p = self.scale(level)?;
        let n = BigUint::from(source.ring_dimension());
        let j = BigUint::from(branches);
        let s = &n + 1u8;
        let twice_p = p * 2u8;
        let inherited = &j *
            &n *
            (BigUint::one() << next.base_bits()) *
            next.modulus_digits() *
            (&input.error + &self.encoding_error_bound) *
            2u8;
        let fresh = &j * &input.payload_l1 * fresh_error_gain * &self.encoding_error_bound;
        let rounding = ((&j + 1u8) * s + 1u8) * (p - 1u8);
        let numerator = inherited + fresh * &twice_p + rounding;
        Ok(RnsBggNoiseBound {
            error: (numerator + &twice_p - 1u8) / &twice_p,
            payload_l1: output_payload_l1,
        })
    }

    pub fn noise_fits(&self, level: usize, noise: &RnsBggNoiseBound) -> Result<bool, RnsBggError> {
        let range_fits = self.level(level)?.modulus().as_ref() > &(&noise.error * 8u8);
        // Nonterminal states must resolve adjacent p_i-spaced payloads. A
        // held logical level has p_i=1 and needs no spacing check; the last
        // level is a terminal carrier with a separately audited range.
        Ok(range_fits &&
            (level + 1 == self.levels.len() ||
                self.scale(level)? == &BigUint::one() ||
                self.scale(level)? > &(&noise.error * 2u8)))
    }

    /// Search nested prime-prefix chains using the actual gate noise transfer.
    /// At equal last two depths, retain the Pareto frontier of error and peak
    /// memory. A larger prefix modulus is tried only if smaller prefixes fail.
    pub fn plan(
        request: &RnsBggPlanRequest,
    ) -> Result<(Self, RnsBggParameterManifest), RnsBggError> {
        let mut request = request.clone();
        request.encoding_error_sigma = canonical_encoding_sigma(request.encoding_error_sigma)?;
        request.validate()?;
        let stages = request.widths.len();
        let physical_drops = stages.div_ceil(request.operations_per_drop);
        if request.max_crt_depth - request.terminal_depth < physical_drops {
            return Err(RnsBggError::Budget("CRT depth"));
        }
        let pool = DCRTPolyParams::try_new(
            request.ring_dimension,
            request.max_crt_depth,
            request.crt_bits,
            request.crt_bits.div_ceil(2).min(31) as u32,
            None,
            None,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported prime pool"))?;
        let primes = pool.to_crt().0;
        let bases = (1..=request.max_crt_depth)
            .map(|depth| {
                let modulus = primes[..depth].iter().copied().map(BigUint::from).product();
                pool.select_modulus(&modulus).expect("prime prefix in the pool")
            })
            .collect::<Vec<_>>();
        if bases[request.terminal_depth - 1].modulus().as_ref() <=
            &(2u8 * &request.terminal_value_bound)
        {
            return Err(RnsBggError::Budget("terminal range"));
        }
        let mut failure = RnsBggError::Budget("evaluation noise");
        for start in request.terminal_depth + physical_drops..=request.max_crt_depth {
            let initial = RnsBggNoiseBound {
                error: encoding_error_bound(&request.encoding_error_sigma)?,
                payload_l1: BigUint::one(),
            };
            let mut frontier =
                BTreeMap::from([((start, start), vec![(vec![start], initial, BigUint::zero())])]);
            for stage in 0..stages {
                let mut following: BTreeMap<_, Vec<(Vec<usize>, RnsBggNoiseBound, BigUint)>> =
                    BTreeMap::new();
                for ((_, high), candidates) in frontier {
                    let must_drop =
                        (stage + 1) % request.operations_per_drop == 0 || stage + 1 == stages;
                    let future_drops = ((stage + 1)..stages)
                        .filter(|operation| {
                            (*operation + 1) % request.operations_per_drop == 0 ||
                                *operation + 1 == stages
                        })
                        .count();
                    let minimum = request.terminal_depth + future_drops;
                    if high < minimum {
                        continue;
                    }
                    let lows = if must_drop {
                        (minimum..=if stage + 1 == stages {
                            request.terminal_depth
                        } else {
                            high - 1
                        })
                            .collect::<Vec<_>>()
                    } else {
                        vec![high]
                    };
                    for low in lows {
                        for (path, input, peak) in &candidates {
                            let mut extended = path.clone();
                            extended.push(low);
                            let parameters = Self::new_unchecked_schedule(
                                extended.iter().map(|depth| bases[*depth - 1].clone()).collect(),
                                request.encoding_error_sigma.clone(),
                                encoding_error_bound(&request.encoding_error_sigma)?,
                                request.operations_per_drop,
                            )?;
                            if !parameters.noise_fits(stage, input)? {
                                continue;
                            }
                            let output = parameters.lut_noise_bound(
                                stage,
                                input,
                                request.widths[stage],
                                &BigUint::from(request.slots[stage]),
                                BigUint::one(),
                            )?;
                            // Output spacing is checked when its next drop is chosen.
                            if !parameters.noise_fits(stage + 1, &output)? {
                                continue;
                            }
                            let mut peak = peak.clone();
                            if stage > 0 {
                                peak = peak.max(parameters.lut_stage_bytes(
                                    stage - 1,
                                    request.widths[stage - 1],
                                    request.slots[stage - 1],
                                )?);
                            }
                            if stage + 1 == stages {
                                peak = peak.max(parameters.lut_stage_bytes(
                                    stage,
                                    request.widths[stage],
                                    request.slots[stage],
                                )?);
                                if parameters.level(stage + 1)?.modulus().as_ref() <=
                                    &(2u8 * (&request.terminal_value_bound + &output.error))
                                {
                                    failure = RnsBggError::Budget("terminal range");
                                    continue;
                                }
                            }
                            if peak > BigUint::from(request.memory_budget_bytes) {
                                failure = RnsBggError::Budget("memory");
                                continue;
                            }
                            let states = following.entry((high, low)).or_default();
                            if states.iter().any(|(_, bound, memory)| {
                                bound.error <= output.error && *memory <= peak
                            }) {
                                continue;
                            }
                            states.retain(|(_, bound, memory)| {
                                !(output.error <= bound.error && peak <= *memory)
                            });
                            states.push((extended, output, peak));
                        }
                    }
                }
                frontier = following;
                if frontier.is_empty() {
                    break;
                }
            }
            if let Some((path, _, _)) = frontier
                .into_values()
                .flatten()
                .filter(|(path, _, _)| path.len() == stages + 1)
                .min_by(|a, b| a.2.cmp(&b.2).then_with(|| a.1.error.cmp(&b.1.error)))
            {
                let parameters = Self::new(
                    path.into_iter().map(|depth| bases[depth - 1].clone()).collect(),
                    request.encoding_error_sigma.clone(),
                    request.operations_per_drop,
                )?;
                let manifest = parameters.audit(&request)?;
                return Ok((parameters, manifest));
            }
        }
        Err(failure)
    }

    /// Graph-value storage for one LUT branch wave on the compact-digit GPU
    /// backend, excluding allocator/NTT workspaces. Previous stages can be
    /// streamed to the artifact store, so chain peak is the maximum stage.
    fn lut_stage_bytes(
        &self,
        level: usize,
        branches: usize,
        slots: usize,
    ) -> Result<BigUint, RnsBggError> {
        let high = self.level(level)?;
        let low = self.level(level + 1)?;
        let d = BigUint::from(low.modulus_digits());
        let output_d = BigUint::from(self.component_columns(level + 1)?);
        let j = BigUint::from(branches);
        let high_entries = 6u8 * &d * (&j + 1u8) + 10u8 * &output_d;
        let low_entries = 4u8 * &j * &output_d * (BigUint::from(slots) + 1u8) + 6u8 * &output_d;
        // U and V: two 2-by-2 blocks of D-by-D_out digits per branch.
        // One additional pair covers source/destination conversion scratch.
        let compact_entries = 8u8 * (&j + 1u8) * &d * &output_d;
        let compact_bytes = BigUint::from(1 + low.base_bits().div_ceil(8));
        Ok((8u8 * (high_entries * high.crt_depth() + low_entries * low.crt_depth()) +
            compact_entries * compact_bytes) *
            high.ring_dimension())
    }

    /// Recompute the certificate from public workload data and exact bases.
    /// This chain planner covers monomial LUT stages; arbitrary Program DAGs
    /// must use their full noise trace before claiming parameter acceptance.
    pub(super) fn audit(
        &self,
        request: &RnsBggPlanRequest,
    ) -> Result<RnsBggParameterManifest, RnsBggError> {
        request.validate()?;
        let stages = request.widths.len();
        if self.levels.len() != stages + 1 ||
            self.levels[0].ring_dimension() != request.ring_dimension ||
            self.encoding_error_sigma != request.encoding_error_sigma ||
            self.operations_per_drop != request.operations_per_drop ||
            self.levels[0].crt_depth() > request.max_crt_depth ||
            self.levels[stages].crt_depth() != request.terminal_depth ||
            self.levels.iter().any(|level| level.crt_bits() > request.crt_bits)
        {
            return Err(RnsBggError::Parameters("modulus chain does not match workload"));
        }
        let mut bounds = vec![RnsBggNoiseBound {
            error: self.encoding_error_bound.clone(),
            payload_l1: BigUint::one(),
        }];
        let mut bytes = BigUint::zero();
        for i in 0..stages {
            let must_drop = (i + 1) % request.operations_per_drop == 0 || i + 1 == stages;
            if (self.levels[i] == self.levels[i + 1]) == must_drop {
                return Err(RnsBggError::Parameters("modulus chain does not match drop cadence"));
            }
            if !self.noise_fits(i, &bounds[i])? {
                return Err(RnsBggError::Budget("evaluation noise"));
            }
            let bound = self.lut_noise_bound(
                i,
                &bounds[i],
                request.widths[i],
                &BigUint::from(request.slots[i]),
                BigUint::one(),
            )?;
            if !self.noise_fits(i + 1, &bound)? {
                return Err(RnsBggError::Budget("evaluation noise"));
            }
            bounds.push(bound);
            bytes = bytes.max(self.lut_stage_bytes(i, request.widths[i], request.slots[i])?);
        }
        if bytes > BigUint::from(request.memory_budget_bytes) {
            return Err(RnsBggError::Budget("memory"));
        }
        let terminal = self.level(stages)?;
        let range = (&request.terminal_value_bound + &bounds[stages].error) * 2u8;
        if terminal.modulus().as_ref() <= &range {
            return Err(RnsBggError::Budget("terminal range"));
        }
        let manifest = RnsBggParameterManifest {
            workload: request.clone(),
            ring_dimension: request.ring_dimension,
            encoding_error_sigma: request.encoding_error_sigma.clone(),
            encoding_error_bound: self.encoding_error_bound.clone(),
            operations_per_drop: request.operations_per_drop,
            levels: self.levels.iter().map(|level| level.to_crt().0).collect(),
            base_bits: self.levels.iter().map(PolyParams::base_bits).collect(),
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
pub struct RnsBggRefreshPlanRequest {
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
pub struct RnsBggRefreshPlan {
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
    /// Persistent bounded U/V, input keys, raw outputs, and shared helpers on
    /// the compact-digit GPU backend. This is artifact storage, not peak VRAM.
    pub terminal_artifact_bytes: BigUint,
}

impl RnsBggParameters {
    /// Solve counts against the same aggregate-before-ModDown recurrence used
    /// by evaluation, including decoder noise exactly once. Large target
    /// weights do not multiply these errors: RHS errors are sampled afterwards.
    /// Each input is a traced Program value (or an input sampling contract)
    /// under these parameters, supplying one terminal member envelope. Data and
    /// PRF may come from different Programs. Setup establishes this binding;
    /// the planner does not rerun either Program or inspect ciphertexts.
    pub fn plan_refresh(
        &self,
        request: &RnsBggRefreshPlanRequest,
        data: &RnsBggNoiseValue,
        prf: &RnsBggNoiseValue,
        average: Option<&AverageCaseConfig>,
    ) -> Result<RnsBggRefreshPlan, RnsBggError> {
        if data.level != prf.level {
            return Err(RnsBggError::Parameters("refresh inputs need a common terminal level"));
        }
        let input_level = data.level;
        let source = self.level(input_level)?;
        let output = self.level(input_level + 1)?;
        if output.crt_depth() < 2 || request.decoder_eta == 0 {
            return Err(RnsBggError::Parameters("refresh needs two CRT primes and decoder error"));
        }
        let q = output.modulus();
        let spacings =
            output.to_crt().0.iter().map(|prime| q.as_ref() / *prime).collect::<Vec<_>>();
        let mut digits = RefreshDigitPlan::new(
            request.base,
            &vec![1u8.into(); spacings.len()],
            &request.fresh_support,
        )
        .map_err(|_| RnsBggError::Parameters("refresh digit support"))?;
        let coordinates =
            2usize * self.component_columns(input_level + 1)? * source.ring_dimension() as usize;
        let prf_coordinates = (0..=input_level).try_fold(BigUint::zero(), |sum, level| {
            Ok::<_, RnsBggError>(
                sum + BigUint::from(self.component_columns(level)?) *
                    self.level(level)?.ring_dimension() *
                    2u8,
            )
        })?;
        let average = average.filter(|config| config.allow_average_acceptance);
        let tail =
            |plan: &RefreshDigitPlan| -> Result<Option<NonnegativeRational>, DigitPlanError> {
                average
                    .map(|config| {
                        // Final exposed coordinates and every PRF state coordinate.
                        // The count follows the current fixed-point digit schedule.
                        let outputs = (plan.mask_counts.iter().sum::<usize>() + plan.fresh_count) *
                            coordinates;
                        let events =
                            &request.observations + BigUint::from(outputs) * &prf_coordinates;
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
                DigitPlanError::MaskRange => RnsBggError::Budget("refresh mask spacing"),
                DigitPlanError::FreshRange => RnsBggError::Budget("refresh fresh-error support"),
                DigitPlanError::Parameters => {
                    RnsBggError::Parameters("refresh digit/error workload")
                }
            })?;
        let fresh_error = digits.fresh_interval().1.magnitude().clone();
        if !self.noise_fits(
            input_level + 1,
            &RnsBggNoiseBound { error: fresh_error, payload_l1: data.bound.payload_l1.clone() },
        )? {
            return Err(RnsBggError::Budget("refreshed state scale"));
        }
        let masks = digits.mask_counts.iter().map(|count| BigUint::from(*count)).sum::<BigUint>();
        let distinct_prf_outputs = (&masks + digits.fresh_count) * coordinates;
        let terminal_digit_emissions =
            (&masks + BigUint::from(digits.fresh_count) * spacings.len()) * coordinates;
        let high = BigUint::from(source.crt_depth());
        let low = BigUint::from(output.crt_depth());
        let d = BigUint::from(output.modulus_digits());
        let out_d = BigUint::from(self.component_columns(input_level + 1)?);
        let compact_bytes = BigUint::from(1 + output.base_bits().div_ceil(8));
        let action_bytes = 8u8 * &d * &out_d * compact_bytes;
        let key_and_raw_bytes = 32u8 * (&d + &out_d) * &high;
        let digit_bytes =
            &terminal_digit_emissions * (&action_bytes * request.prf_branches + &key_and_raw_bytes);
        let data_bytes = BigUint::from(spacings.len()) *
            (&action_bytes * request.data_branches + &key_and_raw_bytes);
        // Captured mask matrices are stored once per structural group; input
        // keys and raw term outputs remain per member. CRT target/offset rows
        // retain the full resumable output width.
        let shared_bytes = BigUint::from(spacings.len()) *
            (32u8 *
                &d *
                &high *
                (BigUint::from(request.prf_branches) * 2u8 + request.data_branches) +
                96u8 * &out_d * &high +
                48u8 * &out_d * &low) +
            32u8 * &out_d * &low;
        let terminal_artifact_bytes =
            (digit_bytes + data_bytes + shared_bytes) * source.ring_dimension();
        Ok(RnsBggRefreshPlan {
            model: if average.is_some() {
                NoiseModelKind::AverageCase
            } else {
                NoiseModelKind::WorstCase
            },
            z_squared: tail(&digits)
                .map_err(|_| RnsBggError::Parameters("refresh event budget"))?,
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
    fn test_rns_bgg_refresh_plan_uses_emitted_digits_in_noise_fixed_point() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(4);
        let (parameters, _) = RnsBggParameters::plan(&super::super::RnsBggPlanRequest {
            ring_dimension: n,
            // One composed PRF product precedes refresh and continuation.
            widths: vec![1, n as usize, 1],
            slots: vec![2, 1, 1],
            encoding_error_sigma: bigdecimal::BigDecimal::from(2u8),
            crt_bits: 30,
            terminal_depth: 3,
            max_crt_depth: 24,
            memory_budget_bytes: u64::MAX,
            terminal_value_bound: 32u8.into(),
            operations_per_drop: 1,
        })
        .unwrap();
        let request = RnsBggRefreshPlanRequest {
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
                RnsBggNoiseValue {
                    level,
                    bound: RnsBggNoiseBound { error: 2u8.into(), payload_l1: 1u8.into() },
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
        let (material, rhs_bound) = RnsBggGateNoiseMaterial::from_one_hot_labels(
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
pub struct RnsBggTerminalNoise {
    pub error: BigUint,
    pub coefficient_bound: BigUint,
    pub branches: usize,
}

impl RnsBggParameters {
    /// Setup-time integer recovery and evaluation-noise checks at the terminal
    /// output level. Weights affect the required range, never the error gain.
    pub fn terminal_noise_fits(
        &self,
        output_level: usize,
        bound: &RnsBggTerminalNoise,
    ) -> Result<bool, RnsBggError> {
        let q = self.level(output_level)?.modulus();
        let error = &bound.error;
        Ok(q.as_ref() > &((&bound.coefficient_bound + error) * 2u8) && q.as_ref() > &(error * 8u8))
    }

    /// Conditional terminal moments from the same raw aggregate as
    /// `terminal_noise_bound`. H1/H3/H4 decorrelate centered right actions
    /// across freshly prepared targets, including reused input/helper errors;
    /// this is an explicit heuristic, not independence inferred from loop indices.
    /// Fresh target Gaussian errors have distinct sampling identities. All internal
    /// remainders are counted, while the two outer remainders occur only once.
    pub fn terminal_noise_moments<'a>(
        &self,
        level: usize,
        stages: impl IntoIterator<Item = (&'a RnsBggNoiseValue, usize)>,
    ) -> Result<NoiseMoments, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let low = self.level(level + 1)?;
        let p_squared = p * p;
        let gaussian = self.encoding_error_variance();
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
                .ok_or(RnsBggError::Parameters("terminal input moment model is missing"))?;
            let j = BigUint::from(count);
            let input_helper = m.variance.add(&gaussian);
            let inherited_helper = NonnegativeRational::new(
                &input_helper.numerator *
                    &j *
                    &n *
                    (BigUint::one() << (2 * low.base_bits())) *
                    low.modulus_digits(),
                &input_helper.denominator * 2u8 * &p_squared,
            )
            .unwrap();
            let fresh = gaussian.scaled(&(&j * &input.payload_l2_squared));
            let internal_rounding = NonnegativeRational::new(
                &j * (&n * 2u8 + 3u8) * (&p_squared - 1u8),
                36u8 * &p_squared,
            )
            .unwrap();
            let mean = m.linear_image(
                &NonnegativeRational::new(
                    &j * &n * (BigUint::one() << low.base_bits()) * low.modulus_digits(),
                    p.clone(),
                )
                .unwrap(),
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
        stages: impl IntoIterator<Item = (&'a RnsBggNoiseBound, usize, &'a BigUint)>,
        coefficient_bound: BigUint,
    ) -> Result<RnsBggTerminalNoise, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let low = self.level(level + 1)?;
        let twice_p = p * 2u8;
        let mut numerator = (&n + 2u8) * (p - 1u8);
        let mut branches = 0usize;
        for (input, count, gain) in stages {
            if count == 0 {
                return Err(RnsBggError::Parameters("empty terminal stage"));
            }
            branches =
                branches.checked_add(count).ok_or(RnsBggError::Budget("terminal branches"))?;
            let j = BigUint::from(count);
            numerator += &j *
                &n *
                (BigUint::one() << low.base_bits()) *
                low.modulus_digits() *
                (&input.error + &self.encoding_error_bound) *
                2u8;
            numerator += &twice_p * &j * &input.payload_l1 * gain * &self.encoding_error_bound;
            numerator += &j * (&n + 1u8) * (p - 1u8);
        }
        if branches == 0 {
            return Err(RnsBggError::Parameters("empty terminal batch"));
        }
        Ok(RnsBggTerminalNoise {
            error: (numerator + &twice_p - 1u8) / twice_p,
            coefficient_bound,
            branches,
        })
    }
}
