//! Noise contracts, transfer rules, and audits for an RNS-BGG Program.
//!
//! This module is the analysis side of the RNS-BGG implementation. It builds
//! a symbolic graph from public setup contracts; it does not execute a GPU
//! graph or inspect ciphertext samples. Deterministic bounds are kept separate
//! from conditional average-case moments. Fresh encoding errors use the
//! configured Gaussian sigma and runtime cutoff; a decoder adds the unmask
//! error of its trapdoor preimage (see the `decoder` submodule).
#![warn(missing_docs)]
use super::{RhsMode, RnsBggError, RnsBggParameters, workload::RnsBggDerivedWorkload};
use crate::{
    decoder::{
        decoding::{Decoder, DecoderDeclaration, DecoderError},
        simulation::NoiseModelKind,
    },
    noise_utils::{
        NoiseCorrelation, NoiseHeuristicId, NoiseHeuristicLedger, NoiseMomentPolicy, NoiseMoments,
        NoiseSourceProvenance, NonnegativeRational, RightActionNoiseGain, ceil_div_nonnegative,
        encoding_variance_from_sigma, stable_source_id, variance_log2, zero_mean_moments,
    },
    program::{
        Environment, ExponentLUTProgramLowering, ExponentLutProgram, FamilyExecution, FamilyRange,
        InputBinding, LutId, LutTable, ProgramLowerError, ProgramLoweringContext, RhsId, RhsRef,
        WireId,
    },
};
use bigdecimal::BigDecimal;
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

use crate::noise_utils::{NoiseBreakdown, ReductionCounts, SampleCoordinate};

mod audit;
mod build;
mod decoder;
mod dynamic;
mod evaluator;
mod rules;

pub(crate) use self::decoder::simulate_decoder_noise;
pub use self::{
    audit::{RnsBggNoiseTrace, RnsBggParameterManifest, RnsBggPlanRequest},
    decoder::RnsBggDecoderNoise,
};

/// Evaluator-owned noise contracts for one Program lowering.
///
/// These bindings deliberately stay out of the shared Program API: RHS and
/// helper material are concrete noise-analysis inputs, not logical wires.
#[derive(Default)]
pub struct RnsBggNoiseBindings {
    /// Input contracts retain the declaration's scalar/family shape.  The
    /// family envelope is one symbolic member plus its support contract; it
    /// is never expanded into one sampled value per loop index.
    pub inputs: BTreeMap<WireId, InputBinding<RnsBggNoiseValue, RnsBggNoiseFamily>>,
    /// RHS contracts are references to logical inputs.  A family has one
    /// logical level and one aggregate support bound across all members.
    pub rhs: BTreeMap<RhsId, InputBinding<RnsBggNoiseRhs, RnsBggNoiseFamily<RnsBggNoiseRhs>>>,
    /// How the Program's outputs are consumed; the simulated outputs are the
    /// carriers [`RnsBggParameters::output_carrier`] resolves for it.
    pub output_use: super::RnsOutputUse,
    /// Dynamic: the package error contracts of the Program's setup.
    pub dynamic: Option<RnsBggDynamicNoise>,
}

/// The error contract of one RHS package read by a Program.
#[derive(Clone, Debug)]
pub struct RnsBggPackageNoise {
    /// The setup coordinate of the package.
    pub coordinate: u32,
    /// Coefficient bound of the gadget ciphertext error `e_C`.
    pub ciphertext_error: BigUint,
    /// The Gaussian sigma of `e_C` when encryption samples it; a supplied
    /// ciphertext has only its bound.
    pub ciphertext_sigma: Option<BigDecimal>,
}

/// The Dynamic material contracts of one Program: the setup's sampling
/// contracts of companions and helpers, each RHS package's ciphertext error
/// contract, and the setup automorphism group. Built from a setup
/// ([`Self::from_setup`]), its contracts are the setup's, so a sampled width
/// and an analysed bound cannot differ; parameter planning, which precedes
/// the setup, states them for a Program directly ([`Self::for_program`]).
/// `namespace` names the package sources: repeated reads of one package are
/// one sampled object.
#[derive(Clone, Debug)]
pub struct RnsBggDynamicNoise {
    /// The namespace of the package sources: the setup identity.
    pub namespace: [u8; 32],
    /// The sampling contracts of companions and helpers.
    pub distributions: super::RnsBggDynamicDistributions,
    /// The package contract of every RHS of the Program.
    pub rhs: BTreeMap<RhsId, RnsBggPackageNoise>,
    /// The setup automorphism group, ascending.
    pub automorphisms: Vec<usize>,
}

impl RnsBggDynamicNoise {
    /// The contracts of `program`, bound to `setup` through `binding`.
    pub fn from_setup(
        setup: &super::RnsBggDynamicSetup,
        binding: &super::RnsBggDynamicBinding,
        program: &ExponentLutProgram,
    ) -> Result<Self, RnsBggError> {
        binding.validate(setup, program)?;
        let distributions = setup.distributions();
        let rhs = program
            .rhs_inputs()
            .keys()
            .map(|id| {
                let coordinate = binding.rhs(*id)?;
                let (ciphertext_error, ciphertext_sigma) = match &setup.rhs()[&coordinate].source {
                    super::RnsBggPackageSource::Sampled => (
                        distributions.ciphertext.cutoff().clone(),
                        Some(distributions.ciphertext.sigma().clone()),
                    ),
                    super::RnsBggPackageSource::Supplied { error_bound } => {
                        (error_bound.clone(), None)
                    }
                };
                Ok((*id, RnsBggPackageNoise { coordinate, ciphertext_error, ciphertext_sigma }))
            })
            .collect::<Result<_, RnsBggError>>()?;
        Ok(Self {
            namespace: setup.identity(),
            distributions: distributions.clone(),
            rhs,
            automorphisms: setup.automorphisms().to_vec(),
        })
    }

    /// The planning contracts of `program` before its setup exists: every
    /// RHS `RhsId(c)` at coordinate `c`, sampled under `distributions`, and
    /// the automorphism group of `layout` on rings of dimension
    /// `ring_dimension`.
    pub fn for_program(
        namespace: [u8; 32],
        program: &ExponentLutProgram,
        distributions: super::RnsBggDynamicDistributions,
        layout: crate::program::ExpLayout,
        ring_dimension: usize,
    ) -> Result<Self, RnsBggError> {
        let automorphisms = layout
            .automorphism_indices(ring_dimension)
            .map_err(|_| RnsBggError::Parameters("invalid automorphism layout"))?;
        let rhs = program
            .rhs_inputs()
            .keys()
            .map(|id| {
                (
                    *id,
                    RnsBggPackageNoise {
                        coordinate: id.0,
                        ciphertext_error: distributions.ciphertext.cutoff().clone(),
                        ciphertext_sigma: Some(distributions.ciphertext.sigma().clone()),
                    },
                )
            })
            .collect();
        Ok(Self { namespace, distributions, rhs, automorphisms })
    }
}

impl RnsBggNoiseBindings {
    /// Construct validated noise bindings. Family support bounds are checked
    /// against their declared active ranges before any lowering occurs.
    pub fn new(
        inputs: BTreeMap<WireId, InputBinding<RnsBggNoiseValue, RnsBggNoiseFamily>>,
        rhs: BTreeMap<RhsId, InputBinding<RnsBggNoiseRhs, RnsBggNoiseFamily<RnsBggNoiseRhs>>>,
    ) -> Result<Self, RnsBggError> {
        for family in inputs.values().filter_map(|binding| match binding {
            InputBinding::Family(family) => Some(family),
            _ => None,
        }) {
            family.validate_support()?;
        }
        for family in rhs.values().filter_map(|binding| match binding {
            InputBinding::Family(family) => Some(family),
            _ => None,
        }) {
            family.validate_support()?;
        }
        Ok(Self { inputs, rhs, output_use: super::RnsOutputUse::default(), dynamic: None })
    }

    /// These bindings with the Dynamic package contracts `dynamic`.
    pub fn with_dynamic(mut self, dynamic: RnsBggDynamicNoise) -> Self {
        self.dynamic = Some(dynamic);
        self
    }

    /// These bindings with the output use `output_use`.
    pub fn with_output_use(mut self, output_use: super::RnsOutputUse) -> Self {
        self.output_use = output_use;
        self
    }

    /// Build noise bindings from inputs in the declaration order of `program`.
    ///
    /// The declaration is the only source of wire/RHS identifiers at this
    /// boundary.  Callers provide values positionally, so gate and LUT
    /// material remains evaluator-owned and cannot be accidentally attached to
    /// the wrong lowering site.
    pub fn from_inputs(
        program: &ExponentLutProgram,
        lhs: &[RnsBggNoiseValue],
        rhs: &[RnsBggNoiseRhs],
    ) -> Result<Self, RnsBggError> {
        if lhs.len() != program.inputs().len() || rhs.len() != program.rhs_inputs().len() {
            return Err(RnsBggError::Binding("invalid noise binding"));
        }
        let mut bindings = Self::default();
        for ((id, declaration), value) in program.inputs().iter().zip(lhs) {
            match declaration.shape() {
                InputBinding::Scalar(()) => {
                    bindings.inputs.insert(*id, InputBinding::Scalar(value.clone()));
                }
                InputBinding::Family(count) => {
                    let range = FamilyRange::full(count.clone())
                        .map_err(|_| RnsBggError::Binding("invalid family range"))?;
                    bindings.inputs.insert(
                        *id,
                        InputBinding::Family(RnsBggNoiseFamily {
                            element: value.clone(),
                            range,
                            max_nonzero_payloads: count
                                .clone()
                                .evaluate(&ParamEnv::default())
                                .ok()
                                .and_then(|n| n.to_biguint())
                                .ok_or(RnsBggError::Binding("family count is not concrete"))?,
                        }),
                    );
                }
            }
        }
        for ((id, declaration), value) in program.rhs_inputs().iter().zip(rhs) {
            match declaration {
                InputBinding::Scalar(()) => {
                    bindings.rhs.insert(*id, InputBinding::Scalar(value.clone()));
                }
                InputBinding::Family(count) => {
                    let range = FamilyRange::full(count.clone())
                        .map_err(|_| RnsBggError::Binding("invalid RHS family range"))?;
                    let n = count
                        .clone()
                        .evaluate(&ParamEnv::default())
                        .ok()
                        .and_then(|n| n.to_biguint())
                        .ok_or(RnsBggError::Binding("RHS family count is not concrete"))?;
                    bindings.rhs.insert(
                        *id,
                        InputBinding::Family(RnsBggNoiseFamily {
                            element: value.clone(),
                            range,
                            max_nonzero_payloads: n,
                        }),
                    );
                }
            }
        }
        Ok(bindings)
    }
}

/// Deterministic coefficient-error and payload envelope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggNoiseBound {
    /// Coefficient error bound.
    pub error: BigUint,
    /// Payload L1 bound.
    pub payload_l1: BigUint,
}

impl RnsBggNoiseBound {
    /// Add polynomial envelopes.
    pub fn add(&self, rhs: &Self) -> Self {
        Self { error: &self.error + &rhs.error, payload_l1: &self.payload_l1 + &rhs.payload_l1 }
    }
    /// Apply a public multiplier.
    pub fn mul_public(&self, multiplier_l1: &BigUint) -> Self {
        Self { error: &self.error * multiplier_l1, payload_l1: &self.payload_l1 * multiplier_l1 }
    }
    /// Apply a public action decomposition.
    pub fn mul_public_with_action(
        &self,
        public: &Self,
        plaintext_l1: &BigUint,
        action: &RightActionNoiseGain,
    ) -> Self {
        Self {
            error: &self.error * plaintext_l1 + &public.error * &action.l1,
            payload_l1: &self.payload_l1 * &public.payload_l1,
        }
    }
}

/// A symbolic wire value used by deterministic and average-case noise audits.
/// Knowledge and logical level are paired by the Program/workload owners.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggNoiseValue {
    /// Deterministic coefficient and payload envelope for this value.
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
    /// Stable source/route identities for shared correlations.
    ///
    /// A value may aggregate several sampled sources, such as the members of
    /// a decoder smudge. Keeping the complete source set (rather than one
    /// representative source) lets a later combination recognize a shared
    /// sample.
    pub provenance: BTreeMap<[u8; 32], NoiseSourceProvenance>,
    /// Conservative assumptions used while composing moments.
    pub heuristics: NoiseHeuristicLedger,
    /// Source-aware contributions whose total is `moments`. `None` means
    /// only the total envelope is known; it is then treated as one opaque
    /// contribution, coherent with every other.
    #[serde(skip)]
    pub(crate) breakdown: Option<Arc<NoiseBreakdown>>,
}

impl RnsBggNoiseValue {
    #[cfg(test)]
    fn single_provenance(
        provenance: NoiseSourceProvenance,
    ) -> BTreeMap<[u8; 32], NoiseSourceProvenance> {
        let source = provenance.source;
        BTreeMap::from([(source, provenance)])
    }

    fn l2_add_bound(left: &BigUint, right: &BigUint) -> BigUint {
        // The two coefficient vectors may be aligned.  Adding their squared
        // norms is therefore not a valid envelope; 2(a²+b²) is the integer
        // Cauchy bound and remains valid without a hidden independence claim.
        (left + right) * 2u8
    }

    /// Known public coefficients provide the convolution gain directly. The
    /// coefficients are the chosen integer lift, with negacyclic terms reduced
    /// and combined before constructing this value.
    pub fn public(error_bound: BigUint, coefficients: &[BigInt]) -> Self {
        let moments = error_bound.is_zero().then(|| zero_mean_moments(NonnegativeRational::zero()));
        Self {
            breakdown: moments.as_ref().map(|_| Arc::new(NoiseBreakdown::empty())),
            moments,
            payload_l2_squared: coefficients.iter().map(|c| c.magnitude().pow(2)).sum(),
            bound: RnsBggNoiseBound {
                error: error_bound,
                payload_l1: coefficients.iter().map(|c| c.magnitude()).sum(),
            },
            provenance: BTreeMap::new(),
            heuristics: NoiseHeuristicLedger::new(),
        }
    }

    /// Construct a fresh encoding-error carrier with the configured Gaussian
    /// moment model.  The deterministic cutoff is retained in `bound`, while
    /// the conditional model records a zero mean and the exact configured
    /// variance.  `source` is caller-owned stable context (for example an
    /// input wire label), so repeated uses of the same encoding sample can be
    /// represented as a shared provenance source by later composition.
    pub fn sampled_encoding_sigma(
        sigma: &BigDecimal,
        source: &[u8],
        coefficients: &[BigInt],
    ) -> Result<Self, RnsBggError> {
        let sigma = super::canonical_encoding_sigma(sigma.clone())?;
        let source = stable_source_id(&(b"rns-bgg/encoding", source));
        let moments = zero_mean_moments(encoding_variance_from_sigma(&sigma));
        Ok(Self {
            bound: RnsBggNoiseBound {
                error: super::encoding_error_bound(&sigma)?,
                payload_l1: coefficients.iter().map(|c| c.magnitude()).sum(),
            },
            payload_l2_squared: coefficients.iter().map(|c| c.magnitude().pow(2)).sum(),
            breakdown: Some(Arc::new(NoiseBreakdown::sampled(
                source,
                "encoding",
                moments.clone(),
                true,
            ))),
            moments: Some(moments),
            provenance: BTreeMap::from([(
                source,
                NoiseSourceProvenance { source, route: Vec::new() },
            )]),
            heuristics: NoiseHeuristicLedger::new(),
        })
    }

    /// A fresh encoding-error carrier sampled under `contract`: its bound is
    /// the contract's cutoff and its moments the contract's Gaussian. A
    /// Dynamic setup samples each encoding role under its own contract.
    pub fn sampled_gaussian(
        contract: &super::RnsBggGaussian,
        source: &[u8],
        coefficients: &[BigInt],
    ) -> Self {
        let mut value = Self::sampled_encoding_sigma(contract.sigma(), source, coefficients)
            .expect("a validated Gaussian contract");
        value.bound.error = contract.cutoff().clone();
        value
    }

    /// Declare that the members of a family bound with this element are
    /// encrypted independently, each with its own error sample. Without this
    /// binding contract every member read is treated as the same sample.
    pub fn independent_family_members(mut self) -> Self {
        self.breakdown = self
            .breakdown
            .map(|breakdown| Arc::new(breakdown.as_ref().clone().independent_members()));
        self
    }

    /// The image under a right action by constant integer digits whose `l1`
    /// norm is `l1`, such as a gadget inverse `G^-1(w)`, followed by a
    /// monomial: the payload is scaled by `weight`. The digits combine the
    /// same coefficient of several columns, so the image is coherent: mean
    /// gain `l1` and variance gain `l1^2`, with no independence claim.
    pub(crate) fn constant_digit_action(mut self, l1: &BigUint, weight: &BigUint) -> Self {
        self.bound.error *= l1;
        self.bound.payload_l1 *= weight;
        self.payload_l2_squared *= weight.pow(2u32);
        let mean_gain =
            NonnegativeRational::new(l1.clone(), BigUint::one()).expect("unit denominator");
        let variance_gain = l1.pow(2u32);
        let breakdown = self.contributions("constant-digit-action").map(|terms| {
            terms.transform(|term| {
                (mean_gain.clone(), term.moments.variance.scaled(&variance_gain), false)
            })
        });
        self.with_breakdown(breakdown)
    }

    /// Replace the moments by the total of `breakdown`, when one exists.
    fn with_breakdown(mut self, breakdown: Option<NoiseBreakdown>) -> Self {
        if let Some(breakdown) = breakdown {
            self.moments = Some(breakdown.total());
            self.breakdown = Some(Arc::new(breakdown));
        }
        self
    }

    /// The source-aware contributions of this value. A value that only has
    /// total moments is one opaque contribution, coherent with every other.
    fn contributions(&self, origin: &'static str) -> Option<NoiseBreakdown> {
        match (&self.breakdown, &self.moments) {
            (Some(breakdown), Some(_)) => Some(breakdown.as_ref().clone()),
            (None, Some(moments)) => Some(NoiseBreakdown::opaque(origin, moments.clone())),
            (_, None) => None,
        }
    }

    /// Diagnostic summary of the largest contribution and the variance by
    /// contribution kind.
    fn describe_contributions(&self) -> String {
        let Some(breakdown) = self.breakdown.as_ref() else {
            return "no breakdown".into();
        };
        let dominant = breakdown.dominant().map_or("none".into(), |term| {
            format!(
                "{} {:?} {:?} log2 {}",
                term.origin,
                term.kind,
                term.sample,
                variance_log2(&term.moments.variance)
            )
        });
        let by_kind = breakdown
            .variance_by_kind()
            .iter()
            .map(|(kind, variance)| format!("{kind:?}: log2 {}", variance_log2(variance)))
            .collect::<Vec<_>>()
            .join(", ");
        format!("dominant {dominant}; {by_kind}")
    }

    /// Construct a fresh encoding-error carrier with a parameter set's
    /// already-validated sigma.  This is a convenience wrapper around
    /// [`Self::sampled_encoding_sigma`].
    pub fn sampled_encoding(
        parameters: &RnsBggParameters,
        source: &[u8],
        coefficients: &[BigInt],
    ) -> Self {
        Self::sampled_encoding_sigma(&parameters.encoding_error_sigma, source, coefficients)
            .expect("validated encoding sigma")
    }

    /// Convert a public-role noise carrier along the same alignment transfer
    /// used by public-key and encoding preparation. This helper only changes
    /// the symbolic level and noise envelope; it does not reinterpret a
    /// private carrier or manufacture a missing secret convolution.
    pub fn align_public(
        &self,
        parameters: &RnsBggParameters,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &RnsBggNoiseValue,
        public_key_rounding: &RnsBggNoiseValue,
        mask_secret: &[RnsBggMaskSecretComponent],
        policy: &NoiseMomentPolicy,
    ) -> Result<Self, RnsBggError> {
        if *alpha == BigUint::one() {
            return Ok(self.clone());
        }
        let transfer = parameters.public_alignment_noise_transfer(
            self,
            h,
            alpha,
            ciphertext_rounding,
            public_key_rounding,
            mask_secret,
            policy,
        )?;
        Ok(transfer)
    }
}

/// Public payload and selector contract for one private-product RHS input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggNoiseRhs {
    /// L1 payload bound of the RHS plaintext.
    pub payload_l1: BigUint,
    /// Squared coefficient-norm bound of the RHS plaintext.
    pub payload_l2_squared: BigUint,
}

/// One secret component participating in the public-alignment convolution
/// `s * rho_A`. `coefficient_l1` is the deterministic hard-action norm. The
/// second field is a centered second-moment action; it may be non-integral for
/// a random secret model and therefore uses exact rationals.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggMaskSecretComponent {
    /// L1 convolution gain used by the hard bound.
    pub coefficient_l1: BigUint,
    /// Sum of squared coefficients, or their modelled second moments.
    pub coefficient_l2_squared: NonnegativeRational,
}

impl RnsBggMaskSecretComponent {
    /// Construct a component from a deterministic integer coefficient norm.
    pub fn deterministic(coefficient_l1: BigUint, coefficient_l2_squared: BigUint) -> Self {
        Self {
            coefficient_l1,
            coefficient_l2_squared: NonnegativeRational::new(
                coefficient_l2_squared,
                BigUint::one(),
            )
            .expect("unit second-moment denominator"),
        }
    }
}

impl RnsBggNoiseValue {
    /// Combine two converted carriers while preserving the declared route
    /// correlation. This is used when one source public encoding feeds several
    /// labels/buckets; a different route does not imply a fresh sampler.
    pub fn combine(&self, rhs: &Self, correlation: NoiseCorrelation) -> Self {
        let same_source = self.provenance.keys().any(|source| rhs.provenance.contains_key(source));
        let correlation = if same_source && matches!(correlation, NoiseCorrelation::Independent) {
            NoiseCorrelation::SharedSource
        } else {
            correlation
        };
        let moments = match (&self.moments, &rhs.moments) {
            (Some(left), Some(right)) => Some(left.add_with_covariance(right, correlation)),
            _ => None,
        };
        let mut provenance = self.provenance.clone();
        for (source, right) in &rhs.provenance {
            if let Some(left) = provenance.get_mut(source) {
                left.route.extend_from_slice(&right.route);
            } else {
                provenance.insert(*source, right.clone());
            }
        }
        let mut heuristics = self.heuristics.clone();
        for id in rhs.heuristics.ids() {
            heuristics.insert(*id);
        }
        if matches!(correlation, NoiseCorrelation::Unknown) {
            heuristics.insert(NoiseHeuristicId::UnknownCovarianceBound);
        }
        Self {
            bound: self.bound.add(&rhs.bound),
            payload_l2_squared: Self::l2_add_bound(
                &self.payload_l2_squared,
                &rhs.payload_l2_squared,
            ),
            moments,
            provenance,
            heuristics,
            breakdown: None,
        }
    }
}

/// Symbolic family contract. `max_nonzero_payloads` bounds payload support,
/// never sampled error support.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggNoiseFamily<T = RnsBggNoiseValue> {
    /// One symbolic family member.
    pub element: T,
    /// Active family range.
    pub range: FamilyRange,
    /// Maximum number of nonzero payload members in this range.
    pub max_nonzero_payloads: BigUint,
}

impl<T> RnsBggNoiseFamily<T> {
    fn validate_support(&self) -> Result<BigUint, RnsBggError> {
        let n = self
            .range
            .count()
            .evaluate(&ParamEnv::default())
            .ok()
            .and_then(|value| value.to_biguint())
            .ok_or(RnsBggError::Parameters("noise family needs a concrete active count"))?;
        if self.max_nonzero_payloads > n {
            return Err(RnsBggError::Binding("family support exceeds active range"));
        }
        Ok(n)
    }
}

/// Stateless family adapter shared by workload and noise lowerings.
#[derive(Default)]
pub struct RnsBggNoiseFamilies {
    /// Number of enclosing parallel families of the member being lowered.
    depth: usize,
}

/// Evaluator-owned symbolic noise state.
pub struct RnsBggNoiseEvaluator<'a> {
    parameters: &'a RnsBggParameters,
    program: &'a ExponentLutProgram,
    workload: &'a RnsBggDerivedWorkload,
    bindings: &'a RnsBggNoiseBindings,
    trace: RnsBggNoiseTrace,
    current_site: Option<crate::program::LoweringSite>,
    event_multiplier: BigUint,
    finalized_outputs: BTreeMap<WireId, RnsBggNoiseValue>,
    /// The decoder declaration when this evaluator lowers its smudge Program.
    decoder_smudge: Option<Arc<DecoderDeclaration>>,
    /// The unmask trapdoor sigma when this evaluator composes a decoder.
    unmask_trapdoor_sigma: Option<f64>,
    /// Lane index of each enclosing parallel family, innermost last.
    lane_indices: Vec<IntExpr>,
    /// Wires whose payload is nonzero in at most `count` lanes of the family
    /// at `depth` (the number of enclosing parallel families), because they
    /// are weighted by distinct members of a sparse RHS family.
    payload_support: BTreeMap<WireId, PayloadSupport>,
    /// How the family sums of this Program combined their contributions.
    reduction_counts: ReductionCounts,
}

#[derive(Clone, Debug)]
struct PayloadSupport {
    depth: usize,
    count: BigUint,
}

/// Common API noise-stage adapter.  It owns only graph handles and the
/// evaluator; sampled matrices and decoder artifacts stay at their stages.
pub struct RnsNoiseBuild<'a> {
    evaluator: Option<RnsBggNoiseEvaluator<'a>>,
    families: RnsBggNoiseFamilies,
    environment: Environment<
        (WireId, RnsBggNoiseValue),
        RhsRef<IntExpr>,
        (WireId, RnsBggNoiseFamily),
        (RhsId, FamilyRange),
    >,
    pending_decoder: Option<RnsBggDecoderNoise>,
}

#[cfg(test)]
mod test {
    use super::*;

    pub(super) fn value(payload: &[i64]) -> RnsBggNoiseValue {
        RnsBggNoiseValue::public(
            BigUint::zero(),
            &payload.iter().copied().map(BigInt::from).collect::<Vec<_>>(),
        )
    }

    pub(super) fn nested_alignment_parameters() -> RnsBggParameters {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![high, low], 1, 1).unwrap()
    }

    #[test]
    fn add_payload_l2_is_conservative_for_aligned_vectors() {
        let left = value(&[1, 1]);
        let right = value(&[1, 1]);
        let combined = left.combine(&right, NoiseCorrelation::SharedSource);
        assert_eq!(combined.payload_l2_squared, BigUint::from(8u8));
    }

    #[test]
    fn family_support_rejects_more_active_members_than_range() {
        let range = FamilyRange::full(IntExpr::constant(2)).expect("valid family range");
        let family = RnsBggNoiseFamily {
            element: value(&[1]),
            range,
            max_nonzero_payloads: BigUint::from(3u8),
        };
        assert!(family.validate_support().is_err());
    }

    #[test]
    fn family_support_preserves_authorized_sparse_budget() {
        let range = FamilyRange::full(IntExpr::constant(4)).expect("valid family range");
        let family = RnsBggNoiseFamily {
            element: value(&[1]),
            range,
            max_nonzero_payloads: BigUint::from(1u8),
        };
        assert_eq!(family.validate_support().unwrap(), BigUint::from(4u8));
        assert_eq!(family.max_nonzero_payloads, BigUint::from(1u8));
    }

    #[test]
    fn cadence_maps_logical_levels_to_equal_modulus_windows() {
        let depths = (0..=8)
            .map(|level| crate::rns_bgg::planned_crt_depth(level, 3, 1, 6, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(depths, vec![6, 6, 6, 5, 5, 5, 4, 4, 4]);
    }

    #[test]
    fn sampled_encoding_exposes_zero_mean_variance_and_stable_provenance() {
        let parameters = nested_alignment_parameters();
        let first =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"input-0", &[1.into(), (-2).into()]);
        let repeated =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"input-0", &[1.into(), (-2).into()]);
        let distinct =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"input-1", &[1.into(), (-2).into()]);
        let moments = first.moments.as_ref().expect("sampled encoding moments");

        assert_eq!(moments.mean_numerator, BigInt::zero());
        assert_eq!(moments.mean_denominator, BigUint::one());
        assert_eq!(moments.mean_radius, NonnegativeRational::zero());
        assert_eq!(moments.variance, parameters.encoding_error_variance());
        assert_eq!(first.bound.error, *parameters.encoding_error_bound());
        assert_eq!(first.provenance, repeated.provenance);
        assert_ne!(first.provenance, distinct.provenance);
    }
}

// These assertions were coupled to the pre-lowering bindings API
// (family input bindings, RHS values, and explicit gate positions).  The current
// evaluator keeps those details private to the lowering context.
