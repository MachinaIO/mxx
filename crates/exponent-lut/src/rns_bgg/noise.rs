//! Noise contracts, transfer rules, and audits for an RNS-BGG Program.
//!
//! This module is the analysis side of the RNS-BGG implementation. It builds
//! a symbolic graph from public setup contracts; it does not execute a GPU
//! graph or inspect ciphertext samples. Deterministic bounds are kept separate
//! from conditional average-case moments. Fresh encoding errors use the
//! configured Gaussian sigma and runtime cutoff, while refresh decoder noise
//! remains the separate direct-decoder residual contract carried by the
//! refresh noise parameters.
#![warn(missing_docs)]
use super::{RnsBggError, RnsBggParameters};
use crate::{
    noise_refresh::{
        prf::PrfLabel,
        refresh::{NoiseRefresh, RefreshDeclaration, RefreshError, RefreshPrfLabel},
        simulation::{NoiseModelKind, RefreshNoiseParameters},
    },
    noise_utils::{
        NoiseCorrelation, NoiseHeuristicId, NoiseHeuristicLedger, NoiseMomentPolicy, NoiseMoments,
        NoiseSourceProvenance, NonnegativeRational, RightActionNoiseGain,
    },
    program::{
        Environment, ExponentLUTProgramLowering, ExponentLutProgram, FamilyExecution, FamilyRange,
        InputBinding, LutId, LutTable, ProgramLowerError, ProgramLoweringContext, RhsId, RhsRef,
        WireId,
    },
};
use bigdecimal::BigDecimal;
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_ir_core::{IntExpr, ParamEnv, types::WireType};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

use crate::noise_utils::{NoiseBreakdown, ReductionCounts, SampleCoordinate};

mod bigdecimal_string {
    use bigdecimal::BigDecimal;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::str::FromStr;

    pub fn serialize<S>(value: &BigDecimal, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D>(deserializer: D) -> Result<BigDecimal, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        BigDecimal::from_str(&value).map_err(serde::de::Error::custom)
    }
}

fn zero_mean_moments(variance: NonnegativeRational) -> NoiseMoments {
    NoiseMoments {
        mean_numerator: BigInt::zero(),
        mean_denominator: BigUint::one(),
        mean_radius: NonnegativeRational::zero(),
        variance,
    }
}

fn encoding_variance_from_sigma(sigma: &BigDecimal) -> NonnegativeRational {
    let (mut numerator, exponent) = sigma.as_bigint_and_exponent();
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

fn ceil_div_nonnegative(numerator: &BigUint, denominator: &BigUint) -> BigUint {
    debug_assert!(!denominator.is_zero());
    if numerator.is_zero() {
        BigUint::zero()
    } else {
        (numerator + denominator - 1u8) / denominator
    }
}

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

fn refresh_crt_weight(plaintext_moduli: &[BigUint], slot: usize) -> Result<BigUint, RnsBggError> {
    let total = plaintext_moduli.iter().product::<BigUint>();
    let qt = plaintext_moduli.get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let base = &total / qt;
    let inverse = mxx_backends::utils::mod_inverse_biguints(&(&base % qt), qt)
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    Ok(base * inverse)
}

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
        Ok(Self { inputs, rhs })
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

    /// Bind the checked envelope produced by a refresh as a scalar input for
    /// a subsequent Program.  The declaration and trace are both required so
    /// a stale trace, wrong target ring, or wrong output schema cannot be
    /// reinterpreted as an ordinary level-zero input.
    pub fn bind_refreshed_input(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &RnsBggParameters,
        input_id: WireId,
        declaration: &RefreshDeclaration,
        trace: &RnsBggNoiseTrace,
    ) -> Result<(), RnsBggError> {
        if self.inputs.contains_key(&input_id) {
            return Err(RnsBggError::Binding("refreshed input is already bound"));
        }
        let input = program
            .inputs()
            .get(&input_id)
            .ok_or(RnsBggError::Binding("refreshed input is not declared"))?;
        if !matches!(input.shape(), InputBinding::Scalar(())) ||
            input.knowledge() != crate::program::AttributeKnowledge::PrivatePayload
        {
            return Err(RnsBggError::Binding("refreshed input has incompatible shape or knowledge"));
        }
        if trace.refreshed_declaration_commitment != Some(*declaration.commitment()) ||
            trace.refreshed_output_schema.as_deref() != Some(declaration.output_schema()) ||
            trace.refreshed_target_level != Some(declaration.target_level())
        {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        let target = parameters.level(declaration.target_level())?;
        if trace.refreshed_target_modulus.as_ref() != Some(target.modulus().as_ref()) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        let value = trace
            .refreshed_output
            .clone()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidOutputSchema))?;
        self.inputs.insert(input_id, InputBinding::Scalar(value));
        Ok(())
    }
}

impl RnsBggParameters {
    /// Check the terminal coefficient representation after its single final
    /// ModDown using the q/4 correctness threshold.
    pub fn coefficient_output_noise_fits(
        &self,
        level: usize,
        noise: &RnsBggNoiseValue,
    ) -> Result<bool, RnsBggError> {
        let modulus = self
            .levels
            .get(level + 1)
            .or_else(|| self.levels.get(level))
            .ok_or(RnsBggError::Parameters("level outside modulus chain"))?
            .modulus();
        Ok(modulus.as_ref() > &(&noise.bound.error * 4u8))
    }

    /// Flat-LUT noise after its single scale-and-round by `p = q_l/q_(l+1)`
    /// (spec §8.2): the branch sum of [`Self::lut_noise_bound_before_mod_down`]
    /// converted once by [`Self::lut_mod_down_bound`].
    pub fn lut_noise_bound(
        &self,
        level: usize,
        input: &RnsBggNoiseBound,
        branches: usize,
        fresh_error_gain: &BigUint,
        output_payload_l1: BigUint,
    ) -> Result<RnsBggNoiseBound, RnsBggError> {
        let before = self.lut_noise_bound_before_mod_down(
            level,
            input,
            branches,
            fresh_error_gain,
            output_payload_l1,
        )?;
        Ok(RnsBggNoiseBound {
            error: self.lut_mod_down_bound(level, &before.error)?,
            payload_l1: before.payload_l1,
        })
    }

    /// Flat-LUT noise of the summed branches before the scale-and-round, in
    /// level-`level` units: `J K (e + e_h) + J p ||x||_1 e_C` plus the `J`
    /// mask-rounding remainders `s_sigma rho`, each at most `(n+1)(p-1)/2`.
    /// The RHS fresh error is multiplied by `p`, so the one later division
    /// leaves it undivided.
    pub fn lut_noise_bound_before_mod_down(
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
        let inherited = &j *
            &n *
            (BigUint::one() << next.base_bits()) *
            next.modulus_digits() *
            (&input.error + self.encoding_error_bound()) *
            2u8;
        let fresh =
            &j * &input.payload_l1 * fresh_error_gain * self.encoding_error_bound() * p * 2u8;
        let rounding = &j * (&n + 1u8) * (p - 1u8);
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + 1u8) / 2u8,
            payload_l1: output_payload_l1,
        })
    }

    /// Scale-and-round a level-`level` LUT noise bound once by `p`, adding the
    /// public-key rounding under the mask secret and the encoding rounding,
    /// `(n+1)/2 + 1/2`: `ceil((2 e + (n+2)(p-1)) / 2p)`.
    pub fn lut_mod_down_bound(
        &self,
        level: usize,
        error: &BigUint,
    ) -> Result<BigUint, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let twice_p = p * 2u8;
        Ok((error * 2u8 + (&n + 2u8) * (p - 1u8) + &twice_p - 1u8) / twice_p)
    }

    /// Check the deterministic q/4 correctness threshold and q/2 message
    /// scaling condition.
    pub fn noise_fits(&self, level: usize, noise: &RnsBggNoiseBound) -> Result<bool, RnsBggError> {
        let q = self.level(level)?.modulus();
        Ok(q.as_ref() > &(&noise.error * 4u8) &&
            (level + 1 == self.levels().len() ||
                self.scale(level)? == &BigUint::one() ||
                self.scale(level)? > &(&noise.error * 2u8)))
    }

    /// Deterministic public-key/encoding conversion transfer for
    /// `e_t = e_s H / alpha + rho_c - s rho_A`.
    ///
    /// `h` is the actual source-to-target gadget action, `alpha` is the
    /// exact source/target modulus ratio, and the two rounding arguments are
    /// coefficientwise hard residual bounds. The payload is unchanged by a
    /// public-role conversion. An absent mask component is an error: unknown
    /// secret convolution is never silently treated as zero.
    pub fn public_alignment_noise_bound(
        &self,
        source: &RnsBggNoiseValue,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &RnsBggNoiseValue,
        public_key_rounding: &RnsBggNoiseValue,
        mask_secret: &[RnsBggMaskSecretComponent],
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        if alpha.is_zero() || h.l1.is_zero() || h.l2_squared.is_zero() {
            return Err(RnsBggError::Parameters("invalid public alignment action"));
        }
        let secret_l1: BigUint =
            mask_secret.iter().map(|component| &component.coefficient_l1).sum();
        if secret_l1.is_zero() {
            return Err(RnsBggError::Parameters("missing mask-secret components"));
        }
        let action = &source.bound.error * &h.l1;
        let divided = ceil_div_nonnegative(&action, alpha);
        let error = divided +
            &ciphertext_rounding.bound.error +
            &public_key_rounding.bound.error * secret_l1;
        Ok(RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error, payload_l1: source.bound.payload_l1.clone() },
            payload_l2_squared: source.payload_l2_squared.clone(),
            moments: None,
            provenance: source.provenance.clone(),
            heuristics: source.heuristics.clone(),
            breakdown: None,
        })
    }

    /// Conditional moment transfer for the same public alignment equation.
    ///
    /// The three signed terms are composed through `NoiseMoments::add` and
    /// `subtract`, whose Cauchy covariance rule is deliberately retained for
    /// unknown/shared sources. This method therefore never turns a missing
    /// covariance into zero. `h.l2_squared/alpha²` is the action variance,
    /// while the mask-secret contribution sums every component's second
    /// moment, including nonzero secret means when supplied by the model.
    pub fn public_alignment_noise_moments(
        &self,
        source: &NoiseMoments,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &NoiseMoments,
        public_key_rounding: &NoiseMoments,
        mask_secret: &[RnsBggMaskSecretComponent],
    ) -> Result<NoiseMoments, RnsBggError> {
        if alpha.is_zero() || h.l1.is_zero() || h.l2_squared.is_zero() {
            return Err(RnsBggError::Parameters("invalid public alignment action"));
        }
        let secret_l1: BigUint =
            mask_secret.iter().map(|component| &component.coefficient_l1).sum();
        let secret_l2_squared = mask_secret
            .iter()
            .map(|component| component.coefficient_l2_squared.clone())
            .fold(NonnegativeRational::zero(), |a, b| a.add(&b));
        if secret_l1.is_zero() {
            return Err(RnsBggError::Parameters("missing mask-secret components"));
        }
        let alpha_squared = alpha.pow(2u32);
        let source_variance = source.variance.scaled_ratio(&h.l2_squared, &alpha_squared);
        let source_gain = NonnegativeRational::new(h.l1.clone(), alpha.clone())
            .expect("positive alignment ratio");
        let source_term = source.linear_image(&source_gain, source_variance);
        let mask_variance = public_key_rounding
            .variance
            .scaled_ratio(&secret_l2_squared.numerator, &secret_l2_squared.denominator);
        let mask_gain = NonnegativeRational::new(secret_l1, BigUint::one())
            .expect("positive mask convolution gain");
        let mask_term = public_key_rounding.linear_image(&mask_gain, mask_variance);
        // e_t = x + y - z.  `subtract` preserves the sign of known means and
        // uses the conservative unknown-covariance envelope for all pairs.
        Ok(source_term
            .add_with_covariance(ciphertext_rounding, crate::noise_utils::NoiseCorrelation::Unknown)
            .subtract(&mask_term))
    }

    /// Compute hard and optional moment transfers together, attaching the
    /// assumptions used by the selected discrete/continuous residue policy.
    /// `provenance` is carried unchanged so repeated source routes can be
    /// joined by the caller without being reclassified as fresh samples.
    pub fn public_alignment_noise_transfer(
        &self,
        source: &RnsBggNoiseValue,
        h: &RightActionNoiseGain,
        alpha: &BigUint,
        ciphertext_rounding: &RnsBggNoiseValue,
        public_key_rounding: &RnsBggNoiseValue,
        mask_secret: &[RnsBggMaskSecretComponent],
        policy: &NoiseMomentPolicy,
    ) -> Result<RnsBggNoiseValue, RnsBggError> {
        // A physically equal source/target ring is an exact no-op.  In
        // particular, do not require a synthetic action or attach rounding
        // provenance merely because the caller is taking the alignment path.
        if *alpha == BigUint::one() {
            return Ok(source.clone());
        }
        let bound = self.public_alignment_noise_bound(
            source,
            h,
            alpha,
            ciphertext_rounding,
            public_key_rounding,
            mask_secret,
        )?;
        if let NoiseMomentPolicy::Discrete { alpha: policy_alpha } = policy {
            if policy_alpha != alpha {
                return Err(RnsBggError::Parameters(
                    "discrete alignment policy does not match modulus ratio",
                ));
            }
        }
        let policy_rounding = policy
            .rounding_moments()
            .map_err(|_| RnsBggError::Parameters("invalid alignment moment policy"))?;
        let moments = source
            .moments
            .as_ref()
            .map(|source| {
                let ciphertext = ciphertext_rounding.moments.as_ref().unwrap_or(&policy_rounding);
                let public_key = public_key_rounding.moments.as_ref().unwrap_or(&policy_rounding);
                self.public_alignment_noise_moments(
                    source,
                    h,
                    alpha,
                    ciphertext,
                    public_key,
                    mask_secret,
                )
            })
            .transpose()?;
        let mut provenance = source.provenance.clone();
        for rounding in [ciphertext_rounding, public_key_rounding] {
            for (source_id, identity) in &rounding.provenance {
                if let Some(existing) = provenance.get_mut(source_id) {
                    existing.route.extend_from_slice(&identity.route);
                } else {
                    provenance.insert(*source_id, identity.clone());
                }
            }
        }
        let mut heuristics = source.heuristics.clone();
        for rounding in [ciphertext_rounding, public_key_rounding] {
            for id in rounding.heuristics.ids() {
                heuristics.insert(*id);
            }
        }
        heuristics.insert(NoiseHeuristicId::H1StateUncorrelated);
        heuristics.insert(NoiseHeuristicId::UnknownCovarianceBound);
        match policy {
            NoiseMomentPolicy::Discrete { .. } => {
                heuristics.insert(NoiseHeuristicId::ExactDistribution);
            }
            NoiseMomentPolicy::Continuous => {
                heuristics.insert(NoiseHeuristicId::H2ResidueUniform);
            }
        }
        Ok(RnsBggNoiseValue {
            bound: bound.bound,
            payload_l2_squared: source.payload_l2_squared.clone(),
            moments,
            provenance,
            heuristics,
            breakdown: None,
        })
    }

    /// L1 bound and coefficient-energy cap of the signed decomposition used
    /// for a level-preserving `MulPublic`.
    ///
    /// The current-level key decomposition uses the current ring's CRT
    /// decomposition base and modulus digits. An output column has `2D`
    /// digit polynomials of `N` coefficients bounded by `B / 2`, so the L1
    /// bound is `N * D * B` and the energy `sum u_i^2` is at most
    /// `2 D N (B/2)^2 = N D B^2 / 2`. The energy is the variance gain of an
    /// error whose covariance is bounded by its variance times the identity,
    /// under the fixed-action model (H1/H4); callers use the squared L1 bound
    /// for any other error. Keeping
    /// this derivation here makes the noise recurrence use the same
    /// decomposition contract as the public-key evaluator.
    fn mul_public_action_gain(&self, level: usize) -> Result<RightActionNoiseGain, RnsBggError> {
        let current = self.level(level)?;
        let ring_dimension = BigUint::from(current.ring_dimension());
        let decomposition_columns = BigUint::from(current.modulus_digits());
        let base = BigUint::one() << current.base_bits();
        let l1 = &ring_dimension * &decomposition_columns * &base;
        let l2_squared = ring_dimension * decomposition_columns * base.pow(2) / 2u8;
        Ok(RightActionNoiseGain { l2_squared, l1 })
    }

    /// Return the physical-coordinate variance of one encoding error.
    ///
    /// The stored decimal sigma is converted exactly to a rational before it
    /// is squared. This value is used by conditional moment propagation only;
    /// deterministic acceptance continues to use the hard cutoff.
    pub fn encoding_error_variance(&self) -> NonnegativeRational {
        encoding_variance_from_sigma(&self.encoding_error_sigma)
    }

    /// Separately identified components of the direct product
    /// `e*U/p - mu*F + s*rho(A*U)/p - rho(c*U)/p`, in output-level units.
    ///
    /// Conditional model: the prepared right action is fixed (H4) and the
    /// input error deviations are uncorrelated with a common variance bound
    /// (H1); the RHS error is the encoding Gaussian; both ModDown remainders
    /// are independent centered uniforms, independent of inherited/fresh
    /// errors and the ternary secret (H2). These are average-case
    /// heuristics, not deterministic independence claims. `payload_gain`
    /// bounds the actual reduced input plaintext norms.
    pub(crate) fn private_mul_noise_terms(
        &self,
        level: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<PrivateMulNoiseTerms, RnsBggError> {
        let p = self.scale(level)?;
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let p_squared = p * p;
        // e*U/p: 2D bounded digit polynomials per output column, each with
        // N coefficients bounded by b/2. H4 conditions on the prepared U;
        // this is an energy cap, not a uniform-digit independence assumption.
        let inherited_variance_gain = NonnegativeRational::new(
            &n * next.modulus_digits() * (BigUint::one() << (2 * next.base_bits())),
            &p_squared * 2u8,
        )
        .expect("positive scale");
        let inherited_mean_gain = NonnegativeRational::new(
            &n * next.modulus_digits() * (BigUint::one() << next.base_bits()),
            p.clone(),
        )
        .expect("positive scale");
        // -mu*sum(a_j*F_j): combine both deterministic actions on each F_j.
        // Weighted F coefficients are correlated, so multiplying squared l2
        // norms is not valid. Young's two convolution bounds give a cap using
        // the retained norms. The sum of squared l1 gains is bounded by the
        // square of their sum. Monomial payloads preserve the exact fresh gain.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        let rhs_error = self.encoding_error_variance().scaled(&fresh_gain);
        // s*rho(A*C)/p - rho(c*C)/p: E||[s_bar,1]||_2^2 = 2N/3+1.
        // Adding the online remainder gives (2N/3+2)*(p^2-1)/(12*p^2).
        let rounding = NonnegativeRational::new((&n + 3u8) * (&p_squared - 1u8), 18u8 * &p_squared)
            .expect("positive scale");
        Ok(PrivateMulNoiseTerms {
            inherited_variance_gain,
            inherited_mean_gain,
            rhs_error,
            rounding,
        })
    }

    /// Total variance of the direct product when the input error, the RHS
    /// error, and the remainders are pairwise uncorrelated; the sum of the
    /// components of [`Self::private_mul_noise_terms`].
    pub fn private_mul_noise_variance(
        &self,
        level: usize,
        input_variance: &NonnegativeRational,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let terms = self.private_mul_noise_terms(level, payload_gain, fresh_error_gain)?;
        Ok(terms.inherited_variance(input_variance).add(&terms.rhs_error).add(&terms.rounding))
    }

    /// Separately identified components of the explicit LUT equation (6.6)
    /// with `branches = J`, before the scale-and-round, in level-`level`
    /// units (a factor `p` above the output units).
    ///
    /// The input error enters every branch through a different automorphism
    /// and decomposition. The flat-branch decorrelation heuristic (H3) is not
    /// assumed: the input action is the coherent envelope of the `J` branch
    /// actions. Helper and B-key errors are sampled per automorphism, so
    /// their branch images are distinct samples. Branch fresh gains describe
    /// ciphertext weighting, not fresh plaintext targets.
    pub(crate) fn lut_noise_terms(
        &self,
        level: usize,
        branches: usize,
        payload_gain: &RightActionNoiseGain,
        fresh_error_gain: &RightActionNoiseGain,
    ) -> Result<LutNoiseTerms, RnsBggError> {
        let p = self.scale(level)?.clone();
        let next = self.level(level + 1)?;
        let n = BigUint::from(next.ring_dimension());
        let j = BigUint::from(branches);
        let p_squared = &p * &p;
        let gaussian = self.encoding_error_variance();
        // sigma_k(e)*U_k and ed_k*V_k: both low-ring decompositions have 2D
        // entries per column with coefficients <= b/2. Automorphisms preserve
        // both norms.
        let branch_energy =
            &n * (BigUint::one() << (2 * next.base_bits())) * next.modulus_digits() / 2u8;
        let branch_l1 = &n * (BigUint::one() << next.base_bits()) * next.modulus_digits();
        // Combine deterministic actions on each F_k before estimating
        // convolution, just as for the direct gate.
        let fresh_gain = (payload_gain.l1.pow(2) * &fresh_error_gain.l2_squared)
            .min(&payload_gain.l2_squared * fresh_error_gain.l1.pow(2));
        // The J internal public remainders; the outer public and encoding
        // remainders occur once, in the scale-and-round after the branch sum.
        // E||[s_bar,1]||_2^2 = (2N+3)/3.
        let branch_rounding =
            NonnegativeRational::new((&n * 2u8 + 3u8) * (&p_squared - 1u8), BigUint::from(36u8))
                .expect("positive denominator");
        let outer_rounding = self.lut_public_rounding_variance(level)?.add(
            &NonnegativeRational::new(&p_squared - 1u8, 12u8 * &p_squared).expect("positive scale"),
        );
        Ok(LutNoiseTerms {
            input_variance_gain: (&j * &j) * &branch_energy,
            input_mean_gain: &j * &branch_l1,
            helper: gaussian.scaled(&branch_energy),
            fresh: gaussian.scaled(&(fresh_gain * &p_squared)),
            internal_rounding: branch_rounding.scaled(&(&j * &j)),
            scale: p,
            outer_rounding,
        })
    }

    /// Variance of the public-key rounding under the mask secret after the
    /// scale-and-round: `E||[s_bar,1]||_2^2 (p^2-1)/(12 p^2)` with
    /// `E||[s_bar,1]||_2^2 = (2N+3)/3`.
    pub fn lut_public_rounding_variance(
        &self,
        level: usize,
    ) -> Result<NonnegativeRational, RnsBggError> {
        let p = self.scale(level)?;
        let n = BigUint::from(self.level(level)?.ring_dimension());
        let p_squared = p * p;
        Ok(NonnegativeRational::new((n * 2u8 + 3u8) * (&p_squared - 1u8), 36u8 * p_squared)
            .expect("positive scale"))
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

/// Components of one private product's error; see
/// [`RnsBggParameters::private_mul_noise_terms`].
pub(crate) struct PrivateMulNoiseTerms {
    /// Variance gain of the inherited input error, `N D B^2 / (2 p^2)`.
    pub inherited_variance_gain: NonnegativeRational,
    /// Mean gain of the inherited input error, `l1(U) / p`.
    pub inherited_mean_gain: NonnegativeRational,
    /// Variance of the RHS error term `-mu * F`.
    pub rhs_error: NonnegativeRational,
    /// Variance of both ModDown remainders.
    pub rounding: NonnegativeRational,
}

impl PrivateMulNoiseTerms {
    fn inherited_variance(&self, input: &NonnegativeRational) -> NonnegativeRational {
        input.scaled_ratio(
            &self.inherited_variance_gain.numerator,
            &self.inherited_variance_gain.denominator,
        )
    }
}

/// Components of one flat LUT's error before its scale-and-round; see
/// [`RnsBggParameters::lut_noise_terms`].
pub(crate) struct LutNoiseTerms {
    /// Coherent variance gain of the input error over all branches.
    pub input_variance_gain: BigUint,
    /// Mean gain of the input error over all branches.
    pub input_mean_gain: BigUint,
    /// Variance of one branch's helper-error image.
    pub helper: NonnegativeRational,
    /// Variance of one branch's B-key error image.
    pub fresh: NonnegativeRational,
    /// Coherent variance of the branches' internal public remainders.
    pub internal_rounding: NonnegativeRational,
    /// The scale `p` of the scale-and-round.
    pub scale: BigUint,
    /// Variance of the outer public and encoding remainders, output units.
    pub outer_rounding: NonnegativeRational,
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
    /// A refresh may aggregate several independently sampled PRF labels.
    /// Keeping the complete source set (rather than one representative
    /// source) lets later CRT routes recognize that the same fresh-error
    /// sample is shared by every slot while treating distinct labels as
    /// independent.
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

    /// Declare that the members of a family bound with this element are
    /// encrypted independently, each with its own error sample. Without this
    /// binding contract every member read is treated as the same sample.
    pub fn independent_family_members(mut self) -> Self {
        self.breakdown = self
            .breakdown
            .map(|breakdown| Arc::new(breakdown.as_ref().clone().independent_members()));
        self
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

/// Collects the derived logical level map for a Program.
pub(crate) struct WorkloadCollector<'a> {
    program: &'a ExponentLutProgram,
    parameters: Option<&'a RnsBggParameters>,
    levels: BTreeMap<WireId, usize>,
    rhs_levels: BTreeMap<RhsId, usize>,
    /// For each public Program input, the lowest level at which it is
    /// consumed, or `None` once any use needs it at its own level.
    public_input_uses: BTreeMap<WireId, Option<usize>>,
    current_site: Option<crate::program::LoweringSite>,
}

impl<'a> WorkloadCollector<'a> {
    /// Record one use of `wire`. A public input consumed only as the public
    /// left operand of a product with a private right operand can be encoded
    /// at that product's level, which is set by the right operand alone;
    /// `consumed_at` is then that level. Any other use pins it.
    fn note_use(&mut self, wire: WireId, consumed_at: Option<usize>) {
        if let Some(entry) = self.public_input_uses.get_mut(&wire) {
            *entry = match (*entry, consumed_at) {
                (Some(current), Some(level)) => Some(current.min(level)),
                _ => None,
            };
        }
    }

    fn site(&self) -> Result<&crate::program::LoweringSite, RnsBggError> {
        self.current_site
            .as_ref()
            .ok_or(RnsBggError::Graph("workload lowering site is missing".into()))
    }
}

/// The single workload map retained outside the validated Program.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsBggDerivedWorkload {
    program: crate::program::ExponentLutProgramId,
    levels: BTreeMap<WireId, usize>,
    rhs_levels: BTreeMap<RhsId, usize>,
}

impl RnsBggDerivedWorkload {
    /// Derive logical levels by lowering the Program once.
    pub fn from_program(program: &ExponentLutProgram) -> Result<Self, RnsBggError> {
        Self::derive(program, None)
    }

    /// Derive logical levels while validating public operand routes against
    /// the physical modulus chain. Scheme-independent callers can continue
    /// using [`Self::from_program`]; RNS noise paths use this variant.
    pub fn from_program_with_parameters(
        parameters: &RnsBggParameters,
        program: &ExponentLutProgram,
    ) -> Result<Self, RnsBggError> {
        Self::derive(program, Some(parameters))
    }

    /// Public inputs are encoded at the level where they are first consumed,
    /// so that no stage stores or aligns them at a higher modulus than they
    /// are used at. The first pass finds those levels with every input at
    /// level zero; the second derives the workload with them.
    fn derive<'a>(
        program: &'a ExponentLutProgram,
        parameters: Option<&'a RnsBggParameters>,
    ) -> Result<Self, RnsBggError> {
        let first = Self::derive_with_input_levels(program, parameters, &BTreeMap::new())?;
        let input_levels = first
            .1
            .into_iter()
            .filter_map(|(wire, level)| level.filter(|level| *level > 0).map(|level| (wire, level)))
            .collect::<BTreeMap<_, _>>();
        if input_levels.is_empty() {
            return Ok(first.0);
        }
        Ok(Self::derive_with_input_levels(program, parameters, &input_levels)?.0)
    }

    fn derive_with_input_levels<'a>(
        program: &'a ExponentLutProgram,
        parameters: Option<&'a RnsBggParameters>,
        input_levels: &BTreeMap<WireId, usize>,
    ) -> Result<(Self, BTreeMap<WireId, Option<usize>>), RnsBggError> {
        let input_level = |id: &WireId| input_levels.get(id).copied().unwrap_or(0);
        let mut collector = WorkloadCollector {
            program,
            parameters,
            levels: program.inputs().keys().map(|id| (*id, input_level(id))).collect(),
            rhs_levels: BTreeMap::new(),
            public_input_uses: program
                .inputs()
                .iter()
                .filter(|(_, declaration)| {
                    matches!(
                        declaration.knowledge(),
                        crate::program::AttributeKnowledge::PublicPayload
                    )
                })
                .map(|(id, _)| (*id, Some(usize::MAX)))
                .collect(),
            current_site: None,
        };
        // A public input that is itself an output keeps its own level.
        for output in program.outputs() {
            let (crate::program::Operand::Scalar(wire) |
            crate::program::Operand::Member { family: wire, .. }) = output;
            collector.public_input_uses.insert(*wire, None);
        }
        let mut environment = Environment::new();
        for id in program.inputs().keys() {
            match program.inputs().get(id).expect("input declaration").shape() {
                InputBinding::Scalar(()) => {
                    environment.bind_wire(*id, InputBinding::Scalar((*id, input_level(id))))
                }
                InputBinding::Family(count) => environment.bind_wire(
                    *id,
                    InputBinding::Family((
                        *id,
                        input_level(id),
                        FamilyRange::full(count.clone())
                            .map_err(|_| RnsBggError::Binding("invalid family range"))?,
                    )),
                ),
            }
            .map_err(|_| RnsBggError::Binding("invalid workload binding"))?;
        }
        for (id, declaration) in program.rhs_inputs() {
            match declaration {
                InputBinding::Scalar(()) => environment.bind_rhs(*id, InputBinding::Scalar(*id)),
                InputBinding::Family(count) => environment.bind_rhs(
                    *id,
                    InputBinding::Family((
                        *id,
                        FamilyRange::full(count.clone())
                            .map_err(|_| RnsBggError::Binding("invalid RHS family range"))?,
                    )),
                ),
            }
            .map_err(|_| RnsBggError::Binding("invalid workload RHS binding"))?;
        }
        let mut families = RnsBggNoiseFamilies::default();
        crate::program::lower_program(program, &mut collector, &mut families, &mut environment)
            .map_err(|error| match error {
                ProgramLowerError::Backend(error) => error,
                ProgramLowerError::Program(error) => RnsBggError::Graph(error.to_string()),
                ProgramLowerError::Dsl(error) => RnsBggError::Graph(error.to_string()),
                ProgramLowerError::UnsupportedStructure { reason } => RnsBggError::Graph(reason),
            })?;
        let uses = collector
            .public_input_uses
            .iter()
            .filter(|(wire, _)| program.inputs().contains_key(wire))
            .map(|(wire, level)| (*wire, level.filter(|level| *level != usize::MAX)))
            .collect();
        Ok((
            Self {
                program: program.id(),
                levels: collector.levels,
                rhs_levels: collector.rhs_levels,
            },
            uses,
        ))
    }
    /// Return the logical level of a wire.
    pub fn level(&self, wire: WireId) -> Result<usize, RnsBggError> {
        self.levels
            .get(&wire)
            .copied()
            .ok_or(RnsBggError::Graph("missing wire logical level".into()))
    }
    /// Return the one consuming logical level of an RHS ID.
    pub fn rhs_level(&self, rhs: RhsId) -> Result<usize, RnsBggError> {
        self.rhs_levels
            .get(&rhs)
            .copied()
            .ok_or(RnsBggError::Graph("RHS has no consuming level".into()))
    }
    /// Program identity committed by this workload.
    pub fn program(&self) -> crate::program::ExponentLutProgramId {
        self.program
    }
}

impl<'a> ExponentLUTProgramLowering for WorkloadCollector<'a> {
    type Wire = (WireId, usize);
    type Rhs = RhsId;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        self.note_use(lhs.0, None);
        self.note_use(rhs.0, None);
        if lhs.1 != rhs.1 {
            return Err(RnsBggError::Parameters("add operands have different logical levels"));
        }
        let output = self.site()?.output;
        self.levels.insert(output, lhs.1);
        Ok((output, lhs.1))
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let output = self.site()?.output;
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
        let private_rhs = rhs_knowledge == crate::program::AttributeKnowledge::PrivatePayload;
        self.note_use(lhs.0, private_rhs.then_some(rhs.1));
        self.note_use(rhs.0, None);
        let target = self.parameters.map_or(Ok(rhs.1), |parameters| {
            parameters.mul_public_target_level(lhs.1, rhs.1, rhs_knowledge)
        })?;
        self.levels.insert(output, target);
        Ok((output, target))
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
        self.note_use(input.0, None);
        let output = self.site()?.output;
        let level = input.1;
        if let Some(&rhs_level) = self.rhs_levels.get(rhs) {
            if rhs_level != level {
                return Err(RnsBggError::RhsLevelMismatch {
                    rhs: *rhs,
                    expected: level,
                    actual: rhs_level,
                });
            }
        } else {
            self.rhs_levels.insert(*rhs, level);
        }
        let next = level.checked_add(1).ok_or(RnsBggError::Parameters("logical level overflow"))?;
        self.levels.insert(output, next);
        Ok((output, next))
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid LUT input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Parameters("RNS-BGG LUT requires a private input"));
        }
        self.note_use(input.0, None);
        let output = self.site()?.output;
        let next = if matches!(
            table.output_representation(),
            crate::program::LutOutputRepresentation::Coefficient
        ) {
            input.1
        } else {
            input.1.checked_add(1).ok_or(RnsBggError::Parameters("logical level overflow"))?
        };
        self.levels.insert(output, next);
        Ok((output, next))
    }
}

impl<'a> ProgramLoweringContext for WorkloadCollector<'a> {
    fn replace_site(
        &mut self,
        site: Option<crate::program::LoweringSite>,
    ) -> Option<crate::program::LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl<'a> FamilyExecution<WorkloadCollector<'a>> for RnsBggNoiseFamilies {
    type WireFamily = (WireId, usize, FamilyRange);
    type RhsFamily = (RhsId, FamilyRange);

    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "workload family index is not canonical".into(),
                ),
            ));
        }
        Ok((family.0, family.1))
    }
    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<RhsId, ProgramLowerError<RnsBggError>> {
        if index.clone().canonicalize() != *index {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::InvalidStructure(
                    "workload RHS index is not canonical".into(),
                ),
            ));
        }
        Ok(family.0)
    }
    fn parallel(
        &mut self,
        evaluator: &mut WorkloadCollector<'a>,
        range: &FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut WorkloadCollector<'a>,
            IntExpr,
        ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, ProgramLowerError<RnsBggError>> {
        let index = IntExpr::Add(Box::new(range.start().clone()), Box::new(IntExpr::LoopIndex(0)))
            .canonicalize();
        let (wire, level) = body(self, evaluator, index)?;
        Ok((wire, level, range.clone()))
    }
    fn sum(
        &mut self,
        evaluator: &mut WorkloadCollector<'a>,
        family: &Self::WireFamily,
    ) -> Result<(WireId, usize), ProgramLowerError<RnsBggError>> {
        let output = evaluator.site().map_err(ProgramLowerError::Backend)?.output;
        evaluator.levels.insert(output, family.1);
        Ok((output, family.1))
    }
}

/// Approximate `log2` of a nonnegative rational variance.
fn variance_log2(variance: &NonnegativeRational) -> i64 {
    variance.numerator.bits() as i64 - variance.denominator.bits() as i64
}

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

/// Hash a serializable sampling-domain label into a stable source identity.
///
/// Callers must include every value that distinguishes independent sampling
/// domains. Equal identities mean that the audit must conservatively combine
/// the corresponding linear actions as correlated uses of one source.
pub fn stable_source_id<T: Serialize>(value: &T) -> [u8; 32] {
    Sha256::digest(serde_json::to_vec(value).expect("stable source identity serialization")).into()
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
        let divisor = gcd_biguint(numerator.magnitude(), &denominator);
        moments.mean_numerator = numerator / BigInt::from(divisor.clone());
        moments.mean_denominator = denominator / divisor;
    }
}

fn gcd_biguint(left: &BigUint, right: &BigUint) -> BigUint {
    let mut left = left.clone();
    let mut right = right.clone();
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    if left.is_zero() { BigUint::one() } else { left }
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
    finalized_coefficient_wires: BTreeMap<WireId, RnsBggNoiseValue>,
    refresh_parameters: Option<RefreshNoiseParameters>,
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

/// The canonical lane index of a family over `range`.
fn family_lane(range: &FamilyRange) -> IntExpr {
    IntExpr::Add(Box::new(range.start().clone()), Box::new(IntExpr::LoopIndex(0))).canonicalize()
}

/// Whether distinct lanes of a family read distinct members through
/// `index`: the lane index itself (up to a constant offset), or a selection
/// by the lane among pairwise distinct constants.
fn lane_injective(index: &IntExpr, lane: &IntExpr) -> bool {
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
    fn new(
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
            finalized_coefficient_wires: BTreeMap::new(),
            refresh_parameters: None,
            lane_indices: Vec::new(),
            payload_support: BTreeMap::new(),
            reduction_counts: ReductionCounts::default(),
        }
    }

    /// Sparse payload support of `wire` in the innermost enclosing family.
    fn lane_support(&self, wire: WireId) -> Option<&BigUint> {
        self.payload_support
            .get(&wire)
            .filter(|support| support.depth == self.lane_indices.len())
            .map(|support| &support.count)
    }

    fn set_lane_support(&mut self, wire: WireId, count: BigUint) {
        self.payload_support.insert(wire, PayloadSupport { depth: self.lane_indices.len(), count });
    }

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

    fn site(&self) -> Result<&crate::program::LoweringSite, RnsBggError> {
        self.current_site
            .as_ref()
            .ok_or(RnsBggError::Graph("noise lowering site is missing".into()))
    }
    fn level(&self, wire: WireId) -> Result<usize, RnsBggError> {
        self.workload.level(wire)
    }
    fn carry_provenance(
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
                identity.route.extend_from_slice(&suffix);
                if let Some(existing) = provenance.get_mut(&source) {
                    existing.route.extend_from_slice(&identity.route);
                } else {
                    provenance.insert(source, identity);
                }
            }
        }
        value.provenance = provenance;
        value
    }

    fn add_events(&mut self, count: &BigUint) {
        self.trace.joint_event_count += count * &self.event_multiplier;
    }

    fn node_for_wire(&self, wire: WireId) -> Result<crate::noise_utils::NoiseNodeId, RnsBggError> {
        self.trace.dependencies.wires.get(&wire).copied().ok_or_else(|| {
            RnsBggError::Graph(format!("noise dependency parent is missing for wire {wire:?}"))
        })
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

    fn rhs_node(&mut self, rhs: &RhsRef<IntExpr>) -> crate::noise_utils::NoiseNodeId {
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
    fn lut_material_provenance(
        &self,
        evaluation_level: usize,
        lut: LutId,
        table: &LutTable,
    ) -> Result<BTreeMap<[u8; 32], NoiseSourceProvenance>, RnsBggError> {
        let decomposition_level = evaluation_level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("LUT decomposition level overflow"))?;
        self.parameters.level(decomposition_level)?;
        let layout = table.input_layout();
        let ring_dimension = self.parameters.level(evaluation_level)?.ring_dimension() as usize;
        let sigmas = layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
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
            let b = stable_source_id(&(
                b"rns-bgg/lut/B",
                evaluation_level,
                decomposition_level,
                lut,
                layout,
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

    fn event_coordinates(&self, wire: WireId) -> Result<BigUint, RnsBggError> {
        let level = self.level(wire)?;
        Ok(BigUint::from(self.parameters.level(level)?.ring_dimension()) *
            2u8 *
            BigUint::from(self.parameters.component_columns(level)?))
    }

    fn operation_node_with_multiplier(
        &mut self,
        wire: WireId,
        value: RnsBggNoiseValue,
        node: crate::noise_utils::NoiseDependencyNode,
        multiplier: BigUint,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        let coordinates = self.event_coordinates(wire)?;
        trace_noise_value(wire, self.level(wire).ok(), &node, &value);
        self.trace.wires.insert(wire, value.clone());
        let node_id = crate::noise_utils::NoiseNodeId(self.trace.dependencies.nodes.len());
        self.trace.dependencies.nodes.push(node);
        self.trace.dependencies.wires.insert(wire, node_id);
        self.trace.node_event_counts.insert(node_id.0, multiplier * coordinates);
        self.add_events(&BigUint::one());
        Ok((wire, value))
    }

    fn operation_node(
        &mut self,
        wire: WireId,
        value: RnsBggNoiseValue,
        node: crate::noise_utils::NoiseDependencyNode,
    ) -> Result<(WireId, RnsBggNoiseValue), RnsBggError> {
        self.operation_node_with_multiplier(wire, value, node, self.event_multiplier.clone())
    }

    fn merge_refresh_trace(&mut self, other: &RnsBggNoiseTrace) {
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
                &family_lane(&family.range),
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
        self.trace.wires.insert(wire, value.clone());
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
            let crt_weight = refresh_crt_weight(&plaintext_moduli, slot_index)?;
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
        input_terms: NoiseBreakdown,
        coefficient_output: bool,
    ) -> Result<NoiseBreakdown, RnsBggError> {
        let branches = table.values().len();
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
        let automorphisms = table
            .input_layout()
            .automorphism_indices(ring_dimension)
            .map_err(|_| RnsBggError::Lut)?
            .len();
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

    fn finish_outputs(
        &mut self,
        outputs: Vec<(WireId, RnsBggNoiseValue)>,
    ) -> Result<Vec<RnsBggNoiseValue>, RnsBggError> {
        let mut finalized = Vec::with_capacity(outputs.len());
        for (wire, mut value) in outputs {
            // Coefficient LUT output is finalized once, after the complete
            // aggregation topology.  Exponent outputs already carry their
            // normal next-level result and must not be rounded again.
            if self
                .program
                .wire_is_coefficient_output(wire)
                .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?
            {
                if let Some(previous) = self.finalized_coefficient_wires.get(&wire) {
                    finalized.push(previous.clone());
                    continue;
                }
                let level = self.level(wire)?;
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
                    value.bound.error =
                        self.parameters.lut_mod_down_bound(level, &value.bound.error)?;
                    let rounding =
                        zero_mean_moments(self.parameters.lut_public_rounding_variance(level)?)
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
                            None => NoiseBreakdown::opaque(
                                "coefficient-final-rounding",
                                rounding.clone(),
                            ),
                        };
                        terms
                            .transform(|term| {
                                (
                                    mean_gain.clone(),
                                    term.moments
                                        .variance
                                        .scaled_ratio(&BigUint::one(), &ratio.pow(2u32)),
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
                    value
                        .provenance
                        .insert(source, NoiseSourceProvenance { source, route: vec![level] });
                }
                self.add_events(&BigUint::one());
                if let Some(node) = self.trace.dependencies.wires.get(&wire).copied() {
                    let coordinates = self.event_coordinates(wire)?;
                    *self.trace.node_event_counts.entry(node.0).or_default() += coordinates;
                }
                self.finalized_coefficient_wires.insert(wire, value.clone());
            }
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
    fn wire_at(
        &mut self,
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
        let element = &family.1.element;
        let member = element.breakdown.as_ref().map(|breakdown| {
            let identity = *index == family_lane(&family.1.range);
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
        let index = family_lane(range);
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
        self.depth += 1;
        let result = body(self, evaluator, index);
        let body_depth = self.depth;
        self.depth -= 1;
        let (wire, mut element) = match result {
            Ok(value) => value,
            Err(error) => {
                evaluator.lane_indices.pop();
                evaluator.event_multiplier = previous_multiplier;
                return Err(error);
            }
        };
        element.breakdown = element
            .breakdown
            .map(|breakdown| Arc::new(breakdown.as_ref().clone().with_lane_depth(body_depth)));
        // Members weighted by distinct members of a sparse RHS family have
        // nonzero payloads in at most that family's sparse count of lanes.
        let max_nonzero_payloads = evaluator
            .lane_support(wire)
            .map_or(count.clone(), |support| support.min(&count).clone());
        evaluator.lane_indices.pop();
        evaluator.event_multiplier = previous_multiplier;
        Ok((wire, RnsBggNoiseFamily { element, range: range.clone(), max_nonzero_payloads }))
    }
    fn sum(
        &mut self,
        evaluator: &mut RnsBggNoiseEvaluator<'a>,
        family: &Self::WireFamily,
    ) -> Result<(WireId, RnsBggNoiseValue), ProgramLowerError<RnsBggError>> {
        let output = evaluator.site().map_err(ProgramLowerError::Backend)?.output;
        evaluator.reduce_add(output, family.0, &family.1).map_err(ProgramLowerError::Backend)
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
        let logical_level = self.level(input.0)?;
        let level = logical_level;
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
        if let (Some(count), Some(index), Some(lane)) =
            (sparse_count, rhs.index.as_ref(), self.lane_indices.last())
        {
            if lane_injective(index, lane) {
                self.set_lane_support(wire, count);
            }
        }
        self.operation_node(wire, value, node)
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.0)
            .map_err(|_| RnsBggError::Graph("invalid LUT input metadata".into()))? ==
            crate::program::AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Parameters("RNS-BGG LUT requires a private input"));
        }
        let lut = self.program.lut_id_at(self.site()?.output).map_err(|_| RnsBggError::Lut)?;
        let level = self.level(input.0)?;
        let fresh_error_gain = self.fresh_gain(level)?;
        let wire = self.site()?.output;
        // Exponent outputs are signed monomials, irrespective of the
        // numeric exponent stored in the table.  Coefficient outputs instead
        // carry the table's integer coefficient bound.
        let output_payload_l1 = match table.output_representation() {
            crate::program::LutOutputRepresentation::Exponent(_) => BigUint::one(),
            crate::program::LutOutputRepresentation::Coefficient => table
                .values()
                .iter()
                .copied()
                .max()
                .map(BigUint::from)
                .ok_or(RnsBggError::Parameters("LUT table has no output values"))?,
        };
        let output_payload_l2_squared = output_payload_l1.pow(2u32);
        let coefficient_output = matches!(
            table.output_representation(),
            crate::program::LutOutputRepresentation::Coefficient
        );
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
                table.values().len(),
                &fresh_error_gain.l1,
                output_payload_l1,
            )?
        } else {
            self.parameters.lut_noise_bound(
                level,
                &input.1.bound,
                table.values().len(),
                &fresh_error_gain.l1,
                output_payload_l1,
            )?
        };
        let breakdown = input
            .1
            .contributions("lut-input")
            .map(|input_terms| {
                if terminal_coefficient {
                    return Ok(input_terms);
                }
                self.lut_contributions(level, lut, table, &input.1, input_terms, coefficient_output)
            })
            .transpose()?;
        let material = self.lut_material_provenance(level, lut, table)?;
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
        self.operation_node_with_multiplier(
            wire,
            value,
            node,
            self.event_multiplier.clone() * BigUint::from(table.values().len()),
        )
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

/// Noise trace with finalized declared outputs separated from raw wires.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RnsBggNoiseTrace {
    /// Accepted model, when an audit completed.
    pub acceptance: Option<NoiseModelKind>,
    /// Joint physical event count.
    pub joint_event_count: BigUint,
    /// Tail threshold for average-case acceptance.
    pub z_squared: Option<NonnegativeRational>,
    /// Dependency graph for shared sources.
    pub dependencies: crate::noise_utils::NoiseDependencyGraph,
    /// Raw symbolic wire values.
    pub wires: BTreeMap<WireId, RnsBggNoiseValue>,
    /// Physical event budget for each dependency node. A family body and LUT
    /// branch set are represented once in the symbolic trace, so their active
    /// multiplicities are expanded before this map is recorded. Refresh
    /// transcript nodes store their already-expanded physical event count.
    pub node_event_counts: BTreeMap<usize, BigUint>,
    /// Finalized declared outputs in declaration order.
    pub outputs: Vec<RnsBggNoiseValue>,
    /// Target logical level of the most recently composed refresh output.
    pub refreshed_target_level: Option<usize>,
    /// Commitment of the declaration that produced the refreshed envelope.
    pub refreshed_declaration_commitment: Option<[u8; 32]>,
    /// Target coefficient modulus of the refreshed envelope.
    pub refreshed_target_modulus: Option<BigUint>,
    /// Authoritative schema frozen by the refresh declaration.
    pub refreshed_output_schema: Option<Vec<WireType>>,
    /// Refreshed target envelope after CRT recomposition and one ModDown.
    pub refreshed_output: Option<RnsBggNoiseValue>,
    /// Explicit residue model used by the final refresh ModDown.
    pub refreshed_moment_policy: Option<NoiseMomentPolicy>,
}

/// Trace one Program using one workload and one evaluator path.
pub fn trace_program(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
) -> Result<RnsBggNoiseTrace, ProgramLowerError<RnsBggError>> {
    let workload = RnsBggDerivedWorkload::from_program_with_parameters(parameters, program)
        .map_err(ProgramLowerError::Backend)?;
    let mut environment = Environment::new();
    for (id, declaration) in program.inputs() {
        let value = bindings.inputs.get(id).ok_or(ProgramLowerError::Backend(
            RnsBggError::Binding("missing input noise contract"),
        ))?;
        let lowered = match value.clone() {
            InputBinding::Scalar(value) => InputBinding::Scalar((*id, value)),
            InputBinding::Family(family) => InputBinding::Family((*id, family)),
        };
        environment.bind_wire(*id, lowered).map_err(ProgramLowerError::Program)?;
        if matches!(declaration.shape(), InputBinding::Family(_)) &&
            !matches!(value, InputBinding::Family(_))
        {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::ExpectedFamilyWire(*id),
            ));
        }
    }
    for (id, declaration) in program.rhs_inputs() {
        let value = bindings.rhs.get(id).ok_or(ProgramLowerError::Backend(
            RnsBggError::Binding("missing RHS noise contract"),
        ))?;
        let lowered = match value {
            InputBinding::Scalar(_) => InputBinding::Scalar(RhsRef { id: *id, index: None }),
            InputBinding::Family(family) => InputBinding::Family((*id, family.range.clone())),
        };
        environment.bind_rhs(*id, lowered).map_err(ProgramLowerError::Program)?;
        if matches!(declaration, InputBinding::Family(_)) &&
            !matches!(value, InputBinding::Family(_))
        {
            return Err(ProgramLowerError::Program(
                crate::program::ProgramValidationError::ExpectedFamilyRhs(*id),
            ));
        }
    }
    let mut evaluator = RnsBggNoiseEvaluator::new(parameters, program, &workload, bindings);
    for (wire, binding) in bindings.inputs.iter() {
        let value = match binding {
            InputBinding::Scalar(value) => value.clone(),
            InputBinding::Family(family) => family.element.clone(),
        };
        evaluator.trace.wires.insert(*wire, value);
        let node = crate::noise_utils::NoiseNodeId(evaluator.trace.dependencies.nodes.len());
        evaluator.trace.dependencies.nodes.push(crate::noise_utils::NoiseDependencyNode::Binding(
            crate::noise_utils::NoiseBinding::Input(*wire),
        ));
        evaluator.trace.dependencies.wires.insert(*wire, node);
        let multiplicity = match binding {
            InputBinding::Scalar(_) => BigUint::one(),
            InputBinding::Family(family) => {
                family.validate_support().map_err(|error| ProgramLowerError::Backend(error))?
            }
        };
        let coordinates = evaluator.event_coordinates(*wire).map_err(ProgramLowerError::Backend)?;
        evaluator.trace.node_event_counts.insert(node.0, multiplicity * coordinates);
    }
    let mut families = RnsBggNoiseFamilies::default();
    let outputs =
        crate::program::lower_program(program, &mut evaluator, &mut families, &mut environment)?;
    evaluator.trace.outputs =
        evaluator.finish_outputs(outputs).map_err(ProgramLowerError::Backend)?;
    Ok(evaluator.trace)
}

/// Public parameter-search request. LUT widths are read from the Program.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggPlanRequest {
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Fresh encoding sigma.
    #[serde(with = "bigdecimal_string")]
    pub encoding_error_sigma: bigdecimal::BigDecimal,
    /// Target CRT prime bits.
    pub crt_bits: usize,
    /// Gadget decomposition base bits, at most 31.
    pub base_bits: u32,
    /// Terminal CRT depth.
    pub terminal_depth: usize,
    /// Maximum candidate CRT depth.
    pub max_crt_depth: usize,
    /// Declared terminal value bound.
    pub terminal_value_bound: BigUint,
    /// Logical operations per physical drop.
    pub operations_per_drop: usize,
    /// CRT primes removed by each physical drop. The drop quotient is the
    /// message scale that a level's noise must stay below half of, so its
    /// width, not only the drop cadence, decides whether a level fits.
    pub primes_per_drop: usize,
}

/// Serialized parameter-search evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggParameterManifest {
    /// Original request.
    pub request: RnsBggPlanRequest,
    /// Frozen Program identity.
    pub program: crate::program::ExponentLutProgramId,
    /// Commitment to public noise contracts.
    pub noise_contract_commitment: [u8; 32],
    /// Commitment to the selected parameter bases and request.
    pub parameter_commitment: [u8; 32],
    /// Actual selected CRT prime bases.
    pub levels: Vec<Vec<u64>>,
    /// Base widths for each level.
    pub base_bits: Vec<u32>,
    /// Accepted noise trace.
    pub trace: RnsBggNoiseTrace,
    /// Average-case policy used when this trace was accepted.
    pub average: Option<crate::noise_refresh::simulation::AverageCaseConfig>,
    /// Estimated persistent artifact bytes.
    pub estimated_artifact_bytes: BigUint,
    /// Commitment to the complete accepted trace, including dependencies.
    pub trace_commitment: [u8; 32],
}

fn parameter_artifact_bytes(parameters: &RnsBggParameters) -> BigUint {
    parameters
        .levels()
        .iter()
        .map(|level| {
            let modulus_bytes = BigUint::from(level.modulus().bits().div_ceil(8));
            BigUint::from(level.ring_dimension()) * modulus_bytes * 16u8
        })
        .sum()
}

fn noise_contract_commitment(
    program: &ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    terminal_value_bounds: &[BigUint],
    average: Option<&crate::noise_refresh::simulation::AverageCaseConfig>,
) -> [u8; 32] {
    stable_source_id(&(
        program.id(),
        &bindings.inputs,
        &bindings.rhs,
        terminal_value_bounds,
        average,
    ))
}

fn parameter_commitment(
    request: &RnsBggPlanRequest,
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    noise_contract_commitment: [u8; 32],
) -> [u8; 32] {
    stable_source_id(&(
        request,
        program.id(),
        parameters.levels().iter().map(|level| level.to_crt().0).collect::<Vec<_>>(),
        parameters.levels().iter().map(PolyParams::base_bits).collect::<Vec<_>>(),
        noise_contract_commitment,
    ))
}

fn trace_commitment(trace: &RnsBggNoiseTrace) -> [u8; 32] {
    stable_source_id(&(
        &trace.acceptance,
        &trace.z_squared,
        &trace.dependencies,
        &trace.wires,
        &trace.node_event_counts,
        &trace.outputs,
        &trace.joint_event_count,
        &trace.refreshed_target_level,
        &trace.refreshed_declaration_commitment,
        &trace.refreshed_target_modulus,
        &trace.refreshed_output_schema,
        &trace.refreshed_output,
        &trace.refreshed_moment_policy,
    ))
}

fn average_event_budget(trace: &RnsBggNoiseTrace) -> Result<BigUint, RnsBggError> {
    let events = trace.node_event_counts.values().sum::<BigUint>();
    Ok(events.max(BigUint::one()))
}

impl RnsBggParameters {
    fn construct_program_parameters(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
    ) -> Result<Self, RnsBggError> {
        if request.operations_per_drop == 0 ||
            request.primes_per_drop == 0 ||
            request.terminal_depth == 0 ||
            request.max_crt_depth < request.terminal_depth
        {
            return Err(RnsBggError::Parameters("invalid public parameter request"));
        }
        let workload = RnsBggDerivedWorkload::from_program(program)?;
        let logical_count = workload.levels.values().copied().max().unwrap_or(0);
        let pool = DCRTPolyParams::try_new(
            request.ring_dimension,
            request.max_crt_depth,
            request.crt_bits,
            request.base_bits,
            None,
            None,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported prime pool"))?;
        let primes = pool.to_crt().0;
        let mut levels = Vec::with_capacity(logical_count + 1);
        for index in 0..=logical_count {
            let depth = super::planned_crt_depth(
                index,
                request.operations_per_drop,
                request.primes_per_drop,
                request.max_crt_depth,
                request.terminal_depth,
            )?;
            let modulus = primes[..depth].iter().copied().map(BigUint::from).product();
            levels.push(
                pool.select_modulus(&modulus)
                    .ok_or(RnsBggError::Parameters("invalid prime basis"))?,
            );
        }
        RnsBggParameters::new(
            levels,
            request.encoding_error_sigma.clone(),
            request.operations_per_drop,
        )
    }

    /// Evaluate the selected Program noise contract and terminal envelopes.
    pub fn audit_program(
        &self,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
        average: Option<&crate::noise_refresh::simulation::AverageCaseConfig>,
    ) -> Result<RnsBggNoiseTrace, RnsBggError> {
        if terminal_value_bounds.len() != program.outputs().len() {
            return Err(RnsBggError::Parameters("terminal bounds must match declared outputs"));
        }
        let started = std::time::Instant::now();
        let mut trace = trace_program(self, program, bindings).map_err(|error| match error {
            ProgramLowerError::Backend(error) => error,
            ProgramLowerError::Program(error) => RnsBggError::Graph(error.to_string()),
            ProgramLowerError::Dsl(error) => RnsBggError::Graph(error.to_string()),
            ProgramLowerError::UnsupportedStructure { reason } => RnsBggError::Graph(reason),
        })?;
        tracing::info!(
            wires = trace.wires.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "noise trace finished"
        );
        let workload = RnsBggDerivedWorkload::from_program_with_parameters(self, program)?;
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "noise audit workload derived"
        );
        let average = average.filter(|config| config.allow_average_acceptance);
        let z_squared = average
            .map(|config| {
                let events = average_event_budget(&trace)?;
                let z_squared = config
                    .z_squared_for_events(&events)
                    .map_err(|_| RnsBggError::Parameters("invalid average-case event count"))?;
                let correction = config
                    .tail_correction_bits
                    .checked_mul(2)
                    .ok_or(RnsBggError::Parameters("invalid average-case tail correction"))?;
                Ok::<_, RnsBggError>(z_squared.scaled(&(BigUint::one() << correction)))
            })
            .transpose()?;

        if let Some(z_squared) = z_squared.as_ref() {
            let mut values = BTreeMap::<usize, (WireId, &RnsBggNoiseValue)>::new();
            for (wire, value) in &trace.wires {
                let node =
                    trace.dependencies.wires.get(wire).ok_or_else(|| {
                        RnsBggError::Graph("noise dependency node is missing".into())
                    })?;
                values.entry(node.0).or_insert((*wire, value));
            }
            for (wire, value) in values.values() {
                let moments = value
                    .moments
                    .as_ref()
                    .ok_or(RnsBggError::Parameters("average-case input moment model is missing"))?;
                let level = workload.level(*wire)?;
                let modulus_limit = NonnegativeRational::new(
                    self.level(level)?.modulus().as_ref().clone(),
                    4u8.into(),
                )
                .map_err(|_| RnsBggError::Parameters("invalid average-case rational budget"))?;
                // A coefficient LUT output keeps its noise at the scale
                // before its one scale-and-round; the finalized output is
                // checked against the terminal modulus below.
                let coefficient_output = program
                    .wire_is_coefficient_output(*wire)
                    .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?;
                let scale_fits = coefficient_output ||
                    level + 1 == self.levels.len() ||
                    self.scale(level)? == &BigUint::one() ||
                    moments.fits(
                        &NonnegativeRational::new(self.scale(level)?.clone(), 2u8.into()).map_err(
                            |_| RnsBggError::Parameters("invalid average-case rational budget"),
                        )?,
                        z_squared,
                    );
                let modulus_fits = moments.fits(&modulus_limit, z_squared);
                if !modulus_fits || !scale_fits {
                    tracing::info!(
                        wire = wire.0,
                        level,
                        levels = self.levels.len(),
                        variance_log2 = moments.variance.numerator.bits() as i64 -
                            moments.variance.denominator.bits() as i64,
                        modulus_bits = self.level(level)?.modulus().bits(),
                        scale_bits = self.scale(level)?.bits(),
                        modulus_fits,
                        scale_fits,
                        z_squared_log2 = variance_log2(z_squared),
                        top_modulus_bits = self.levels[0].modulus().bits(),
                        first_scale_bits = self.scale(0)?.bits(),
                        contributions = %value.describe_contributions(),
                        "average-case evaluation noise exceeds the level budget"
                    );
                    return Err(RnsBggError::Budget("evaluation noise"));
                }
            }
        }

        for ((index, output), terminal_bound) in
            program.outputs().iter().enumerate().zip(terminal_value_bounds)
        {
            let wire = match output {
                crate::program::Operand::Scalar(id) |
                crate::program::Operand::Member { family: id, .. } => *id,
            };
            let value = trace
                .outputs
                .get(index)
                .ok_or(RnsBggError::Graph("missing audited output".into()))?;
            if let Some(z_squared) = z_squared.as_ref() {
                let moments = value
                    .moments
                    .as_ref()
                    .ok_or(RnsBggError::Parameters("average-case input moment model is missing"))?;
                let level = workload.level(wire)?;
                let modulus = if program
                    .wire_is_coefficient_output(wire)
                    .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?
                {
                    self.levels
                        .get(level + 1)
                        .or_else(|| self.levels.get(level))
                        .ok_or(RnsBggError::Parameters("level outside modulus chain"))?
                        .modulus()
                } else {
                    self.level(level)?.modulus()
                };
                let payload = terminal_bound * 2u8;
                if modulus.as_ref() <= &payload {
                    return Err(RnsBggError::Budget("terminal range"));
                }
                let modulus_limit = NonnegativeRational::new(modulus.as_ref().clone(), 4u8.into())
                    .map_err(|_| RnsBggError::Parameters("invalid average-case rational budget"))?;
                if !moments.fits(&modulus_limit, z_squared) {
                    tracing::info!(
                        wire = wire.0,
                        level,
                        variance_log2 = variance_log2(&moments.variance),
                        modulus_bits = modulus.bits(),
                        contributions = %value.describe_contributions(),
                        "average-case terminal noise exceeds the output budget"
                    );
                    return Err(RnsBggError::Budget("terminal range"));
                }
            } else {
                let level = workload.level(wire)?;
                let fits = if program
                    .wire_is_coefficient_output(wire)
                    .map_err(|_| RnsBggError::Graph("invalid coefficient output".into()))?
                {
                    self.coefficient_output_noise_fits(level, value)?
                } else {
                    self.noise_fits(level, &value.bound)?
                };
                if !fits {
                    return Err(RnsBggError::Budget(
                        "evaluation noise does not fit the selected modulus",
                    ));
                }
            }
            if &value.bound.payload_l1 > terminal_bound {
                return Err(RnsBggError::Budget("terminal payload exceeds declared bound"));
            }
        }
        trace.acceptance = Some(if z_squared.is_some() {
            NoiseModelKind::AverageCase
        } else {
            NoiseModelKind::WorstCase
        });
        trace.z_squared = z_squared;
        Ok(trace)
    }

    /// Construct a deterministic chain from the Program's actual logical levels.
    pub fn plan_program(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
    ) -> Result<(Self, RnsBggParameterManifest), RnsBggError> {
        let parameters = Self::construct_program_parameters(request, program)?;
        let trace = parameters.audit_program(
            program,
            bindings,
            &vec![request.terminal_value_bound.clone(); program.outputs().len()],
            None,
        )?;
        let terminal_value_bounds =
            vec![request.terminal_value_bound.clone(); program.outputs().len()];
        let noise_contract_commitment =
            noise_contract_commitment(program, bindings, &terminal_value_bounds, None);
        let parameter_commitment =
            parameter_commitment(request, &parameters, program, noise_contract_commitment);
        let estimated_artifact_bytes = parameter_artifact_bytes(&parameters);
        let trace_digest = trace_commitment(&trace);
        let manifest = RnsBggParameterManifest {
            request: request.clone(),
            program: program.id(),
            noise_contract_commitment,
            parameter_commitment,
            levels: parameters.levels().iter().map(|level| level.to_crt().0).collect(),
            base_bits: parameters.levels().iter().map(PolyParams::base_bits).collect(),
            trace,
            average: None,
            estimated_artifact_bytes,
            trace_commitment: trace_digest,
        };
        Ok((parameters, manifest))
    }

    /// Search candidates, preserving the exact evaluator/audit path.
    pub fn search_program(
        request: &RnsBggPlanRequest,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
        average: Option<&crate::noise_refresh::simulation::AverageCaseConfig>,
    ) -> Result<(Self, RnsBggParameterManifest), RnsBggError> {
        if terminal_value_bounds.is_empty() {
            return Err(RnsBggError::Parameters("terminal bounds must not be empty"));
        }
        let trial_bound = terminal_value_bounds
            .iter()
            .max()
            .cloned()
            .ok_or(RnsBggError::Parameters("terminal bounds must not be empty"))?;
        let mut last_error = RnsBggError::Budget("no CRT candidate satisfies the noise contract");
        for depth in request.terminal_depth..=request.max_crt_depth {
            let mut candidate = request.clone();
            candidate.max_crt_depth = depth;
            candidate.terminal_value_bound = trial_bound.clone();
            let Ok(parameters) = Self::construct_program_parameters(&candidate, program) else {
                continue;
            };
            match parameters.audit_program(program, bindings, terminal_value_bounds, average) {
                Ok(trace) => {
                    let noise_contract_commitment = noise_contract_commitment(
                        program,
                        bindings,
                        terminal_value_bounds,
                        average,
                    );
                    let manifest = RnsBggParameterManifest {
                        request: candidate.clone(),
                        program: program.id(),
                        noise_contract_commitment,
                        parameter_commitment: parameter_commitment(
                            &candidate,
                            &parameters,
                            program,
                            noise_contract_commitment,
                        ),
                        levels: parameters.levels().iter().map(|level| level.to_crt().0).collect(),
                        base_bits: parameters.levels().iter().map(PolyParams::base_bits).collect(),
                        trace_commitment: trace_commitment(&trace),
                        trace,
                        average: average.cloned(),
                        estimated_artifact_bytes: parameter_artifact_bytes(&parameters),
                    };
                    return Ok((parameters, manifest));
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }
}

impl RnsBggParameterManifest {
    /// Rehydrate and validate a manifest against the frozen Program and
    /// contracts.  No serialized level or commitment is trusted on its own.
    pub fn validate(
        &self,
        program: &ExponentLutProgram,
        bindings: &RnsBggNoiseBindings,
        terminal_value_bounds: &[BigUint],
    ) -> Result<RnsBggParameters, RnsBggError> {
        if self.program != program.id() ||
            self.levels.is_empty() ||
            self.levels.len() != self.base_bits.len()
        {
            return Err(RnsBggError::Binding("parameter manifest does not match the Program"));
        }
        let commitment = noise_contract_commitment(
            program,
            bindings,
            terminal_value_bounds,
            self.average.as_ref(),
        );
        if commitment != self.noise_contract_commitment {
            return Err(RnsBggError::Binding("noise contract commitment mismatch"));
        }
        let pool = DCRTPolyParams::try_new(
            self.request.ring_dimension,
            self.request.max_crt_depth,
            self.request.crt_bits,
            self.request.base_bits,
            None,
            None,
        )
        .map_err(|_| RnsBggError::Parameters("unsupported manifest prime pool"))?;
        let mut levels = Vec::with_capacity(self.levels.len());
        for (primes, base_bits) in self.levels.iter().zip(&self.base_bits) {
            if primes.is_empty() || primes.iter().any(|prime| *prime < 3) {
                return Err(RnsBggError::Binding("invalid manifest CRT basis"));
            }
            let modulus = primes.iter().copied().map(BigUint::from).product();
            let level = pool
                .select_modulus(&modulus)
                .ok_or(RnsBggError::Binding("manifest CRT basis is not in the prime pool"))?;
            if level.base_bits() != *base_bits {
                return Err(RnsBggError::Binding("manifest base width mismatch"));
            }
            levels.push(level);
        }
        let parameters = RnsBggParameters::new(
            levels,
            self.request.encoding_error_sigma.clone(),
            self.request.operations_per_drop,
        )?;
        let expected_parameter_commitment = parameter_commitment(
            &self.request,
            &parameters,
            program,
            self.noise_contract_commitment,
        );
        if expected_parameter_commitment != self.parameter_commitment {
            return Err(RnsBggError::Binding("parameter commitment mismatch"));
        }
        if parameter_artifact_bytes(&parameters) != self.estimated_artifact_bytes {
            return Err(RnsBggError::Binding("artifact size estimate mismatch"));
        }
        let trace = parameters.audit_program(
            program,
            bindings,
            terminal_value_bounds,
            self.average.as_ref(),
        )?;
        if trace_commitment(&trace) != self.trace_commitment {
            return Err(RnsBggError::Binding("manifest noise trace mismatch"));
        }
        Ok(parameters)
    }
}

/// Common API noise-stage adapter.  It owns only graph handles and the
/// evaluator; sampled matrices and refresh artifacts stay at their stages.
pub struct RnsNoiseBuild<'a> {
    evaluator: Option<RnsBggNoiseEvaluator<'a>>,
    families: RnsBggNoiseFamilies,
    environment: Environment<
        (WireId, RnsBggNoiseValue),
        RhsRef<IntExpr>,
        (WireId, RnsBggNoiseFamily),
        (RhsId, FamilyRange),
    >,
    refresh_prf: Option<(usize, [u8; 32], [u8; 32])>,
    pending_refresh: Option<(usize, (WireId, RnsBggNoiseValue))>,
}

impl<'a> RnsNoiseBuild<'a> {
    /// Construct a noise stage from the already validated workload.
    pub fn new(
        parameters: &'a RnsBggParameters,
        program: &'a ExponentLutProgram,
        workload: &'a RnsBggDerivedWorkload,
        bindings: &'a RnsBggNoiseBindings,
    ) -> Result<Self, RnsBggError> {
        let mut environment = Environment::new();
        for (id, declaration) in program.inputs() {
            let value = bindings
                .inputs
                .get(id)
                .ok_or(RnsBggError::Binding("missing input noise contract"))?
                .clone();
            environment
                .bind_wire(
                    *id,
                    match (declaration.shape(), value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(value)) => {
                            InputBinding::Scalar((*id, value))
                        }
                        (InputBinding::Family(_), InputBinding::Family(value)) => {
                            InputBinding::Family((*id, value))
                        }
                        _ => return Err(RnsBggError::Binding("input binding shape mismatch")),
                    },
                )
                .map_err(|_| RnsBggError::Binding("invalid input binding"))?;
        }
        for (id, declaration) in program.rhs_inputs() {
            let value = bindings
                .rhs
                .get(id)
                .ok_or(RnsBggError::Binding("missing RHS noise contract"))?
                .clone();
            environment
                .bind_rhs(
                    *id,
                    match (declaration, value) {
                        (InputBinding::Scalar(()), InputBinding::Scalar(_)) => {
                            InputBinding::Scalar(RhsRef { id: *id, index: None })
                        }
                        (InputBinding::Family(_), InputBinding::Family(family)) => {
                            InputBinding::Family((*id, family.range.clone()))
                        }
                        _ => return Err(RnsBggError::Binding("RHS binding shape mismatch")),
                    },
                )
                .map_err(|_| RnsBggError::Binding("invalid RHS binding"))?;
        }
        let mut evaluator = RnsBggNoiseEvaluator::new(parameters, program, workload, bindings);
        for (wire, binding) in &bindings.inputs {
            let value = match binding {
                InputBinding::Scalar(value) => value.clone(),
                InputBinding::Family(family) => family.element.clone(),
            };
            evaluator.trace.wires.insert(*wire, value);
            let node = crate::noise_utils::NoiseNodeId(evaluator.trace.dependencies.nodes.len());
            evaluator.trace.dependencies.nodes.push(
                crate::noise_utils::NoiseDependencyNode::Binding(
                    crate::noise_utils::NoiseBinding::Input(*wire),
                ),
            );
            evaluator.trace.dependencies.wires.insert(*wire, node);
            let multiplicity = match binding {
                InputBinding::Scalar(_) => BigUint::one(),
                InputBinding::Family(family) => family.validate_support()?,
            };
            let coordinates = evaluator.event_coordinates(*wire)?;
            evaluator.trace.node_event_counts.insert(node.0, multiplicity * coordinates);
        }
        Ok(Self {
            evaluator: Some(evaluator),
            families: RnsBggNoiseFamilies::default(),
            environment,
            refresh_prf: None,
            pending_refresh: None,
        })
    }

    /// Configure this builder for the prepared PRF Program belonging to one
    /// refresh CRT channel.  This binding is intentionally separate from the
    /// source Program validation: the prepared declaration is the authority
    /// for the PRF output wires and labels.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<(), RnsBggError> {
        if slot >= declaration.slots().len() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        evaluator.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        let source_q = declaration
            .slots()
            .get(slot)
            .and_then(|slot| slot.q.to_biguint())
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        if evaluator.parameters.level(0)?.modulus().as_ref() != &source_q {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        if evaluator.program.id() != declaration.prepared().program().id() {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        if self.refresh_prf.is_some() {
            return Err(RnsBggError::Binding("refresh PRF is already bound"));
        }
        self.refresh_prf =
            Some((slot, *declaration.commitment(), evaluator.parameters.parameter_identity()));
        Ok(())
    }

    /// Lower the prepared PRF Program once and return its outputs in the
    /// declaration's exact label order.  No ordinary output finalization is
    /// applied to these wires.
    fn lower_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
    ) -> Result<BTreeMap<PrfLabel, (WireId, RnsBggNoiseValue)>, RnsBggError> {
        let (slot, commitment, parameter_identity) =
            self.refresh_prf.ok_or(RnsBggError::Binding("refresh PRF is not bound"))?;
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        if parameter_identity != evaluator.parameters.parameter_identity() {
            return Err(RnsBggError::Binding("refresh PRF parameter identity changed"));
        }
        if commitment != *declaration.commitment() || slot >= declaration.slots().len() {
            return Err(RnsBggError::Refresh(RefreshError::Commitment));
        }
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        let expected = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        let mut families = RnsBggNoiseFamilies::default();
        let outputs = crate::program::lower_program(
            evaluator.program,
            evaluator,
            &mut families,
            &mut self.environment,
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        expected
            .into_iter()
            .zip(outputs)
            .map(|((label, expected_wire), (wire, value))| {
                if wire != expected_wire {
                    return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                }
                Ok((label, (wire, value)))
            })
            .collect()
    }

    /// Apply one declared refresh using one separately configured PRF builder
    /// per CRT slot. The common refresh trait is invoked exactly once; the
    /// channel Programs are lowered once each by its callback.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        inputs: BTreeMap<usize, (WireId, RnsBggNoiseValue)>,
        mut channels: BTreeMap<usize, RnsNoiseBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() {
            return Err(RnsBggError::Binding("refresh output was already selected"));
        }
        if channels.len() != declaration.slots().len() ||
            channels.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let mut channel_traces = Vec::with_capacity(channels.len());
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        let source_level = evaluator.workload.level(declaration.source_output())?;
        evaluator.parameters.validate_refresh_level_contract(declaration, source_level)?;
        for (slot, channel) in &channels {
            let expected_child_parameters =
                evaluator.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if channel.evaluator.as_ref().is_none_or(|child| {
                child.parameters.parameter_identity() != expected_child_parameters
            }) || channel.refresh_prf.map(|(_, _, identity)| identity) !=
                Some(expected_child_parameters)
            {
                return Err(RnsBggError::Binding("refresh PRF child parameter identity mismatch"));
            }
        }
        let output = {
            let mut refresh = super::RnsBggRefresh::new(evaluator, inputs);
            crate::api::apply_refresh(declaration, &mut refresh, |callback_declaration, labels| {
                let mut per_label =
                    BTreeMap::<PrfLabel, BTreeMap<usize, (WireId, RnsBggNoiseValue)>>::new();
                let label_positions = labels
                    .iter()
                    .enumerate()
                    .map(|(index, label)| (label.clone(), index))
                    .collect::<BTreeMap<_, _>>();
                for slot in 0..callback_declaration.slots().len() {
                    let mut channel = channels
                        .remove(&slot)
                        .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                    let outputs = channel.lower_refresh_prf(callback_declaration)?;
                    for (label, output) in outputs {
                        // Every slot's PRF Program yields every label; a mask
                        // label is kept only from the slot that owns it, as
                        // in preprocessing.
                        if let Some(&label_index) = label_positions.get(&label) {
                            let refresh_label = declaration
                                .labels()
                                .label(label_index)
                                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                            if !refresh_label
                                .slots(callback_declaration.slots().len())
                                .map_err(RnsBggError::Refresh)?
                                .contains(&slot)
                            {
                                continue;
                            }
                        }
                        if per_label.entry(label).or_default().insert(slot, output).is_some() {
                            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
                        }
                    }
                    channel_traces.push(
                        channel
                            .evaluator
                            .as_ref()
                            .ok_or(RnsBggError::Binding("noise evaluator consumed"))?
                            .trace
                            .clone(),
                    );
                }
                Ok(per_label)
            })
            .map_err(|error| match error {
                crate::api::RefreshApplicationError::Refresh(error) => error,
                crate::api::RefreshApplicationError::OutputMismatch => {
                    RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
                }
            })?
        };
        let evaluator =
            self.evaluator.as_mut().ok_or(RnsBggError::Binding("noise evaluator consumed"))?;
        for channel_trace in &channel_traces {
            evaluator.merge_refresh_trace(channel_trace);
        }
        self.pending_refresh = Some((output_index, output));
        Ok(())
    }

    /// Compose refresh noise on this builder using the existing evaluator.
    /// The same method is used by simulation and by checked resumption paths;
    /// refresh metadata is retained in the evaluator and never wrapped in a
    /// second carrier.
    pub fn with_refresh_noise_parameters(
        &mut self,
        parameters: RefreshNoiseParameters,
    ) -> Result<(), RnsBggError> {
        self.evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("noise evaluator consumed"))?
            .with_refresh_noise_parameters(parameters)
    }
}

/// Noise-side implementation of the common one-shot refresh protocol.
///
/// PRF results are symbolic graph carriers, not sampled plaintexts. Every
/// mask label carries one canonical slot channel; every fresh-error label
/// carries all CRT channels. The shared fresh-error source is retained by
/// combining carriers according to their provenance instead of treating each
/// label or slot as an independent term.
impl NoiseRefresh for super::RnsBggRefresh<'_, RnsBggNoiseEvaluator<'_>> {
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

impl<'a> crate::api::NoiseBuild for RnsNoiseBuild<'a> {
    type Evaluator = RnsBggNoiseEvaluator<'a>;
    type Families = RnsBggNoiseFamilies;
    type Report = RnsBggNoiseTrace;
    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<
                (WireId, RnsBggNoiseValue),
                RhsRef<IntExpr>,
                (WireId, RnsBggNoiseFamily),
                (RhsId, FamilyRange),
            >,
        ),
        crate::api::ApiError,
    > {
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| crate::api::ApiError::input("noise evaluator consumed"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
    fn build_report(
        mut self,
        outputs: Vec<(WireId, RnsBggNoiseValue)>,
    ) -> Result<Self::Report, crate::api::ApiError> {
        let mut evaluator = self
            .evaluator
            .take()
            .ok_or_else(|| crate::api::ApiError::input("noise evaluator consumed"))?;
        let finalized = evaluator
            .finish_outputs(outputs)
            .map_err(|error| crate::api::ApiError::scheme(error))?;
        let mut trace = evaluator.trace;
        let mut finalized = finalized;
        if let Some((index, refreshed)) = self.pending_refresh {
            if index >= finalized.len() {
                return Err(crate::api::ApiError::input("refresh output index is out of range"));
            }
            finalized[index] = refreshed.1;
        }
        trace.outputs = finalized;
        Ok(trace)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::program::ExpLayout;
    use std::collections::BTreeSet;

    fn value(payload: &[i64]) -> RnsBggNoiseValue {
        RnsBggNoiseValue::public(
            BigUint::zero(),
            &payload.iter().copied().map(BigInt::from).collect::<Vec<_>>(),
        )
    }

    fn nested_alignment_parameters() -> RnsBggParameters {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![high, low], 1, 1).unwrap()
    }

    fn alignment_rounding(
        alpha: &BigUint,
        source_level: usize,
        key_level: usize,
        target_level: usize,
        role: &'static [u8],
    ) -> RnsBggNoiseValue {
        let source = stable_source_id(&(
            b"rns-bgg/public-alignment/rounding".as_slice(),
            role,
            source_level,
            key_level,
            target_level,
        ));
        RnsBggNoiseValue {
            bound: RnsBggNoiseBound { error: BigUint::one(), payload_l1: BigUint::zero() },
            payload_l2_squared: BigUint::zero(),
            moments: Some(NoiseMoments::discrete_rounding_residue(alpha).unwrap()),
            provenance: BTreeMap::from([(
                source,
                NoiseSourceProvenance {
                    source,
                    route: vec![source_level, key_level, target_level],
                },
            )]),
            heuristics: NoiseHeuristicLedger::new().with([NoiseHeuristicId::ExactDistribution]),
            breakdown: None,
        }
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

    /// A public input consumed only as the public left operand of products
    /// with private right operands takes the lowest of their levels; one also
    /// consumed any other way stays at level zero.
    #[test]
    fn workload_encodes_public_inputs_at_their_first_consuming_level() {
        use crate::program::{AttributeKnowledge, Operand};
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let state = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let moved = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let pinned = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let level_one = builder.mul_private(Operand::Scalar(state), Operand::Scalar(rhs)).unwrap();
        let level_two =
            builder.mul_private(Operand::Scalar(level_one), Operand::Scalar(second_rhs)).unwrap();
        let first = builder.mul_public(Operand::Scalar(moved), Operand::Scalar(level_one)).unwrap();
        let second =
            builder.mul_public(Operand::Scalar(moved), Operand::Scalar(level_two)).unwrap();
        let third =
            builder.mul_public(Operand::Scalar(pinned), Operand::Scalar(level_two)).unwrap();
        let program = builder
            .finish(vec![
                Operand::Scalar(first),
                Operand::Scalar(second),
                Operand::Scalar(third),
                Operand::Scalar(pinned),
            ])
            .unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();
        assert_eq!(workload.level(state).unwrap(), 0);
        assert_eq!(workload.level(moved).unwrap(), 1);
        assert_eq!(workload.level(pinned).unwrap(), 0);
        assert_eq!(workload.level(first).unwrap(), 1);
        assert_eq!(workload.level(second).unwrap(), 2);
        assert_eq!(workload.level(third).unwrap(), 2);
    }

    #[test]
    fn workload_rejects_one_rhs_at_two_logical_levels_before_sampling() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();

        assert!(matches!(
            RnsBggDerivedWorkload::from_program(&program),
            Err(RnsBggError::RhsLevelMismatch { rhs: actual, expected: 1, actual: 0 })
                if actual == rhs
        ));
    }

    #[test]
    fn workload_allows_equal_plaintext_through_separate_rhs_ids() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let first_rhs = builder.rhs_input().unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(first_rhs),
            )
            .unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert_eq!(workload.rhs_level(first_rhs).unwrap(), 0);
        assert_eq!(workload.rhs_level(second_rhs).unwrap(), 1);
    }

    #[test]
    fn workload_allows_same_level_rhs_reuse() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let left = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let right = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let output = builder
            .add(crate::program::Operand::Scalar(left), crate::program::Operand::Scalar(right))
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();

        assert_eq!(
            RnsBggDerivedWorkload::from_program(&program).unwrap().rhs_level(rhs).unwrap(),
            0
        );
    }

    #[test]
    fn workload_rejects_public_mul_private_and_lut_inputs() {
        let mut public_mul_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input =
            public_mul_builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap();
        let rhs = public_mul_builder.rhs_input().unwrap();
        let output = public_mul_builder
            .mul_private(
                crate::program::Operand::Scalar(public_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let public_mul_program =
            public_mul_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&public_mul_program).is_err());

        let mut private_mul_builder = crate::program::ExponentLutProgramBuilder::new();
        let private_input =
            private_mul_builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = private_mul_builder.rhs_input().unwrap();
        let output = private_mul_builder
            .mul_private(
                crate::program::Operand::Scalar(private_input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let private_mul_program =
            private_mul_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&private_mul_program).is_ok());

        let table = LutTable::unary(
            2,
            crate::program::ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(crate::program::ExpLayout {
                stride_log2: 0,
                width_log2: 1,
            }),
        )
        .unwrap();
        let mut public_lut_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input =
            public_lut_builder.input(crate::program::AttributeKnowledge::PublicPayload).unwrap();
        let lut = public_lut_builder.register_lut(table.clone()).unwrap();
        let output =
            public_lut_builder.lut(crate::program::Operand::Scalar(public_input), lut).unwrap();
        let public_lut_program =
            public_lut_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&public_lut_program).is_err());

        let mut private_lut_builder = crate::program::ExponentLutProgramBuilder::new();
        let private_input =
            private_lut_builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let lut = private_lut_builder.register_lut(table).unwrap();
        let output =
            private_lut_builder.lut(crate::program::Operand::Scalar(private_input), lut).unwrap();
        let private_lut_program =
            private_lut_builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&private_lut_program).is_ok());
    }

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

    #[test]
    fn workload_tracks_nested_family_rhs_consumers_once() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder
            .input_family(IntExpr::constant(4), crate::program::AttributeKnowledge::PrivatePayload)
            .unwrap();
        let rhs = builder.rhs_family(IntExpr::constant(4)).unwrap();
        let outer = builder
            .parallel(FamilyRange::full(IntExpr::constant(2)).unwrap(), |builder, outer_index| {
                let inner = builder.parallel(
                    FamilyRange::full(IntExpr::constant(2)).unwrap(),
                    |builder, inner_index| {
                        let index =
                            IntExpr::Add(Box::new(outer_index.clone()), Box::new(inner_index))
                                .canonicalize();
                        builder
                            .mul_private(
                                crate::program::Operand::Member {
                                    family: input,
                                    index: index.clone(),
                                },
                                crate::program::Operand::Member { family: rhs, index },
                            )
                            .map(crate::program::Operand::Scalar)
                    },
                )?;
                builder.sum(inner).map(crate::program::Operand::Scalar)
            })
            .unwrap();
        let output = builder.sum(outer).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap();

        assert_eq!(
            RnsBggDerivedWorkload::from_program(&program).unwrap().rhs_level(rhs).unwrap(),
            0
        );
    }

    #[test]
    fn workload_distinguishes_logical_levels_when_physical_moduli_are_equal() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let first = builder
            .mul_private(
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(rhs),
            )
            .unwrap();
        let second_rhs = builder.rhs_input().unwrap();
        let second = builder
            .mul_private(
                crate::program::Operand::Scalar(first),
                crate::program::Operand::Scalar(second_rhs),
            )
            .unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(second)]).unwrap();
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters =
            RnsBggParameters::new(vec![level.clone(), level.clone(), level], 1, 1).unwrap();
        assert_eq!(parameters.scale(0).unwrap(), &BigUint::from(1u8));
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert_eq!(workload.rhs_level(rhs).unwrap(), 0);
        assert_eq!(workload.rhs_level(second_rhs).unwrap(), 1);
    }

    #[test]
    fn workload_does_not_prepare_unused_rhs_declarations() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        let workload = RnsBggDerivedWorkload::from_program(&program).unwrap();

        assert!(matches!(workload.rhs_level(rhs), Err(RnsBggError::Graph(_))));
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
            .lut_material_provenance(0, first_table, &projection.luts()[first_table.0 as usize])
            .unwrap();
        let same = evaluator
            .lut_material_provenance(0, first_table, &projection.luts()[first_table.0 as usize])
            .unwrap();
        let different = evaluator
            .lut_material_provenance(0, second_table, &projection.luts()[second_table.0 as usize])
            .unwrap();
        let different_level = evaluator
            .lut_material_provenance(1, first_table, &projection.luts()[first_table.0 as usize])
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
    fn cadence_maps_logical_levels_to_equal_modulus_windows() {
        let depths = (0..=8)
            .map(|level| crate::rns_bgg::planned_crt_depth(level, 3, 1, 6, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(depths, vec![6, 6, 6, 5, 5, 5, 4, 4, 4]);
    }

    #[test]
    fn equal_modulus_finalization_adds_no_rounding_noise() {
        assert_eq!(
            refresh_channel_mod_down_bound(&BigUint::from(17u8), &BigUint::one()).unwrap(),
            BigUint::from(17u8)
        );
    }

    #[test]
    fn public_alignment_uses_the_parameter_nested_drop() {
        let parameters = nested_alignment_parameters();
        let alignment = parameters.public_alignment(0, 1).unwrap();
        assert!(alignment.ratio() > &BigUint::one());
        assert_eq!(
            parameters
                .mul_public_target_level(0, 1, crate::program::AttributeKnowledge::PrivatePayload)
                .unwrap(),
            1
        );

        let alpha = alignment.ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let mut source = value(&[1]);
        source.bound.error = BigUint::one();
        source.moments = Some(NoiseMoments {
            mean_numerator: BigInt::zero(),
            mean_denominator: BigUint::one(),
            mean_radius: NonnegativeRational::zero(),
            variance: NonnegativeRational::new(1u8.into(), 1u8.into()).unwrap(),
        });
        let result = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &[RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into())],
                &NoiseMomentPolicy::Discrete { alpha: alpha.clone() },
            )
            .unwrap();
        assert!(result.bound.error > source.bound.error);
        assert_eq!(result.provenance.len(), 2);
        assert!(result.moments.is_some());
    }

    #[test]
    fn public_alignment_accounts_for_both_mask_row_components() {
        let parameters = nested_alignment_parameters();
        let alpha = parameters.public_alignment(0, 1).unwrap().ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let source = value(&[1]);
        let policy = NoiseMomentPolicy::Discrete { alpha: alpha.clone() };
        let component = RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into());
        let one = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                std::slice::from_ref(&component),
                &policy,
            )
            .unwrap();
        let two = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &[component.clone(), component],
                &policy,
            )
            .unwrap();
        assert!(two.bound.error > one.bound.error);
        let one_variance = &one.moments.as_ref().unwrap().variance;
        let two_variance = &two.moments.as_ref().unwrap().variance;
        assert!(
            &two_variance.numerator * &one_variance.denominator >
                &one_variance.numerator * &two_variance.denominator
        );
        assert_eq!(two.provenance, one.provenance);
    }

    #[test]
    fn public_alignment_equal_physical_ring_is_an_exact_noop() {
        let parameters = nested_alignment_parameters();
        let mut source = value(&[1]);
        let source_id = stable_source_id(&b"alignment-source".as_slice());
        source.provenance = RnsBggNoiseValue::single_provenance(NoiseSourceProvenance {
            source: source_id,
            route: vec![7],
        });
        source.heuristics.insert(NoiseHeuristicId::H4RhsIndependence);
        let result = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain { l1: BigUint::zero(), l2_squared: BigUint::zero() },
                &BigUint::one(),
                &RnsBggNoiseValue::public(BigUint::zero(), &[]),
                &RnsBggNoiseValue::public(BigUint::zero(), &[]),
                &[],
                &NoiseMomentPolicy::Continuous,
            )
            .unwrap();
        assert_eq!(result.bound, source.bound);
        assert_eq!(result.provenance, source.provenance);
        assert_eq!(result.heuristics, source.heuristics);
    }

    #[test]
    fn public_alignment_reuses_rounding_sources_across_repeated_routes() {
        let parameters = nested_alignment_parameters();
        let alpha = parameters.public_alignment(0, 1).unwrap().ratio().clone();
        let ciphertext = alignment_rounding(&alpha, 0, 0, 1, b"ciphertext");
        let key = alignment_rounding(&alpha, 0, 0, 1, b"key");
        let source = value(&[1]);
        let policy = NoiseMomentPolicy::Discrete { alpha: alpha.clone() };
        let mask = [RnsBggMaskSecretComponent::deterministic(1u8.into(), 1u8.into())];
        let first = parameters
            .public_alignment_noise_transfer(
                &source,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &mask,
                &policy,
            )
            .unwrap();
        let second = parameters
            .public_alignment_noise_transfer(
                &first,
                &RightActionNoiseGain::unit(),
                &alpha,
                &ciphertext,
                &key,
                &mask,
                &policy,
            )
            .unwrap();
        assert_eq!(second.provenance.len(), first.provenance.len());
        for source in ["ciphertext", "key"] {
            let id = stable_source_id(&(
                b"rns-bgg/public-alignment/rounding".as_slice(),
                source.as_bytes(),
                0usize,
                0usize,
                1usize,
            ));
            assert_eq!(second.provenance[&id].route.len(), 6);
        }
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

    fn identity_program() -> (ExponentLutProgram, RnsBggParameters) {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let program = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        (program, nested_alignment_parameters())
    }

    #[test]
    fn average_case_can_accept_a_candidate_rejected_by_worst_case() {
        let (program, parameters) = identity_program();
        let mut input =
            RnsBggNoiseValue::sampled_encoding(&parameters, b"average-input", &[1.into()]);
        input.bound.error = parameters.level(0).unwrap().modulus().as_ref().clone() * 2u8;
        let bindings = RnsBggNoiseBindings::from_inputs(&program, &[input], &[]).unwrap();
        let terminal = [BigUint::one()];
        assert!(parameters.audit_program(&program, &bindings, &terminal, None).is_err());
        let average = crate::noise_refresh::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        let trace =
            parameters.audit_program(&program, &bindings, &terminal, Some(&average)).unwrap();
        assert_eq!(trace.acceptance, Some(NoiseModelKind::AverageCase));
    }

    #[test]
    fn average_case_rejects_missing_input_moments() {
        let (program, parameters) = identity_program();
        let bindings = RnsBggNoiseBindings::from_inputs(
            &program,
            &[RnsBggNoiseValue::public(BigUint::one(), &[1.into()])],
            &[],
        )
        .unwrap();
        let average = crate::noise_refresh::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        };
        assert!(matches!(
            parameters.audit_program(&program, &bindings, &[BigUint::one()], Some(&average)),
            Err(RnsBggError::Parameters("average-case input moment model is missing"))
        ));
    }

    #[test]
    fn average_event_budget_deduplicates_nodes_and_applies_tail_correction() {
        let parameters = nested_alignment_parameters();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let program = builder
            .finish(vec![
                crate::program::Operand::Scalar(input),
                crate::program::Operand::Scalar(input),
            ])
            .unwrap();
        let input = RnsBggNoiseValue::sampled_encoding(&parameters, b"budget-input", &[1.into()]);
        let bindings = RnsBggNoiseBindings::from_inputs(&program, &[input], &[]).unwrap();
        let trace = trace_program(&parameters, &program, &bindings).unwrap();
        let events = average_event_budget(&trace).unwrap();
        let expected = BigUint::from(parameters.level(0).unwrap().ring_dimension()) *
            2u8 *
            BigUint::from(parameters.component_columns(0).unwrap());
        assert_eq!(events, expected);
        let base = crate::noise_refresh::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 0,
            allow_average_acceptance: true,
        }
        .z_squared_for_events(&events)
        .unwrap();
        let corrected = crate::noise_refresh::simulation::AverageCaseConfig {
            failure_exponent: 1,
            tail_correction_bits: 2,
            allow_average_acceptance: true,
        }
        .z_squared_for_events(&events)
        .unwrap()
        .scaled(&(BigUint::one() << 4));
        assert!(corrected.numerator > base.numerator);
    }

    #[test]
    fn correctness_uses_q_over_four_at_terminal_level() {
        let parameters = nested_alignment_parameters();
        let q = parameters.level(1).unwrap().modulus().as_ref().clone();
        let below = (&q - 1u8) / 4u8;
        let above = &q / 4u8 + 1u8;
        assert!(
            parameters
                .noise_fits(1, &RnsBggNoiseBound { error: below, payload_l1: BigUint::zero() })
                .unwrap()
        );
        assert!(
            !parameters
                .noise_fits(1, &RnsBggNoiseBound { error: above, payload_l1: BigUint::zero() })
                .unwrap()
        );
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
        // The aggregated pre-rounding noise is scaled and rounded once.
        let expected = parameters.lut_mod_down_bound(0, &value(&[1]).bound.error).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].bound.error, expected);
        assert_eq!(outputs[0].bound, outputs[1].bound);
        assert_eq!(evaluator.trace.joint_event_count, BigUint::one());
        assert_eq!(outputs[0].provenance.len(), 1);
        assert_eq!(outputs[0].provenance, outputs[1].provenance);
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
        assert_eq!(refresh_crt_weight(&[3u8.into(), 5u8.into()], 0).unwrap(), BigUint::from(10u8));
        assert_eq!(refresh_crt_weight(&[3u8.into(), 5u8.into()], 1).unwrap(), BigUint::from(6u8));

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
                let mask_divisor =
                    gcd_biguint(mask_mean_numerator.magnitude(), &mask_mean_denominator);
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
                let fresh_divisor =
                    gcd_biguint(fresh_mean_numerator.magnitude(), &fresh_mean_denominator);
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

// These assertions were coupled to the pre-lowering bindings API
// (family input bindings, RHS values, and explicit gate positions).  The current
// evaluator keeps those details private to the lowering context.
