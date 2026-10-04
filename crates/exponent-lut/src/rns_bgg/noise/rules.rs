//! Per-operation noise transfer rules of the RNS-BGG parameters.
use super::*;

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
        let companion = &j * self.companion_noise_gain(level)? * self.encoding_error_bound() * 2u8;
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + companion + 1u8) / 2u8,
            payload_l1: output_payload_l1,
        })
    }

    /// Coefficient gain of the Dynamic digit-companion error
    /// `max_j sum_a eps_{a,j} U_{:,a}` of one product or LUT branch at
    /// `level` (spec Eq. 075): `m_i` digit rows, each acting through a column
    /// of `M_i = 2 D_i` key-decomposition digits with `n` coefficients
    /// bounded by `b/2`, so `m_i M_i n b / 2`. Static products have no
    /// companions and a zero gain.
    fn companion_noise_gain(&self, level: usize) -> Result<BigUint, RnsBggError> {
        if self.rhs_mode() == RhsMode::Static {
            return Ok(BigUint::zero());
        }
        let current = self.level(level)?;
        let m = BigUint::from(self.component_columns(level)?) * 2u8;
        let public_columns = BigUint::from(current.modulus_digits()) * 2u8;
        Ok(
            m * public_columns * current.ring_dimension() * (BigUint::one() << current.base_bits()) /
                2u8,
        )
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

    /// Check the deterministic q/4 correctness threshold.
    ///
    /// The noise need not stay below half of the level's drop: the message is
    /// part of the gadget term `p_level x r G`, which ModDown divides exactly,
    /// so noise above `p_level / 2` only remains as noise at the next level,
    /// where [`Self::lut_mod_down_bound`] and the inherited LUT term carry it.
    pub fn noise_fits(&self, level: usize, noise: &RnsBggNoiseBound) -> Result<bool, RnsBggError> {
        let q = self.level(level)?.modulus();
        Ok(q.as_ref() > &(&noise.error * 4u8))
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
    pub(super) fn mul_public_action_gain(
        &self,
        level: usize,
    ) -> Result<RightActionNoiseGain, RnsBggError> {
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
        let companion =
            self.companion_noise_variance(level)?.scaled_ratio(&BigUint::one(), &p_squared);
        Ok(PrivateMulNoiseTerms {
            inherited_variance_gain,
            inherited_mean_gain,
            rhs_error,
            rounding,
            companion,
        })
    }

    /// Diagnostic variance of the Dynamic digit-companion error
    /// `sum_a eps_{a,j} U_{:,a}` of one product or LUT branch at `level`,
    /// before its scale-and-round: the `m_i` fresh companion errors act
    /// through key-decomposition columns of energy at most `M_i n (b/2)^2`.
    /// Acceptance of Dynamic parameters uses the deterministic bound only.
    fn companion_noise_variance(&self, level: usize) -> Result<NonnegativeRational, RnsBggError> {
        let base = BigUint::one() << self.level(level)?.base_bits();
        Ok(self
            .encoding_error_variance()
            .scaled_ratio(&(self.companion_noise_gain(level)? * base), &BigUint::from(2u8)))
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
        Ok(terms
            .inherited_variance(input_variance)
            .add(&terms.rhs_error)
            .add(&terms.rounding)
            .add(&terms.companion))
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
            companion: self.companion_noise_variance(level)?,
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
        // Dynamic: the digit companions' error, divided once by p.
        let companion = self.companion_noise_gain(level)? * &self.encoding_error_bound * 2u8;
        Ok(RnsBggNoiseBound {
            error: (inherited + fresh + rounding + companion + &twice_p - 1u8) / twice_p,
            payload_l1: output_payload_l1,
        })
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
    /// Variance of the Dynamic digit-companion error; zero when Static.
    pub companion: NonnegativeRational,
}

impl PrivateMulNoiseTerms {
    pub(super) fn inherited_variance(&self, input: &NonnegativeRational) -> NonnegativeRational {
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
    /// Variance of one branch's Dynamic digit-companion error image; zero
    /// when Static.
    pub companion: NonnegativeRational,
    /// Coherent variance of the branches' internal public remainders.
    pub internal_rounding: NonnegativeRational,
    /// The scale `p` of the scale-and-round.
    pub scale: BigUint,
    /// Variance of the outer public and encoding remainders, output units.
    pub outer_rounding: NonnegativeRational,
}

#[cfg(test)]
mod test {
    use super::{
        super::test::{nested_alignment_parameters, value},
        *,
    };

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

    /// The Dynamic product and LUT bounds are the Static recurrences plus the
    /// digit-companion term `m_i M_i n b E / 2` per product or branch before
    /// the single division by `p` (spec Eq. 075, 094-095).
    #[test]
    fn dynamic_bounds_add_the_companion_term_to_the_static_recurrence() {
        let parameters = nested_alignment_parameters();
        let dynamic = parameters.clone().with_rhs_mode(RhsMode::Dynamic);
        let input = RnsBggNoiseBound { error: BigUint::from(1000u16), payload_l1: BigUint::one() };
        let gain = BigUint::from(3u8);
        let level = 0;
        let current = parameters.level(level).unwrap();
        let n = BigUint::from(current.ring_dimension());
        let b = BigUint::one() << current.base_bits();
        let m = BigUint::from(2 * parameters.component_columns(level).unwrap());
        let public_columns = BigUint::from(2 * current.modulus_digits());
        let p = parameters.scale(level).unwrap().clone();
        let e = parameters.encoding_error_bound().clone();
        let numerator = &m * &n * &b * &input.error +
            &m * &public_columns * &n * &b * &e +
            &p * 2u8 * &input.payload_l1 * &gain * &e +
            (&n + 2u8) * (&p - 1u8);
        let twice_p = &p * 2u8;
        assert_eq!(
            dynamic.private_mul_noise_bound(level, &input, &gain, BigUint::one()).unwrap().error,
            (numerator + &twice_p - 1u8) / twice_p
        );
        assert!(
            parameters.private_mul_noise_bound(level, &input, &gain, BigUint::one()).unwrap().error <
                dynamic
                    .private_mul_noise_bound(level, &input, &gain, BigUint::one())
                    .unwrap()
                    .error
        );
        let branches = 4;
        let static_lut = parameters
            .lut_noise_bound_before_mod_down(level, &input, branches, &gain, BigUint::one())
            .unwrap()
            .error;
        let dynamic_lut = dynamic
            .lut_noise_bound_before_mod_down(level, &input, branches, &gain, BigUint::one())
            .unwrap()
            .error;
        assert_eq!(
            dynamic_lut - static_lut,
            BigUint::from(branches) * &m * &public_columns * &n * &b * &e / 2u8
        );
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
    fn intermediate_noise_may_exceed_half_the_drop() {
        let parameters = nested_alignment_parameters();
        let q = parameters.level(0).unwrap().modulus().as_ref().clone();
        let drop = parameters.scale(0).unwrap().clone();
        assert!(&drop * 4u8 < q);
        assert!(
            parameters
                .noise_fits(0, &RnsBggNoiseBound { error: drop, payload_l1: BigUint::zero() })
                .unwrap()
        );
    }
}
